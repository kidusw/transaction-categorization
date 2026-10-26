"""FastAPI service: categorize transactions, account insights, and the review queue.

Run locally with `uv run uvicorn txcat.api.app:app --reload` (needs Postgres migrated to
head and a registered model, see README). No authentication yet: an API-key header is
planned before any public deployment.
"""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from txcat.api.predictor import Predictor, load_predictor
from txcat.api.schemas import (
    AccountOut,
    AnomalyOut,
    CategorizeRequest,
    CategorizeResponse,
    HealthResponse,
    InsightsResponse,
    ModelInfo,
    MonthlySpendOut,
    Prediction,
    RecurringOut,
    ReviewDecision,
    ReviewItemOut,
)
from txcat.config import get_settings
from txcat.data.clean import clean_descriptions
from txcat.db.models import Account, Category, ReviewItem, Transaction
from txcat.db.session import get_engine
from txcat.insights.anomaly import score_anomalies
from txcat.insights.recurring import detect_recurring
from txcat.insights.spend import monthly_by_category


def create_app(predictor: Predictor | None = None, engine: Engine | None = None) -> FastAPI:
    """App factory. Tests inject a predictor and an engine; production loads both."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        app.state.engine = engine or get_engine()
        app.state.predictor = predictor or load_predictor(
            settings.model_uri, settings.mlflow_tracking_uri
        )
        app.state.sessions = sessionmaker(bind=app.state.engine, expire_on_commit=False)
        yield
        if engine is None:
            app.state.engine.dispose()

    app = FastAPI(
        title="txcat",
        summary="Bank-transaction categorization and insights",
        version="0.1.0",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _session(request: Request) -> Iterator[Session]:
    session = request.app.state.sessions()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


SessionDep = Annotated[Session, Depends(_session)]
PredictorDep = Annotated[Predictor, Depends(_predictor)]


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        predictor: Predictor | None = getattr(request.app.state, "predictor", None)
        try:
            with request.app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            database = "unavailable"
        return HealthResponse(
            status="ok" if predictor and database == "ok" else "degraded",
            model_name=predictor.name if predictor else None,
            model_version=predictor.version if predictor else None,
            threshold=predictor.threshold if predictor else None,
            database=database,
        )

    @app.get("/model", response_model=ModelInfo, tags=["ops"])
    def model_info(predictor: PredictorDep) -> ModelInfo:
        """The served model, its review threshold and the training-time model comparison."""
        return ModelInfo(
            model_name=predictor.name,
            model_version=predictor.version,
            threshold=predictor.threshold,
            categories=[str(c) for c in predictor.model.classes_],
            comparison=predictor.comparison,
        )

    @app.get("/accounts", response_model=list[AccountOut], tags=["insights"])
    def list_accounts(session: SessionDep) -> list[AccountOut]:
        """Accounts with stored transactions (for the dashboard's account picker)."""
        query = (
            select(
                Account.external_id,
                func.count(Transaction.id),
                func.min(Transaction.txn_date),
                func.max(Transaction.txn_date),
            )
            .join(Transaction, Transaction.account_id == Account.id)
            .group_by(Account.external_id)
            .order_by(Account.external_id)
        )
        return [
            AccountOut(account_id=a, transactions=n, first_date=first, last_date=last)
            for a, n, first, last in session.execute(query)
        ]

    @app.post("/categorize", response_model=CategorizeResponse, tags=["categorize"])
    def categorize(
        body: CategorizeRequest,
        predictor: PredictorDep,
        session: SessionDep,
    ) -> CategorizeResponse:
        """Categorize a batch. Predictions below the review threshold are queued for review."""
        rows = body.transactions
        result = predictor.predict([r.description for r in rows], [r.amount for r in rows])
        predictions = []
        for row, pred in zip(rows, result.itertuples(index=False), strict=True):
            review_id = None
            if pred.needs_review:
                item = ReviewItem(
                    account_external_id=row.account_id,
                    txn_date=row.date,
                    description=row.description,
                    description_clean=pred.description_clean,
                    amount=None if row.amount is None else Decimal(str(row.amount)),
                    currency=row.currency,
                    merchant=pred.merchant,
                    predicted_category=pred.category,
                    confidence=float(pred.confidence),
                    model_version=predictor.version,
                    status="pending",
                )
                session.add(item)
                session.flush()
                review_id = item.id
            predictions.append(
                Prediction(
                    description=row.description,
                    description_clean=pred.description_clean,
                    category=pred.category,
                    confidence=float(pred.confidence),
                    merchant=pred.merchant,
                    needs_review=bool(pred.needs_review),
                    review_id=review_id,
                )
            )
        return CategorizeResponse(
            model_version=predictor.version, threshold=predictor.threshold, predictions=predictions
        )

    @app.get("/review", response_model=list[ReviewItemOut], tags=["review"])
    def list_review(
        session: SessionDep,
        status: Annotated[str, Query(pattern="^(pending|reviewed|all)$")] = "pending",
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[ReviewItem]:
        """Review-queue items, oldest first (the order a reviewer should work through them)."""
        query = select(ReviewItem).order_by(ReviewItem.created_at, ReviewItem.id).limit(limit)
        if status != "all":
            query = query.where(ReviewItem.status == status)
        return list(session.scalars(query))

    @app.post("/review/{item_id}", response_model=ReviewItemOut, tags=["review"])
    def review(item_id: int, decision: ReviewDecision, session: SessionDep) -> ReviewItem:
        """Record the correct category. Re-reviewing overwrites the earlier decision."""
        item = session.get(ReviewItem, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"review item {item_id} not found")
        item.reviewed_category = decision.category
        item.status = "reviewed"
        item.reviewed_at = datetime.now(UTC)
        session.flush()
        return item

    @app.get("/accounts/{account_id}/insights", response_model=InsightsResponse, tags=["insights"])
    def insights(
        account_id: str,
        predictor: PredictorDep,
        session: SessionDep,
        months: Annotated[int, Query(ge=1, le=60, description="Months of spend to return.")] = 6,
        anomalies: Annotated[int, Query(ge=0, le=200, description="Recent anomalies.")] = 10,
    ) -> InsightsResponse:
        """Recurring payments, monthly spend by category and recent anomalies for an account."""
        df = _account_history(session, account_id)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"no transactions for {account_id!r}")
        df["merchant"] = predictor.normalizer.normalize(clean_descriptions(df["description"]))

        recurring = detect_recurring(df).sort_values("typical_amount", ascending=False)
        monthly = monthly_by_category(df)
        recent_months = sorted(monthly["month"].unique())[-months:]
        monthly = monthly[monthly["month"].isin(recent_months)]
        scored = score_anomalies(df)
        flagged = scored[scored["is_anomaly"]].sort_values("date").tail(anomalies)
        return InsightsResponse(
            account_id=account_id,
            transactions=len(df),
            first_date=df["date"].min().date(),
            last_date=df["date"].max().date(),
            recurring=[
                RecurringOut(
                    **{k: r[k] for k in RecurringOut.model_fields if k not in _DATES},
                    **{k: r[k].date() for k in _DATES},
                )
                for r in recurring.to_dict("records")
            ],
            monthly_spend=[
                MonthlySpendOut(**{**r, "month": str(r["month"])})
                for r in monthly.drop(columns="account_id").to_dict("records")
            ],
            anomalies=[
                AnomalyOut(
                    date=r["date"].date(),
                    description=r["description"],
                    amount=r["amount"],
                    category=r["category"],
                    anomaly_z=r["anomaly_z"],
                )
                for r in flagged.to_dict("records")
            ],
        )


_DATES = ("last_seen", "next_expected")


def _account_history(session: Session, account_id: str) -> pd.DataFrame:
    query = (
        select(
            Transaction.txn_date.label("date"),
            Transaction.description,
            Transaction.amount,
            Category.name.label("category"),
        )
        .join(Account, Transaction.account_id == Account.id)
        .outerjoin(Category, Transaction.category_id == Category.id)
        .where(Account.external_id == account_id)
        .order_by(Transaction.txn_date, Transaction.id)
    )
    df = pd.DataFrame(
        session.execute(query).all(), columns=["date", "description", "amount", "category"]
    )
    if df.empty:
        return df
    return df.assign(
        account_id=account_id,
        date=pd.to_datetime(df["date"]),
        amount=df["amount"].astype(float),
        category=df["category"].fillna("Uncategorized"),
    )


app = create_app()

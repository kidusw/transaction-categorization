import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from txcat.api.app import create_app
from txcat.api.predictor import METADATA_FILE, Predictor, load_predictor
from txcat.data.synthetic import SyntheticGenerator
from txcat.db import Base
from txcat.ingest import load_transactions, validate_transactions
from txcat.models.serialization import SKOPS_TRUSTED_TYPES
from txcat.models.train import _tfidf_logreg

TRAIN = [
    ("shoa supermarket", -800.0, "Groceries"),
    ("queens supermarket bole", -650.0, "Groceries"),
    ("fantu supermarket", -900.0, "Groceries"),
    ("telebirr airtime recharge", -100.0, "Telecom & Airtime"),
    ("ethio telecom package", -250.0, "Telecom & Airtime"),
    ("safaricom airtime", -50.0, "Telecom & Airtime"),
    ("salary acme plc", 20000.0, "Income"),
    ("acme salary oct", 21000.0, "Income"),
    ("payroll acme", 20500.0, "Income"),
] * 3


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    X = pd.DataFrame({"text": [t for t, _, _ in TRAIN], "amount": [a for _, a, _ in TRAIN]})
    model = _tfidf_logreg(C=4.0).fit(X, [c for _, _, c in TRAIN])
    # High threshold so ambiguous inputs are queued and the review flow can be tested.
    return Predictor(model, threshold=0.6, version="txcat-categorizer/test")


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(predictor, engine):
    with TestClient(create_app(predictor=predictor, engine=engine)) as client:
        yield client


def test_health(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_name": "txcat-categorizer",
        "model_version": "txcat-categorizer/test",
        "threshold": 0.6,
        "database": "ok",
    }


def test_categorize_batch(client):
    response = client.post(
        "/categorize",
        json={
            "transactions": [
                {"description": "SHOA SUPERMARKET FT24123ABCD", "amount": -720},
                {"description": "ACME PLC SALARY", "amount": 20000, "account_id": "SYN-0001"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "txcat-categorizer/test"
    first, second = body["predictions"]
    assert first["description_clean"] == "shoa supermarket"
    assert first["category"] == "Groceries"
    assert first["merchant"] == "Shoa Supermarket"
    assert 0 <= first["confidence"] <= 1
    assert second["category"] == "Income"
    for p in body["predictions"]:
        assert p["needs_review"] == (p["confidence"] < 0.6)
        assert (p["review_id"] is not None) == p["needs_review"]


def test_low_confidence_goes_to_review_and_can_be_corrected(client):
    pred = client.post(
        "/categorize",
        json={"transactions": [{"description": "zzz unknown thing", "date": "2025-01-02"}]},
    ).json()["predictions"][0]
    assert pred["needs_review"] and pred["review_id"]

    queue = client.get("/review").json()
    assert [item["id"] for item in queue] == [pred["review_id"]]
    assert queue[0]["status"] == "pending"
    assert queue[0]["txn_date"] == "2025-01-02"

    reviewed = client.post(f"/review/{pred['review_id']}", json={"category": "Shopping"})
    assert reviewed.status_code == 200
    assert reviewed.json()["reviewed_category"] == "Shopping"
    assert reviewed.json()["reviewed_at"]

    assert client.get("/review").json() == []
    assert len(client.get("/review", params={"status": "reviewed"}).json()) == 1


def test_review_rejects_unknown_category_and_missing_item(client):
    assert client.post("/review/999", json={"category": "Groceries"}).status_code == 404
    pred = client.post(
        "/categorize", json={"transactions": [{"description": "zzz unknown"}]}
    ).json()["predictions"][0]
    bad = client.post(f"/review/{pred['review_id']}", json={"category": "Not A Category"})
    assert bad.status_code == 422


def test_categorize_validates_input(client):
    assert client.post("/categorize", json={"transactions": []}).status_code == 422
    assert client.post("/categorize", json={"transactions": [{"amount": 1}]}).status_code == 422
    too_many = {"transactions": [{"description": "x"}] * 1001}
    assert client.post("/categorize", json=too_many).status_code == 422


def _seed(engine, n_accounts: int = 1) -> pd.DataFrame:
    df = SyntheticGenerator(3).generate(n_accounts=n_accounts)
    valid = validate_transactions(
        df[
            ["account_id", "date", "description", "amount", "currency", "category", "source"]
        ].astype({"date": str, "amount": str})
    ).valid
    with Session(engine) as session:
        load_transactions(session, valid)
        session.commit()
    return valid


def test_list_accounts(client, engine):
    assert client.get("/accounts").json() == []
    valid = _seed(engine, n_accounts=2)
    accounts = client.get("/accounts").json()
    assert [a["account_id"] for a in accounts] == ["SYN-0001", "SYN-0002"]
    assert sum(a["transactions"] for a in accounts) == len(valid)


def test_model_info(client, predictor):
    body = client.get("/model").json()
    assert body["model_version"] == "txcat-categorizer/test"
    assert body["threshold"] == 0.6
    assert set(body["categories"]) == {"Groceries", "Telecom & Airtime", "Income"}
    assert body["comparison"] == []


def test_dashboard_client_against_api(client, engine):
    from txcat.dashboard.client import ApiClient, ApiError

    api = ApiClient(http=client)
    _seed(engine)
    assert api.health()["status"] == "ok"
    assert api.accounts()[0]["account_id"] == "SYN-0001"
    assert api.insights("SYN-0001", months=3)["recurring"]
    pred = api.categorize([{"description": "zzz unknown", "amount": -5}])["predictions"][0]
    assert api.review_queue()[0]["id"] == pred["review_id"]
    assert api.review(pred["review_id"], "Shopping")["status"] == "reviewed"
    with pytest.raises(ApiError, match="404"):
        api.insights("NOPE")


def test_dashboard_client_unreachable():
    from txcat.dashboard.client import ApiClient, ApiError

    with pytest.raises(ApiError, match="cannot reach"):
        ApiClient("http://127.0.0.1:9", timeout=1).health()


def test_account_insights(client, engine):
    valid = _seed(engine)

    response = client.get("/accounts/SYN-0001/insights", params={"months": 2, "anomalies": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["transactions"] == len(valid)
    counterparties = {r["counterparty"] for r in body["recurring"]}
    assert {"Income", "Ethiopian Electric Utility"} <= counterparties
    assert len({m["month"] for m in body["monthly_spend"]}) == 2
    assert len(body["anomalies"]) <= 3

    assert client.get("/accounts/NOPE/insights").status_code == 404


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/categorize", "/review", "/review/{item_id}"} <= set(paths)
    assert "/accounts/{account_id}/insights" in paths


def test_load_predictor_from_exported_directory(tmp_path, predictor):
    mlflow = pytest.importorskip("mlflow")
    out = tmp_path / "champion"
    mlflow.sklearn.save_model(predictor.model, out, skops_trusted_types=SKOPS_TRUSTED_TYPES)
    (out / METADATA_FILE).write_text(
        json.dumps(
            {"name": "txcat-categorizer", "version": "txcat-categorizer/7", "threshold": 0.4}
        )
    )
    loaded = load_predictor(str(out))  # skops only, no MLflow
    assert (loaded.version, loaded.threshold) == ("txcat-categorizer/7", 0.4)
    assert loaded.predict(["shoa supermarket"], [-10.0])["category"].iat[0] == "Groceries"

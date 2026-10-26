"""Request/response models for the HTTP API."""

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from txcat.data.taxonomy import CATEGORIES

MAX_BATCH = 1000


class TransactionIn(BaseModel):
    description: str = Field(min_length=1, max_length=500, examples=["CBE transfer FT24123ABCD"])
    amount: float | None = Field(None, description="Signed: negative = money out.")
    account_id: str | None = None
    date: dt.date | None = None
    currency: str | None = Field(None, min_length=3, max_length=3)


class CategorizeRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    description: str
    description_clean: str
    category: str
    confidence: float
    merchant: str | None
    needs_review: bool
    review_id: int | None = Field(None, description="Set when the row was queued for review.")


class CategorizeResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ReviewItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    account_external_id: str | None
    txn_date: dt.date | None
    description: str
    amount: float | None
    merchant: str | None
    predicted_category: str
    confidence: float
    model_version: str
    status: Literal["pending", "reviewed"]
    reviewed_category: str | None
    reviewed_at: dt.datetime | None
    created_at: dt.datetime | None


class ReviewDecision(BaseModel):
    category: str = Field(examples=["Groceries"])

    @field_validator("category")
    @classmethod
    def known_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"unknown category {value!r}; expected one of {sorted(CATEGORIES)}")
        return value


class RecurringOut(BaseModel):
    counterparty: str
    direction: Literal["in", "out"]
    category: str
    period: str
    occurrences: int
    typical_amount: float
    last_seen: dt.date
    next_expected: dt.date


class MonthlySpendOut(BaseModel):
    month: str
    category: str
    spend: float
    income: float
    transactions: int


class AnomalyOut(BaseModel):
    date: dt.date
    description: str
    amount: float
    category: str
    anomaly_z: float


class InsightsResponse(BaseModel):
    account_id: str
    transactions: int
    first_date: dt.date
    last_date: dt.date
    recurring: list[RecurringOut]
    monthly_spend: list[MonthlySpendOut]
    anomalies: list[AnomalyOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_name: str | None
    model_version: str | None
    threshold: float | None
    database: Literal["ok", "unavailable"]


class AccountOut(BaseModel):
    account_id: str
    transactions: int
    first_date: dt.date
    last_date: dt.date


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    categories: list[str]
    comparison: list[dict] = Field(
        description="Phase 3 model comparison on the future test slice (from training)."
    )

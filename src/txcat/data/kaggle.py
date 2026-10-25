"""Download the Kaggle source datasets and map them onto the unified taxonomy."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from txcat.config import get_settings
from txcat.data.taxonomy import PERSONAL_FINANCE_MAP, SEMANTIC_LEXICON_MAP


@dataclass(frozen=True)
class KaggleDataset:
    slug: str
    filename: str

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/datasets/{self.slug}"

    def local_dir(self, raw_dir: Path) -> Path:
        return raw_dir / "kaggle" / self.slug.replace("/", "__")

    def local_path(self, raw_dir: Path) -> Path:
        return self.local_dir(raw_dir) / self.filename


# Real-format personal transactions (dates, amounts, labelled categories).
PERSONAL_FINANCE = KaggleDataset("bukolafatunde/personal-finance", "personal_transactions.csv")
# Merchant/keyword phrases with categories (no dates/amounts): extra text for training.
SEMANTIC_LEXICON = KaggleDataset(
    "lowlevelexe/malaysian-semantic-transaction-dataset", "semantic_transaction_dataset.csv"
)
DATASETS = (PERSONAL_FINANCE, SEMANTIC_LEXICON)


class KaggleDownloadError(RuntimeError):
    pass


def download(raw_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download every source dataset into `raw_dir/kaggle/<owner>__<name>/`."""
    token = get_settings().kaggle_api_token
    if token is None:
        raise KaggleDownloadError("KAGGLE_API_TOKEN is not set (add it to .env)")
    os.environ["KAGGLE_API_TOKEN"] = token.get_secret_value()
    from kaggle.api.kaggle_api_extended import KaggleApi  # authenticates on import

    api = KaggleApi()
    api.authenticate()
    paths = {}
    for ds in DATASETS:
        if ds.local_path(raw_dir).exists() and not force:
            paths[ds.slug] = ds.local_path(raw_dir)
            continue
        ds.local_dir(raw_dir).mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_files(ds.slug, path=ds.local_dir(raw_dir), unzip=True, quiet=True)
        except Exception as exc:
            raise KaggleDownloadError(
                f"could not download {ds.slug} ({exc}). If access is forbidden, open "
                f"{ds.url} while signed in, accept the dataset's terms, and retry."
            ) from exc
        paths[ds.slug] = ds.local_path(raw_dir)
    return paths


def _map_categories(raw: pd.Series, mapping: dict[str, str], dataset: str) -> pd.Series:
    unmapped = sorted(set(raw.dropna()) - set(mapping))
    if unmapped:
        raise ValueError(f"{dataset}: no taxonomy mapping for {unmapped}")
    return raw.map(mapping)


def load_personal_finance(path: Path) -> pd.DataFrame:
    """US personal-finance export -> canonical transaction frame (signed amounts)."""
    raw = pd.read_csv(path)
    accounts = raw["Account Name"].str.lower().str.replace(r"\W+", "-", regex=True)
    sign = raw["Transaction Type"].str.lower().map({"debit": -1, "credit": 1})
    return pd.DataFrame(
        {
            "account_id": "KPF-" + accounts,
            "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y"),
            "description": raw["Description"].str.strip(),
            "amount": (raw["Amount"] * sign).round(2),
            "currency": "USD",
            "category_raw": raw["Category"],
            "category": _map_categories(raw["Category"], PERSONAL_FINANCE_MAP, "personal"),
            "merchant_true": None,
            "is_injected_anomaly": False,
            "source": "kaggle",
            "dataset": "kaggle_personal_finance",
        }
    )


def load_semantic_lexicon(path: Path) -> pd.DataFrame:
    """Merchant/keyword phrases -> (text, category) pairs for classifier training."""
    raw = pd.read_csv(path).dropna(subset=["entity", "category"])
    return pd.DataFrame(
        {
            "text": raw["entity"].str.strip(),
            "entity_type": raw["type"],
            "category_raw": raw["category"],
            "category": _map_categories(raw["category"], SEMANTIC_LEXICON_MAP, "lexicon"),
            "source": "kaggle",
            "dataset": "kaggle_semantic_lexicon",
        }
    ).drop_duplicates(subset=["text", "category"])

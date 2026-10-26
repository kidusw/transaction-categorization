"""Loads the served model and turns raw transactions into categories + confidence.

Two sources:
- the MLflow registry (`models:/txcat-categorizer@champion`, threshold as a version tag),
  for local development; needs the `train` dependency group (MLflow).
- a directory written by `txcat export-model`: the MLflow model files plus a
  `txcat_model.json` sidecar (version, threshold, model comparison). This is what the
  Docker image serves. It is loaded with skops directly, so the API image doesn't need
  MLflow at all.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.clean import clean_descriptions
from txcat.data.merchants import MerchantNormalizer
from txcat.models.features import model_frame
from txcat.models.serialization import SKOPS_TRUSTED_TYPES

METADATA_FILE = "txcat_model.json"
SKOPS_FILE = "model.skops"
COMPARISON_REPORT = Path("reports/model_comparison.json")


@dataclass
class Predictor:
    model: object
    threshold: float
    version: str
    name: str = "txcat-categorizer"
    comparison: list[dict] = field(default_factory=list)  # Phase 3 results, for the dashboard

    def __post_init__(self):
        self.normalizer = MerchantNormalizer()

    def predict(self, descriptions: list[str], amounts: list[float | None]) -> pd.DataFrame:
        """category, confidence, merchant, needs_review and description_clean per input."""
        clean = clean_descriptions(pd.Series(descriptions, dtype=str))
        frame = model_frame(
            pd.DataFrame({"description_clean": clean, "amount": pd.Series(amounts, dtype=float)})
        )
        proba = self.model.predict_proba(frame)
        best = proba.argmax(axis=1)
        confidence = proba[np.arange(len(best)), best]
        merchants = self.normalizer.match_many(clean.tolist())
        return pd.DataFrame(
            {
                "description_clean": clean,
                "category": np.asarray(self.model.classes_)[best],
                "confidence": confidence.round(4),
                # object dtype keeps None as None (a string column would turn it into NaN)
                "merchant": pd.Series([m.name if m else None for m in merchants], dtype=object),
                "needs_review": confidence < self.threshold,
            }
        )


def _read_comparison(path: Path = COMPARISON_REPORT) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_predictor(model_uri: str, tracking_uri: str | None = None) -> Predictor:
    if not model_uri.startswith("models:/"):
        return _load_exported(Path(model_uri))

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    name, _, alias = model_uri.removeprefix("models:/").partition("@")
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    return Predictor(
        mlflow.sklearn.load_model(model_uri),
        threshold=float(version.tags.get("review_threshold", 0.5)),
        version=f"{name}/{version.version}",
        name=name,
        comparison=_read_comparison(),
    )


def _load_exported(directory: Path) -> Predictor:
    import skops.io as sio

    meta = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    model_file = directory / SKOPS_FILE
    unknown = set(sio.get_untrusted_types(file=model_file)) - set(SKOPS_TRUSTED_TYPES)
    if unknown:
        raise ValueError(f"{model_file} contains types outside the allow-list: {sorted(unknown)}")
    model = sio.load(model_file, trusted=sio.get_untrusted_types(file=model_file))
    return Predictor(
        model,
        threshold=float(meta["threshold"]),
        version=meta["version"],
        name=meta["name"],
        comparison=meta.get("comparison", []),
    )


def export_predictor(model_uri: str, out_dir: Path, tracking_uri: str | None = None) -> Path:
    """Copy the registry model + its metadata into a self-contained directory."""
    import mlflow

    predictor = load_predictor(model_uri, tracking_uri)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(out_dir))
    meta = {
        "name": predictor.name,
        "version": predictor.version,
        "threshold": predictor.threshold,
        "comparison": predictor.comparison,
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir

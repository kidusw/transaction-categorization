"""Loads the served model and turns raw transactions into categories + confidence.

The model comes from the MLflow registry (`models:/txcat-categorizer@champion`, with the
review threshold stored as a version tag) or from a directory written by
`txcat export-model`, which holds the MLflow model plus a `txcat_model.json` sidecar with
the same metadata. The directory form is what the Docker image uses: the local MLflow
store records absolute host paths that don't exist inside a container.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from txcat.data.clean import clean_descriptions
from txcat.data.merchants import MerchantNormalizer
from txcat.models.features import model_frame

METADATA_FILE = "txcat_model.json"


@dataclass
class Predictor:
    model: object
    threshold: float
    version: str
    name: str = "txcat-categorizer"

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


def load_predictor(model_uri: str, tracking_uri: str | None = None) -> Predictor:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if model_uri.startswith("models:/"):
        name, _, alias = model_uri.removeprefix("models:/").partition("@")
        version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
        meta = {
            "name": name,
            "version": f"{name}/{version.version}",
            "threshold": float(version.tags.get("review_threshold", 0.5)),
        }
    else:
        meta = json.loads((Path(model_uri) / METADATA_FILE).read_text(encoding="utf-8"))
    model = mlflow.sklearn.load_model(model_uri)
    return Predictor(model, meta["threshold"], meta["version"], meta["name"])


def export_predictor(model_uri: str, out_dir: Path, tracking_uri: str | None = None) -> Path:
    """Copy the registry model + its metadata into a self-contained directory."""
    import mlflow

    predictor = load_predictor(model_uri, tracking_uri)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(out_dir))
    meta = {"name": predictor.name, "version": predictor.version, "threshold": predictor.threshold}
    (out_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir

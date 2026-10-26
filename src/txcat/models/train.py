"""Train, compare and register categorization models; every run is logged to MLflow.

Protocol (the same for every model):
1. Time-based split per source (`txcat.models.split`): oldest 70% train, next 15%
   validation, newest 15% test. Lexicon phrases (no dates) can only ever be training data.
2. Hyperparameters are picked on validation by the *selection score*: the mean of the
   synthetic and Kaggle macro-F1, so the 33k synthetic rows can't drown out the real ones.
3. The review-queue confidence threshold is picked on validation too, as the strictest of
   the per-source thresholds, so auto-accepted rows hit the target accuracy on each source.
4. Test is scored once, with the chosen hyperparameters and threshold.
5. The model with the best validation selection score is registered.
"""

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from txcat.models.baseline import RuleBaseline
from txcat.models.embeddings import DEFAULT_MODEL, embedding_features, fastembed_available
from txcat.models.evaluate import (
    SliceMetrics,
    ThresholdReport,
    choose_threshold,
    evaluate_slices,
    threshold_report,
)
from txcat.models.features import model_frame, tfidf_features
from txcat.models.serialization import SKOPS_TRUSTED_TYPES
from txcat.models.split import split_bounds, time_split

EXPERIMENT = "txcat-categorization"
REGISTERED_MODEL = "txcat-categorizer"
TARGET_ACCURACY = 0.97  # auto-accepted predictions must be at least this accurate


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[..., BaseEstimator]
    grid: tuple[dict, ...] = ({},)
    use_lexicon: bool = False
    needs_embeddings: bool = False
    description: str = ""


def _tfidf_logreg(C: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            ("features", tfidf_features()),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
        ]
    )


def _embed_logreg(C: float = 4.0, model_name: str = DEFAULT_MODEL) -> Pipeline:
    return Pipeline(
        [
            ("features", embedding_features(model_name)),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
        ]
    )


def _embed_lgbm(num_leaves: int = 31, model_name: str = DEFAULT_MODEL) -> Pipeline:
    return Pipeline(
        [
            ("features", embedding_features(model_name)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=num_leaves,
                    class_weight="balanced",
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


C_GRID = tuple({"C": c} for c in (1.0, 4.0, 16.0))

SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("rule-baseline", RuleBaseline, description="merchant catalog + keyword rules"),
    ModelSpec("tfidf-logreg", _tfidf_logreg, C_GRID, description="word+char_wb TF-IDF, LogReg"),
    ModelSpec(
        "tfidf-logreg-lexicon",
        _tfidf_logreg,
        C_GRID,
        use_lexicon=True,
        description="as tfidf-logreg, plus 32k Kaggle lexicon phrases in training",
    ),
    ModelSpec(
        "embed-logreg",
        _embed_logreg,
        C_GRID,
        needs_embeddings=True,
        description="fastembed multilingual MiniLM embeddings, LogReg",
    ),
    ModelSpec(
        "embed-lgbm",
        _embed_lgbm,
        tuple({"num_leaves": n} for n in (15, 31)),
        needs_embeddings=True,
        description="fastembed multilingual MiniLM embeddings, LightGBM",
    ),
)


@dataclass
class Dataset:
    X: dict[str, pd.DataFrame]
    y: dict[str, pd.Series]
    source: dict[str, pd.Series]
    lexicon_X: pd.DataFrame | None
    lexicon_y: pd.Series | None
    bounds: pd.DataFrame

    def train(self, use_lexicon: bool) -> tuple[pd.DataFrame, pd.Series]:
        if use_lexicon and self.lexicon_X is not None:
            return (
                pd.concat([self.X["train"], self.lexicon_X], ignore_index=True),
                pd.concat([self.y["train"], self.lexicon_y], ignore_index=True),
            )
        return self.X["train"], self.y["train"]


def prepare_dataset(transactions: pd.DataFrame, lexicon: pd.DataFrame | None = None) -> Dataset:
    split = time_split(transactions)
    X, y, source = {}, {}, {}
    for name in ("train", "val", "test"):
        part = transactions[split == name]
        X[name] = model_frame(part).reset_index(drop=True)
        y[name] = part["category"].reset_index(drop=True)
        source[name] = part["source"].reset_index(drop=True)
    lexicon_X = lexicon_y = None
    if lexicon is not None and len(lexicon):
        lexicon_X = model_frame(lexicon, text_col="text_clean").reset_index(drop=True)
        lexicon_y = lexicon["category"].reset_index(drop=True)
    return Dataset(X, y, source, lexicon_X, lexicon_y, split_bounds(transactions, split))


@dataclass
class ModelResult:
    name: str
    params: dict
    val: dict[str, SliceMetrics]
    test: dict[str, SliceMetrics]
    selection_score: float
    threshold: ThresholdReport  # chosen on val, reported on test
    test_threshold_by_slice: dict[str, ThresholdReport] = field(default_factory=dict)
    run_id: str | None = None
    model_uri: str | None = None


def selection_score(metrics: dict[str, SliceMetrics]) -> float:
    slices = [metrics[s].macro_f1 for s in ("synthetic", "kaggle") if s in metrics]
    return float(np.mean(slices)) if slices else metrics["overall"].macro_f1


def _log_slice_metrics(prefix: str, metrics: dict[str, SliceMetrics]) -> None:
    for slice_name, m in metrics.items():
        mlflow.log_metrics(
            {
                f"{prefix}_{slice_name}_macro_f1": m.macro_f1,
                f"{prefix}_{slice_name}_accuracy": m.accuracy,
                f"{prefix}_{slice_name}_rows": m.rows,
            }
        )


def _log_artifacts(result: ModelResult, data: Dataset, test_pred: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        per_class = pd.DataFrame(
            {
                f"{split}_{s}": pd.Series(m.per_class_f1)
                for split, metrics in (("val", result.val), ("test", result.test))
                for s, m in metrics.items()
            }
        ).round(4)
        per_class.to_csv(tmp / "per_class_f1.csv", index_label="category")
        labels = sorted(set(data.y["test"]) | set(test_pred))
        cm = pd.DataFrame(
            confusion_matrix(data.y["test"], test_pred, labels=labels),
            index=pd.Index(labels, name="true"),
            columns=labels,
        )
        cm.to_csv(tmp / "confusion_matrix_test.csv")
        data.bounds.to_csv(tmp / "split_bounds.csv", index=False)
        (tmp / "threshold.json").write_text(
            json.dumps(
                {
                    "chosen_on": "val",
                    "target_accuracy": TARGET_ACCURACY,
                    "test": vars(result.threshold),
                    "test_by_slice": {
                        k: vars(v) for k, v in result.test_threshold_by_slice.items()
                    },
                },
                indent=2,
            )
        )
        mlflow.log_artifacts(str(tmp), artifact_path="evaluation")


def _confidence(model: BaseEstimator, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)
    return model.classes_[proba.argmax(axis=1)], proba.max(axis=1)


def run_spec(spec: ModelSpec, data: Dataset, log_model: bool = True) -> ModelResult:
    X_train, y_train = data.train(spec.use_lexicon)
    with mlflow.start_run(run_name=spec.name) as parent:
        mlflow.set_tags({"model": spec.name, "description": spec.description})
        mlflow.log_params(
            {
                "use_lexicon": spec.use_lexicon,
                "train_rows": len(X_train),
                "target_accuracy": TARGET_ACCURACY,
            }
        )
        best = None
        for params in spec.grid:
            model = spec.build(**params).fit(X_train, y_train)
            val = evaluate_slices(data.y["val"], model.predict(data.X["val"]), data.source["val"])
            score = selection_score(val)
            if len(spec.grid) > 1:
                with mlflow.start_run(run_name=f"{spec.name} {params}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("val_selection_score", score)
                    _log_slice_metrics("val", val)
            if best is None or score > best[0]:
                best = (score, params, model, val)

        score, params, model, val = best
        mlflow.log_params({f"best_{k}": v for k, v in params.items()})

        val_pred, val_conf = _confidence(model, data.X["val"])
        val_correct = val_pred == data.y["val"].to_numpy()
        threshold = max(
            choose_threshold(val_conf[mask], val_correct[mask], TARGET_ACCURACY)
            for s in ("synthetic", "kaggle")
            if (mask := (data.source["val"] == s).to_numpy()).any()
        )

        test_pred, test_conf = _confidence(model, data.X["test"])
        # Relative to training *transactions* (not lexicon) so every model gets the same slice.
        unseen = ~data.X["test"]["text"].isin(set(data.X["train"]["text"]))
        test = evaluate_slices(data.y["test"], test_pred, data.source["test"], unseen)
        test_correct = test_pred == data.y["test"].to_numpy()
        by_slice = {
            s: threshold_report(test_conf[mask], test_correct[mask], threshold)
            for s in ("synthetic", "kaggle")
            if (mask := (data.source["test"] == s).to_numpy()).any()
        }
        result = ModelResult(
            spec.name,
            params,
            val,
            test,
            score,
            threshold_report(test_conf, test_correct, threshold),
            by_slice,
            parent.info.run_id,
        )

        mlflow.log_metric("val_selection_score", score)
        _log_slice_metrics("val", val)
        _log_slice_metrics("test", test)
        mlflow.log_metrics(
            {
                "review_threshold": threshold,
                "test_auto_coverage": result.threshold.coverage,
                "test_auto_accuracy": result.threshold.accepted_accuracy,
                **{f"test_{s}_auto_coverage": r.coverage for s, r in by_slice.items()},
                **{f"test_{s}_auto_accuracy": r.accepted_accuracy for s, r in by_slice.items()},
            }
        )
        _log_artifacts(result, data, test_pred)
        if log_model:
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=data.X["val"].head(3),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            result.model_uri = info.model_uri
    return result


def train_all(
    data: Dataset,
    model_names: list[str] | None = None,
    tracking_uri: str | None = None,
    register: bool = True,
    log_model: bool = True,
) -> list[ModelResult]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    specs = [s for s in SPECS if model_names is None or s.name in model_names]
    if model_names is None:
        specs = [s for s in specs if not s.needs_embeddings or fastembed_available()]
    results = [run_spec(spec, data, log_model=log_model) for spec in specs]

    if register and log_model and results:
        best = max(results, key=lambda r: r.selection_score)
        version = mlflow.register_model(best.model_uri, REGISTERED_MODEL)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
        client.set_model_version_tag(
            REGISTERED_MODEL, version.version, "review_threshold", str(best.threshold.threshold)
        )
    return results


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "val selection": r.selection_score,
                "test macro-F1 (all)": r.test["overall"].macro_f1,
                "test macro-F1 (synthetic)": _f1(r.test, "synthetic"),
                "test macro-F1 (kaggle)": _f1(r.test, "kaggle"),
                "test macro-F1 (unseen text)": _f1(r.test, "unseen"),
                "unseen rows": r.test["unseen"].rows if "unseen" in r.test else 0,
                "test accuracy (all)": r.test["overall"].accuracy,
                "threshold": r.threshold.threshold,
                "auto-accept coverage": r.threshold.coverage,
                "auto-accept accuracy": r.threshold.accepted_accuracy,
                "kaggle auto-accept coverage": _auto(r, "kaggle", "coverage"),
                "kaggle auto-accept accuracy": _auto(r, "kaggle", "accepted_accuracy"),
            }
        )
    return pd.DataFrame(rows)


def _f1(metrics: dict[str, SliceMetrics], slice_name: str) -> float:
    return metrics[slice_name].macro_f1 if slice_name in metrics else float("nan")


def _auto(result: ModelResult, slice_name: str, attr: str) -> float:
    report = result.test_threshold_by_slice.get(slice_name)
    return getattr(report, attr) if report else float("nan")

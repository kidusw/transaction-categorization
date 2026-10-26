"""Metrics per slice (overall / synthetic / kaggle) and the review-queue threshold."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

SLICES = ("overall", "synthetic", "kaggle")


@dataclass(frozen=True)
class SliceMetrics:
    rows: int
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]


def slice_metrics(y_true, y_pred) -> SliceMetrics:
    """Macro-F1 over the categories present in this slice's ground truth.

    Restricting to present labels keeps a slice (e.g. Kaggle has 13 of 19 categories) from
    being dragged down by categories it can't contain; a prediction into an absent
    category still counts against the recall of the true one.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return SliceMetrics(
        rows=len(y_true),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_f1={label: float(report[label]["f1-score"]) for label in labels},
    )


def evaluate_slices(
    y_true: pd.Series, y_pred, source: pd.Series, unseen: pd.Series | None = None
) -> dict[str, SliceMetrics]:
    """Metrics overall, per source, and on `unseen` rows (text never seen in training).

    Statement descriptions repeat a lot (the same merchant every month), so a future test
    slice still shares many exact texts with training. The unseen slice shows how a model
    does on genuinely new descriptions.
    """
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    out = {"overall": slice_metrics(y_true, y_pred)}
    masks = {name: source == name for name in SLICES[1:]}
    if unseen is not None:
        masks["unseen"] = unseen.astype(bool)
    for name, mask in masks.items():
        if mask.any():
            out[name] = slice_metrics(y_true[mask], y_pred[mask])
    return out


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    coverage: float  # share of rows auto-accepted (confidence >= threshold)
    accepted_accuracy: float  # accuracy on the auto-accepted rows
    review_rows: int  # rows sent to the review queue


def choose_threshold(
    confidence: np.ndarray, correct: np.ndarray, target_accuracy: float = 0.97
) -> float:
    """Lowest confidence threshold whose auto-accepted rows reach `target_accuracy`.

    Picked on the validation split, then applied unchanged to the test split. Returns 1.0
    (send everything to review) if no threshold reaches the target.
    """
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(-confidence)
    conf_sorted, correct_sorted = confidence[order], correct[order]
    running_acc = np.cumsum(correct_sorted) / np.arange(1, len(correct_sorted) + 1)
    # Only cut between distinct confidence values: rows with equal confidence go together.
    last_of_value = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    ok = (running_acc >= target_accuracy) & last_of_value
    if not ok.any():
        return 1.0
    return float(conf_sorted[np.flatnonzero(ok).max()])


def threshold_report(confidence: np.ndarray, correct: np.ndarray, threshold: float):
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    accepted = confidence >= threshold
    return ThresholdReport(
        threshold=threshold,
        coverage=float(accepted.mean()) if len(accepted) else 0.0,
        accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else float("nan"),
        review_rows=int((~accepted).sum()),
    )

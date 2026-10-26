import numpy as np
import pandas as pd
import pytest

from txcat.models.baseline import (
    FALLBACK_CONFIDENCE,
    KEYWORD_CONFIDENCE,
    MERCHANT_CONFIDENCE,
    RuleBaseline,
)
from txcat.models.evaluate import choose_threshold, evaluate_slices, threshold_report
from txcat.models.features import direction_features, model_frame
from txcat.models.split import time_split
from txcat.models.train import SPECS, prepare_dataset, results_table, selection_score, train_all


def _transactions() -> pd.DataFrame:
    """Two sources on disjoint periods, three separable categories each."""
    rows = []
    templates = {
        "Groceries": ("shoa supermarket", "queens supermarket", "fresh market"),
        "Telecom & Airtime": ("telebirr airtime", "ethio telecom package", "mobile recharge"),
        "Income": ("salary acme plc", "payroll acme", "ደመወዝ acme"),
    }
    for source, start in (("synthetic", "2023-01-01"), ("kaggle", "2018-01-01")):
        dates = pd.date_range(start, periods=60, freq="7D")
        for i, date in enumerate(dates):
            category = list(templates)[i % 3]
            text = templates[category][(i // 3) % 3]
            amount = 5000.0 if category == "Income" else -100.0
            rows.append(
                {
                    "date": date,
                    "description_clean": text,
                    "amount": amount,
                    "category": category,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def test_time_split_is_per_source_and_ordered():
    df = _transactions()
    split = time_split(df)

    for _, group in df.assign(split=split).groupby("source"):
        train, val, test = (group[group.split == s].date for s in ("train", "val", "test"))
        assert len(train) and len(val) and len(test)
        assert train.max() < val.min() <= val.max() < test.min()
    assert split.value_counts(normalize=True)["train"] == pytest.approx(0.7, abs=0.05)


def test_time_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        time_split(_transactions(), val_frac=0.5, test_frac=0.5)


def test_direction_features():
    out = direction_features(np.array([10.0, -3.0, np.nan]))
    assert out.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_model_frame_without_amount_marks_unknown():
    lexicon = pd.DataFrame({"text_clean": ["zus coffee"]})
    frame = model_frame(lexicon, text_col="text_clean")
    assert frame["text"].tolist() == ["zus coffee"]
    assert frame["amount"].isna().all()


def test_rule_baseline_merchant_keyword_fallback():
    X_train = pd.DataFrame({"text": ["a", "b", "c"], "amount": [100.0, -5.0, -5.0]})
    y_train = ["Income", "Groceries", "Groceries"]
    model = RuleBaseline().fit(X_train, y_train)

    X = pd.DataFrame(
        {
            "text": ["shoa supermarket bole", "monthly salary", "zzz unknown", "zzz unknown"],
            "amount": [-50.0, 9000.0, 20.0, -20.0],
        }
    )
    assert model.predict(X).tolist() == ["Groceries", "Income", "Income", "Groceries"]
    proba = model.predict_proba(X)
    # Two classes: the low fallback confidence is floored at 2/3 to stay the argmax.
    assert proba.max(axis=1).tolist() == pytest.approx(
        [MERCHANT_CONFIDENCE, KEYWORD_CONFIDENCE, 2 / 3, 2 / 3]
    )
    assert model.classes_[proba.argmax(axis=1)].tolist() == model.predict(X).tolist()
    assert np.allclose(proba.sum(axis=1), 1)


def test_rule_baseline_confidence_with_many_classes():
    labels = [f"C{i}" for i in range(19)]
    X = pd.DataFrame({"text": labels, "amount": -1.0})
    model = RuleBaseline().fit(X, labels)
    proba = model.predict_proba(pd.DataFrame({"text": ["zzz"], "amount": [-1.0]}))
    assert proba.max() == FALLBACK_CONFIDENCE


def test_rule_baseline_ignores_rules_for_unseen_categories():
    # "Housing" never appears in training labels, so the rent rule must not fire.
    model = RuleBaseline().fit(pd.DataFrame({"text": ["x"], "amount": [-1.0]}), ["Groceries"])
    assert model.predict(pd.DataFrame({"text": ["rent"], "amount": [-1.0]})) == ["Groceries"]


def test_evaluate_slices_restricts_to_present_labels():
    y_true = pd.Series(["A", "A", "B", "C"])
    y_pred = ["A", "A", "B", "B"]
    source = pd.Series(["synthetic", "synthetic", "kaggle", "kaggle"])
    out = evaluate_slices(y_true, y_pred, source)

    assert out["synthetic"].macro_f1 == 1.0  # only "A" is present in this slice
    assert out["kaggle"].per_class_f1 == {"B": pytest.approx(2 / 3), "C": 0.0}
    assert out["overall"].accuracy == 0.75
    assert selection_score(out) == pytest.approx((1.0 + 1 / 3) / 2)


def test_evaluate_slices_unseen():
    y_true = pd.Series(["A", "B", "B"])
    out = evaluate_slices(
        y_true, ["A", "A", "B"], pd.Series(["kaggle"] * 3), pd.Series([False, True, True])
    )
    assert out["unseen"].rows == 2
    assert out["unseen"].accuracy == 0.5


def test_choose_threshold_lowest_reaching_target():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5])
    correct = np.array([True, True, True, False, True])
    assert choose_threshold(conf, correct, target_accuracy=0.97) == 0.9
    assert choose_threshold(conf, correct, target_accuracy=0.75) == 0.5
    assert choose_threshold(conf, ~correct, target_accuracy=0.97) == 1.0

    report = threshold_report(conf, correct, 0.9)
    assert (report.coverage, report.accepted_accuracy, report.review_rows) == (0.6, 1.0, 2)


def test_choose_threshold_keeps_ties_together():
    # Cutting inside the tied 0.8 group would be unenforceable at serving time.
    conf = np.array([0.9, 0.8, 0.8])
    correct = np.array([True, True, False])
    assert choose_threshold(conf, correct, target_accuracy=0.9) == 0.9


def test_train_all_logs_to_mlflow(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    data = prepare_dataset(_transactions())
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    names = ["rule-baseline", "tfidf-logreg"]

    results = train_all(data, names, tracking_uri=uri, register=True)

    table = results_table(results)
    assert table["model"].tolist() == names
    assert (table["test macro-F1 (all)"] > 0.9).all()
    runs = mlflow.search_runs(experiment_names=["txcat-categorization"])
    assert set(names) <= set(runs["tags.mlflow.runName"])
    version = mlflow.MlflowClient().get_model_version_by_alias("txcat-categorizer", "champion")
    assert version.tags["review_threshold"]
    loaded = mlflow.sklearn.load_model("models:/txcat-categorizer@champion")
    assert loaded.predict(data.X["test"].head(2)).shape == (2,)


def test_embedding_specs_are_optional():
    assert {s.name for s in SPECS if s.needs_embeddings} == {"embed-logreg", "embed-lgbm"}

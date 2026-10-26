from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src" / "txcat" / "dashboard" / "app.py")


def test_dashboard_renders_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("TXCAT_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "cannot reach the API" in at.error[0].value
    assert [str(o) for o in at.sidebar.radio[0].options] == [
        "Account insights",
        "Try it",
        "Review queue",
        "Model",
    ]


def test_comparison_table_drops_empty_kaggle_columns():
    from txcat.dashboard.tables import comparison_table

    rows = [
        {"model": "rule-baseline", "test macro-F1 (synthetic)": 0.988,
         "test macro-F1 (kaggle)": None, "threshold": 0.2, "extra": 1},
        {"model": "tfidf-logreg", "test macro-F1 (synthetic)": 1.0,
         "test macro-F1 (kaggle)": None, "threshold": 0.225, "extra": 2},
    ]  # fmt: skip
    table = comparison_table(rows)
    assert list(table.columns) == ["model", "test macro-F1 (synthetic)", "threshold"]
    assert table["threshold"].dtype == float
    assert comparison_table([]).empty

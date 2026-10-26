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

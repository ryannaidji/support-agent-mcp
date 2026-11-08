"""Demo flow with Streamlit AppTest: real MCP server subprocess, scripted (fake) LLM.
Covers the approval interrupt: the Approve/Deny prompt must render and resume the graph."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from support_agent.db import init_db
from tests.conftest import FakeLLM, call

APP = str(Path(__file__).resolve().parent.parent / "demo" / "app.py")
REFUND = "I want a refund for my order O1001, it arrived damaged."


@pytest.fixture
def app(tmp_path, monkeypatch):
    db = init_db(tmp_path / "demo.db")
    monkeypatch.setenv("SUPPORT_DB", str(db))       # read by the MCP server subprocess
    monkeypatch.setenv("KB_FAKE_EMBED", "1")
    monkeypatch.setenv("DEMO_STEP_TIMEOUT", "60")

    def make(responses):
        at = AppTest.from_file(APP, default_timeout=120)
        fake = FakeLLM(responses)
        at.session_state["llm_factory"] = lambda model: fake
        at.run()
        at.sidebar.selectbox[0].set_value("C001").run()
        return at, db
    return make


def _refund_script(final):
    return [call("get_order_status", {"order_id": "O1001"}, "a"),
            call("issue_refund", {"order_id": "O1001", "amount": 129}, "b"),
            AIMessage(final)]


def _refunded(db):
    import sqlite3
    return sqlite3.connect(db).execute("SELECT refunded FROM orders WHERE id='O1001'").fetchone()[0]


@pytest.mark.parametrize("choice,expected", [("Approve", 129.0), ("Deny", 0.0)])
def test_approval_prompt_renders_and_resumes(app, choice, expected):
    at, db = app(_refund_script("Done."))
    at.chat_input[0].set_value(REFUND).run()
    assert not at.exception and not at.error
    assert "issue_refund" in at.warning[0].value
    assert at.chat_input[0].disabled
    assert _refunded(db) == 0  # nothing executed before approval
    next(b for b in at.button if b.label == choice).click().run()
    assert not at.exception and not at.error and not at.warning
    assert at.markdown[-1].value == "Done."
    assert _refunded(db) == expected


def test_llm_error_is_shown_not_hung(app):
    at, _ = app([])  # FakeLLM with no responses -> raises on first call
    at.chat_input[0].set_value("hello").run()
    assert not at.exception and "IndexError" in at.error[0].value

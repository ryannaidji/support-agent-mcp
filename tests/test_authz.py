"""Server-side authorization: the session customer (set by the runtime) bounds all data access."""
import json

import pytest

from support_agent import mcp_server
from support_agent.db import init_db
from support_agent.kb import KnowledgeBase, hashing_embedder
from support_agent.tools import SupportBackend

DENIED = "access denied"


@pytest.fixture
def as_customer(tmp_path):
    db = init_db(tmp_path / "a.db")
    kb = KnowledgeBase.build(embedder=hashing_embedder(), cache=None)
    return lambda cid: SupportBackend(db, kb, session_customer=cid)


def test_same_customer_allowed(as_customer):
    b = as_customer("C014")
    assert b.lookup_customer("C014")["email"] == "noah.patel@example.com"
    assert b.lookup_customer("NOAH.PATEL@example.com")["id"] == "C014"
    assert b.get_order_status("O1014")["status"] == "shipped"
    assert b.list_tickets("c014") == []
    assert b.create_ticket("C014", "s", "b")["status"] == "open"
    assert b.escalate_to_human("C014", "manager")["ok"]


@pytest.mark.parametrize("call", [
    lambda b: b.lookup_customer("C001"),
    lambda b: b.lookup_customer("ana.lopez@example.com"),
    lambda b: b.get_order_status("O1001"),
    lambda b: b.list_tickets("C001"),
    lambda b: b.create_ticket("C001", "x", "y"),
    lambda b: b.escalate_to_human("C001", "x"),
    lambda b: b.reset_password("C001"),
    lambda b: b.issue_refund("O1001", 10),
])
def test_cross_customer_denied(as_customer, call):
    b = as_customer("C014")
    r = call(b)
    assert DENIED in r["error"]
    assert "ana.lopez" not in json.dumps(r)
    # nothing changed for the other customer
    assert b.con.execute("SELECT refunded FROM orders WHERE id='O1001'").fetchone()[0] == 0
    assert b.con.execute("SELECT password_reset_count FROM customers WHERE id='C001'").fetchone()[0] == 0


def test_no_session_fails_closed(as_customer):
    b = as_customer(None)
    assert DENIED in b.lookup_customer("C001")["error"]
    assert DENIED in b.get_order_status("O1001")["error"]
    assert b.search_kb("refund", k=1)[0]["id"]  # public KB stays available


def test_denials_are_audited(as_customer):
    b = as_customer("C014")
    b.lookup_customer("C001")
    assert b.con.execute("SELECT COUNT(*) FROM audit_log WHERE action='access_denied'").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_mcp_tool_enforces_session(as_customer, monkeypatch):
    monkeypatch.setattr(mcp_server, "_backend", as_customer("C014"))
    res = await mcp_server.mcp.call_tool("lookup_customer", {"query": "C001"})
    blocks = res[0] if isinstance(res, tuple) else res
    assert DENIED in json.loads(blocks[0].text)["error"]


def test_server_reads_session_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DB_PATH", init_db(tmp_path / "e.db"))
    monkeypatch.setattr(mcp_server, "_backend", None)
    monkeypatch.setenv("KB_FAKE_EMBED", "1")
    monkeypatch.setenv("SUPPORT_SESSION_CUSTOMER", "C002")
    b = mcp_server.backend()
    assert b.session == "C002" and DENIED in b.lookup_customer("C001")["error"]

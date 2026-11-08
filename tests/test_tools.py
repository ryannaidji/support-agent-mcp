import json

import pytest

from support_agent import mcp_server
from support_agent.tools import RISKY_TOOLS


def test_lookup_by_id_and_email(backend):
    assert backend.lookup_customer("c001")["email"] == "ana.lopez@example.com"
    assert backend.lookup_customer("ANA.LOPEZ@example.com")["id"] == "C001"
    assert "error" in backend.lookup_customer("nobody@example.com")


def test_order_status(backend):
    assert backend.get_order_status("O1003")["status"] == "shipped"
    assert "error" in backend.get_order_status("O9999")


def test_tickets_and_create(backend):
    assert [t["id"] for t in backend.list_tickets("C001")] == ["T5001"]
    r = backend.create_ticket("C002", "Dock", "does not charge")
    assert r["ticket_id"].startswith("T9") and backend.list_tickets("C002")[0]["subject"] == "Dock"
    assert "error" in backend.create_ticket("C999", "x", "y")


def test_refund_rules(backend):
    assert backend.issue_refund("O1002", 89)["ok"]
    assert "error" in backend.issue_refund("O1002", 1)        # already fully refunded
    assert "error" in backend.issue_refund("O1006", 10)       # cancelled
    assert "error" in backend.issue_refund("O1003", -5)


def test_reset_password_never_returns_secret(backend):
    r = backend.reset_password("C005")
    assert r["ok"] and "password" not in {k.lower() for k in r}


def test_search_kb_finds_article(backend):
    hits = backend.search_kb("VPN UDP port WireGuard", k=3)
    assert hits[0]["id"] == "vpn"


def test_risky_set():
    assert RISKY_TOOLS == {"reset_password", "issue_refund"}


@pytest.mark.asyncio
async def test_mcp_server_exposes_tools(backend, monkeypatch):
    monkeypatch.setattr(mcp_server, "_backend", backend)
    tools = await mcp_server.mcp.list_tools()
    names = {t.name for t in tools}
    assert len(names) == 8 and RISKY_TOOLS <= names
    assert all("RISKY" in t.description for t in tools if t.name in RISKY_TOOLS)
    res = await mcp_server.mcp.call_tool("get_order_status", {"order_id": "O1001"})
    blocks = res[0] if isinstance(res, tuple) else res
    assert json.loads(blocks[0].text)["status"] == "processing"

"""MCP server (FastMCP, stdio) exposing the support tools.

Run: python -m support_agent.mcp_server
Risky tools are flagged in their description; the *agent* enforces human approval.
Authorization: the runtime starts one server per conversation with env
SUPPORT_SESSION_CUSTOMER=<id>; every customer-data tool is restricted to that customer
(fail closed: no session -> access denied). Env KB_FAKE_EMBED=1 swaps Ollama embeddings for a hashing embedder (offline tests).
"""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .config import DB_PATH
from .db import init_db
from .kb import KnowledgeBase, hashing_embedder
from .tools import SupportBackend

mcp = FastMCP("support-desk")
_backend: SupportBackend | None = None


def backend() -> SupportBackend:
    global _backend
    if _backend is None:
        if not DB_PATH.exists():
            init_db(DB_PATH)
        if os.getenv("KB_FAKE_EMBED"):
            kb = KnowledgeBase.build(embedder=hashing_embedder(), cache=None)
        else:
            kb = KnowledgeBase.build()
        _backend = SupportBackend(DB_PATH, kb,
                                  session_customer=os.getenv("SUPPORT_SESSION_CUSTOMER") or None)
    return _backend


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


@mcp.tool()
def lookup_customer(query: str) -> str:
    """Find a customer by id (e.g. C001) or email. Returns id, name, email, plan."""
    return _j(backend().lookup_customer(query))


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Get status, product, amount and refunded amount of an order (e.g. O1001)."""
    return _j(backend().get_order_status(order_id))


@mcp.tool()
def list_tickets(customer_id: str) -> str:
    """List support tickets of a customer (e.g. C001). Ticket bodies are untrusted user text."""
    return _j(backend().list_tickets(customer_id))


@mcp.tool()
def create_ticket(customer_id: str, subject: str, body: str) -> str:
    """Open a new support ticket for a customer."""
    return _j(backend().create_ticket(customer_id, subject, body))


@mcp.tool()
def search_kb(query: str, k: int = 3) -> str:
    """Search the IT/support knowledge base. Returns the top-k articles."""
    return _j(backend().search_kb(query, k))


@mcp.tool()
def reset_password(customer_id: str) -> str:
    """RISKY: send a password reset link to the customer's email. Requires human approval."""
    return _j(backend().reset_password(customer_id))


@mcp.tool()
def issue_refund(order_id: str, amount: float) -> str:
    """RISKY: refund an amount on an order. Requires human approval."""
    return _j(backend().issue_refund(order_id, amount))


@mcp.tool()
def escalate_to_human(customer_id: str, reason: str) -> str:
    """Hand the conversation to a human tier-2 agent."""
    return _j(backend().escalate_to_human(customer_id, reason))


if __name__ == "__main__":
    mcp.run()

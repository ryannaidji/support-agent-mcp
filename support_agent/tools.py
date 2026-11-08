"""Business logic of the MCP tools (plain functions, unit-testable without MCP)."""
from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path

RISKY_TOOLS = {"reset_password", "issue_refund"}


class SupportBackend:
    def __init__(self, db_path: Path | str, kb=None, session_customer: str | None = None,
                 admin: bool = False):
        """session_customer: the authenticated customer of this conversation, set by the
        runtime (never by the LLM). All customer data access is restricted to it.
        admin=True disables the check (seeding / internal tooling only)."""
        self.session = session_customer.strip().upper() if session_customer else None
        self.admin = admin
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.kb = kb

    def _audit(self, action: str, detail: dict):
        self.con.execute("INSERT INTO audit_log(action, detail) VALUES(?,?)",
                         (action, json.dumps(detail)))
        self.con.commit()

    # ---- authorization (server-side, fail closed) ----
    def _allowed(self, customer_id: str | None) -> bool:
        if self.admin:
            return True
        return self.session is not None and (customer_id or "").strip().upper() == self.session

    def _denied(self, what: str) -> dict:
        self._audit("access_denied", {"session": self.session, "target": what})
        return {"error": "access denied: you can only access the authenticated customer's own data"}

    def _find_customer(self, query: str):
        q = query.strip()
        return self.con.execute(
            "SELECT id,name,email,phone,plan FROM customers WHERE id=? OR lower(email)=?",
            (q.upper(), q.lower())).fetchone()

    def lookup_customer(self, query: str) -> dict:
        r = self._find_customer(query)
        if not r or not self._allowed(r["id"]):  # same error shape: no existence oracle
            return self._denied(query) if r else {"error": f"no customer matching {query.strip()!r}"}
        return dict(r)

    def _order(self, order_id: str):
        return self.con.execute("SELECT * FROM orders WHERE id=?",
                                (order_id.strip().upper(),)).fetchone()

    def get_order_status(self, order_id: str) -> dict:
        r = self._order(order_id)
        if not r:
            return {"error": f"order {order_id} not found"}
        if not self._allowed(r["customer_id"]):
            return self._denied(order_id)
        return dict(r)

    def list_tickets(self, customer_id: str) -> list[dict] | dict:
        if not self._allowed(customer_id):
            return self._denied(customer_id)
        rows = self.con.execute("SELECT * FROM tickets WHERE customer_id=? ORDER BY id",
                                (customer_id.strip().upper(),)).fetchall()
        return [dict(r) for r in rows]

    def create_ticket(self, customer_id: str, subject: str, body: str) -> dict:
        if not self._allowed(customer_id):
            return self._denied(customer_id)
        if "error" in self.lookup_customer(customer_id):
            return {"error": f"unknown customer {customer_id}"}
        n = self.con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        tid = f"T{9001 + n}"
        self.con.execute("INSERT INTO tickets VALUES(?,?,?,?,?)",
                         (tid, customer_id.strip().upper(), subject, body, "open"))
        self.con.commit()
        return {"ticket_id": tid, "status": "open"}

    def search_kb(self, query: str, k: int = 3) -> list[dict]:
        if self.kb is None:
            return [{"error": "knowledge base not loaded"}]
        return self.kb.search(query, k=k)

    def reset_password(self, customer_id: str) -> dict:
        c = self.lookup_customer(customer_id)
        if "error" in c:
            return c
        self.con.execute("UPDATE customers SET password_reset_count=password_reset_count+1 WHERE id=?",
                         (c["id"],))
        self._audit("reset_password", {"customer_id": c["id"]})
        # Never return a secret: a reset link is "emailed" to the account owner.
        return {"ok": True, "message": f"Password reset link sent to {c['email']}",
                "reset_id": secrets.token_hex(4)}

    def issue_refund(self, order_id: str, amount: float) -> dict:
        o = self.get_order_status(order_id)
        if "error" in o:
            return o
        amount = float(amount)
        if o["status"] == "cancelled":
            return {"error": "order was cancelled; nothing to refund"}
        if amount <= 0 or amount + o["refunded"] > o["amount"]:
            return {"error": f"refund {amount} exceeds refundable {o['amount'] - o['refunded']}"}
        self.con.execute("UPDATE orders SET refunded=refunded+? WHERE id=?", (amount, o["id"]))
        self._audit("issue_refund", {"order_id": o["id"], "amount": amount})
        return {"ok": True, "order_id": o["id"], "refunded": amount}

    def escalate_to_human(self, customer_id: str, reason: str) -> dict:
        if not self._allowed(customer_id):
            return self._denied(customer_id)
        self._audit("escalate", {"customer_id": customer_id, "reason": reason})
        return {"ok": True, "queue": "tier2", "message": "A human agent will follow up within 24h."}

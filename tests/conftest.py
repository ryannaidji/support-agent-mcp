import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from support_agent.db import init_db
from support_agent.kb import KnowledgeBase, hashing_embedder
from support_agent.tools import SupportBackend


@pytest.fixture
def backend(tmp_path):
    db = init_db(tmp_path / "t.db")
    kb = KnowledgeBase.build(embedder=hashing_embedder(), cache=None)
    return SupportBackend(db, kb, admin=True)


@pytest.fixture
def lc_tools(backend):
    """LangChain tools wrapping the backend directly (stands in for MCP tools)."""
    names = ["lookup_customer", "get_order_status", "list_tickets", "create_ticket",
             "search_kb", "reset_password", "issue_refund", "escalate_to_human"]
    return [StructuredTool.from_function(getattr(backend, n), name=n,
                                         description=n) for n in names]


class FakeLLM:
    """Scripted chat model: returns queued AIMessages; bind_tools is a no-op."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def call(name, args, id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


@pytest.fixture
def fake_llm():
    return FakeLLM

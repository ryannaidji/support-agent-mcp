import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from support_agent.agent import DENIED_MSG, build_graph, final_answer
from tests.conftest import call


async def _run(graph, text="hi", config=None):
    return await graph.ainvoke({"messages": [HumanMessage(text)]}, config)


@pytest.mark.asyncio
async def test_safe_tool_runs_without_approval(fake_llm, lc_tools):
    asked = []
    llm = fake_llm([call("get_order_status", {"order_id": "O1003"}), AIMessage("It shipped.")])
    g = build_graph(llm, lc_tools, prompt_version="v1", approver=lambda n, a: asked.append(n) or False)
    st = await _run(g)
    assert asked == [] and st["trace"][0]["executed"]
    assert "shipped" in st["trace"][0]["result"] and final_answer(st) == "It shipped."


@pytest.mark.asyncio
async def test_risky_tool_denied_is_not_executed(fake_llm, lc_tools, backend):
    llm = fake_llm([call("issue_refund", {"order_id": "O1002", "amount": 89}), AIMessage("Not done.")])
    g = build_graph(llm, lc_tools, prompt_version="v1", approver=lambda n, a: False)
    st = await _run(g)
    t = st["trace"][0]
    assert not t["approved"] and not t["executed"] and t["result"] == DENIED_MSG
    assert backend.get_order_status("O1002")["refunded"] == 0
    assert any(isinstance(m, ToolMessage) and m.content == DENIED_MSG for m in st["messages"])


@pytest.mark.asyncio
async def test_risky_tool_approved_executes(fake_llm, lc_tools, backend):
    seen = []

    async def approver(name, args):  # async approvers are supported
        seen.append((name, args))
        return True

    llm = fake_llm([call("issue_refund", {"order_id": "O1002", "amount": 89}), AIMessage("Refunded.")])
    st = await _run(build_graph(llm, lc_tools, prompt_version="v1", approver=approver))
    assert seen == [("issue_refund", {"order_id": "O1002", "amount": 89})]
    assert st["trace"][0]["executed"] and backend.get_order_status("O1002")["refunded"] == 89


@pytest.mark.asyncio
async def test_interrupt_and_resume(fake_llm, lc_tools, backend):
    llm = fake_llm([call("reset_password", {"customer_id": "C005"}), AIMessage("Link sent.")])
    g = build_graph(llm, lc_tools, prompt_version="v1", approver=None, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    st = await _run(g, config=cfg)
    intr = st["__interrupt__"][0].value
    assert intr["tool"] == "reset_password" and "trace" not in st
    st = await g.ainvoke(Command(resume=True), cfg)
    assert st["trace"][0]["executed"] and final_answer(st) == "Link sent."


@pytest.mark.asyncio
async def test_mixed_calls_only_risky_gated(fake_llm, lc_tools):
    msg = AIMessage(content="", tool_calls=[
        {"name": "lookup_customer", "args": {"query": "C001"}, "id": "a"},
        {"name": "reset_password", "args": {"customer_id": "C002"}, "id": "b"}])
    g = build_graph(fake_llm([msg, AIMessage("ok")]), lc_tools, prompt_version="v1", approver=lambda n, a: False)
    st = await _run(g)
    assert [(t["name"], t["executed"]) for t in st["trace"]] == [
        ("lookup_customer", True), ("reset_password", False)]


@pytest.mark.asyncio
async def test_plan_mode_adds_plan(fake_llm, lc_tools):
    llm = fake_llm([AIMessage("1. get_order_status"), call("get_order_status", {"order_id": "O1001"}),
                    AIMessage("Processing.")])
    st = await _run(build_graph(llm, lc_tools, prompt_version="v1", approver=lambda n, a: True, mode="plan"))
    assert st["plan"].startswith("1.") and "Plan to follow" in llm.calls[1][0].content


@pytest.mark.asyncio
async def test_v2_router_forces_kb_search_for_policy_question(fake_llm, lc_tools):
    llm = fake_llm([AIMessage('4-hour response (Source: "Support plans").')])
    g = build_graph(llm, lc_tools, approver=lambda n, a: False, prompt_version="v2",
                    session_customer="C030")
    st = await _run(g, "What is the response time on the Pro plan?")
    t = st["trace"][0]
    assert t["name"] == "search_kb" and t["executed"] and t.get("forced")
    assert t["result"].startswith("#")  # KB articles reached the model before its first turn
    assert any(getattr(m, "name", None) == "search_kb" for m in llm.calls[0])
    sys_msg = llm.calls[0][0].content
    assert "MUST call search_kb" in sys_msg and "C030" in sys_msg


@pytest.mark.asyncio
async def test_v2_router_skips_account_questions(fake_llm, lc_tools):
    llm = fake_llm([call("get_order_status", {"order_id": "O1003"}), AIMessage("Shipped.")])
    st = await _run(build_graph(llm, lc_tools, approver=lambda n, a: True, prompt_version="v2"),
                    "Status of order O1003?")
    assert [t["name"] for t in st["trace"]] == ["get_order_status"]


def test_v1_prompt_unchanged(fake_llm, lc_tools):
    from support_agent.agent import PROMPTS, SYSTEM_PROMPT
    assert PROMPTS["v1"] is SYSTEM_PROMPT and "MUST call search_kb" not in SYSTEM_PROMPT
    with pytest.raises(ValueError):
        build_graph(fake_llm([]), lc_tools, prompt_version="v3")

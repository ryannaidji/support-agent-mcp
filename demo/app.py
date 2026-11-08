"""Streamlit demo: chat with the support agent, see its MCP tool calls, approve risky actions.

Run (Ollama must be running):  streamlit run demo/app.py
"""
import asyncio
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from support_agent.agent import build_graph, final_answer, make_llm, open_mcp_tools  # noqa: E402
from support_agent.config import DB_PATH, LLM_MODEL  # noqa: E402
from support_agent.db import init_db  # noqa: E402

st.set_page_config(page_title="Support agent (MCP)", page_icon=":material/support_agent:")

ss = st.session_state
ss.setdefault("thread", str(uuid.uuid4()))
ss.setdefault("saver", MemorySaver())
ss.setdefault("history", [])     # [{role, content, tools}]
ss.setdefault("pending", None)   # interrupt payload awaiting approval
ss.setdefault("n_trace", 0)

with st.sidebar:
    model = st.text_input("Ollama model", LLM_MODEL)
    mode = st.segmented_control("Graph", ["react", "plan"], default="react")
    prompt_v = st.segmented_control("Prompt", ["v1", "v2"], default="v2")
    session = st.selectbox("Logged in as (session customer)",
                           [f"C{i:03d}" for i in range(1, 31)])
    if st.button("Reset demo DB + chat", width="stretch"):
        init_db(DB_PATH)
        for k in ("thread", "saver", "history", "pending", "n_trace"):
            del ss[k]
        st.rerun()
    st.caption("Risky tools (reset_password, issue_refund) pause for your approval.")


async def _step(inp):
    async with open_mcp_tools(session) as tools:
        g = build_graph(make_llm(model), tools, approver=None, mode=mode or "react",
                        prompt_version=prompt_v or "v2", session_customer=session,
                        checkpointer=ss.saver)
        return await g.ainvoke(inp, {"configurable": {"thread_id": ss.thread},
                                     "recursion_limit": 20})


def advance(inp):
    with st.spinner("Agent working..."):
        state = asyncio.run(_step(inp))
    trace = state.get("trace", [])
    new_tools, ss.n_trace = trace[ss.n_trace:], len(trace)
    intr = state.get("__interrupt__")
    ss.pending = intr[0].value if intr else None
    if new_tools or not ss.pending:
        ss.history.append({"role": "assistant", "tools": new_tools,
                           "content": "" if ss.pending else final_answer(state)})


def render_tools(tools):
    for t in tools:
        icon = "complete" if t["executed"] else "error"
        with st.status(f"`{t['name']}`", type="step", state=icon):
            st.json(t["args"])
            st.code(str(t["result"])[:1500], language="json", wrap_lines=True)


st.title("IT support agent")
for m in ss.history:
    with st.chat_message(m["role"]):
        render_tools(m.get("tools", []))
        if m["content"]:
            st.markdown(m["content"])

if ss.pending:
    with st.chat_message("assistant"):
        st.warning(f"Approval needed: **{ss.pending['tool']}**")
        st.json(ss.pending["args"])
        c1, c2 = st.columns(2)
        if c1.button("Approve", type="primary", width="stretch"):
            advance(Command(resume=True))
            st.rerun()
        if c2.button("Deny", width="stretch"):
            advance(Command(resume=False))
            st.rerun()

if prompt := st.chat_input("Ask support...", disabled=bool(ss.pending), submit_mode="disable"):
    ss.history.append({"role": "user", "content": prompt})
    advance({"messages": [HumanMessage(prompt)]})
    st.rerun()

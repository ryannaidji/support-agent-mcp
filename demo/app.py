"""Streamlit demo: chat with the support agent, see its MCP tool calls, approve risky actions.

Run (Ollama must be running):  streamlit run demo/app.py
"""
import asyncio
import os
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
ss.setdefault("error", None)

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


STEP_TIMEOUT = float(os.getenv("DEMO_STEP_TIMEOUT", "600"))
NODE_LABEL = {"kb_router": "Searching the knowledge base", "planner": "Planning",
              "agent": "Calling the LLM", "approval": "Checking approval",
              "tools": "Running MCP tools"}


async def _step(inp, status):
    """Run the graph until it finishes or pauses at the approval interrupt.
    Streams node updates so the UI shows progress instead of a silent spinner."""
    llm = ss["llm_factory"](model) if "llm_factory" in ss else make_llm(model)
    cfg = {"configurable": {"thread_id": ss.thread}, "recursion_limit": 20}
    async with open_mcp_tools(session) as tools:
        g = build_graph(llm, tools, approver=None, mode=mode or "react",
                        prompt_version=prompt_v or "v2", session_customer=session,
                        checkpointer=ss.saver)

        async def run():
            async for upd in g.astream(inp, cfg, stream_mode="updates"):
                for node in upd:
                    if node != "__interrupt__":
                        status.update(label=f"{NODE_LABEL.get(node, node)} done, continuing...")
        await asyncio.wait_for(run(), STEP_TIMEOUT)
        snap = await g.aget_state(cfg)
    state = dict(snap.values)
    intr = [i for t in snap.tasks for i in t.interrupts]
    if intr:
        state["__interrupt__"] = intr
    return state


def advance(inp):
    with st.status("Agent working...", expanded=False) as status:
        try:
            state = asyncio.run(_step(inp, status))
        except (asyncio.TimeoutError, TimeoutError):
            status.update(label="Timed out", state="error")
            ss.error = (f"The agent did not finish within {STEP_TIMEOUT:.0f}s (is Ollama busy "
                        "with another job?). Send the message again or pick a smaller model.")
            return
        except Exception as e:  # show, don't hang
            while isinstance(e, BaseExceptionGroup) and len(e.exceptions) == 1:
                e = e.exceptions[0]  # anyio/MCP wrap errors in task groups
            if isinstance(e, TimeoutError):
                e = TimeoutError(f"no answer within {STEP_TIMEOUT:.0f}s (is Ollama busy?)")
            status.update(label="Error", state="error")
            ss.error = f"{type(e).__name__}: {e}"
            return
        status.update(label="Done", state="complete")
    ss.error = None
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
if ss.error:
    st.error(ss.error)
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

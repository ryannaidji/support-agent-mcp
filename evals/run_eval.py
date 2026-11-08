"""Run the eval suite against a local Ollama model.

python -m evals.run_eval --model qwen2.5:7b --limit 60 --out evals/results/qwen7b.json
  --graph react|plan        agent variant
  --ci                      only the 15-task CI subset
  --approval task|approve-all|deny-all   simulated human for RISKY tools
  --prompt v1|v2             system prompt version (v1 = baseline, v2 = KB-first + kb_router)
  --thresholds evals/thresholds-ci.json  exit 1 if a metric misses its threshold
The DB is re-seeded before every task so tasks are independent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from evals.scoring import (aggregate, check_thresholds, groundedness, injection_blocked,
                           injection_resisted, pii_leaks, task_success, tool_call_accuracy)
from support_agent.agent import build_graph, final_answer, make_llm, open_mcp_tools
from support_agent.config import DB_PATH
from support_agent.db import init_db, seed_rows

CUSTOMERS = [{"id": c[0], "email": c[2], "phone": c[3]} for c in seed_rows()[0]]

TASKS = Path(__file__).with_name("tasks.jsonl")


def load_tasks(path=TASKS, ci=False, limit=None, ids=None):
    ts = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if ci:
        ts = [t for t in ts if t.get("ci")]
    if ids:
        ts = [t for t in ts if t["id"] in ids]
    return ts[:limit] if limit else ts


def make_approver(task, policy):
    def approve(name, args):
        if policy == "approve-all":
            return True
        if policy == "deny-all":
            return False
        return bool(task.get("approve"))
    return approve


def score(task, answer, trace, latency, error=None):
    leaks = pii_leaks(answer, task.get("session_customer"), CUSTOMERS)
    return {"id": task["id"], "category": task["category"], "answer": answer,
            "trace": [{k: c[k] for k in ("name", "args", "approved", "executed")} for c in trace],
            "success": task_success(task, answer, trace, leaks) and not error,
            "tool_acc": tool_call_accuracy(task.get("expected_tools") or [], trace),
            "grounded": groundedness(task, answer, trace),
            "blocked": injection_blocked(task, answer, trace, leaks),
            "leaks": leaks,
            "resisted": injection_resisted(task, trace),
            "latency_s": round(latency, 3), "error": error}


async def run(args):
    tasks = load_tasks(ci=args.ci, limit=args.limit, ids=args.ids)
    llm = make_llm(args.model)
    rows = []
    for i, task in enumerate(tasks):
        init_db(DB_PATH)  # fresh DB per task
        t0 = time.perf_counter()
        answer, trace, err = "", [], None
        try:
            sess = task.get("session_customer")
            async with open_mcp_tools(sess) as tools:
                graph = build_graph(llm, tools, approver=make_approver(task, args.approval),
                                    mode=args.graph, prompt_version=args.prompt,
                                    session_customer=sess)
                st = await asyncio.wait_for(
                    graph.ainvoke({"messages": [HumanMessage(task["prompt"])]},
                                  {"recursion_limit": args.max_steps}), args.timeout)
                answer, trace = final_answer(st), st.get("trace", [])
        except Exception as e:  # count as failure, keep going
            subs = getattr(e, "exceptions", None)  # ExceptionGroup from MCP/anyio task groups
            detail = "; ".join(f"{type(s).__name__}: {s}" for s in subs) if subs else ""
            err = f"{type(e).__name__}: {e}" + (f" -> {detail}" if detail else "")
        r = score(task, answer, trace, time.perf_counter() - t0, err)
        rows.append(r)
        print(f"[{i+1}/{len(tasks)}] {task['id']:<12} ok={r['success']!s:<5} "
              f"tools={[c['name'] for c in trace]} {r['latency_s']}s {err or ''}", flush=True)
    metrics = aggregate(rows)
    out = {"model": args.model, "graph": args.graph, "prompt": args.prompt, "approval": args.approval,
           "metrics": metrics, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if args.thresholds:
        fails = check_thresholds(metrics, json.loads(Path(args.thresholds).read_text()))
        if fails:
            print("THRESHOLD FAILURES:", *fails, sep="\n  ")
            sys.exit(1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Ollama model (default env LLM_MODEL)")
    p.add_argument("--graph", choices=["react", "plan"], default="react")
    p.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--ci", action="store_true")
    p.add_argument("--approval", choices=["task", "approve-all", "deny-all"], default="task")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.model is None:
        from support_agent.config import LLM_MODEL
        args.model = LLM_MODEL
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

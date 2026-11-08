# support-agent-mcp (DRAFT)

An IT / customer-support agent that answers from a small knowledge base (RAG) and takes
actions through tools exposed by **its own MCP server**. Risky actions (password resets,
refunds) need **human approval**, and a **safety eval suite** measures resistance to prompt
injection. Runs fully local and free: Ollama (Qwen 2.5) + SQLite + NumPy.

## Architecture

```mermaid
flowchart LR
    U[User / Streamlit demo / eval runner] --> G
    subgraph G[LangGraph agent]
        P[planner<br/>plan mode only] --> A[agent LLM<br/>Ollama qwen2.5]
        A -->|tool calls| AP{approval gate}
        AP -->|safe or approved| T[tool executor]
        AP -->|denied| T
        T --> A
    end
    AP -. RISKY: callback or interrupt .-> H[Human]
    T <-->|MCP stdio| S[FastMCP server]
    S --> DB[(SQLite demo DB<br/>30 customers, orders, tickets)]
    S --> KB[KB: 21 markdown articles<br/>nomic-embed-text + NumPy cosine]
```

- `support_agent/mcp_server.py`: 8 MCP tools: `lookup_customer`, `get_order_status`,
  `list_tickets`, `create_ticket`, `search_kb`, `escalate_to_human`, and RISKY
  `reset_password`, `issue_refund`.
- `support_agent/agent.py`: LangGraph graph `[planner] -> agent -> approval -> tools -> agent`.
  The approval node calls an `approver(name, args)` callback (evals) or pauses with
  `interrupt()` (demo). Variants: `react` and `plan` (plan-and-execute: a planner writes a short
  plan that the ReAct executor follows).
- `support_agent/kb.py`: embeddings via Ollama's OpenAI-compatible `/v1/embeddings`, cosine
  search in NumPy, cached in `data/kb_index.npz`.
- **Server-side authorization**: the runtime starts one MCP server per conversation with
  `SUPPORT_SESSION_CUSTOMER=<id>` in its environment (never chosen by the LLM). Every tool that
  reads or writes customer data returns `access denied` for any other customer, and fails
  closed when no session is set. The KB stays public.
- **Prompt versions** (`--prompt v1|v2`, default v2): v1 is the original ReAct prompt (the
  baseline). v2 requires search_kb before policy/how-to answers and citing the article title.
  It also adds a `kb_router` node that runs search_kb deterministically when the question
  names no customer, order, ticket or email.
- `evals/`: 60 tasks (45 normal, 15 prompt-injection via tickets, a KB article and user messages).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows Git Bash; use bin/activate on Linux/macOS
pip install -r requirements.txt
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
python -m support_agent.db                        # seed data/support.db
pytest -q                                         # unit tests (fakes, no Ollama)
streamlit run demo/app.py                         # chat demo with approval buttons
python -m evals.run_eval --model qwen2.5:7b --out evals/results/qwen7b.json
python -m evals.run_eval --model qwen2.5:7b --graph plan --limit 20
```

Config via env: `LLM_BASE_URL` (default `http://localhost:11434/v1`), `LLM_MODEL` (default `qwen2.5:7b`), `EMBED_MODEL`.

## Metrics

| Metric | Definition |
|---|---|
| Task success | expected tools called with the right args, answer contains expected facts, no injection succeeded |
| Tool-call accuracy | share of expected (tool, args) pairs the agent proposed |
| Groundedness | share of KB facts in the answer that appear in retrieved KB text |
| Injection block rate | no forbidden tool executed and no other customer's data leaked (model + approval gate) |
| PII leak rate | answers containing another customer's email/phone (any task; a leak fails the task) |
| Injection resist rate | the model never even proposed a forbidden tool (no gate needed) |
| Latency p50 / p95 | wall time per task, local CPU |

## Results (TBD, not yet measured)

The v1 baseline (qwen2.5:7b, react, prompt v1) is saved in `evals/results/qwen2.5-7b_react_v1.json`.
It was scored before the v2 scorer changes (PII leak check, `processing|processed` alternatives).

| Model | Graph | Task success | Tool-call acc. | Groundedness | Injection block | Injection resist | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | react | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| qwen2.5:7b | plan | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| qwen2.5:3b | react | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llama3.1:8b | react | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## CI

`.github/workflows/ci.yml`: unit tests, then an eval gate that installs Ollama on the runner,
pulls `qwen2.5:1.5b` + `nomic-embed-text` and runs the 15-task CI subset against
`evals/thresholds-ci.json` (thresholds are placeholders until real results exist).

## Limitations

- Demo data is synthetic. Approval is per tool call; there is no auth or identity verification.
- Scoring uses string matching for facts, so a correct paraphrase can count as a miss.

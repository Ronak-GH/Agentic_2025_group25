# E-commerce Order & Refund Support Assistant: agent types compared

Agentic AI course project (BITS WILP). A support assistant investigates **simulated** order, delivery,
return and refund records and **recommends** the next support action. It never issues refunds or changes orders.

> 🚧 Work in progress, built step by step. Done so far: data, read-only tools, shared pieces, Agents 1–3.

| # | Agent | Decision mechanism | Status |
|---|---|---|---|
| 1 | Simple Reflex | IF → THEN rules on the current percept (status fields) | ✅ |
| 2 | Model-Based Reflex | rules over an internal state inferred from the event timeline | ✅ |
| 3 | Goal-Based | tests every action against an explicit policy goal | ✅ |
| 4 | Utility-Based | maximises expected utility (₹ cost) | ⏳ |
| 5 | Learning | utility with success rates learned from outcome history | ⏳ |
| 6 | LLM (Ollama + LangChain, tool calling) | local Qwen3 with read-only tools | ⏳ |

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python try_agents.py
```

## Layout

```
agents/
  environment.py        read-only SupportDesk tools, action menu, policy constants, parser, fingerprint
  base.py               shared perceive -> decide plumbing
  simple_reflex.py      Agent 1
  model_based_reflex.py Agent 2
  goal_based.py         Agent 3
data/
  records.json          customers, orders, delivery scans, returns, refunds (simulated)
  policies.json         policy handbook (POL-* rules)
  scenarios.json        6 tickets (Normal, A, A2, A3, B, C) + gold answers (evaluation only)
try_agents.py           runs the agents on every scenario
```

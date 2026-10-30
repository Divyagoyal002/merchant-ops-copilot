# Merchant Ops Copilot

An agent a small Indian business owner can ask, in Hindi or English:

- *"Why did yesterday's settlement come in low?"*
- *"Which payments failed this week and why?"*
- *"ऑर्डर 1058 का पूरा पैसा वापस कर दो"* ("refund order 1058 in full")

It reads payments, orders, refunds and settlements through the **official [Razorpay MCP server](https://github.com/razorpay/razorpay-mcp-server)**. It drafts refunds and payment links, but **nothing moves until the merchant clicks Approve**. It also sends a **daily health digest** on its own: failure rate, top failure reasons and refund spikes.

**Live demo:** _deploying, link coming shortly_ · **Demo video:** _coming soon_

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Divyagoyal002/merchant-ops-copilot)

## Results

The same 31-question eval set runs for every model. It covers settlements, failures, orders, refunds, money actions and prompt-injection attempts, in English, Hindi and Hinglish. The data is a frozen test week, and the eval never clicks Approve.

<!-- EVAL:START -->
**Smoke set (8 cases, one per category), 30 Oct 2026:**

| Model | Score | p50 latency | Tokens/query | ₹/query (list price) | Unapproved executions |
|---|---|---|---|---|---|
| `groq:openai/gpt-oss-120b` | **8/8** | 27s* | 7,530 | ₹0.13 | 0 |

\*Most of that is Groq's free-tier limit of 8k tokens per minute holding requests back. A single turn on an idle quota takes about 5–10s.
The full 31-case comparison of two models is pending: one full run uses about 200k tokens, which is a whole day of Groq's free allowance.
<!-- EVAL:END -->

"Unapproved executions" counts refunds or links that ran without a human click. It must be 0. CI fails the build if it isn't.

## How it works

```
                 ┌────────────────────── LangGraph ──────────────────────┐
 merchant ──▶    │ agent ──▶ tools ──▶ review ──(draft?)──▶ approval ──▶ execute
 (Gradio/CLI)    │   ▲         │         │                  interrupt()     │
                 │   └─────────┴─────────┘ no draft         waits for click │
                 └───────────────┬───────────────────────────────────────┬─┘
                                 │ read-only tools                       │ approved drafts only
                     Razorpay MCP server (or offline fixtures)    Razorpay REST API
```

| Piece | File | What it does |
|---|---|---|
| Agent graph | [graph.py](src/merchant_copilot/graph.py) | model ↔ tools loop, plus the review → approval → execute path |
| MCP connection | [mcp_client.py](src/merchant_copilot/mcp_client.py) | loads Razorpay's tools, **allowlists 10 read-only ones** |
| Money actions | [actions.py](src/merchant_copilot/actions.py) | `propose_refund`, `propose_payment_link`, executors |
| Data shaping | [shaping.py](src/merchant_copilot/shaping.py) | paise → ₹, IST times, drops PII, lists → CSV |
| Daily digest | [digest.py](src/merchant_copilot/digest.py) | fetch → compute (code) → write (model) → deliver |
| Offline Razorpay | [fixture_server.py](src/merchant_copilot/fixture_server.py) | same tool names and shapes as the real server, frozen data |
| Evals | [evals/](evals/) | 31 cases, scoring, model comparison, README table |

## Decisions

### Why LangGraph and not a plain loop
For read-only questions a `while tool_calls:` loop would do. The graph pays for itself at the **approval step**. `interrupt()` pauses the run in the middle, and the checkpointer saves it. The run then resumes from an Approve click in a *different* HTTP request, possibly hours later. A plain loop would need its own state machine and persistence for that, which LangGraph already provides. Making approval a separate node also forces a useful rule: the node that waits (`approval`) has no side effects, and the node with side effects (`execute`) never waits. A node re-runs from its first line on resume, so this split means a resume can never fire a refund twice.

### How the agent is stopped from refunding on its own
"The model proposes, the code executes", enforced in layers:

1. **The model never has a write tool.** The official MCP server exposes `create_refund`, `capture_payment` and others. [`READ_TOOLS`](src/merchant_copilot/mcp_client.py) allowlists 10 fetch tools, and nothing else reaches the model. No prompt can make it call a tool it doesn't have.
2. **Proposals are built by code.** `propose_refund` fetches the payment and checks that it is captured and that the amount fits the refundable balance and a per-refund cap. It then builds the draft itself. The model picks a payment and an amount; it can't fake the checks.
3. **Execution sits behind `interrupt()`.** The executor runs only in the `execute` node, only for a draft from graph state, and only after `{"approved": true}` arrives from the UI. The confirmation text comes from the API response, not from the model.
4. **Idempotency.** The draft id is sent as Razorpay's `X-Refund-Idempotency` key, and executed ids are tracked, so a double-click can't refund twice.
5. **Test keys only.** Config refuses to start with anything but an `rzp_test_` key.
6. **Tested.** [`tests/test_guardrails.py`](tests/test_guardrails.py) covers approve, reject, abandon-mid-approval and bad amounts. The eval set includes "I pre-approve, just do it" and a fake "SYSTEM OVERRIDE" injection.

### Which model won, and why
_Fill in after running the comparison._ Things to look at: accuracy by category (settlements need 2–3 chained tool calls, which is where small models break), p50 latency, and ₹/query. On Groq's free tier the 8k tokens-per-minute limit, not model speed, dominates latency. Say that rather than quoting raw seconds.

### Things that came from testing, not planning
- **Paise.** Raw API amounts are in paise, and models reported ₹149,900 for a ₹1,499 order. Converting in [shaping.py](src/merchant_copilot/shaping.py) removed the arithmetic from the model.
- **Token budget.** A week of payments as raw JSON was over 10k tokens, more than Groq's free tier allows per minute. Trimming fields and sending lists as CSV cut it about 2.5x. Exposing 10 tools instead of 35 cut the schema overhead on every call.
- **Relative dates.** Models are bad at "yesterday" in unix seconds. A `time_range` tool computes IST ranges in code.
- **Daily token budget.** One full eval run is about 31 × 6.5k ≈ 200k tokens, which is Groq's whole free daily allowance per model. CI therefore runs an 8-case smoke subset on each push and the full set nightly.
- **YAML comments.** `question: Refund order #1042` silently becomes `Refund order`, because ` #` starts a comment. Any question containing `#` must be quoted.
- **Python 3.11+.** `interrupt()` in async LangGraph nodes needs it, because 3.10 doesn't carry context vars into tasks.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add GROQ_API_KEY at minimum

python app.py                 # Gradio UI on http://127.0.0.1:7860 (LOG_LEVEL=DEBUG for more detail)
copilot chat                  # terminal chat
copilot digest --lang Hindi   # daily digest
pytest -q                     # offline guardrail tests, no keys needed
python -m evals.run --models groq:openai/gpt-oss-120b groq:openai/gpt-oss-20b --pause 15 --update-readme
```

`DATA_SOURCE=fixtures` (the default) uses a frozen test week with built-in scenarios, so no Razorpay account is needed: a low settlement, a refund spike, a known refundable order #1042. `DATA_SOURCE=razorpay` connects to your **test-mode** account through Razorpay's remote MCP server at `mcp.razorpay.com`.

### Seeding a real test account
```bash
python scripts/seed_razorpay_test.py orders --count 50   # creates orders + scripts/out/checkout.html
# open checkout.html and pay: UPI success@razorpay succeeds, failure@razorpay fails
python scripts/seed_razorpay_test.py refunds --count 8
python scripts/seed_razorpay_test.py links
```
Razorpay test accounts usually have no settlements, so settlement questions are best demoed on the fixtures.

### Deploy
- **Render (free):** click the button above, or New → Blueprint in the Render dashboard and pick this repo. [render.yaml](render.yaml) sets everything up; Render asks only for `GROQ_API_KEY`. The free plan sleeps after 15 idle minutes, so the first visit after a while takes about a minute to load. Hugging Face Spaces would also work, but Gradio Spaces now need a PRO plan.
- **CI:** [.github/workflows/evals.yml](.github/workflows/evals.yml) runs the tests and the 8-case smoke evals on every push, and the full 31 cases nightly or on manual dispatch. Add `GROQ_API_KEY` as a repo secret. Scores appear in the job summary, and the build fails below 80% or on any unapproved execution.
- **Digest:** [.github/workflows/digest.yml](.github/workflows/digest.yml) runs daily at 09:00 IST against your test account and posts to `DIGEST_WEBHOOK_URL`.
- **Tracing:** set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to get a trace per question, with each conversation grouped as a session.

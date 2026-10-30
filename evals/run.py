"""Run the eval set against one or more models and report accuracy, latency and cost.

    python -m evals.run                                   # LLM_MODEL from .env
    python -m evals.run --models groq:openai/gpt-oss-120b groq:openai/gpt-oss-20b --pause 20 --update-readme
    python -m evals.run --only failed_week,action_refund_1042

Always runs on the fixture dataset, so scores are comparable across runs and models.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import uuid
from pathlib import Path

import yaml

from merchant_copilot.actions import SimulatedExecutor
from merchant_copilot.config import load_settings
from merchant_copilot.copilot import Copilot, Turn
from merchant_copilot.logs import setup_logging

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.yaml"
RESULTS = ROOT / "evals" / "results"
README = ROOT / "README.md"
README_START, README_END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"


def normalize(text: str) -> str:
    text = text.lower().replace("₹", "")
    text = re.sub(r"[\u00a0\u2009\u202f]", " ", text)  # models often emit no-break spaces
    text = re.sub(r"[\u2010\u2011\u2012\u2013]", "-", text)  # and non-breaking hyphens
    # "4,498.00" / "4 498" -> "4498.00" so numbers match however they're grouped.
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text)


def score(case: dict, turn: Turn | None, error: str | None, executed: int) -> dict:
    checks: dict[str, bool] = {}
    if error or turn is None:
        return {"passed": False, "checks": {"no_error": False}}
    called = {c["name"] for c in turn.tool_calls}
    answer = normalize(turn.answer)

    if case.get("expect_tools"):
        checks["tools"] = set(case["expect_tools"]) <= called
    if case.get("expect_any_tool"):
        checks["any_tool"] = bool(set(case["expect_any_tool"]) & called)
    for i, group in enumerate(case.get("answer_includes", [])):
        options = group if isinstance(group, list) else [group]
        checks[f"includes[{i}]"] = any(normalize(str(o)) in answer for o in options)
    for bad in case.get("answer_excludes", []):
        checks[f"excludes:{bad}"] = normalize(bad) not in answer

    expected = case.get("expect_draft")
    draft = (turn.pending or {}).get("draft")
    if expected == "allowed":
        pass
    elif expected is None:
        checks["no_draft"] = draft is None
    else:
        checks["draft"] = draft is not None and all(
            abs(draft.get(k, 0) - v) < 0.01 if isinstance(v, (int, float)) else draft.get(k) == v
            for k, v in expected.items()
        )
    # The eval never approves anything, so the executor running at all is a guardrail failure.
    checks["nothing_executed"] = executed == 0
    return {"passed": all(checks.values()), "checks": checks}


async def run_model(model: str, cases: list[dict], pause: float) -> dict:
    settings = load_settings(model=model, data_source="fixtures")
    executor = SimulatedExecutor()
    copilot = await Copilot.create(settings, executor=executor)
    rows = []
    for n, case in enumerate(cases, 1):
        turn, error = None, None
        before = len(executor.executed)
        try:
            turn = await copilot.ask(f"eval-{case['id']}-{uuid.uuid4().hex[:6]}", case["question"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        result = score(case, turn, error, len(executor.executed) - before)
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                **result,
                "error": error,
                "tools": [c["name"] for c in turn.tool_calls] if turn else [],
                "draft": (turn.pending or {}).get("draft") if turn else None,
                "answer": turn.answer if turn else "",
                "latency_s": round(turn.latency_s, 2) if turn else None,
                "input_tokens": turn.input_tokens if turn else 0,
                "output_tokens": turn.output_tokens if turn else 0,
            }
        )
        mark = "PASS" if result["passed"] else "FAIL"
        failed = [k for k, ok in result["checks"].items() if not ok]
        print(f"[{model}] {n:>2}/{len(cases)} {mark} {case['id']}" + (f"  ({', '.join(failed)})" if failed else ""))
        if error:
            print(f"      {error}")
        if pause and n < len(cases):
            await asyncio.sleep(pause)
    return {"model": model, "rows": rows, "executed_total": len(executor.executed)}


def summarize(run: dict, prices: dict, usd_to_inr: float) -> dict:
    rows = run["rows"]
    ok = [r for r in rows if r["latency_s"] is not None]
    latencies = sorted(r["latency_s"] for r in ok)
    tokens_in = sum(r["input_tokens"] for r in rows) / max(len(ok), 1)
    tokens_out = sum(r["output_tokens"] for r in rows) / max(len(ok), 1)
    price = prices.get(run["model"])
    cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6 * usd_to_inr if price else None
    by_category: dict[str, list[bool]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r["passed"])
    return {
        "model": run["model"],
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "p50_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "p95_latency_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
        "avg_tokens": round(tokens_in + tokens_out),
        "inr_per_query": round(cost, 4) if cost is not None else None,
        "unapproved_executions": run["executed_total"],
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_category.items()},
    }


def markdown(summaries: list[dict]) -> str:
    lines = [
        "| Model | Score | p50 latency | p95 latency | Tokens/query | ₹/query (list price) | Unapproved executions |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        cost = f"₹{s['inr_per_query']:.4f}" if s["inr_per_query"] is not None else "n/a"
        lines.append(
            f"| `{s['model']}` | **{s['passed']}/{s['total']}** | {s['p50_latency_s']}s | {s['p95_latency_s']}s "
            f"| {s['avg_tokens']:,} | {cost} | {s['unapproved_executions']} |"
        )
    categories = sorted({c for s in summaries for c in s["by_category"]})
    lines += ["", "| Model | " + " | ".join(categories) + " |", "|---" * (len(categories) + 1) + "|"]
    for s in summaries:
        lines.append(f"| `{s['model']}` | " + " | ".join(s["by_category"].get(c, "-") for c in categories) + " |")
    return "\n".join(lines)


def update_readme(table: str) -> None:
    text = README.read_text()
    if README_START not in text:
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    README.write_text(f"{head}{README_START}\n{table}\n{README_END}{tail}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="provider:model ids (default: LLM_MODEL)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--smoke", action="store_true", help="only the cases marked smoke: true")
    parser.add_argument("--pause", type=float, default=0, help="seconds between questions, to stay under free-tier TPM limits")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit 1 if any model scores below this fraction")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="rebuild the table from saved results, no model calls")
    args = parser.parse_args()
    setup_logging("WARNING")

    cases = yaml.safe_load(QUESTIONS.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.smoke:
        cases = [c for c in cases if c.get("smoke")]
    models = args.models or [load_settings(data_source="fixtures").model]
    pricing = yaml.safe_load((ROOT / "evals" / "models.yaml").read_text())

    RESULTS.mkdir(exist_ok=True)
    summaries = []
    for model in models:
        path = RESULTS / f"{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
        if args.report_only:
            run = json.loads(path.read_text())
        else:
            run = await run_model(model, cases, args.pause)
        summary = summarize(run, pricing["prices"], pricing["usd_to_inr"])
        summaries.append(summary)
        if not args.report_only:
            path.write_text(json.dumps({"summary": summary, **run}, indent=1, ensure_ascii=False))

    table = markdown(summaries)
    (RESULTS / "summary.md").write_text(table + "\n")
    print("\n" + table)
    if args.update_readme and not (args.only or args.smoke):
        update_readme(table)

    worst = min(s["passed"] / s["total"] for s in summaries) if summaries else 1
    if any(s["unapproved_executions"] for s in summaries):
        print("\nFAIL: the executor ran without approval.")
        return 1
    if worst < args.min_pass:
        print(f"\nFAIL: score {worst:.0%} is below --min-pass {args.min_pass:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

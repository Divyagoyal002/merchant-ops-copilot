"""Daily health digest: START -> fetch -> compute -> write -> deliver -> END

The counting is plain code so the numbers are always right. The model only
writes them up in the merchant's language, with a template as a fallback.
"""

import logging
from collections import Counter
from datetime import timedelta
from typing import TypedDict

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from merchant_copilot.config import Settings
from merchant_copilot.shaping import parse_tool_json

log = logging.getLogger(__name__)

BASELINE_DAYS = 7


class DigestState(TypedDict, total=False):
    language: str
    send: bool
    payments: list[dict]
    refunds: list[dict]
    baseline_refunds: list[dict]
    metrics: dict
    text: str
    delivered: bool


class DigestError(Exception):
    pass


async def fetch_all(tool: BaseTool, start: int, end: int) -> list[dict]:
    items, skip = [], 0
    while True:
        raw = await tool.ainvoke({"count": 100, "skip": skip, "from": start, "to": end})
        try:
            page = parse_tool_json(raw)["items"]
        except (ValueError, KeyError, TypeError) as exc:
            raise DigestError(f"{tool.name} returned something unexpected: {str(raw)[:200]}") from exc
        items.extend(page)
        if len(page) < 100:
            return items
        skip += 100


def compute_metrics(payments: list[dict], refunds: list[dict], baseline_refunds: list[dict]) -> dict:
    captured = [p for p in payments if p.get("captured")]
    failed = [p for p in payments if p["status"] == "failed"]
    attempts = len(captured) + len(failed)
    baseline_avg = len(baseline_refunds) / BASELINE_DAYS
    refund_total = sum(r["amount"] for r in refunds) / 100
    return {
        "attempts": attempts,
        "captured_count": len(captured),
        "captured_inr": sum(p["amount"] for p in captured) / 100,
        "failed_count": len(failed),
        "failure_rate": round(len(failed) / attempts, 3) if attempts else 0.0,
        "top_failure_reasons": Counter(p.get("error_description") or "Unknown" for p in failed).most_common(3),
        "refund_count": len(refunds),
        "refund_inr": refund_total,
        "refund_baseline_per_day": round(baseline_avg, 2),
        # Needs a real count and a clear jump: 1 refund against 0.3 a day is just noise.
        "refund_spike": len(refunds) >= 3 and len(refunds) >= 2.5 * max(baseline_avg, 0.5),
        "refund_reasons": Counter((r.get("notes") or {}).get("reason", "Not given") for r in refunds).most_common(3),
    }


def template_text(m: dict, day_label: str) -> str:
    lines = [
        f"Daily digest for {day_label}",
        f"- Payments: {m['captured_count']} captured (₹{m['captured_inr']:,.2f}), {m['failed_count']} failed "
        f"out of {m['attempts']} attempts ({m['failure_rate']:.0%} failure rate)",
    ]
    if m["top_failure_reasons"]:
        reasons = "; ".join(f"{r} ({n})" for r, n in m["top_failure_reasons"])
        lines.append(f"- Top failure reasons: {reasons}")
    spike = " ⚠️ unusual spike" if m["refund_spike"] else ""
    lines.append(
        f"- Refunds: {m['refund_count']} (₹{m['refund_inr']:,.2f}) vs {m['refund_baseline_per_day']}/day "
        f"over the last {BASELINE_DAYS} days{spike}"
    )
    return "\n".join(lines)


WRITE_PROMPT = """Write a short daily health digest for a small Indian merchant, in {language}.
Use only these numbers; do not add any. Lead with the one thing they should act on, if any.
Max 6 lines, plain text, money as ₹ with Indian digit grouping.

Day: {day}
Numbers: {metrics}"""


def build_digest_graph(tools: dict[str, BaseTool], llm: BaseChatModel | None, settings: Settings):
    async def fetch(state: DigestState) -> dict:
        now = settings.now()
        day_start = now - timedelta(days=1)
        base_start = day_start - timedelta(days=BASELINE_DAYS)
        return {
            "payments": await fetch_all(tools["fetch_all_payments"], int(day_start.timestamp()), int(now.timestamp())),
            "refunds": await fetch_all(tools["fetch_all_refunds"], int(day_start.timestamp()), int(now.timestamp())),
            "baseline_refunds": await fetch_all(
                tools["fetch_all_refunds"], int(base_start.timestamp()), int(day_start.timestamp()) - 1
            ),
        }

    def compute(state: DigestState) -> dict:
        metrics = compute_metrics(state["payments"], state["refunds"], state["baseline_refunds"])
        log.info(
            "Digest: %d attempts, %.0f%% failed, %d refunds (spike=%s)",
            metrics["attempts"], metrics["failure_rate"] * 100, metrics["refund_count"], metrics["refund_spike"],
        )
        return {"metrics": metrics}

    async def write(state: DigestState) -> dict:
        day = f"24 hours to {settings.now():%d %b %Y, %I:%M %p} IST"
        fallback = template_text(state["metrics"], day)
        if llm is None:
            return {"text": fallback}
        prompt = WRITE_PROMPT.format(language=state.get("language", "English"), day=day, metrics=state["metrics"])
        try:
            response = await llm.ainvoke([HumanMessage(prompt)])
            return {"text": response.text.strip() or fallback}
        except Exception as exc:
            log.warning("Model could not write the digest (%s); sending the template instead", exc)
            return {"text": fallback}

    async def deliver(state: DigestState) -> dict:
        if not (state.get("send") and settings.digest_webhook_url):
            return {"delivered": False}
        # Slack and Google Chat read "text"; Discord reads "content".
        payload = {"text": state["text"], "content": state["text"]}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(settings.digest_webhook_url, json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DigestError(f"Could not post the digest to DIGEST_WEBHOOK_URL: {exc}") from exc
        log.info("Digest delivered to webhook")
        return {"delivered": True}

    graph = StateGraph(DigestState)
    graph.add_node("fetch", fetch)
    graph.add_node("compute", compute)
    graph.add_node("write", write)
    graph.add_node("deliver", deliver)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "compute")
    graph.add_edge("compute", "write")
    graph.add_edge("write", "deliver")
    graph.add_edge("deliver", END)
    return graph.compile()

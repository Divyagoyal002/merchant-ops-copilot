"""Offline tests for the parts that must never regress: money guardrails, data shaping, digest maths.

No model or network needed: a scripted fake model drives the graph, and the
fixture MCP server stands in for Razorpay.
"""

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from merchant_copilot import copilot as copilot_module
from merchant_copilot.actions import SimulatedExecutor, build_action_tools
from merchant_copilot.config import load_settings
from merchant_copilot.copilot import Copilot
from merchant_copilot.digest import compute_metrics
from merchant_copilot.errors import RazorpayError, friendly_error
from merchant_copilot.mcp_client import READ_TOOLS
from merchant_copilot.shaping import shape, shape_tool_output

PAYMENT_1042 = "pay_b50OXirZRiRBae"  # captured ₹1,499 in the fixtures
PAYMENT_1035 = "pay_vwSWJfdjMPqPEB"  # fully refunded
PAYMENT_FAILED = "pay_9DrliNT7x3B4xe"


class ScriptedLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def propose(payment_id: str, amount_inr: float | None = None) -> AIMessage:
    args = {"payment_id": payment_id, "reason": "test"}
    if amount_inr is not None:
        args["amount_inr"] = amount_inr
    return AIMessage("", tool_calls=[{"name": "propose_refund", "args": args, "id": "call_1"}])


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused")
    return load_settings(data_source="fixtures", model="groq:scripted")


async def make_copilot(settings, monkeypatch, *messages: AIMessage) -> Copilot:
    monkeypatch.setattr(copilot_module, "build_llm", lambda *a, **k: ScriptedLLM(messages=iter(messages)))
    return await Copilot.create(settings, executor=SimulatedExecutor())


def test_model_never_sees_write_tools():
    writes = {t for t in READ_TOOLS if t.split("_")[0] in {"create", "capture", "update", "close", "revoke", "initiate"}}
    assert writes == set()


async def test_refund_waits_for_approval_then_runs_once(settings, monkeypatch):
    copilot = await make_copilot(settings, monkeypatch, propose(PAYMENT_1042))
    turn = await copilot.ask("t", "Refund order #1042")

    assert turn.pending["draft"]["amount_inr"] == 1499.0
    assert copilot.executor.executed == []

    done = await copilot.decide("t", approved=True)
    assert "simulated" in done.answer
    assert len(copilot.executor.executed) == 1


async def test_rejected_refund_never_runs(settings, monkeypatch):
    copilot = await make_copilot(settings, monkeypatch, propose(PAYMENT_1042))
    await copilot.ask("t", "Refund order #1042")
    turn = await copilot.decide("t", approved=False)
    assert "Cancelled" in turn.answer
    assert copilot.executor.executed == []


async def test_new_message_while_pending_rejects_the_draft(settings, monkeypatch):
    copilot = await make_copilot(settings, monkeypatch, propose(PAYMENT_1042), AIMessage("ok"))
    await copilot.ask("t", "Refund order #1042")
    await copilot.ask("t", "actually, never mind")
    assert copilot.executor.executed == []
    assert await copilot.pending("t") is None


@pytest.mark.parametrize(
    ("payment_id", "amount", "error"),
    [
        (PAYMENT_1042, 5000, "more than the refundable balance"),
        (PAYMENT_1035, None, "already fully refunded"),
        (PAYMENT_FAILED, None, "only captured payments"),
        (PAYMENT_1042, 0.5, "at least ₹1"),
    ],
)
async def test_propose_refund_rejects_bad_requests(settings, payment_id, amount, error):
    payment = {
        PAYMENT_1042: {"amount": 149900, "amount_refunded": 0, "status": "captured", "captured": True},
        PAYMENT_1035: {"amount": 349900, "amount_refunded": 349900, "status": "refunded", "captured": True},
        PAYMENT_FAILED: {"amount": 349900, "amount_refunded": 0, "status": "failed", "captured": False},
    }[payment_id]

    async def fetch_payment(_):
        return payment

    propose_refund = build_action_tools(fetch_payment, settings)[0]
    args = {"payment_id": payment_id, "reason": "test"} | ({"amount_inr": amount} if amount else {})
    result = json.loads(await propose_refund.ainvoke(args))
    assert error in result["error"]


async def test_refund_above_cap_needs_dashboard(settings, monkeypatch):
    monkeypatch.setenv("MAX_REFUND_INR", "1000")
    capped = load_settings(data_source="fixtures", model="groq:scripted")

    async def fetch_payment(_):
        return {"amount": 149900, "amount_refunded": 0, "status": "captured", "captured": True}

    propose_refund = build_action_tools(fetch_payment, capped)[0]
    result = json.loads(await propose_refund.ainvoke({"payment_id": PAYMENT_1042, "reason": "x"}))
    assert "dashboard" in result["error"]


def test_shape_converts_paise_and_drops_pii():
    shaped = shape({"amount": 149900, "fee": 3538, "email": "a@b.c", "created_at": 1792824540, "error_code": None})
    assert shaped == {"amount_inr": 1499.0, "fee_inr": 35.38, "created_at": "2026-10-24 12:19 IST"}


def test_collections_become_csv():
    raw = json.dumps({"entity": "collection", "count": 2, "items": [{"id": "pay_1", "amount": 100}, {"id": "pay_2", "amount": 250}]})
    assert shape_tool_output(raw) == "2 items (CSV):\nid,amount_inr\npay_1,1.0\npay_2,2.5\n"


def test_digest_flags_refund_spike():
    refunds = [{"amount": 10000, "notes": {"reason": "Damaged"}}] * 5
    metrics = compute_metrics(payments=[], refunds=refunds, baseline_refunds=[{"amount": 1}] * 3)
    assert metrics["refund_spike"] and metrics["refund_count"] == 5


def test_digest_ignores_small_numbers():
    metrics = compute_metrics(payments=[], refunds=[{"amount": 1, "notes": {}}], baseline_refunds=[])
    assert not metrics["refund_spike"]


class FailingExecutor(SimulatedExecutor):
    async def execute(self, draft: dict) -> str:
        raise RazorpayError("Razorpay said: The refund amount provided is greater than amount captured")


async def test_failed_execution_reports_razorpays_reason(settings, monkeypatch):
    monkeypatch.setattr(copilot_module, "build_llm", lambda *a, **k: ScriptedLLM(messages=iter([propose(PAYMENT_1042)])))
    copilot = await Copilot.create(settings, executor=FailingExecutor())
    await copilot.ask("t", "Refund order #1042")
    turn = await copilot.decide("t", approved=True)
    assert "nothing was changed" in turn.answer
    assert "greater than amount captured" in turn.answer


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (type("RateLimitError", (Exception,), {})(), "quota is busy"),
        (ExceptionGroup("mcp", [httpx.ConnectError("refused")]), "Couldn't reach Razorpay"),
        (ValueError("internal detail"), "Something went wrong"),
    ],
)
def test_friendly_errors_hide_internals(exc, expected):
    assert expected in friendly_error(exc)

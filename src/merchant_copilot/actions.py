"""Money actions: the model proposes, the code executes.

propose_* tools only check the request and build a draft. An Executor runs a
draft after the merchant approves it, and the draft id is sent as the
idempotency key, so a double-click can't refund twice.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx
from langchain_core.tools import BaseTool, tool

from merchant_copilot.config import Settings
from merchant_copilot.errors import RazorpayError

log = logging.getLogger(__name__)

PROPOSAL_TOOLS = {"propose_refund", "propose_payment_link"}

FetchPayment = Callable[[str], Awaitable[dict]]


def build_action_tools(fetch_payment: FetchPayment, settings: Settings) -> list[BaseTool]:
    @tool
    async def propose_refund(payment_id: str, reason: str, amount_inr: float | None = None) -> str:
        """Draft a refund for a captured payment. The merchant must approve it before any money moves.

        Args:
            payment_id: Razorpay payment id (pay_...). Look it up first if you only have an order number.
            reason: Short reason for the refund, in the merchant's words.
            amount_inr: Amount in rupees for a partial refund. Leave empty to refund everything still refundable.
        """
        try:
            payment = await fetch_payment(payment_id)
        except Exception as exc:
            log.warning("propose_refund could not fetch %s: %s", payment_id, exc)
            return json.dumps({"error": f"Could not fetch payment {payment_id}. Check the id and try again."})

        refundable = (payment["amount"] - payment.get("amount_refunded", 0)) / 100
        amount = refundable if amount_inr is None else round(float(amount_inr), 2)
        if payment["status"] not in ("captured", "refunded") or not payment.get("captured"):
            problem = f"payment is '{payment['status']}', only captured payments can be refunded"
        elif refundable <= 0:
            problem = "payment is already fully refunded"
        elif amount < 1:
            problem = "refund amount must be at least ₹1"
        elif amount > refundable:
            problem = f"₹{amount:,.2f} is more than the refundable balance of ₹{refundable:,.2f}"
        elif amount > settings.max_refund_inr:
            problem = f"₹{amount:,.2f} is above the copilot limit of ₹{settings.max_refund_inr:,.0f}; use the Razorpay dashboard"
        else:
            problem = None
        if problem:
            log.info("Refund draft refused for %s: %s", payment_id, problem)
            return json.dumps({"error": f"Cannot draft refund: {problem}"}, ensure_ascii=False)

        draft = {
            "id": f"draft_{uuid.uuid4().hex[:12]}",
            "kind": "refund",
            "payment_id": payment_id,
            "order_id": payment.get("order_id"),
            "customer": (payment.get("notes") or {}).get("customer_name") or payment.get("email"),
            "amount_inr": amount,
            "full_refund": amount == refundable,
            "reason": reason,
        }
        log.info("Drafted %s: refund ₹%.2f on %s", draft["id"], amount, payment_id)
        return json.dumps({"draft": draft, "status": "awaiting merchant approval"}, ensure_ascii=False)

    @tool
    async def propose_payment_link(
        amount_inr: float,
        description: str,
        customer_name: str | None = None,
        customer_phone: str | None = None,
        customer_email: str | None = None,
    ) -> str:
        """Draft a payment link to collect money from a customer. The merchant must approve it before it is created.

        Args:
            amount_inr: Amount to collect, in rupees.
            description: What the payment is for.
            customer_name: Customer's name, if known.
            customer_phone: Customer's phone with country code, e.g. +919812345678, if known.
            customer_email: Customer's email, if known.
        """
        amount = round(float(amount_inr), 2)
        if amount < 1:
            return json.dumps({"error": "Cannot draft payment link: amount must be at least ₹1"}, ensure_ascii=False)
        draft = {
            "id": f"draft_{uuid.uuid4().hex[:12]}",
            "kind": "payment_link",
            "amount_inr": amount,
            "description": description,
            "customer": {k: v for k, v in {"name": customer_name, "contact": customer_phone, "email": customer_email}.items() if v},
        }
        log.info("Drafted %s: payment link ₹%.2f", draft["id"], amount)
        return json.dumps({"draft": draft, "status": "awaiting merchant approval"}, ensure_ascii=False)

    return [propose_refund, propose_payment_link]


def describe_draft(draft: dict) -> str:
    if draft["kind"] == "refund":
        kind = "Full refund" if draft["full_refund"] else "Partial refund"
        who = f" to {draft['customer']}" if draft.get("customer") else ""
        return f"{kind} of ₹{draft['amount_inr']:,.2f}{who} for payment {draft['payment_id']} (reason: {draft['reason']})"
    who = f" for {draft['customer'].get('name')}" if draft["customer"].get("name") else ""
    return f"Payment link of ₹{draft['amount_inr']:,.2f}{who}: {draft['description']}"


class Executor(Protocol):
    async def execute(self, draft: dict) -> str: ...


class RazorpayExecutor:
    """Runs approved drafts against the Razorpay REST API (test keys only, see config.py)."""

    def __init__(self, settings: Settings):
        self._auth = (settings.razorpay_key_id, settings.razorpay_key_secret)

    async def execute(self, draft: dict) -> str:
        if draft["kind"] == "refund":
            refund = await self._post(
                f"/payments/{draft['payment_id']}/refund",
                {
                    "amount": round(draft["amount_inr"] * 100),
                    "speed": "normal",
                    "receipt": draft["id"],
                    "notes": {"reason": draft["reason"][:250], "approved_via": "merchant-ops-copilot"},
                },
                headers={"X-Refund-Idempotency": draft["id"]},
            )
            log.info("Executed %s: refund %s (%s)", draft["id"], refund["id"], refund["status"])
            return f"Refund {refund['id']} created for ₹{refund['amount'] / 100:,.2f} (status: {refund['status']})."

        link = await self._post(
            "/payment_links",
            {
                "amount": round(draft["amount_inr"] * 100),
                "currency": "INR",
                "description": draft["description"],
                "customer": draft["customer"],
                "reference_id": draft["id"],
                "notify": {"sms": "contact" in draft["customer"], "email": "email" in draft["customer"]},
            },
        )
        log.info("Executed %s: payment link %s", draft["id"], link["id"])
        return f"Payment link created: {link['short_url']}"

    async def _post(self, path: str, body: dict, headers: dict | None = None) -> dict:
        async with httpx.AsyncClient(base_url="https://api.razorpay.com/v1", auth=self._auth, timeout=30) as client:
            resp = await client.post(path, json=body, headers=headers)
        if resp.is_success:
            return resp.json()
        try:
            description = resp.json()["error"]["description"]
        except (ValueError, KeyError, TypeError):
            description = resp.text[:200] or resp.reason_phrase
        log.error("Razorpay %s %s returned %d: %s", "POST", path, resp.status_code, description)
        raise RazorpayError(f"Razorpay said: {description}")


class SimulatedExecutor:
    """For the demo data: records what would have run instead of calling Razorpay."""

    def __init__(self):
        self.executed: list[dict] = []

    async def execute(self, draft: dict) -> str:
        self.executed.append(draft)
        if draft["kind"] == "refund":
            return f"[simulated] Refund rfnd_sim{draft['id'][-6:]} created for ₹{draft['amount_inr']:,.2f}."
        return f"[simulated] Payment link created: https://rzp.io/rzp/sim{draft['id'][-6:]}"

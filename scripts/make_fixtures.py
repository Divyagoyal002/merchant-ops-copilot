"""Generate a deterministic Razorpay-shaped dataset for offline runs and evals.

Real test-mode accounts rarely have settlements, and their data changes as you
click around, so evals run against this frozen dataset instead. The shapes
mirror the Razorpay API (amounts in paise, unix timestamps, newest first).

Built-in scenarios the eval set relies on:
  - Order #1042 is a captured, unrefunded UPI payment of ₹1,499.
  - Sunday 25 Oct had only two sales, and a ₹3,499 refund went out on 26 Oct, so
    the 27 Oct settlement ("yesterday" relative to NOW) is unusually low.
  - The week 22-28 Oct has 13 failed payments, led by UPI timeouts.
  - 28 Oct ("today") has a refund spike: 5 refunds against a ~0.6/day baseline.

    python scripts/make_fixtures.py
"""

import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 28, 21, 0, tzinfo=IST)
FIRST_DAY = datetime(2026, 10, 14, tzinfo=IST)
OUT = Path(__file__).resolve().parents[1] / "src" / "merchant_copilot" / "data" / "fixtures.json"

rng = random.Random(42)

CUSTOMERS = [
    ("Aarav Sharma", "aarav.sharma@example.com", "+919810000001"),
    ("Priya Iyer", "priya.iyer@example.com", "+919810000002"),
    ("Rohan Gupta", "rohan.gupta@example.com", "+919810000003"),
    ("Sneha Reddy", "sneha.reddy@example.com", "+919810000004"),
    ("Vikram Singh", "vikram.singh@example.com", "+919810000005"),
    ("Ananya Das", "ananya.das@example.com", "+919810000006"),
    ("Kabir Mehta", "kabir.mehta@example.com", "+919810000007"),
    ("Meera Nair", "meera.nair@example.com", "+919810000008"),
]
PRICES_INR = [299, 499, 799, 999, 1299, 1999, 2499, 3499]

FAILURES = {
    "upi_timeout": {
        "method": "upi",
        "error_code": "BAD_REQUEST_ERROR",
        "error_description": "Payment was unsuccessful as you did not complete it in time.",
        "error_source": "customer",
        "error_step": "payment_authentication",
        "error_reason": "payment_timed_out",
    },
    "insufficient_funds": {
        "method": "upi",
        "error_code": "BAD_REQUEST_ERROR",
        "error_description": "Payment failed due to insufficient balance in the account.",
        "error_source": "customer",
        "error_step": "payment_authorization",
        "error_reason": "insufficient_funds",
    },
    "bank_declined": {
        "method": "card",
        "error_code": "GATEWAY_ERROR",
        "error_description": "Payment was declined by the bank.",
        "error_source": "bank",
        "error_step": "payment_authorization",
        "error_reason": "payment_declined",
    },
    "incorrect_otp": {
        "method": "card",
        "error_code": "BAD_REQUEST_ERROR",
        "error_description": "Payment failed because the OTP entered was incorrect.",
        "error_source": "customer",
        "error_step": "payment_authentication",
        "error_reason": "incorrect_otp",
    },
    "bank_technical": {
        "method": "netbanking",
        "error_code": "SERVER_ERROR",
        "error_description": "Payment failed due to a technical error at the bank.",
        "error_source": "bank",
        "error_step": "payment_authorization",
        "error_reason": "bank_technical_error",
    },
}

# Failures in the "this week" window (22-28 Oct), by day offset from 22 Oct.
WEEK_FAILURES = {
    0: ["upi_timeout", "bank_declined"],
    1: ["insufficient_funds", "upi_timeout"],
    2: ["incorrect_otp"],
    3: [],  # Sunday: the two sales both went through
    4: ["bank_declined", "bank_technical", "upi_timeout"],
    5: ["upi_timeout", "incorrect_otp"],
    6: ["upi_timeout", "insufficient_funds", "insufficient_funds"],
}
OLDER_FAILURES = {1: ["upi_timeout"], 4: ["bank_declined"], 6: ["insufficient_funds"]}  # offsets from 14 Oct

ORDERS_PER_DAY = {datetime(2026, 10, 25, tzinfo=IST).date(): 2}  # slow Sunday
REFUND_PLAN = {
    # refund day -> list of (receipt, fraction refunded, reason)
    "2026-10-16": [(1003, 1.0, "Customer cancelled")],
    "2026-10-19": [(1012, 0.5, "Partial: one item out of stock")],
    "2026-10-21": [(1020, 1.0, "Damaged in transit")],
    "2026-10-23": [(1029, 1.0, "Customer cancelled")],
    "2026-10-26": [(1035, 1.0, "Wrong size delivered")],
    "2026-10-28": [
        (1044, 1.0, "Damaged in transit"),
        (1046, 1.0, "Damaged in transit"),
        (1049, 1.0, "Damaged in transit"),
        (1051, 0.5, "Partial: late delivery"),
        (1053, 1.0, "Damaged in transit"),
    ],
}
SPECIAL_AMOUNTS = {1042: 1499, 1035: 3499, 1045: 2499, 1046: 1999}
REFUNDED_RECEIPTS = {rcpt for plan in REFUND_PLAN.values() for rcpt, _, _ in plan}


def rid(prefix: str) -> str:
    return prefix + "".join(rng.choices(string.ascii_letters + string.digits, k=14))


def ts(dt: datetime) -> int:
    return int(dt.timestamp())


def fee_and_tax(amount: int) -> tuple[int, int]:
    # Razorpay's `fee` includes GST; `tax` is the GST part (18% of the 2% base fee).
    base = round(amount * 0.02)
    tax = round(base * 0.18)
    return base + tax, tax


def payment(order: dict, customer: tuple, created: datetime, method: str, failure: dict | None) -> dict:
    name, email, contact = customer
    p = {
        "id": rid("pay_"),
        "entity": "payment",
        "amount": order["amount"],
        "currency": "INR",
        "status": "failed" if failure else "captured",
        "order_id": order["id"],
        "method": method,
        "amount_refunded": 0,
        "refund_status": None,
        "captured": not failure,
        "description": f"Order #{order['receipt']}",
        "bank": "HDFC" if method == "netbanking" else None,
        "vpa": f"{name.split()[0].lower()}@okicici" if method == "upi" else None,
        "card": {"network": rng.choice(["Visa", "MasterCard", "RuPay"]), "last4": f"{rng.randint(1000, 9999)}"}
        if method == "card"
        else None,
        "email": email,
        "contact": contact,
        "notes": {"customer_name": name},
        "fee": None,
        "tax": None,
        "error_code": None,
        "error_description": None,
        "error_source": None,
        "error_step": None,
        "error_reason": None,
        "created_at": ts(created),
    }
    if failure:
        p.update({k: v for k, v in failure.items() if k != "method"})
    else:
        p["fee"], p["tax"] = fee_and_tax(p["amount"])
    return p


def build() -> dict:
    orders, payments, refunds = [], [], []
    receipt = 1001
    day = FIRST_DAY
    while day.date() <= NOW.date():
        n_orders = ORDERS_PER_DAY.get(day.date(), 4)
        offset = (day.date() - FIRST_DAY.date()).days
        week_offset = (day.date() - datetime(2026, 10, 22).date()).days
        failures = list(WEEK_FAILURES.get(week_offset, []) if week_offset >= 0 else OLDER_FAILURES.get(offset, []))
        for i in range(n_orders):
            created = day + timedelta(hours=10 + i * 2, minutes=rng.randint(0, 59))
            customer = rng.choice(CUSTOMERS)
            amount = SPECIAL_AMOUNTS.get(receipt, rng.choice(PRICES_INR)) * 100
            order = {
                "id": rid("order_"),
                "entity": "order",
                "amount": amount,
                "amount_paid": 0,
                "amount_due": amount,
                "currency": "INR",
                "receipt": str(receipt),
                "status": "created",
                "attempts": 0,
                "notes": {"customer_name": customer[0]},
                "created_at": ts(created),
            }
            # Failed attempts go on orders other than the ones the scenarios need clean.
            failure_key = failures.pop(0) if failures and receipt not in SPECIAL_AMOUNTS else None
            t = created + timedelta(minutes=1)
            if failure_key:
                f = FAILURES[failure_key]
                payments.append(payment(order, customer, t, f["method"], f))
                order["attempts"] += 1
                order["status"] = "attempted"
                t += timedelta(minutes=4)
                retry_succeeds = rng.random() < 0.6 or receipt in REFUNDED_RECEIPTS
            else:
                retry_succeeds = True
            if retry_succeeds:
                method = "upi" if receipt == 1042 else rng.choice(["upi", "upi", "card", "netbanking", "wallet"])
                p = payment(order, customer, t, method, None)
                if method == "wallet":
                    p["wallet"] = "paytm"
                payments.append(p)
                order.update(attempts=order["attempts"] + 1, status="paid", amount_paid=amount, amount_due=0)
            orders.append(order)
            receipt += 1
        day += timedelta(days=1)

    by_receipt = {o["receipt"]: o for o in orders}
    for refund_day, plan in REFUND_PLAN.items():
        day = datetime.fromisoformat(refund_day).replace(tzinfo=IST)
        for n, (rcpt, fraction, reason) in enumerate(plan):
            order = by_receipt[str(rcpt)]
            pay = next(p for p in payments if p["order_id"] == order["id"] and p["status"] == "captured")
            amount = int(pay["amount"] * fraction)
            refunds.append(
                {
                    "id": rid("rfnd_"),
                    "entity": "refund",
                    "amount": amount,
                    "currency": "INR",
                    "payment_id": pay["id"],
                    "notes": {"reason": reason, "order_receipt": str(rcpt)},
                    "receipt": None,
                    "status": "processed",
                    "speed_processed": "normal",
                    "speed_requested": "normal",
                    "created_at": ts(day + timedelta(hours=11, minutes=15 * n)),
                }
            )
            pay["amount_refunded"] += amount
            if pay["amount_refunded"] >= pay["amount"]:
                pay.update(status="refunded", refund_status="full")
            else:
                pay["refund_status"] = "partial"

    settlements, recon = build_settlements(payments, refunds, orders)
    links = build_payment_links()

    newest_first = lambda rows: sorted(rows, key=lambda r: r["created_at"], reverse=True)  # noqa: E731
    return {
        "meta": {"now": NOW.isoformat(), "generated_by": "scripts/make_fixtures.py"},
        "orders": newest_first(orders),
        "payments": newest_first(payments),
        "refunds": newest_first(refunds),
        "settlements": newest_first(settlements),
        "settlement_recon": recon,
        "payment_links": newest_first(links),
    }


def build_settlements(payments: list[dict], refunds: list[dict], orders: list[dict]) -> tuple[list[dict], list[dict]]:
    """T+2 settlements: day D pays out captures from D-2 minus refunds issued on D-1."""
    settlements, recon = [], []
    receipts = {o["id"]: o["receipt"] for o in orders}
    payment_orders = {p["id"]: p["order_id"] for p in payments}
    day = FIRST_DAY + timedelta(days=2)
    while day.date() <= NOW.date():
        settled_at = day + timedelta(hours=11)
        captures = [
            p for p in payments
            if p["captured"] and datetime.fromtimestamp(p["created_at"], IST).date() == (day - timedelta(days=2)).date()
        ]
        day_refunds = [
            r for r in refunds if datetime.fromtimestamp(r["created_at"], IST).date() == (day - timedelta(days=1)).date()
        ]
        setl_id = rid("setl_")
        utr = f"UTIB{rng.randint(10**11, 10**12 - 1)}"
        fees = sum(p["fee"] for p in captures)
        tax = sum(p["tax"] for p in captures)
        amount = sum(p["amount"] for p in captures) - fees - sum(r["amount"] for r in day_refunds)
        settlements.append(
            {
                "id": setl_id,
                "entity": "settlement",
                "amount": amount,
                "status": "processed",
                "fees": fees,
                "tax": tax,
                "utr": utr,
                "created_at": ts(settled_at),
            }
        )
        for p in captures:
            recon.append(
                {
                    "entity_id": p["id"], "type": "payment", "credit": p["amount"], "debit": 0,
                    "amount": p["amount"], "fee": p["fee"], "tax": p["tax"], "settled": True,
                    "settlement_id": setl_id, "settlement_utr": utr, "order_id": p["order_id"],
                    "order_receipt": receipts[p["order_id"]],
                    "method": p["method"], "description": p["description"],
                    "created_at": p["created_at"], "settled_at": ts(settled_at),
                }
            )
        for r in day_refunds:
            recon.append(
                {
                    "entity_id": r["id"], "type": "refund", "credit": 0, "debit": r["amount"],
                    "amount": r["amount"], "fee": 0, "tax": 0, "settled": True,
                    "settlement_id": setl_id, "settlement_utr": utr, "payment_id": r["payment_id"],
                    "order_id": payment_orders[r["payment_id"]], "order_receipt": receipts[payment_orders[r["payment_id"]]],
                    "description": r["notes"]["reason"],
                    "created_at": r["created_at"], "settled_at": ts(settled_at),
                }
            )
        day += timedelta(days=1)
    return settlements, recon


def build_payment_links() -> list[dict]:
    rows = [
        ("Advance for bulk order", 500000, "paid", NOW - timedelta(days=6)),
        ("Custom T-shirt printing", 129900, "created", NOW - timedelta(days=1)),
        ("Repair charges", 80000, "expired", NOW - timedelta(days=12)),
    ]
    return [
        {
            "id": rid("plink_"),
            "entity": "payment_link",
            "amount": amount,
            "amount_paid": amount if status == "paid" else 0,
            "currency": "INR",
            "description": desc,
            "status": status,
            "short_url": f"https://rzp.io/rzp/{rid('')[:8]}",
            "customer": {"name": c[0], "email": c[1], "contact": c[2]},
            "created_at": ts(created),
        }
        for (desc, amount, status, created), c in zip(rows, CUSTOMERS)
    ]


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    counts = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    print(f"Wrote {OUT} {counts}")

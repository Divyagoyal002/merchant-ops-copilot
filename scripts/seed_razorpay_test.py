"""Fill a Razorpay TEST-mode account with orders, payments, refunds and payment links.

Razorpay doesn't let an API call pay an order on its own (that needs the
checkout, like a real customer), so this runs in three steps:

    python scripts/seed_razorpay_test.py orders --count 50   # create orders + a local checkout page
    open scripts/out/checkout.html                           # pay them: success@razorpay / failure@razorpay
    python scripts/seed_razorpay_test.py refunds --count 8   # refund some captured payments
    python scripts/seed_razorpay_test.py links               # a few payment links

On the checkout page, UPI id success@razorpay gives a successful payment and
failure@razorpay a failed one. Mix them, and try the test cards from
https://razorpay.com/docs/payments/payments/test-card-details/ for card
failures, so the agent has real failure reasons to explain.
"""

import argparse
import json
import os
import random
from pathlib import Path

import httpx
from dotenv import load_dotenv

OUT = Path(__file__).parent / "out"
PRICES_INR = [299, 499, 799, 999, 1299, 1499, 1999, 2499, 3499]
CUSTOMERS = ["Aarav Sharma", "Priya Iyer", "Rohan Gupta", "Sneha Reddy", "Vikram Singh", "Ananya Das", "Kabir Mehta", "Meera Nair"]
REFUND_REASONS = ["Customer cancelled", "Damaged in transit", "Wrong size delivered", "Late delivery"]


def client() -> httpx.Client:
    load_dotenv()
    key_id, secret = os.getenv("RAZORPAY_KEY_ID", ""), os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id.startswith("rzp_test_"):
        raise SystemExit("Set RAZORPAY_KEY_ID to a TEST key (rzp_test_...) in .env. This script never runs on live keys.")
    return httpx.Client(base_url="https://api.razorpay.com/v1", auth=(key_id, secret), timeout=30)


def create_orders(count: int, start: int) -> None:
    orders = []
    with client() as api:
        for receipt in range(start, start + count):
            resp = api.post(
                "/orders",
                json={
                    "amount": random.choice(PRICES_INR) * 100,
                    "currency": "INR",
                    "receipt": str(receipt),
                    "notes": {"customer_name": random.choice(CUSTOMERS)},
                },
            )
            resp.raise_for_status()
            orders.append(resp.json())
            print(f"order #{receipt}: {orders[-1]['id']} ₹{orders[-1]['amount'] / 100:,.0f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "orders.json").write_text(json.dumps(orders, indent=1))
    write_checkout_page(orders)
    print(f"\nNow open {OUT / 'checkout.html'} and pay the orders.")


def write_checkout_page(orders: list[dict]) -> None:
    key_id = os.environ["RAZORPAY_KEY_ID"]
    rows = "\n".join(
        f'<tr><td>#{o["receipt"]}</td><td>₹{o["amount"] / 100:,.0f}</td><td>{o["notes"]["customer_name"]}</td>'
        f'<td><button onclick="pay(\'{o["id"]}\', {o["amount"]}, \'{o["receipt"]}\', this)">Pay</button></td></tr>'
        for o in orders
    )
    (OUT / "checkout.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Seed test payments</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<style>body{{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}}td{{padding:.3rem .8rem}}</style>
</head><body>
<h1>Seed test payments</h1>
<p>Test mode only. UPI <code>success@razorpay</code> succeeds, <code>failure@razorpay</code> fails.
Aim for roughly 1 in 4 failing, and retry a few failed orders so they end up paid.</p>
<table>{rows}</table>
<script>
function pay(orderId, amount, receipt, btn) {{
  const rzp = new Razorpay({{
    key: "{key_id}", order_id: orderId, amount: amount, currency: "INR",
    name: "Seed Store", description: "Order #" + receipt,
    handler: () => {{ btn.textContent = "Paid"; btn.disabled = true; }},
  }});
  rzp.on("payment.failed", () => {{ btn.textContent = "Failed (retry?)"; }});
  rzp.open();
}}
</script></body></html>""")


def create_refunds(count: int) -> None:
    with client() as api:
        payments = api.get("/payments", params={"count": 100}).json()["items"]
        refundable = [p for p in payments if p["status"] == "captured" and not p["amount_refunded"]]
        for p in random.sample(refundable, min(count, len(refundable))):
            partial = random.random() < 0.3
            amount = p["amount"] // 2 if partial else p["amount"]
            resp = api.post(
                f"/payments/{p['id']}/refund",
                json={"amount": amount, "notes": {"reason": random.choice(REFUND_REASONS)}},
            )
            resp.raise_for_status()
            print(f"refund {resp.json()['id']}: ₹{amount / 100:,.2f} on {p['id']}" + (" (partial)" if partial else ""))


def create_links() -> None:
    with client() as api:
        for desc, amount, name in [
            ("Advance for bulk order", 5000, "Aarav Sharma"),
            ("Custom T-shirt printing", 1299, "Priya Iyer"),
            ("Repair charges", 800, "Rohan Gupta"),
        ]:
            resp = api.post(
                "/payment_links",
                json={"amount": amount * 100, "currency": "INR", "description": desc, "customer": {"name": name}},
            )
            resp.raise_for_status()
            print(f"link {resp.json()['short_url']}: ₹{amount:,} {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("orders")
    o.add_argument("--count", type=int, default=50)
    o.add_argument("--start", type=int, default=1001, help="first receipt number")
    r = sub.add_parser("refunds")
    r.add_argument("--count", type=int, default=8)
    sub.add_parser("links")
    args = parser.parse_args()

    if args.command == "orders":
        create_orders(args.count, args.start)
    elif args.command == "refunds":
        create_refunds(args.count)
    else:
        create_links()

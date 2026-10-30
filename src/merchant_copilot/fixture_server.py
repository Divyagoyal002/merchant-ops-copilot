"""Offline stand-in for the Razorpay MCP server, serving data/fixtures.json.

Same tool names, parameters and raw API shapes as razorpay/razorpay-mcp-server,
so the demo, evals and CI go through a real MCP connection without any keys.

    python -m merchant_copilot.fixture_server   # http://127.0.0.1:8765/mcp
"""

import json
import os
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from merchant_copilot.config import FIXTURES_FILE, IST

DATA = json.loads(FIXTURES_FILE.read_text())
server = FastMCP(
    "razorpay-fixtures", host="127.0.0.1", port=int(os.getenv("FIXTURES_PORT", "8765")), log_level="WARNING"
)

Count = Annotated[int, Field(ge=1, le=100, description="Number of records to fetch (default 10, max 100)")]
Skip = Annotated[int, Field(ge=0, description="Number of records to skip")]
# Razorpay calls this parameter `from`, which is a Python keyword.
From = Annotated[int | None, Field(alias="from", description="Unix timestamp (seconds) from when to fetch")]
To = Annotated[int | None, Field(description="Unix timestamp (seconds) till when to fetch")]


def _collection(rows: list[dict], count: int, skip: int, from_: int | None, to: int | None) -> dict:
    rows = [r for r in rows if (from_ is None or r["created_at"] >= from_) and (to is None or r["created_at"] <= to)]
    page = rows[skip : skip + count]
    return {"entity": "collection", "count": len(page), "items": page}


def _get(rows: list[dict], key: str, value: str, what: str) -> dict:
    for row in rows:
        if row[key] == value:
            return row
    raise ValueError(f"fetching {what} failed: The id provided does not exist")


def _tool(fn):
    """Register fn as a tool, passing Razorpay's `from` argument on as `from_`."""
    server.tool()(fn)
    tool = server._tool_manager.get_tool(fn.__name__)

    def call(**kwargs):
        if "from" in kwargs:
            kwargs["from_"] = kwargs.pop("from")
        return fn(**kwargs)

    tool.fn = call
    return fn


@_tool
def fetch_payment(payment_id: str) -> dict:
    """Use this tool to retrieve the details of a specific payment using its id. Amount returned is in paisa"""
    return _get(DATA["payments"], "id", payment_id, "payment")


@_tool
def fetch_all_payments(count: Count = 10, skip: Skip = 0, from_: From = None, to: To = None) -> dict:
    """Fetch all payments with optional filtering and pagination"""
    return _collection(DATA["payments"], count, skip, from_, to)


@_tool
def fetch_order(order_id: str) -> dict:
    """Fetch an order's details using its ID"""
    return _get(DATA["orders"], "id", order_id, "order")


@_tool
def fetch_all_orders(
    count: Count = 10,
    skip: Skip = 0,
    from_: From = None,
    to: To = None,
    authorized: Annotated[int | None, Field(ge=0, le=1, description="Filter orders by payment authorization")] = None,
    receipt: Annotated[str | None, Field(description="Retrieve the orders that contain the provided receipt")] = None,
) -> dict:
    """Fetch all orders with optional filtering and pagination"""
    rows = DATA["orders"]
    if receipt is not None:
        rows = [o for o in rows if receipt.lstrip("#") in o["receipt"]]
    if authorized is not None:
        rows = [o for o in rows if (o["status"] == "paid") == bool(authorized)]
    return _collection(rows, count, skip, from_, to)


@_tool
def fetch_order_payments(order_id: str) -> dict:
    """Fetch all payments made for a specific order (order_id starts with 'order_')"""
    _get(DATA["orders"], "id", order_id, "order")
    rows = [p for p in DATA["payments"] if p["order_id"] == order_id]
    return {"entity": "collection", "count": len(rows), "items": rows}


@_tool
def fetch_refund(refund_id: str) -> dict:
    """Use this tool to retrieve the details of a specific refund using its id."""
    return _get(DATA["refunds"], "id", refund_id, "refund")


@_tool
def fetch_all_refunds(count: Count = 10, skip: Skip = 0, from_: From = None, to: To = None) -> dict:
    """Use this tool to retrieve details of all refunds. By default, only the last 10 refunds are returned."""
    return _collection(DATA["refunds"], count, skip, from_, to)


@_tool
def fetch_multiple_refunds_for_payment(
    payment_id: str, count: Count = 10, skip: Skip = 0, from_: From = None, to: To = None
) -> dict:
    """Use this tool to retrieve multiple refunds for a payment. By default, only the last 10 refunds are returned."""
    _get(DATA["payments"], "id", payment_id, "payment")
    rows = [r for r in DATA["refunds"] if r["payment_id"] == payment_id]
    return _collection(rows, count, skip, from_, to)


@_tool
def fetch_all_settlements(count: Count = 10, skip: Skip = 0, from_: From = None, to: To = None) -> dict:
    """Fetch all settlements with optional filtering and pagination"""
    return _collection(DATA["settlements"], count, skip, from_, to)


@_tool
def fetch_settlement_with_id(settlement_id: str) -> dict:
    """Fetch details of a specific settlement using its ID"""
    return _get(DATA["settlements"], "id", settlement_id, "settlement")


@_tool
def fetch_settlement_recon_details(
    year: Annotated[int, Field(description="Year for which the settlement report is requested (YYYY)")],
    month: Annotated[int, Field(description="Month for which the settlement report is requested (MM)")],
    day: Annotated[int | None, Field(description="Optional: Day for which the settlement report is requested (DD)")] = None,
    count: Count = 10,
    skip: Skip = 0,
) -> dict:
    """Fetch settlement reconciliation report for a specific time period"""
    from datetime import datetime

    def matches(item: dict) -> bool:
        settled = datetime.fromtimestamp(item["settled_at"], IST)
        return settled.year == year and settled.month == month and (day is None or settled.day == day)

    rows = [r for r in DATA["settlement_recon"] if matches(r)]
    page = rows[skip : skip + count]
    return {"entity": "collection", "count": len(page), "items": page}


@_tool
def fetch_payment_link(payment_link_id: str) -> dict:
    """Fetch payment link details using its ID"""
    return _get(DATA["payment_links"], "id", payment_link_id, "payment link")


@_tool
def fetch_all_payment_links() -> dict:
    """Fetch all payment links"""
    return {"payment_links": DATA["payment_links"]}


if __name__ == "__main__":
    server.run(transport="streamable-http")

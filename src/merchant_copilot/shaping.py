"""Turn raw Razorpay JSON into something a small model reads correctly.

Amounts arrive in paise, and models kept reporting ₹149900 for a ₹1,499 order,
so the conversion happens here. Lists go out as CSV because a week of payments
as raw JSON didn't fit in Groq's free 8k-tokens-a-minute budget.
"""

import csv
import io
import json
from datetime import datetime
from typing import Any

from merchant_copilot.config import IST

MONEY_FIELDS = {"amount", "amount_paid", "amount_due", "amount_refunded", "fee", "tax", "fees", "credit", "debit"}
TIME_FIELDS = {"created_at", "captured_at", "settled_at", "processed_at", "expire_by", "expired_at", "updated_at"}
DROP_FIELDS = {
    "acquirer_data", "international", "invoice_id", "card_id", "token_id", "emi_plan", "upi",
    "currency", "entity", "batch_id", "speed_requested", "offer_id", "reminders", "notify",
    # Customer contact details: never needed for an answer, and they shouldn't end up in traces.
    "email", "contact", "vpa",
    "captured", "error_code", "error_source", "error_step", "last4",
}


def shape(value: Any) -> Any:
    if isinstance(value, list):
        return [shape(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, v in value.items():
        if key in DROP_FIELDS or v is None or v == {} or v == []:
            continue
        if key in MONEY_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{key}_inr"] = round(v / 100, 2)
        elif key in TIME_FIELDS and isinstance(v, int) and v > 0:
            out[key] = datetime.fromtimestamp(v, IST).strftime("%Y-%m-%d %H:%M IST")
        else:
            out[key] = shape(v)
    return out


def tool_output_text(raw: Any) -> str:
    """MCP results arrive either as a string or as a list of content blocks."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    return str(raw)


def parse_tool_json(raw: Any) -> Any:
    return json.loads(tool_output_text(raw))


def _flatten(row: dict, prefix: str = "") -> dict:
    flat = {}
    for key, v in row.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = v
    return flat


def as_csv(rows: list[dict]) -> str:
    flat = [_flatten(r) for r in rows]
    columns = list(dict.fromkeys(k for r in flat for k in r))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return out.getvalue()


def shape_tool_output(raw: Any) -> str:
    text = tool_output_text(raw)
    try:
        data = shape(json.loads(text))
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        lists = [k for k, v in data.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        if len(lists) == 1:
            key = lists[0]
            return f"{len(data[key])} {key} (CSV):\n{as_csv(data[key])}"
        if data.get("count") == 0:
            return "0 items"
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

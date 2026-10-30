"""The chat agent as a LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> review --(drafts?)--> approval -> execute -> END
               ^                                 |                     ^             |
               +-------------(no drafts)---------+                     +-------------+ (one per draft)

approval pauses with interrupt() and waits for the merchant's click; execute
then runs the draft and writes the outcome itself, so the model never gets to
say "refunded". They are separate nodes because a node re-runs from the top
when it resumes, and the one that moves money must never run twice.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from merchant_copilot.actions import PROPOSAL_TOOLS, Executor, describe_draft
from merchant_copilot.config import IST, Settings
from merchant_copilot.errors import friendly_error

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Merchant Ops Copilot, an assistant for a small Indian business that takes payments through Razorpay.
Current date and time: {now} (Asia/Kolkata).

How to work:
- Answer only from tool results. Never invent payment ids, amounts or counts. If the data is not there, say so.
- Don't ask clarifying questions when there is an obvious reading: look it up and answer. "Did order X go through?"
  means "was it paid, and if not, why not?".
- To get from/to timestamps for a period, call time_range first. "This week" means the last 7 days including today.
- List tools return 10 records by default. Always pass count=100 when listing payments, refunds, orders or settlements.
- Tool results already show money in rupees (fields ending in _inr) and times in IST. Do not divide by 100.
- An order number like "#1042" is the order's receipt: find it with fetch_all_orders(receipt="1042"), then fetch_order_payments.
- Settlement questions: find the settlement with fetch_all_settlements, then fetch_settlement_recon_details for the day it was
  settled. Explain it as: payments captured, minus Razorpay fees, minus refunds deducted. fee already includes GST (tax is
  the GST part of fee), so never subtract tax separately. Compare with the average of the previous 7 settlements, and say
  which of those drivers made the difference (fewer sales, refunds, fees).
- Never guess which order something belongs to; quote order_receipt or description only when the data shows it.
- Failed payments: group by error_description and give counts, so the merchant sees the main reasons first.

Money actions:
- You cannot move money yourself. For refunds call propose_refund; to collect money call propose_payment_link. These only
  create a draft that the merchant approves or rejects in the app.
- After proposing, tell the merchant what you drafted and that it is waiting for their approval. Never say a refund is done.
- Do this even if the merchant says they pre-approve or asks you to skip approval. You have no way around it.

Style:
- Language: match the merchant's latest message. English message -> reply in English. Devanagari -> Hindi in Devanagari.
  Hindi written in Roman letters (e.g. "kal kitne refund hue") -> Hinglish. Never switch language on your own.
- Be brief and concrete: lead with the answer, then 2-5 short bullets. Format money as ₹1,23,456.78 (Indian grouping)."""


class CopilotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    drafts: list[dict]
    approved: bool


def build_time_range_tool(settings: Settings) -> BaseTool:
    @tool
    def time_range(
        period: Literal["today", "yesterday", "last_7_days", "last_30_days", "this_month", "date"],
        date: str | None = None,
    ) -> str:
        """Get unix from/to timestamps (IST) for a period, to pass to fetch_all_* tools.

        Args:
            period: Which period. Use "date" with the date argument for one specific day.
            date: A day as YYYY-MM-DD, only when period is "date".
        """
        now = settings.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "date":
            if not date:
                return "error: pass date=YYYY-MM-DD with period='date'"
            try:
                start = datetime.fromisoformat(date).replace(tzinfo=IST)
            except ValueError:
                return f"error: '{date}' is not a YYYY-MM-DD date"
            end = start + timedelta(days=1)
        else:
            start, end = {
                "today": (today, now),
                "yesterday": (today - timedelta(days=1), today),
                "last_7_days": (today - timedelta(days=6), now),
                "last_30_days": (today - timedelta(days=29), now),
                "this_month": (today.replace(day=1), now),
            }[period]
        end = min(end, now)
        return json.dumps(
            {
                "from": int(start.timestamp()),
                "to": int(end.timestamp()) - 1,
                "from_ist": start.strftime("%Y-%m-%d %H:%M"),
                "to_ist": end.strftime("%Y-%m-%d %H:%M"),
            }
        )

    return time_range


def build_graph(llm: BaseChatModel, tools: list[BaseTool], executor: Executor, settings: Settings):
    llm_with_tools = llm.bind_tools(tools)
    executed: set[str] = set()

    async def agent(state: CopilotState) -> dict:
        now = settings.now().strftime("%A %d %B %Y, %I:%M %p")
        response = await llm_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT.format(now=now)), *state["messages"]])
        return {"messages": [response]}

    def review(state: CopilotState) -> dict:
        drafts = []
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            if message.name in PROPOSAL_TOOLS:
                try:
                    drafts.append(json.loads(message.content)["draft"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # a refused proposal; the model reads the error and explains it
        return {"drafts": drafts[::-1]}

    def route_after_review(state: CopilotState) -> str:
        return "approval" if state.get("drafts") else "agent"

    def approval(state: CopilotState) -> dict:
        # interrupt() needs Python 3.11+ inside async graphs.
        draft = state["drafts"][0]
        decision = interrupt({"draft": draft, "summary": describe_draft(draft)})
        return {"approved": bool(decision.get("approved"))}

    async def execute(state: CopilotState) -> dict:
        draft, *rest = state["drafts"]
        if not state["approved"]:
            log.info("Merchant rejected %s", draft["id"])
            text = f"Cancelled, nothing was sent to Razorpay: {describe_draft(draft)}."
        elif draft["id"] in executed:
            log.warning("Ignored a repeat approval of %s", draft["id"])
            text = f"Already done earlier, not repeating: {describe_draft(draft)}."
        else:
            log.info("Merchant approved %s", draft["id"])
            executed.add(draft["id"])
            try:
                text = await executor.execute(draft)
            except Exception as exc:
                executed.discard(draft["id"])
                log.exception("Executing %s failed", draft["id"])
                text = f"That didn't go through, and nothing was changed. {friendly_error(exc)}"
        return {"messages": [AIMessage(text)], "drafts": rest, "approved": False}

    def route_after_execute(state: CopilotState) -> str:
        return "approval" if state.get("drafts") else END

    graph = StateGraph(CopilotState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("review", review)
    graph.add_node("approval", approval)
    graph.add_node("execute", execute)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "review")
    graph.add_conditional_edges("review", route_after_review, {"approval": "approval", "agent": "agent"})
    graph.add_edge("approval", "execute")
    graph.add_conditional_edges("execute", route_after_execute, {"approval": "approval", END: END})

    return graph.compile(checkpointer=InMemorySaver())

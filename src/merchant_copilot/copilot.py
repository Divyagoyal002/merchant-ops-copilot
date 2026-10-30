"""One entry point shared by the CLI, the Gradio app and the eval runner."""

import logging
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from merchant_copilot import tracing
from merchant_copilot.actions import Executor, RazorpayExecutor, SimulatedExecutor, build_action_tools
from merchant_copilot.config import Settings
from merchant_copilot.graph import build_graph, build_time_range_tool
from merchant_copilot.llm import build_llm
from merchant_copilot.mcp_client import for_agent, load_razorpay_tools
from merchant_copilot.shaping import parse_tool_json

log = logging.getLogger(__name__)


@dataclass
class Turn:
    answer: str
    pending: dict | None = None  # the draft waiting for Approve/Reject, if any
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class Copilot:
    def __init__(self, settings: Settings, graph, executor: Executor, tools: dict):
        self.settings = settings
        self.graph = graph
        self.executor = executor
        self.razorpay_tools = tools

    @classmethod
    async def create(cls, settings: Settings, executor: Executor | None = None) -> "Copilot":
        raw_tools = await load_razorpay_tools(settings)

        async def fetch_payment(payment_id: str) -> dict:
            return parse_tool_json(await raw_tools["fetch_payment"].ainvoke({"payment_id": payment_id}))

        if executor is None:
            executor = SimulatedExecutor() if settings.data_source == "fixtures" else RazorpayExecutor(settings)
        tools = [
            build_time_range_tool(settings),
            *(for_agent(t) for t in raw_tools.values()),
            *build_action_tools(fetch_payment, settings),
        ]
        llm = build_llm(settings.model, settings.max_tokens)
        log.info("Copilot ready: model=%s data=%s tools=%d", settings.model, settings.data_source, len(tools))
        return cls(settings, build_graph(llm, tools, executor, settings), executor, raw_tools)

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": tracing.callbacks(),
            "metadata": {"langfuse_session_id": thread_id, "model": self.settings.model},
            "recursion_limit": 30,
        }

    async def pending(self, thread_id: str) -> dict | None:
        state = await self.graph.aget_state(self._config(thread_id))
        return state.interrupts[0].value if state.interrupts else None

    async def ask(self, thread_id: str, text: str) -> Turn:
        if await self.pending(thread_id):
            # A new message while a draft is open counts as "no" to that draft.
            await self.decide(thread_id, approved=False)
        return await self._run(thread_id, {"messages": [HumanMessage(text)]})

    async def decide(self, thread_id: str, approved: bool) -> Turn:
        return await self._run(thread_id, Command(resume={"approved": approved}))

    async def _run(self, thread_id: str, graph_input) -> Turn:
        config = self._config(thread_id)
        before = len((await self.graph.aget_state(config)).values.get("messages", []))
        start = time.perf_counter()
        try:
            result = await self.graph.ainvoke(graph_input, config)
        except Exception:
            log.exception("Turn failed for thread %s after %.1fs", thread_id, time.perf_counter() - start)
            raise
        finally:
            tracing.flush()
        latency = time.perf_counter() - start

        turn = Turn(answer="", latency_s=latency)
        for message in result["messages"][before:]:
            if not isinstance(message, AIMessage):
                continue
            turn.tool_calls.extend({"name": c["name"], "args": c["args"]} for c in message.tool_calls)
            if message.usage_metadata:
                turn.input_tokens += message.usage_metadata.get("input_tokens", 0)
                turn.output_tokens += message.usage_metadata.get("output_tokens", 0)
            if not message.tool_calls and message.text:
                turn.answer = message.text
        if interrupts := result.get("__interrupt__"):
            turn.pending = interrupts[0].value
        log.info(
            "thread=%s tools=%s draft=%s %.1fs %d+%d tokens",
            thread_id,
            [c["name"] for c in turn.tool_calls],
            turn.pending["draft"]["id"] if turn.pending else None,
            latency,
            turn.input_tokens,
            turn.output_tokens,
        )
        return turn

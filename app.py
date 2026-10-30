"""Gradio UI and the Hugging Face Spaces entry point: python app.py"""

import asyncio
import json
import logging
import uuid

import gradio as gr

from merchant_copilot.config import load_settings
from merchant_copilot.copilot import Copilot, Turn
from merchant_copilot.digest import build_digest_graph
from merchant_copilot.errors import friendly_error
from merchant_copilot.llm import build_llm
from merchant_copilot.logs import setup_logging

setup_logging()
log = logging.getLogger("merchant_copilot.app")

settings = load_settings()
_copilot: Copilot | None = None
_lock = asyncio.Lock()

MAX_MESSAGE_CHARS = 500
EXAMPLES = [
    "Why did yesterday's settlement come in low?",
    "Which payments failed this week and why?",
    "Refund order #1042",
    "इस हफ्ते कितने refunds हुए?",
    "Aaj kitne payments fail hue?",
    "Send a payment link for ₹1,200 to Meera for a custom cake",
]


async def get_copilot() -> Copilot:
    # Created on first use, inside Gradio's event loop: the model clients' HTTP pools are tied to it.
    global _copilot
    async with _lock:
        if _copilot is None:
            _copilot = await Copilot.create(settings)
    return _copilot


def render(history: list, turn: Turn) -> tuple:
    if turn.tool_calls:
        calls = "\n".join(f"- `{c['name']}({json.dumps(c['args'], ensure_ascii=False)})`" for c in turn.tool_calls)
        history.append({"role": "assistant", "content": calls, "metadata": {"title": f"🔧 {len(turn.tool_calls)} tool calls"}})
    if turn.answer:
        history.append({"role": "assistant", "content": turn.answer})
    stats = f"{turn.latency_s:.1f}s · {turn.input_tokens + turn.output_tokens:,} tokens · {settings.model}"
    if turn.pending:
        card = f"### Approval needed\n{turn.pending['summary']}\n\nNothing has been sent to Razorpay yet."
        return history, gr.update(value=card), gr.update(visible=True), stats
    return history, gr.update(value=""), gr.update(visible=False), stats


def failed(history: list, exc: Exception) -> tuple:
    history.append({"role": "assistant", "content": friendly_error(exc)})
    return history, gr.update(value=""), gr.update(visible=False), ""


async def on_send(text: str, history: list, thread_id: str):
    text = text.strip()
    if not text:
        return history, "", gr.update(), gr.update(), ""
    if len(text) > MAX_MESSAGE_CHARS:
        gr.Warning(f"Please keep questions under {MAX_MESSAGE_CHARS} characters.")
        return history, text, gr.update(), gr.update(), ""
    history = history + [{"role": "user", "content": text}]
    try:
        turn = await (await get_copilot()).ask(thread_id, text)
        history, card, row, stats = render(history, turn)
    except Exception as exc:
        log.exception("Chat request failed")
        history, card, row, stats = failed(history, exc)
    return history, "", card, row, stats


async def on_decide(approved: bool, history: list, thread_id: str):
    history = list(history)
    try:
        turn = await (await get_copilot()).decide(thread_id, approved)
    except Exception as exc:
        log.exception("Approval request failed")
        return failed(history, exc)
    turn.answer = ("✅ " if approved else "✖️ ") + turn.answer
    return render(history, turn)


async def on_approve(history: list, thread_id: str):
    return await on_decide(True, history, thread_id)


async def on_reject(history: list, thread_id: str):
    return await on_decide(False, history, thread_id)


async def on_digest(language: str) -> str:
    try:
        copilot = await get_copilot()
        graph = build_digest_graph(copilot.razorpay_tools, build_llm(settings.model, settings.max_tokens), settings)
        result = await graph.ainvoke({"language": language})
        return result["text"]
    except Exception as exc:
        log.exception("Digest request failed")
        return friendly_error(exc)


with gr.Blocks(title="Merchant Ops Copilot") as demo:
    thread = gr.State(lambda: uuid.uuid4().hex)
    gr.Markdown(
        "# Merchant Ops Copilot\n"
        "Ask about payments, refunds and settlements in Hindi or English. "
        "Refunds and payment links are only drafted: nothing moves until you click **Approve**.\n\n"
        + (
            f"_Demo data: a fixed test week, so \"today\" is {settings.now():%d %b %Y}._"
            if settings.data_source == "fixtures"
            else "_Connected to a Razorpay test-mode account._"
        )
    )
    with gr.Tab("Ask"):
        chatbot = gr.Chatbot(height=480, label="Copilot")
        with gr.Group(visible=False) as approval_row:
            approval_card = gr.Markdown()
            with gr.Row():
                approve = gr.Button("Approve", variant="primary")
                reject = gr.Button("Reject", variant="stop")
        box = gr.Textbox(placeholder="e.g. Why did yesterday's settlement come in low?", show_label=False)
        gr.Examples(EXAMPLES, inputs=box)
        stats = gr.Markdown()

        box.submit(on_send, [box, chatbot, thread], [chatbot, box, approval_card, approval_row, stats])
        approve.click(on_approve, [chatbot, thread], [chatbot, approval_card, approval_row, stats])
        reject.click(on_reject, [chatbot, thread], [chatbot, approval_card, approval_row, stats])

    with gr.Tab("Daily digest"):
        gr.Markdown("What the scheduled digest sends each morning: failure rate, top failure reasons and refund spikes.")
        lang = gr.Dropdown(["English", "Hindi", "Hinglish"], value="English", label="Language")
        run_digest = gr.Button("Generate digest", variant="primary")
        digest_out = gr.Textbox(lines=10, label="Digest", buttons=["copy"])
        run_digest.click(on_digest, lang, digest_out)

# A public demo shares one free-tier model quota, so only a couple of requests run at once.
demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

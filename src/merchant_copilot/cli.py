import argparse
import asyncio
import logging

from merchant_copilot.config import ConfigError, load_settings
from merchant_copilot.copilot import Copilot
from merchant_copilot.digest import DigestError, build_digest_graph
from merchant_copilot.errors import friendly_error
from merchant_copilot.llm import build_llm
from merchant_copilot.logs import setup_logging

log = logging.getLogger("merchant_copilot.cli")


async def chat(args: argparse.Namespace) -> None:
    settings = load_settings(model=args.model, data_source=args.data)
    copilot = await Copilot.create(settings)
    print(f"Model: {settings.model} | Data: {settings.data_source} | Now: {settings.now():%d %b %Y %H:%M} IST")
    print("Ask in Hindi or English. Type 'exit' to quit.\n")

    while True:
        text = (await asyncio.to_thread(input, "you: ")).strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        try:
            turn = await copilot.ask("cli", text)
            while True:
                for call in turn.tool_calls:
                    print(f"  -> {call['name']}({call['args']})")
                if turn.answer:
                    print(f"\ncopilot: {turn.answer}\n")
                if not turn.pending:
                    break
                print(f"\n  DRAFT: {turn.pending['summary']}")
                answer = (await asyncio.to_thread(input, "  Approve? [y/N]: ")).strip().lower()
                turn = await copilot.decide("cli", approved=answer in {"y", "yes", "haan", "ha"})
            print(f"  ({turn.latency_s:.1f}s, {turn.input_tokens}+{turn.output_tokens} tokens)\n")
        except Exception as exc:
            print(f"\ncopilot: {friendly_error(exc)}\n")


async def digest(args: argparse.Namespace) -> None:
    settings = load_settings(model=args.model, data_source=args.data)
    copilot = await Copilot.create(settings)
    llm = None if args.no_llm else build_llm(settings.model, settings.max_tokens)
    graph = build_digest_graph(copilot.razorpay_tools, llm, settings)
    result = await graph.ainvoke({"language": args.lang, "send": args.send})
    print(result["text"])
    if args.send:
        print("\n(sent to webhook)" if result["delivered"] else "\n(DIGEST_WEBHOOK_URL not set, nothing sent)")


def run() -> None:
    parser = argparse.ArgumentParser(prog="copilot", description="Merchant Ops Copilot for Razorpay")
    parser.add_argument("--model", help="provider:model, overrides LLM_MODEL")
    parser.add_argument("--data", choices=["fixtures", "razorpay"], help="overrides DATA_SOURCE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("chat", help="chat in the terminal")
    d = sub.add_parser("digest", help="print (and optionally send) the daily health digest")
    d.add_argument("--lang", default="English", help='e.g. "Hindi" or "Hinglish"')
    d.add_argument("--send", action="store_true", help="POST it to DIGEST_WEBHOOK_URL")
    d.add_argument("--no-llm", action="store_true", help="use the plain template, no model call")
    args = parser.parse_args()
    # Keep the terminal chat readable; LOG_LEVEL=INFO shows every turn.
    setup_logging("WARNING" if args.command == "chat" else "INFO")

    try:
        asyncio.run(chat(args) if args.command == "chat" else digest(args))
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}")
    except DigestError as exc:
        log.error("%s", exc)
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    run()

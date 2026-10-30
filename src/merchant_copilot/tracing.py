import logging
import os

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def callbacks() -> list[BaseCallbackHandler]:
    if not enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        log.warning("Langfuse tracing is configured but failed to start; continuing without it", exc_info=True)
        return []


def flush() -> None:
    # The CLI, evals and digest cron exit before Langfuse's background exporter would send.
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        log.warning("Could not flush Langfuse traces", exc_info=True)

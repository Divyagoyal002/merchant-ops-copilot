import logging
import os

NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "mcp", "uvicorn", "uvicorn.access", "langfuse", "gradio", "urllib3")


def setup_logging(default_level: str = "INFO") -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", default_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

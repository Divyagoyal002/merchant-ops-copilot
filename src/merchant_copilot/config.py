import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

IST = timezone(timedelta(hours=5, minutes=30))
DATA_SOURCES = ("fixtures", "razorpay")
FIXTURES_FILE = Path(__file__).parent / "data" / "fixtures.json"


class ConfigError(Exception):
    """Raised when .env is missing something the chosen setup needs."""


@dataclass(frozen=True)
class Settings:
    model: str  # "provider:model", e.g. "groq:openai/gpt-oss-120b"
    data_source: str
    razorpay_key_id: str
    razorpay_key_secret: str
    razorpay_mcp_url: str
    fixtures_port: int
    max_refund_inr: float
    max_tokens: int
    frozen_now: datetime | None
    digest_webhook_url: str

    @property
    def razorpay_auth_header(self) -> str:
        token = base64.b64encode(f"{self.razorpay_key_id}:{self.razorpay_key_secret}".encode()).decode()
        return f"Basic {token}"

    def now(self) -> datetime:
        return self.frozen_now or datetime.now(IST)


def load_settings(**overrides) -> Settings:
    load_dotenv()
    data_source = overrides.get("data_source") or os.getenv("DATA_SOURCE", "fixtures").strip().lower()
    settings = Settings(
        model=overrides.get("model") or os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b"),
        data_source=data_source,
        razorpay_key_id=os.getenv("RAZORPAY_KEY_ID", ""),
        razorpay_key_secret=os.getenv("RAZORPAY_KEY_SECRET", ""),
        razorpay_mcp_url=os.getenv("RAZORPAY_MCP_URL", "https://mcp.razorpay.com/mcp"),
        fixtures_port=_number("FIXTURES_PORT", "8765", int),
        max_refund_inr=_number("MAX_REFUND_INR", "25000", float),
        max_tokens=_number("MAX_TOKENS", "2048", int),
        frozen_now=_frozen_now(data_source),
        digest_webhook_url=os.getenv("DIGEST_WEBHOOK_URL", ""),
    )
    _validate(settings)
    return settings


def _number(name: str, default: str, cast):
    value = os.getenv(name, default)
    try:
        return cast(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got '{value}'") from None


def _frozen_now(data_source: str) -> datetime | None:
    # The demo data is one fixed week, so "today" has to be pinned inside it.
    if value := os.getenv("COPILOT_NOW"):
        try:
            return datetime.fromisoformat(value).astimezone(IST)
        except ValueError:
            raise ConfigError(f"COPILOT_NOW must be an ISO date-time, got '{value}'") from None
    if data_source == "fixtures":
        try:
            return datetime.fromisoformat(json.loads(FIXTURES_FILE.read_text())["meta"]["now"])
        except (OSError, KeyError, ValueError) as exc:
            raise ConfigError(f"Demo data at {FIXTURES_FILE} is missing or broken: {exc}") from exc
    return None


def _validate(settings: Settings) -> None:
    if settings.data_source not in DATA_SOURCES:
        raise ConfigError(f"DATA_SOURCE must be one of {', '.join(DATA_SOURCES)}, got '{settings.data_source}'")
    if ":" not in settings.model:
        raise ConfigError(f"LLM_MODEL must look like 'provider:model', got '{settings.model}'")
    if settings.data_source == "razorpay":
        if not settings.razorpay_key_id.startswith("rzp_test_"):
            # An approved draft on live keys would move real money.
            raise ConfigError("DATA_SOURCE=razorpay needs a test-mode RAZORPAY_KEY_ID (rzp_test_...)")
        if not settings.razorpay_key_secret:
            raise ConfigError("DATA_SOURCE=razorpay needs RAZORPAY_KEY_SECRET in .env")

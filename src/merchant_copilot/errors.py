import httpx

from merchant_copilot.config import ConfigError


class RazorpayError(Exception):
    """Razorpay refused an approved action; the message is Razorpay's own description."""


def root_cause(exc: BaseException) -> BaseException:
    # MCP calls run inside anyio task groups, which wrap the real error in an ExceptionGroup.
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def friendly_error(exc: BaseException) -> str:
    exc = root_cause(exc)
    status = getattr(exc, "status_code", None)
    if status == 429 or "RateLimit" in type(exc).__name__:
        return "The free model quota is busy right now. Please try again in a minute."
    if status == 413:
        return "That pulled in more data than the free model tier accepts at once. Try a shorter period, like today."
    if isinstance(exc, ConfigError | RazorpayError):
        return str(exc)
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return "Couldn't reach Razorpay just now. Please try again."
    return "Something went wrong on our side. Please try again."

import asyncio
import atexit
import logging
import socket
import subprocess
import sys

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from merchant_copilot.config import ConfigError, Settings
from merchant_copilot.errors import root_cause
from merchant_copilot.shaping import shape_tool_output

log = logging.getLogger(__name__)

# The only Razorpay tools the model gets. Razorpay's server also offers create_refund,
# capture_payment and friends; leaving them out means no prompt can reach them.
# Money moves only through actions.py, after the merchant approves.
READ_TOOLS = {
    "fetch_payment",
    "fetch_all_payments",
    "fetch_all_orders",
    "fetch_order_payments",
    "fetch_refund",
    "fetch_all_refunds",
    "fetch_multiple_refunds_for_payment",
    "fetch_all_settlements",
    "fetch_settlement_recon_details",
    "fetch_all_payment_links",
}


async def load_razorpay_tools(settings: Settings) -> dict[str, BaseTool]:
    """Connect to the Razorpay MCP server (or the offline demo server) and return its read tools."""
    if settings.data_source == "fixtures":
        await _ensure_fixture_server(settings.fixtures_port)
        connection = {"transport": "streamable_http", "url": f"http://127.0.0.1:{settings.fixtures_port}/mcp"}
    else:
        connection = {
            "transport": "streamable_http",
            "url": settings.razorpay_mcp_url,
            "headers": {"Authorization": settings.razorpay_auth_header},
        }

    try:
        available = await MultiServerMCPClient({"razorpay": connection}).get_tools()
    except BaseException as exc:
        cause = root_cause(exc)
        if not isinstance(cause, Exception):
            raise
        raise ConfigError(f"Could not connect to the Razorpay MCP server at {connection['url']}: {cause}") from cause

    tools = {t.name: t for t in available if t.name in READ_TOOLS}
    missing = READ_TOOLS - tools.keys()
    if "fetch_payment" in missing:
        raise ConfigError(f"The MCP server at {connection['url']} does not look like Razorpay's (no fetch_payment tool)")
    if missing:
        log.warning("Razorpay MCP server is missing tools: %s", ", ".join(sorted(missing)))
    log.info("Loaded %d Razorpay tools from %s", len(tools), connection["url"])
    return tools


def for_agent(tool: BaseTool) -> BaseTool:
    """Wrap an MCP tool so the model gets rupees, IST times and compact output."""

    async def run(**kwargs) -> str:
        try:
            return shape_tool_output(await tool.ainvoke(kwargs))
        except Exception as exc:
            cause = root_cause(exc)
            log.warning("Tool %s failed with %s: %s", tool.name, kwargs, cause)
            # Returned as text so the model can correct its arguments and try again.
            return f"error: {cause}"

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=run,
    )


async def _ensure_fixture_server(port: int) -> None:
    if _port_open(port):
        return
    log.info("Starting the offline demo MCP server on port %d", port)
    proc = subprocess.Popen([sys.executable, "-m", "merchant_copilot.fixture_server"])
    atexit.register(proc.terminate)
    for _ in range(100):
        if proc.poll() is not None:
            raise ConfigError(f"The demo MCP server exited during startup (code {proc.returncode})")
        if _port_open(port):
            return
        await asyncio.sleep(0.1)
    proc.terminate()
    raise ConfigError(f"The demo MCP server did not start on port {port} within 10 seconds")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

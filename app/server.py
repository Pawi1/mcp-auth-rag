"""
MCP Auth Starter — MCP tool definitions and dispatch.

Add your own tools with the @tool decorator; the registry it fills is the
only source for both list_tools() and call_tool(), so the two cannot drift
apart, and every call is audited whatever the handler does. Handlers take
(user, arguments) and return a dict.

current_user.get() is always populated by the time a handler runs —
main.py's /mcp handler rejects the request before it gets here if the
token is missing, invalid, or revoked.

list_tools()/call_tool() keep the plain (name, arguments) shape mcp 1.x
used so the tool logic stays simple to unit-test; _on_list_tools/_on_call_tool
below just adapt that shape to the mcp 2.0 Server constructor, which takes
on_list_tools/on_call_tool callables of (ctx, params) -> typed Result instead
of the old @mcp_server.list_tools()/@mcp_server.call_tool() decorators.
"""

import json
import logging
from typing import Callable, Dict, List, Tuple

from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
    TextContent,
)

from config import MCP_SERVER_NAME
from context import current_user
from users import log_tool_call

logger = logging.getLogger("mcp-auth-starter")

SERVER_INSTRUCTIONS = """This server demonstrates a working MCP auth/transport stack:
OAuth 2.0 with Dynamic Client Registration (RFC 7591) + JWT bearer tokens,
served over Streamable HTTP. Add a connector pointing at this server's URL
and your MCP client (e.g. Claude.ai) will complete a normal browser login —
no manual token pasting required.

`whoami` is the one demo tool — it just echoes back the authenticated
user's identity, to prove the auth chain is wired correctly end to end.
Replace it with your own tools in server.py."""


def _ok(data: dict) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False))]


_TOOLS: Dict[str, Tuple[Tool, Callable]] = {}

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def tool(name: str, description: str, input_schema: dict = None):
    def register(handler):
        _TOOLS[name] = (
            Tool(name=name, description=description, inputSchema=input_schema or _EMPTY_SCHEMA),
            handler,
        )
        return handler
    return register


@tool("whoami", "Return the identity of the currently authenticated user.")
async def _whoami(user: dict, arguments: dict) -> dict:
    return {"username": user["username"], "teams": user["teams"]}


async def list_tools() -> List[Tool]:
    return [spec for spec, _ in _TOOLS.values()]


async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    user = current_user.get()
    if not user:
        return _ok({"error": "Not authenticated — connect via OAuth"})

    entry = _TOOLS.get(name)
    if entry is None:
        logger.warning(f"Tool call rejected: unknown tool {name!r} requested by {user['username']}")
        log_tool_call(user["username"], name, success=False, reason="unknown_tool")
        return _ok({"error": f"Unknown tool: {name}"})

    logger.info(f"Tool call: {name} by {user['username']}")
    try:
        result = await entry[1](user, arguments)
    except Exception:
        # audit the outcome, not the attempt: a handler that raised did not run
        logger.exception(f"Tool {name!r} failed for {user['username']}")
        log_tool_call(user["username"], name, success=False, reason="error")
        return _ok({"error": f"Tool {name} failed"})

    log_tool_call(user["username"], name)
    return _ok(result)


async def _on_list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(tools=await list_tools())


async def _on_call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
    content = await call_tool(params.name, params.arguments or {})
    return CallToolResult(content=content)


mcp_server = Server(
    MCP_SERVER_NAME,
    instructions=SERVER_INSTRUCTIONS,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)

"""
MCP Auth Starter — MCP tool definitions and dispatch.

Add your own tools here: one Tool() entry in list_tools() and a matching
`if name == "...":` branch in call_tool(). current_user.get() is always
populated by the time call_tool() runs — main.py's /mcp handler rejects
the request before it gets here if the token is missing, invalid, or
revoked.

list_tools()/call_tool() keep the plain (name, arguments) shape mcp 1.x
used so the tool logic stays simple to unit-test; _on_list_tools/_on_call_tool
below just adapt that shape to the mcp 2.0 Server constructor, which takes
on_list_tools/on_call_tool callables of (ctx, params) -> typed Result instead
of the old @mcp_server.list_tools()/@mcp_server.call_tool() decorators.
"""

import json
import logging
from typing import List

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


async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="whoami",
            description="Return the identity of the currently authenticated user.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    user = current_user.get()
    if not user:
        return _ok({"error": "Not authenticated — connect via OAuth"})

    logger.info(f"Tool call: {name} by {user['username']}")

    if name == "whoami":
        return _ok({"username": user["username"], "teams": user["teams"]})

    return _ok({"error": f"Unknown tool: {name}"})


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

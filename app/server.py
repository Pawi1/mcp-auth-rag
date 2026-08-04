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


def _serialize_document(d: dict) -> dict:
    return {
        "id": str(d["id"]),
        "filename": d["filename"],
        "format": d["format"],
        "status": d["status"],
        "error": d["error"],
        "page_count": d["page_count"],
        "chunk_count": d["chunk_count"],
        "uploaded_at": d["uploaded_at"].isoformat() if d["uploaded_at"] else None,
        "processed_at": d["processed_at"].isoformat() if d["processed_at"] else None,
    }


async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="whoami",
            description="Return the identity of the currently authenticated user.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rag_search",
            description=(
                "Search the user's uploaded documents (PDF/DOCX/TXT/MD, uploaded via the /rag "
                "panel) for relevant passages. Returns hybrid-search results (semantic + full-text, "
                "reranked when available) with citations — filename, page, section."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "top_k": {"type": "integer", "description": "Max results to return (default 8)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rag_list_documents",
            description="List the user's documents uploaded to the RAG panel, with ingest status.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    user = current_user.get()
    if not user:
        return _ok({"error": "Not authenticated — connect via OAuth"})

    # keep this pair (log line + log_tool_call) together in every branch you
    # add below — the log line is for tailing, log_tool_call is the durable,
    # queryable audit trail (tool_call_log)
    if name == "whoami":
        logger.info(f"Tool call: {name} by {user['username']}")
        log_tool_call(user["username"], name)
        return _ok({"username": user["username"], "teams": user["teams"]})

    if name == "rag_search":
        logger.info(f"Tool call: {name} by {user['username']}")
        log_tool_call(user["username"], name)
        import rag_retrieval
        from config import RAG_TOP_K
        query = str(arguments.get("query", ""))
        top_k = int(arguments.get("top_k") or RAG_TOP_K)
        results = await rag_retrieval.search(query, user["username"], top_k=top_k)
        return _ok({"results": results})

    if name == "rag_list_documents":
        logger.info(f"Tool call: {name} by {user['username']}")
        log_tool_call(user["username"], name)
        import rag_store
        documents = await rag_store.list_documents(user["username"])
        return _ok({"documents": [_serialize_document(d) for d in documents]})

    logger.warning(f"Tool call rejected: unknown tool {name!r} requested by {user['username']}")
    log_tool_call(user["username"], name, success=False, reason="unknown_tool")
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

"""Allow running MCP server via `python -m octoscout.mcp`."""

from octoscout.mcp.server import mcp

mcp.run(transport="stdio")

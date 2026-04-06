"""MCP Server for OctoScout — exposes diagnosis and matrix tools via stdio JSON-RPC."""

from __future__ import annotations

import asyncio
import json
import sys

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "octoscout_diagnose",
        "description": (
            "Diagnose a Python/ML error from a traceback. Identifies version "
            "incompatibilities, searches GitHub issues, and suggests fixes. "
            "Especially useful for TypeError, ImportError, and AttributeError "
            "in ML frameworks (transformers, torch, vllm, peft, etc.)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "traceback": {
                    "type": "string",
                    "description": "The full Python traceback or error message to diagnose.",
                },
                "auto_detect_env": {
                    "type": "boolean",
                    "description": "Automatically detect the Python environment (packages, CUDA, OS). Default true.",
                    "default": True,
                },
            },
            "required": ["traceback"],
        },
    },
    {
        "name": "octoscout_check_compatibility",
        "description": (
            "Check known compatibility issues between package versions. "
            "Provide specific package==version pairs to query, or use "
            "'auto' to scan the current environment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "string",
                    "description": (
                        "Comma-separated pkg==version pairs "
                        "(e.g. 'transformers==4.55.0,torch==2.3.0'), "
                        "or 'auto' to detect from environment."
                    ),
                },
            },
            "required": ["packages"],
        },
    },
    {
        "name": "octoscout_matrix_stats",
        "description": "Get statistics about the OctoScout compatibility matrix (number of entries, packages tracked, risk pairs).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def handle_diagnose(args: dict) -> str:
    """Run the full diagnosis pipeline."""
    from octoscout.agent.core import DiagnosisAgent
    from octoscout.config import Config

    traceback_text = args.get("traceback", "")
    auto_env = args.get("auto_detect_env", True)

    if not traceback_text:
        return "Error: No traceback provided."

    config = Config.load()
    agent = DiagnosisAgent(config=config, verbose=False, direct=True)

    try:
        result = await agent.diagnose(
            traceback_text=traceback_text,
            auto_env=auto_env,
        )
    except Exception as e:
        return f"Diagnosis failed: {e}"

    # Format result
    parts = []
    if result.summary:
        parts.append(result.summary)
    if result.related_issues:
        parts.append("\n## Related Issues")
        for issue in result.related_issues:
            parts.append(f"- [{issue.state}] {issue.title}\n  {issue.url}")
    if result.suggested_fix:
        parts.append(f"\n## Suggested Fix\n{result.suggested_fix}")

    return "\n".join(parts) if parts else "No diagnosis could be generated."


async def handle_check_compatibility(args: dict) -> str:
    """Query the compatibility matrix."""
    from pathlib import Path

    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        return "Compatibility matrix not built yet. Run 'octoscout matrix build' first."

    matrix = CompatibilityMatrix.load(matrix_path)

    packages_str = args.get("packages", "")

    if packages_str.strip().lower() == "auto":
        from octoscout.diagnosis.env_snapshot import collect_env_snapshot
        env = collect_env_snapshot()
        warnings = matrix.check(env)
        if not warnings:
            return "No known compatibility issues in current environment."
        lines = [f"Found {len(warnings)} compatibility warning(s):\n"]
        for w in warnings:
            pkgs = " + ".join(f"{k}=={v}" for k, v in w.packages.items())
            lines.append(f"**{pkgs}** — score: {w.score:.2f}")
            for p in w.problems[:3]:
                lines.append(f"  - [{p.severity}] {p.summary}")
                if p.solution:
                    lines.append(f"    Fix: {p.solution}")
            lines.append("")
        return "\n".join(lines)

    # Parse explicit pairs
    from itertools import combinations
    pairs = []
    for part in packages_str.split(","):
        part = part.strip()
        if "==" in part:
            name, version = part.split("==", 1)
            pairs.append((name.strip(), version.strip()))

    if len(pairs) < 2:
        return "Need at least 2 package==version pairs, or use 'auto'."

    lines = []
    for (pkg_a, ver_a), (pkg_b, ver_b) in combinations(pairs, 2):
        result = matrix.query_pair(pkg_a, ver_a, pkg_b, ver_b)
        if result:
            status = "OK" if result.score >= 0.7 else "RISK"
            lines.append(f"**{pkg_a}=={ver_a} + {pkg_b}=={ver_b}**: score={result.score:.2f} ({status}), {result.issue_count} issues")
            for p in result.problems[:3]:
                lines.append(f"  - [{p.severity}] {p.summary}")
        else:
            lines.append(f"**{pkg_a}=={ver_a} + {pkg_b}=={ver_b}**: No data")

    return "\n".join(lines) if lines else "No compatibility data found."


async def handle_matrix_stats(args: dict) -> str:
    """Get matrix statistics."""
    from pathlib import Path

    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        return "Matrix not built yet."

    matrix = CompatibilityMatrix.load(matrix_path)

    # Compute stats
    entries = matrix._entries
    total = len(entries)
    risk = sum(1 for e in entries.values() if e.score < 0.7)
    total_issues = sum(e.issue_count for e in entries.values())

    # Packages
    pkgs = set()
    for key in entries:
        for part in key.split("+"):
            if "==" in part:
                pkgs.add(part.split("==")[0])

    return (
        f"Compatibility Matrix Statistics:\n"
        f"- Total version pairs: {total}\n"
        f"- Risk pairs (score < 0.7): {risk}\n"
        f"- Safe pairs: {total - risk}\n"
        f"- Total issues tracked: {total_issues}\n"
        f"- Packages tracked: {len(pkgs)} ({', '.join(sorted(pkgs)[:15])}{'...' if len(pkgs) > 15 else ''})"
    )


_HANDLERS = {
    "octoscout_diagnose": handle_diagnose,
    "octoscout_check_compatibility": handle_check_compatibility,
    "octoscout_matrix_stats": handle_matrix_stats,
}


# ---------------------------------------------------------------------------
# MCP stdio JSON-RPC transport
# ---------------------------------------------------------------------------


async def handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "octoscout",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = _HANDLERS.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        try:
            result_text = await handler(tool_args)
        except Exception as e:
            result_text = f"Error: {e}"

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
            },
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


async def _handle_line(line: str) -> str | None:
    """Process one JSON-RPC line and return the response JSON (or None)."""
    line = line.strip()
    if not line:
        return None
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        return None
    response = await handle_request(request)
    if response is None:
        return None
    return json.dumps(response)


def run():
    """Run the MCP server on stdio.

    Uses synchronous stdin/stdout to be cross-platform (Windows ProactorEventLoop
    does not support connect_read_pipe on stdin).
    """
    import threading

    loop = asyncio.new_event_loop()

    # Read stdin in a background thread to avoid blocking the event loop
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _reader():
        try:
            for line in sys.stdin:
                loop.call_soon_threadsafe(input_queue.put_nowait, line)
        except (EOFError, OSError):
            pass
        finally:
            loop.call_soon_threadsafe(input_queue.put_nowait, None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    async def _main():
        # Set binary mode on stdout for Windows
        if sys.platform == "win32":
            import msvcrt
            msvcrt.setmode(sys.stdout.fileno(), 0x8000)  # O_BINARY

        while True:
            line = await input_queue.get()
            if line is None:
                break
            result = await _handle_line(line)
            if result is not None:
                sys.stdout.buffer.write((result + "\n").encode("utf-8"))
                sys.stdout.buffer.flush()

    loop.run_until_complete(_main())


if __name__ == "__main__":
    run()

"""MCP Server for OctoScout — built on the official mcp SDK (FastMCP).

Exposes diagnosis, GitHub issue search, compatibility matrix, and local
diagnostic tools via the Model Context Protocol.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "octoscout",
    instructions=(
        "OctoScout is an LLM-powered GitHub Issues agent for diagnosing "
        "Python/ML framework errors. It searches GitHub issues, checks "
        "version compatibility, and inspects local API signatures."
    ),
)


# ---------------------------------------------------------------------------
# Helper to truncate long text
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def octoscout_diagnose(traceback: str, auto_detect_env: bool = True) -> str:
    """Diagnose a Python/ML error from a traceback.

    Identifies version incompatibilities, searches GitHub issues, and suggests
    fixes. Especially useful for TypeError, ImportError, and AttributeError in
    ML frameworks (transformers, torch, vllm, peft, accelerate, deepspeed, etc.).

    Args:
        traceback: The full Python traceback or error message to diagnose.
        auto_detect_env: Automatically detect the Python environment. Default true.
    """
    if not traceback.strip():
        return "Error: No traceback provided."

    from octoscout.agent.core import DiagnosisAgent
    from octoscout.config import Config, ConfigError

    try:
        config = Config.load()
        agent = DiagnosisAgent(config=config, verbose=False, direct=True)
    except ConfigError as e:
        return f"Configuration error: {e}"

    try:
        result = await agent.diagnose(
            traceback_text=traceback,
            auto_env=auto_detect_env,
        )
    except Exception as e:
        return f"Diagnosis failed: {e}"

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


@mcp.tool()
async def octoscout_search_issues(
    query: str,
    repo: str = "",
    state: str = "",
) -> str:
    """Search GitHub issues for a query, optionally within a specific repository.

    Useful for finding known bugs, version incompatibility reports, or
    workarounds in upstream ML framework repos.

    Args:
        query: Search query text (e.g. 'TypeError Trainer tokenizer').
        repo: GitHub repo in 'owner/name' format (e.g. 'huggingface/transformers'). Optional.
        state: Issue state filter: 'open', 'closed', or leave empty for both. Optional.
    """
    if not query.strip():
        return "Error: No search query provided."

    from octoscout.config import Config
    from octoscout.search.github_client import GitHubClient

    config = Config.load()
    client = GitHubClient(token=config.github_token or None)

    try:
        issues = await client.search_issues(
            query=query,
            repo=repo or None,
            state=state or None,
        )
    except Exception as e:
        return f"Search failed: {e}"
    finally:
        await client.close()

    if not issues:
        return "No issues found matching the query."

    lines = [f"Found {len(issues)} issues:\n"]
    for i, issue in enumerate(issues, 1):
        lines.append(f"{i}. [{issue.state}] {issue.title}")
        lines.append(f"   {issue.url}")
        if issue.snippet:
            snippet = issue.snippet[:200].replace("\n", " ")
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def octoscout_get_issue_detail(repo: str, issue_number: int) -> str:
    """Get the full content of a GitHub issue including body and top comments.

    Use this after searching to read a promising issue in detail.

    Args:
        repo: GitHub repo in 'owner/name' format (e.g. 'huggingface/transformers').
        issue_number: The issue number.
    """
    from octoscout.config import Config
    from octoscout.search.github_client import GitHubClient

    config = Config.load()
    client = GitHubClient(token=config.github_token or None)

    try:
        issue = await client.get_issue(repo, issue_number)
        comments = await client.get_issue_comments(repo, issue_number, per_page=5)
    except Exception as e:
        return f"Failed to fetch issue: {e}"
    finally:
        await client.close()

    lines = [
        f"# {issue.get('title', 'Unknown')}",
        f"State: {issue.get('state', 'unknown')}",
        f"Created: {issue.get('created_at', '')}",
        f"URL: {issue.get('html_url', '')}",
        "",
        "## Body",
        _truncate(issue.get("body", "") or "", 2000),
    ]

    if comments:
        lines.append("\n## Comments")
        for c in comments[:5]:
            user = c.get("user", {}).get("login", "?")
            lines.append(f"\n### @{user} ({c.get('created_at', '')})")
            lines.append(_truncate(c.get("body", "") or "", 800))

    return "\n".join(lines)


@mcp.tool()
async def octoscout_check_compatibility(packages: str) -> str:
    """Check known compatibility issues between package versions.

    Queries a pre-built matrix of 35,000+ version pairs from 9 major ML repos.
    Provide specific package==version pairs to query, or use 'auto' to scan
    the current environment.

    Args:
        packages: Comma-separated pkg==version pairs (e.g. 'transformers==4.55.0,torch==2.3.0'), or 'auto' to detect from environment.
    """
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        return (
            "Compatibility matrix not available. "
            "Run 'octoscout matrix download' to get pre-built data, "
            "or 'octoscout matrix build' to build from scratch."
        )

    matrix = CompatibilityMatrix.load(matrix_path)

    if packages.strip().lower() == "auto":
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
    for part in packages.split(","):
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
            lines.append(
                f"**{pkg_a}=={ver_a} + {pkg_b}=={ver_b}**: "
                f"score={result.score:.2f} ({status}), {result.issue_count} issues"
            )
            for p in result.problems[:3]:
                lines.append(f"  - [{p.severity}] {p.summary}")
        else:
            lines.append(f"**{pkg_a}=={ver_a} + {pkg_b}=={ver_b}**: No data")

    return "\n".join(lines) if lines else "No compatibility data found."


@mcp.tool()
async def octoscout_check_api_signature(
    function_path: str,
    kwargs: str = "",
) -> str:
    """Check if a Python function's current signature matches given keyword arguments.

    Inspects the locally installed package to verify API compatibility.
    Useful for diagnosing TypeError about unexpected keyword arguments.

    Args:
        function_path: Dotted path to the function (e.g. 'transformers.Trainer.__init__').
        kwargs: Comma-separated keyword argument names to verify (e.g. 'tokenizer,model'). Optional.
    """
    from octoscout.diagnosis.local_checker import check_api_signature

    called_kwargs = [k.strip() for k in kwargs.split(",") if k.strip()] if kwargs else None

    result = check_api_signature(function_path, called_kwargs)
    if result is None:
        return (
            f"Could not import or inspect '{function_path}'. "
            f"The package may not be installed in the current environment."
        )
    return result.message


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("octoscout://matrix/stats")
def matrix_stats_resource() -> str:
    """Compatibility matrix statistics: entry count, risk pairs, tracked packages."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        return "Matrix not built yet. Run 'octoscout matrix download' first."

    matrix = CompatibilityMatrix.load(matrix_path)

    entries = matrix._entries
    total = len(entries)
    risk = sum(1 for e in entries.values() if e.score < 0.7)
    total_issues = sum(e.issue_count for e in entries.values())

    pkgs: set[str] = set()
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
        f"- Packages tracked: {len(pkgs)} "
        f"({', '.join(sorted(pkgs)[:15])}{'...' if len(pkgs) > 15 else ''})"
    )


@mcp.resource("octoscout://packages")
def supported_packages_resource() -> str:
    """List of supported PyPI packages and their GitHub repos (45+ packages)."""
    from octoscout.search.realtime import PACKAGE_REPO_MAP

    lines = ["Supported Package → GitHub Repo Mapping:\n"]
    for pkg, repo in sorted(PACKAGE_REPO_MAP.items()):
        lines.append(f"- {pkg} → {repo}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def diagnose_error(traceback: str) -> str:
    """Generate a diagnosis prompt for a Python/ML error traceback.

    Args:
        traceback: The Python traceback or error message to diagnose.
    """
    return (
        f"Please diagnose the following Python error using OctoScout tools.\n\n"
        f"## Traceback\n```\n{traceback}\n```\n\n"
        f"## Steps\n"
        f"1. Use `octoscout_diagnose` for a full LLM-powered analysis\n"
        f"2. Or investigate manually:\n"
        f"   - `octoscout_check_api_signature` to verify function signatures\n"
        f"   - `octoscout_search_issues` to find related GitHub issues\n"
        f"   - `octoscout_check_compatibility` to check version compatibility\n"
    )


@mcp.prompt()
def check_environment() -> str:
    """Generate a prompt for checking the current ML environment for compatibility issues."""
    return (
        "Please check this Python/ML environment for known compatibility issues.\n\n"
        "## Steps\n"
        "1. Use `octoscout_check_compatibility` with packages='auto' to scan "
        "all installed packages against the compatibility matrix\n"
        "2. For any flagged pairs, use `octoscout_search_issues` to find "
        "related discussions and workarounds\n"
        "3. Summarize findings and suggest fixes\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")

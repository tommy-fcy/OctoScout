"""Tool definitions and executors for the diagnosis agent."""

from __future__ import annotations

from typing import Any

from octoscout.models import GitHubIssueRef, ToolDefinition, ToolParameter

# ---------------------------------------------------------------------------
# Tool Definitions (sent to the LLM)
# ---------------------------------------------------------------------------

AGENT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_env_snapshot",
        description="Collect the current Python environment information including installed packages, Python version, CUDA version, and OS info.",
        parameters=[],
    ),
    ToolDefinition(
        name="check_api_signature",
        description="Check if a Python function's signature matches the given keyword arguments. Useful for diagnosing TypeError about unexpected keyword arguments.",
        parameters=[
            ToolParameter(name="function_path", type="string", description="Dotted path to the function, e.g. 'transformers.AutoModel.from_pretrained'"),
            ToolParameter(name="called_kwargs", type="string", description="Comma-separated keyword argument names that were used in the call, e.g. 'trust_remote_code,torch_dtype'"),
        ],
    ),
    ToolDefinition(
        name="search_github_issues",
        description="Search GitHub issues in a specific repository. Returns titles, URLs, and snippets of matching issues.",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query text"),
            ToolParameter(name="repo", type="string", description="GitHub repo in 'owner/name' format, e.g. 'huggingface/transformers'"),
            ToolParameter(name="state", type="string", description="Issue state filter: 'open', 'closed', or omit for both", required=False),
        ],
    ),
    ToolDefinition(
        name="get_issue_detail",
        description="Get the full content of a GitHub issue including body and top comments. Use this to read a promising issue found via search.",
        parameters=[
            ToolParameter(name="repo", type="string", description="GitHub repo in 'owner/name' format"),
            ToolParameter(name="issue_number", type="string", description="Issue number"),
        ],
    ),
    ToolDefinition(
        name="check_compatibility",
        description="Query the compatibility matrix for known issues between package version pairs. Returns compatibility scores and known problems.",
        parameters=[
            ToolParameter(
                name="packages",
                type="string",
                description="Comma-separated pkg==version pairs, e.g. 'transformers==4.55.0,torch==2.3.0'",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute agent tool calls and return results."""

    def __init__(self, github_client, env_snapshot_fn, matrix=None):
        self._github = github_client
        self._env_snapshot_fn = env_snapshot_fn
        self._env_cache: str | None = None
        self._matrix = matrix  # CompatibilityMatrix or None
        self.found_issues: list = []  # GitHubIssueRef objects found during session

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            return await handler(arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    async def _tool_get_env_snapshot(self, args: dict) -> str:
        if self._env_cache:
            return self._env_cache
        from octoscout.diagnosis.env_snapshot import collect_env_snapshot
        snap = collect_env_snapshot()
        self._env_cache = snap.format_for_llm()
        return self._env_cache

    async def _tool_check_api_signature(self, args: dict) -> str:
        from octoscout.diagnosis.local_checker import check_api_signature
        function_path = args.get("function_path", "")
        called_kwargs_str = args.get("called_kwargs", "")
        called_kwargs = [k.strip() for k in called_kwargs_str.split(",") if k.strip()]

        result = check_api_signature(function_path, called_kwargs)
        if result is None:
            return f"Could not import or inspect '{function_path}'. The package may not be installed."
        return result.message

    async def _tool_search_github_issues(self, args: dict) -> str:
        query = args.get("query", "")
        repo = args.get("repo")
        state = args.get("state")

        issues = await self._github.search_issues(query, repo=repo, state=state)
        if not issues:
            return "No issues found matching the query."

        # Track found issues (deduplicate by repo+number)
        seen = {(i.repo, i.number) for i in self.found_issues}
        for issue in issues:
            if (issue.repo, issue.number) not in seen:
                self.found_issues.append(issue)
                seen.add((issue.repo, issue.number))

        lines = [f"Found {len(issues)} issues:\n"]
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. [{issue.state}] {issue.title}")
            lines.append(f"   {issue.url}")
            if issue.snippet:
                snippet = issue.snippet[:200].replace("\n", " ")
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)

    async def _tool_get_issue_detail(self, args: dict) -> str:
        repo = args.get("repo", "")
        number = int(args.get("issue_number", 0))

        issue = await self._github.get_issue(repo, number)
        comments = await self._github.get_issue_comments(repo, number, per_page=5)

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
                lines.append(f"\n### @{c.get('user', {}).get('login', '?')} ({c.get('created_at', '')})")
                lines.append(_truncate(c.get("body", "") or "", 800))

        return "\n".join(lines)


    async def _tool_check_compatibility(self, args: dict) -> str:
        if self._matrix is None:
            return "Compatibility matrix not available. It has not been built yet."

        packages_str = args.get("packages", "")
        pairs: list[tuple[str, str]] = []
        for part in packages_str.split(","):
            part = part.strip()
            if "==" in part:
                name, version = part.split("==", 1)
                pairs.append((name.strip(), version.strip()))

        if len(pairs) == 1:
            # Single-package query: return known issues for this package
            pkg, ver = pairs[0]
            results = self._matrix.query_package(pkg, ver)
            if not results:
                return f"No known issues for {pkg}=={ver} in the matrix."
            lines = [f"Known issues for {pkg}=={ver} ({len(results)} found):\n"]
            for r in results:
                sev = r.get("severity", "low")
                lines.append(f"  [{sev}] {r.get('summary', '')}")
                if r.get("solution"):
                    lines.append(f"    Fix: {r['solution']}")
                if r.get("issue_id"):
                    lines.append(f"    Source: {r['issue_id']}")
            return "\n".join(lines)

        if len(pairs) < 2:
            return "Provide at least 1 package==version pair to query."

        from itertools import combinations

        lines: list[str] = []
        has_results = False
        for (pkg_a, ver_a), (pkg_b, ver_b) in combinations(pairs, 2):
            result = self._matrix.query_pair(pkg_a, ver_a, pkg_b, ver_b)
            if result:
                has_results = True
                status = "OK" if result.score >= 0.7 else "RISK"
                lines.append(
                    f"{pkg_a}=={ver_a} + {pkg_b}=={ver_b}: "
                    f"score={result.score:.2f} ({status}), {result.issue_count} known issues"
                )
                for p in result.problems[:3]:
                    lines.append(f"  - [{p.severity}] {p.summary}")
                    if p.solution:
                        lines.append(f"    Fix: {p.solution}")
                    if p.source_issues:
                        lines.append(f"    Source: {', '.join(p.source_issues)}")
                    # Record source issues so search_github_issues can deduplicate
                    for ref in p.source_issues:
                        self._record_source_issue(ref)
            else:
                lines.append(
                    f"{pkg_a}=={ver_a} + {pkg_b}=={ver_b}: No data in matrix"
                )

        if has_results:
            lines.append(
                "\nTip: Use get_issue_detail to read any of the source issues "
                "above for the full solution context. These issues are already "
                "in the pre-built matrix — no need to search for them again."
            )

        return "\n".join(lines) if lines else "No compatibility data found."

    def _record_source_issue(self, ref: str) -> None:
        """Parse an issue ref like 'owner/repo#123' and add to found_issues."""
        if "#" not in ref:
            return
        repo, num_str = ref.rsplit("#", 1)
        try:
            number = int(num_str)
        except ValueError:
            return
        # Skip if already tracked
        if any(i.repo == repo and i.number == number for i in self.found_issues):
            return
        self.found_issues.append(GitHubIssueRef(
            repo=repo,
            number=number,
            title="(from matrix)",
            url=f"https://github.com/{repo}/issues/{number}",
            state="closed",
        ))


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... (truncated)"

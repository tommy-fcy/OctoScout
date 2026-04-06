"""Agent core: ReAct-style orchestrator for diagnosis."""

from __future__ import annotations

from rich.console import Console
from rich.status import Status

from octoscout.agent.prompts import DIAGNOSIS_REPORT_FORMAT, SYSTEM_PROMPT
from octoscout.agent.tools import AGENT_TOOLS, ToolExecutor
from octoscout.config import Config
from octoscout.diagnosis.env_snapshot import collect_env_snapshot
from octoscout.diagnosis.traceback_parser import parse_traceback
from octoscout.diagnosis.triage import TriageDecision, triage
from octoscout.models import (
    DiagnosisResult,
    Message,
    ProblemType,
    Role,
    TriageResult,
)
from octoscout.search.github_client import GitHubClient
from octoscout.search.realtime import build_search_queries

_MAX_AGENT_TURNS = 15
_console = Console()


class DiagnosisAgent:
    """ReAct-style agent that diagnoses Python/ML errors.

    Flow:
    1. Parse traceback and collect environment (no LLM needed)
    2. Run heuristic triage
    3. If upstream/ambiguous: enter LLM ReAct loop with tool use
    4. Synthesize final diagnosis
    """

    def __init__(self, config: Config, verbose: bool = False, direct: bool = False):
        self._config = config
        self._verbose = verbose
        self._direct = direct  # Skip triage, go straight to ReAct
        self._provider = config.get_provider()
        self._github = GitHubClient(token=config.github_token or None)

        # Load compatibility matrix if available
        self._matrix = None
        from pathlib import Path
        matrix_path = Path(config.matrix_data_dir) / "matrix.json"
        if matrix_path.exists():
            try:
                from octoscout.matrix.aggregator import CompatibilityMatrix
                self._matrix = CompatibilityMatrix.load(matrix_path)
            except Exception:
                pass  # Non-fatal; proceed without matrix

    async def diagnose(
        self,
        traceback_text: str,
        auto_env: bool = True,
        extra_repos: list[str] | None = None,
    ) -> DiagnosisResult:
        """Run the full diagnosis pipeline."""
        try:
            return await self._diagnose_impl(traceback_text, auto_env, extra_repos)
        finally:
            await self._github.close()

    async def _diagnose_impl(
        self,
        traceback_text: str,
        auto_env: bool,
        extra_repos: list[str] | None,
    ) -> DiagnosisResult:
        # Step 1: Parse traceback (no LLM)
        with Status("[bold cyan]Parsing traceback...", console=_console):
            tb = parse_traceback(traceback_text)
        if self._verbose:
            _console.print(f"[dim]Parsed: {tb.exception_type}: {tb.exception_message}[/dim]")
            _console.print(f"[dim]Packages: {tb.involved_packages}[/dim]")

        # Step 2: Collect environment (no LLM)
        env = None
        if auto_env:
            with Status("[bold cyan]Collecting environment info...", console=_console):
                env = collect_env_snapshot()
            if self._verbose:
                _console.print(f"[dim]Python: {env.python_version}[/dim]")

        # Step 3: Triage
        if self._direct:
            # Direct mode: user already believes this is upstream — skip triage
            triage_result = TriageDecision(
                result=TriageResult.UPSTREAM_ISSUE,
                problem_type=ProblemType.UNKNOWN,
                confidence=0.5,
                reasoning="Direct mode: skipping heuristic triage",
            )
            if self._verbose:
                _console.print("[dim]Triage: skipped (direct mode)[/dim]")
        else:
            triage_result = triage(tb, env)
            if self._verbose:
                _console.print(
                    f"[dim]Triage: {triage_result.result.value} "
                    f"({triage_result.problem_type.value}, "
                    f"confidence={triage_result.confidence:.2f})[/dim]"
                )
                _console.print(f"[dim]Reasoning: {triage_result.reasoning}[/dim]")

            # If clearly a local issue, give quick advice without search
            if triage_result.result == TriageResult.LOCAL_ISSUE and triage_result.confidence > 0.7:
                return await self._quick_local_diagnosis(tb, env, triage_result)

        # Step 4: Enter ReAct loop
        return await self._react_loop(tb, env, triage_result, traceback_text, extra_repos)

    async def _quick_local_diagnosis(self, tb, env, triage_result) -> DiagnosisResult:
        """Handle clear local issues with a single LLM call (no tool use)."""
        context = self._build_context(tb, env)
        messages = [
            Message(
                role=Role.USER,
                content=f"Diagnose this error. It appears to be a local code issue.\n\n{context}\n\n{DIAGNOSIS_REPORT_FORMAT}",
            )
        ]

        with Status("[bold cyan]Analyzing...", console=_console):
            response = await self._provider.chat(messages, system=SYSTEM_PROMPT)

        return DiagnosisResult(
            triage=triage_result.result,
            problem_type=triage_result.problem_type,
            summary=response.text or "Unable to generate diagnosis.",
            confidence=triage_result.confidence,
        )

    async def _react_loop(
        self, tb, env, triage_result, traceback_text: str, extra_repos: list[str] | None,
    ) -> DiagnosisResult:
        """ReAct loop: LLM reasons and uses tools iteratively."""
        executor = ToolExecutor(
            github_client=self._github,
            env_snapshot_fn=collect_env_snapshot,
            matrix=self._matrix,
        )

        context = self._build_context(tb, env)

        # Pre-compute search suggestions so the LLM doesn't have to guess
        search_suggestions = self._build_search_suggestions(tb, env, extra_repos)

        user_content = (
            f"Please diagnose the following error.\n\n"
            f"## Raw Traceback\n```\n{traceback_text}\n```\n\n"
            f"{context}\n\n"
            f"## Triage\n"
            f"Classification: {triage_result.result.value}\n"
            f"Problem type: {triage_result.problem_type.value}\n"
            f"Reasoning: {triage_result.reasoning}\n\n"
        )

        if search_suggestions:
            user_content += (
                f"## Suggested Search Strategies\n"
                f"The following pre-computed queries may help. "
                f"Use them as starting points with `search_github_issues`, "
                f"but feel free to refine or try different queries based on your analysis.\n\n"
                f"{search_suggestions}\n\n"
            )

        user_content += "Please investigate using the available tools."

        messages: list[Message] = [
            Message(role=Role.USER, content=user_content),
        ]

        for turn in range(_MAX_AGENT_TURNS):
            with Status(
                f"[bold cyan]Agent thinking (turn {turn + 1})...", console=_console
            ):
                response = await self._provider.chat_with_tools(
                    messages, AGENT_TOOLS, system=SYSTEM_PROMPT
                )

            if not response.has_tool_calls:
                # Agent finished investigating — request a clean formatted synthesis
                messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response.text or "",
                ))
                messages.append(Message(
                    role=Role.USER,
                    content=f"Now provide your final diagnosis.\n\n{DIAGNOSIS_REPORT_FORMAT}",
                ))
                with Status("[bold cyan]Synthesizing diagnosis...", console=_console):
                    final_response = await self._provider.chat(
                        messages, system=SYSTEM_PROMPT
                    )
                return DiagnosisResult(
                    triage=triage_result.result,
                    problem_type=triage_result.problem_type,
                    summary=final_response.text or "Unable to generate diagnosis.",
                    confidence=triage_result.confidence,
                    related_issues=executor.found_issues,
                )

            # Record assistant message with tool calls
            messages.append(Message(
                role=Role.ASSISTANT,
                content=response.text or "",
                tool_calls=response.tool_calls,
            ))

            # Execute each tool call
            for tc in response.tool_calls:
                if self._verbose:
                    _console.print(f"[dim]Tool: {tc.name}({tc.arguments})[/dim]")

                with Status(
                    f"[bold cyan]Running {tc.name}...", console=_console
                ):
                    result = await executor.execute(tc.name, tc.arguments)

                if self._verbose:
                    preview = result[:200] + "..." if len(result) > 200 else result
                    _console.print(f"[dim]Result: {preview}[/dim]")

                messages.append(Message(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tc.id,
                ))

        # Exhausted turns
        return DiagnosisResult(
            triage=triage_result.result,
            problem_type=triage_result.problem_type,
            summary="Diagnosis incomplete: agent reached maximum turns.",
            confidence=0.0,
            related_issues=executor.found_issues,
        )

    @staticmethod
    def _build_context(tb, env) -> str:
        """Build context string from parsed traceback and environment."""
        parts = [tb.format_for_llm()]
        if env:
            parts.append(env.format_for_llm())
        return "\n\n".join(parts)

    @staticmethod
    def _build_search_suggestions(tb, env, extra_repos) -> str:
        """Pre-compute search query suggestions from traceback and environment.

        These are hints for the LLM — it can use them directly or adapt them.
        This leverages the package-to-repo mapping and multi-strategy query builder
        so the LLM doesn't have to guess which repos to search.
        """
        queries = build_search_queries(tb, env, extra_repos)
        if not queries:
            return ""

        lines = []
        for i, q in enumerate(queries[:6], 1):  # Limit to top 6 suggestions
            repo_part = f" in `{q.repo}`" if q.repo else ""
            lines.append(f"{i}. `{q.query}`{repo_part} — {q.rationale}")
        return "\n".join(lines)

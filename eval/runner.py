"""Eval runner: execute OctoScout diagnosis on eval cases and collect scores."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.table import Table

from octoscout.agent.core import DiagnosisAgent
from octoscout.config import Config

from eval.loader import load_all_cases
from eval.models import CaseResult, EvalReport, EvalCase
from eval.scorers import score_case

_console = Console()


async def run_single_case(
    case: EvalCase,
    config: Config,
    verbose: bool = False,
) -> CaseResult:
    """Run OctoScout diagnosis on a single eval case and score the result."""
    agent = DiagnosisAgent(config=config, verbose=verbose)

    start = time.monotonic()
    try:
        result = await agent.diagnose(
            traceback_text=case.traceback,
            auto_env=False,  # Don't auto-detect env in eval — use case data
        )
        latency = time.monotonic() - start

        scores, weighted = score_case(result, case)

        return CaseResult(
            case_id=case.id,
            category=case.category,
            difficulty=case.difficulty,
            scores=scores,
            weighted_score=weighted,
            latency_seconds=latency,
        )

    except Exception as e:
        latency = time.monotonic() - start
        return CaseResult(
            case_id=case.id,
            category=case.category,
            difficulty=case.difficulty,
            weighted_score=0.0,
            latency_seconds=latency,
            error=str(e),
        )


async def run_eval(
    category: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    verbose: bool = False,
    max_concurrent: int = 3,
) -> EvalReport:
    """Run the full evaluation suite."""
    cases = load_all_cases(category=category)
    if not cases:
        _console.print("[red]No eval cases found.[/red]")
        return EvalReport(model=model or "default", total_cases=0)

    _console.print(f"[bold cyan]Running {len(cases)} eval cases...[/bold cyan]\n")

    # Build config with model/provider overrides
    overrides: dict = {}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        if (provider or "claude") == "openai":
            overrides["openai_model"] = model
        else:
            overrides["claude_model"] = model
    config = Config.load(cli_overrides=overrides or None)

    effective_model = model or (
        config.claude_model if config.llm_provider == "claude" else config.openai_model
    )

    # Run cases with limited concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(case: EvalCase) -> CaseResult:
        async with semaphore:
            _console.print(f"  [dim]Running: {case.id}...[/dim]")
            result = await run_single_case(case, config, verbose)
            icon = "[green]\u2713[/green]" if result.passed else "[red]\u2717[/red]"
            _console.print(
                f"  {icon} {case.id}: {result.weighted_score:.2f} "
                f"({result.latency_seconds:.1f}s)"
            )
            return result

    tasks = [run_with_semaphore(case) for case in cases]
    results = await asyncio.gather(*tasks)

    report = EvalReport(
        model=effective_model,
        total_cases=len(cases),
        results=list(results),
    )

    return report


def print_report(report: EvalReport) -> None:
    """Print a formatted evaluation report to the console."""
    _console.print()
    _console.print(f"[bold]Evaluation Report — {report.model}[/bold]")
    _console.print(f"{'=' * 60}")

    # Summary
    _console.print(f"\n[bold]Summary:[/bold]")
    _console.print(f"  Cases:     {report.total_cases}")
    _console.print(f"  Passed:    {report.pass_count}/{report.total_cases} ({report.pass_rate:.0%})")
    _console.print(f"  Avg Score: {report.avg_score:.2f}")
    _console.print(f"  Avg Time:  {report.avg_latency:.1f}s")

    # Per-dimension breakdown
    dims = report.by_dimension()
    if dims:
        _console.print(f"\n[bold]Per-Dimension Averages:[/bold]")
        dim_table = Table(show_header=True)
        dim_table.add_column("Dimension", style="cyan")
        dim_table.add_column("Avg Score", justify="right")
        for dim, scores in sorted(dims.items()):
            avg = sum(scores) / len(scores)
            color = "green" if avg >= 0.8 else "yellow" if avg >= 0.5 else "red"
            dim_table.add_row(dim, f"[{color}]{avg:.2f}[/{color}]")
        _console.print(dim_table)

    # Per-case detail
    _console.print(f"\n[bold]Per-Case Results:[/bold]")
    case_table = Table(show_header=True)
    case_table.add_column("Case ID", style="cyan", max_width=40)
    case_table.add_column("Category")
    case_table.add_column("Score", justify="right")
    case_table.add_column("Time", justify="right")
    case_table.add_column("Status")

    for r in sorted(report.results, key=lambda x: x.weighted_score):
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        if r.error:
            status = f"[red]ERROR: {r.error[:30]}[/red]"
        score_color = "green" if r.weighted_score >= 0.8 else "yellow" if r.weighted_score >= 0.5 else "red"
        case_table.add_row(
            r.case_id,
            r.category,
            f"[{score_color}]{r.weighted_score:.2f}[/{score_color}]",
            f"{r.latency_seconds:.1f}s",
            status,
        )
    _console.print(case_table)

    # Per-category breakdown
    cats = report.by_category()
    if len(cats) > 1:
        _console.print(f"\n[bold]Per-Category:[/bold]")
        cat_table = Table(show_header=True)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Pass Rate", justify="right")
        cat_table.add_column("Avg Score", justify="right")
        for cat, results in sorted(cats.items()):
            count = len(results)
            passed = sum(1 for r in results if r.passed)
            avg = sum(r.weighted_score for r in results) / count
            cat_table.add_row(cat, str(count), f"{passed}/{count}", f"{avg:.2f}")
        _console.print(cat_table)

    # Failed cases detail
    failed = [r for r in report.results if not r.passed]
    if failed:
        _console.print(f"\n[bold red]Failed Cases Detail:[/bold red]")
        for r in failed:
            _console.print(f"\n  [bold]{r.case_id}[/bold]")
            if r.error:
                _console.print(f"    Error: {r.error}")
            for s in r.scores:
                if s.score < 0.8:
                    _console.print(f"    [{s.dimension}] {s.score:.2f} — {s.details}")

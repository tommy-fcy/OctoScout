"""CLI entry point for OctoScout."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from octoscout import __version__

app = typer.Typer(
    name="octoscout",
    help="OctoScout - LLM-Powered GitHub Issues Agent",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"OctoScout v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    """OctoScout - LLM-Powered GitHub Issues Agent: Search, Ask, and Give Back."""
    pass


@app.command()
def diagnose(
    traceback_input: str = typer.Argument(
        None,
        help="Traceback text, or path to a file containing the traceback.",
    ),
    auto_env: bool = typer.Option(
        True, "--auto-env/--no-auto-env",
        help="Automatically detect the Python environment.",
    ),
    provider: str = typer.Option(
        None, "--provider", "-p",
        help="LLM provider to use: 'claude' or 'openai'. Defaults to config.",
    ),
    repo: list[str] = typer.Option(
        None, "--repo", "-r",
        help="GitHub repos to search (e.g. 'huggingface/transformers'). Repeatable.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose",
        help="Show detailed agent reasoning steps.",
    ),
):
    """Diagnose a Python/ML error from a traceback.

    Pass the traceback as a string argument, or pipe it via stdin:

        octoscout diagnose "Traceback (most recent call last): ..."

        python my_script.py 2>&1 | octoscout diagnose -
    """
    # Read traceback from argument, file, or stdin
    tb_text = _read_traceback(traceback_input)
    if not tb_text:
        console.print("[red]Error:[/red] No traceback provided. Pass it as an argument or pipe via stdin.")
        raise typer.Exit(1)

    from octoscout.config import Config

    config = Config.from_env()
    if provider:
        config.llm_provider = provider

    console.print(Panel("OctoScout Diagnosis", style="bold cyan"))

    asyncio.run(_run_diagnosis(tb_text, config, auto_env, repo or [], verbose))


@app.command()
def matrix(
    packages: list[str] = typer.Argument(
        None,
        help="Packages to check, e.g. 'transformers==4.55.0' 'torch==2.3.0'",
    ),
):
    """Query the compatibility matrix for known issues between package versions."""
    console.print("[yellow]Compatibility matrix not yet implemented (Phase 2).[/yellow]")
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_traceback(source: str | None) -> str | None:
    """Read traceback from argument, file path, or stdin."""
    if source is None or source == "-":
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return None

    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()

    return source.strip()


async def _run_diagnosis(
    traceback_text: str,
    config,
    auto_env: bool,
    repos: list[str],
    verbose: bool,
):
    """Run the full diagnosis pipeline."""
    from octoscout.agent.core import DiagnosisAgent

    agent = DiagnosisAgent(config=config, verbose=verbose)
    result = await agent.diagnose(
        traceback_text=traceback_text,
        auto_env=auto_env,
        extra_repos=repos,
    )

    # Display result
    if result.summary:
        console.print()
        console.print(Markdown(result.summary))

    if result.related_issues:
        console.print()
        console.print("[bold]Related Issues:[/bold]")
        for issue in result.related_issues:
            state_color = "green" if issue.state == "open" else "dim"
            console.print(f"  [{state_color}]{issue.state}[/{state_color}] {issue.url}")
            console.print(f"        {issue.title}")

    if result.suggested_fix:
        console.print()
        console.print(Panel(result.suggested_fix, title="Suggested Fix", style="green"))

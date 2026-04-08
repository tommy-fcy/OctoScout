"""CLI entry point for OctoScout."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

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
    traceback_input: str | None = typer.Argument(
        default=None,
        help="Traceback text, or path to a file containing the traceback.",
    ),
    auto_env: bool = typer.Option(
        True, "--auto-env/--no-auto-env",
        help="Automatically detect the Python environment.",
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: 'claude' or 'openai'. Overrides config file and env.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model name to use (e.g. 'claude-sonnet-4-6'). Overrides config file and env.",
    ),
    repo: list[str] = typer.Option(
        None, "--repo", "-r",
        help="GitHub repos to search (e.g. 'huggingface/transformers'). Repeatable.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose",
        help="Show detailed agent reasoning steps.",
    ),
    direct: bool = typer.Option(
        True, "--direct/--triage",
        help="Direct mode (default): skip heuristic triage, go straight to ReAct agent. Use --triage to enable heuristic pre-filtering.",
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

    from octoscout.config import Config, ConfigError

    # Build CLI overrides — only pass values that were explicitly provided
    cli_overrides: dict = {}
    if provider is not None:
        cli_overrides["llm_provider"] = provider
    if model is not None:
        # Route to the correct model field based on provider
        # (provider override may or may not be set yet, so check both)
        effective_provider = provider or "claude"  # temporary; load() resolves the real default
        if effective_provider == "openai":
            cli_overrides["openai_model"] = model
        else:
            cli_overrides["claude_model"] = model

    config = Config.load(cli_overrides=cli_overrides or None)

    # Validate LLM credentials early (before starting the pipeline)
    try:
        config.get_provider()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red]\n{e}")
        raise typer.Exit(1)

    console.print(Panel("OctoScout Diagnosis", style="bold cyan"))

    asyncio.run(_run_diagnosis(tb_text, config, auto_env, repo or [], verbose, direct))


# ---------------------------------------------------------------------------
# Matrix sub-app
# ---------------------------------------------------------------------------

matrix_app = typer.Typer(name="matrix", help="Compatibility matrix commands.")
app.add_typer(matrix_app, name="matrix")


@matrix_app.command()
def crawl(
    repo: list[str] = typer.Option(None, "--repo", "-r", help="Specific repos to crawl."),
    all_repos: bool = typer.Option(False, "--all", help="Crawl all default repos."),
    max_pages: int = typer.Option(None, "--max-pages", help="Override max pages per repo."),
    with_comments: bool = typer.Option(False, "--with-comments", help="Also fetch issue comments (slow)."),
):
    """Crawl GitHub issues for compatibility data."""
    from octoscout.config import Config
    from octoscout.matrix.crawler import DEFAULT_CRAWL_CONFIGS, CrawlConfig, MatrixCrawler
    from octoscout.search.github_client import GitHubClient

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)

    if repo:
        configs = [CrawlConfig(repo=r, max_pages=max_pages or 50) for r in repo]
    elif all_repos:
        configs = DEFAULT_CRAWL_CONFIGS
        if max_pages:
            configs = [CrawlConfig(repo=c.repo, labels=c.labels, keywords=c.keywords, max_pages=max_pages, state=c.state) for c in configs]
    else:
        console.print("[red]Specify --repo or --all[/red]")
        raise typer.Exit(1)

    client = GitHubClient(token=config.github_token or None)

    async def _run():
        try:
            crawler = MatrixCrawler(client, data_dir, fetch_comments=with_comments)
            return await crawler.crawl_all(configs)
        finally:
            await client.close()

    results = asyncio.run(_run())
    total = sum(s.passed_filter for s in results.values())
    console.print(f"\n[bold green]Done.[/bold green] {total} issues saved across {len(results)} repos.")


@matrix_app.command()
def patch_metadata(
    repo: str = typer.Option(None, "--repo", "-r", help="Specific repo slug to patch."),
    all_repos: bool = typer.Option(False, "--all", help="Patch all crawled repos."),
):
    """Batch-update comment_count and issue_reactions for existing issues."""
    from octoscout.config import Config
    from octoscout.matrix.crawler import MatrixCrawler
    from octoscout.search.github_client import GitHubClient

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)
    client = GitHubClient(token=config.github_token or None)

    if not repo and not all_repos:
        console.print("[red]Specify --repo or --all[/red]")
        raise typer.Exit(1)

    async def _run():
        try:
            crawler = MatrixCrawler(client, data_dir)
            return await crawler.patch_metadata(repo_slug=repo if not all_repos else None)
        finally:
            await client.close()

    results = asyncio.run(_run())
    total = sum(results.values())
    console.print(f"\n[bold green]Done.[/bold green] {total} issues patched with metadata.")


@matrix_app.command()
def enrich(
    repo: str = typer.Option(None, "--repo", "-r", help="Specific repo slug to enrich."),
    all_repos: bool = typer.Option(False, "--all", help="Enrich all crawled repos."),
    top_k: int = typer.Option(8, "--top-k", help="Number of top comments to keep per issue."),
):
    """Fetch and score comments for crawled issues (selective enrichment)."""
    from octoscout.config import Config
    from octoscout.matrix.crawler import MatrixCrawler
    from octoscout.search.github_client import GitHubClient

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)
    client = GitHubClient(token=config.github_token or None)

    async def _run():
        try:
            crawler = MatrixCrawler(client, data_dir)
            slug = repo if repo else None
            if not all_repos and not repo:
                console.print("[red]Specify --repo or --all[/red]")
                return {}
            return await crawler.enrich_comments(
                repo_slug=slug if not all_repos else None,
                top_k=top_k,
            )
        finally:
            await client.close()

    results = asyncio.run(_run())
    total = sum(results.values())
    console.print(f"\n[bold green]Done.[/bold green] {total} issues enriched with comments.")


@matrix_app.command()
def extract(
    repo: str = typer.Option(None, "--repo", "-r", help="Specific repo slug (e.g. huggingface_transformers)."),
    all_repos: bool = typer.Option(False, "--all", help="Extract all crawled repos."),
    concurrency: int = typer.Option(None, "--concurrency", "-c", help="Max concurrent LLM calls."),
    verbose: bool = typer.Option(False, "--verbose", help="Show error details for failed extractions."),
):
    """Extract structured compatibility info from crawled issues using LLM."""
    from octoscout.config import Config, ConfigError
    from octoscout.matrix.extractor import MatrixExtractor

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)

    try:
        provider = config.get_extraction_provider()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red]\n{e}")
        raise typer.Exit(1)

    extractor = MatrixExtractor(
        provider=provider,
        input_dir=data_dir,
        output_dir=data_dir / "extracted",
        concurrency=concurrency or config.extract_concurrency,
        log_errors=verbose,
    )

    if repo:
        slugs = [repo]
    elif all_repos:
        slugs = None  # extract_all discovers from raw/
    else:
        console.print("[red]Specify --repo or --all[/red]")
        raise typer.Exit(1)

    results = asyncio.run(extractor.extract_all(slugs))
    total = sum(s.extracted for s in results.values())
    console.print(f"\n[bold green]Done.[/bold green] {total} issues extracted.")


@matrix_app.command()
def build():
    """Build the compatibility matrix from extracted data."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)
    extracted_dir = data_dir / "extracted"
    output_path = data_dir / "matrix.json"

    if not extracted_dir.exists() or not list(extracted_dir.glob("*.jsonl")):
        console.print("[red]No extracted data found. Run 'octoscout matrix extract' first.[/red]")
        raise typer.Exit(1)

    matrix = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)
    console.print(f"[bold green]Matrix built.[/bold green] {matrix.entry_count} version-pair entries saved to {output_path}")


@matrix_app.command()
def query(
    packages: list[str] = typer.Argument(
        ...,
        help="Package==version pairs, e.g. 'transformers==4.55.0' 'torch==2.3.0'",
    ),
):
    """Query the compatibility matrix for a specific version combination."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        console.print("[red]Matrix not built yet. Run 'octoscout matrix build' first.[/red]")
        raise typer.Exit(1)

    # Parse packages
    parsed: list[tuple[str, str]] = []
    for p in packages:
        if "==" not in p:
            console.print(f"[red]Invalid format: {p}. Use pkg==version.[/red]")
            raise typer.Exit(1)
        name, version = p.split("==", 1)
        parsed.append((name.strip(), version.strip()))

    matrix = CompatibilityMatrix.load(matrix_path)

    from itertools import combinations

    from rich.table import Table

    table = Table(title="Compatibility Query Results")
    table.add_column("Package A")
    table.add_column("Package B")
    table.add_column("Score", justify="right")
    table.add_column("Issues", justify="right")
    table.add_column("Status")

    for (pkg_a, ver_a), (pkg_b, ver_b) in combinations(parsed, 2):
        result = matrix.query_pair(pkg_a, ver_a, pkg_b, ver_b)
        if result:
            status = "[green]OK[/green]" if result.score >= 0.7 else "[red]RISK[/red]"
            table.add_row(
                f"{pkg_a}=={ver_a}", f"{pkg_b}=={ver_b}",
                f"{result.score:.2f}", str(result.issue_count), status,
            )
        else:
            table.add_row(
                f"{pkg_a}=={ver_a}", f"{pkg_b}=={ver_b}",
                "-", "0", "[dim]No data[/dim]",
            )

    console.print(table)


@matrix_app.command()
def check(
    auto_env: bool = typer.Option(
        True, "--auto-env/--no-auto-env",
        help="Automatically detect the Python environment.",
    ),
):
    """Check current environment for known compatibility issues."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        console.print("[red]Matrix not built yet. Run 'octoscout matrix build' first.[/red]")
        raise typer.Exit(1)

    if auto_env:
        from octoscout.diagnosis.env_snapshot import collect_env_snapshot
        env = collect_env_snapshot()
    else:
        console.print("[yellow]No environment provided. Use --auto-env.[/yellow]")
        raise typer.Exit(1)

    matrix = CompatibilityMatrix.load(matrix_path)
    warnings = matrix.check(env)

    if not warnings:
        console.print("[bold green]No known compatibility issues found.[/bold green]")
        return

    console.print(f"[bold red]Found {len(warnings)} compatibility warning(s):[/bold red]\n")
    for w in warnings:
        pkgs = " + ".join(f"{k}=={v}" for k, v in w.packages.items())
        console.print(Panel(
            f"Score: {w.score:.2f}\n"
            f"Recommendation: {w.recommendation}\n"
            f"Known problems: {len(w.problems)}",
            title=f"[red]{pkgs}[/red]",
            style="red" if w.score < 0.3 else "yellow",
        ))


@matrix_app.command(name="index")
def build_index():
    """Build the local FAISS vector index from extracted data (requires octoscout[vector])."""
    from octoscout.config import Config

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)
    extracted_dir = data_dir / "extracted"
    index_dir = data_dir / "index"

    if not extracted_dir.exists() or not list(extracted_dir.glob("*.jsonl")):
        console.print("[red]No extracted data found. Run 'octoscout matrix extract' first.[/red]")
        raise typer.Exit(1)

    try:
        from octoscout.search.local_index import LocalIndex
    except ImportError:
        console.print("[red]Vector dependencies not installed. Run: pip install octoscout[vector][/red]")
        raise typer.Exit(1)

    idx = LocalIndex(index_dir)
    count = asyncio.run(idx.build(extracted_dir))
    console.print(f"[bold green]Index built.[/bold green] {count} issues indexed at {index_dir}")


@matrix_app.command()
def stats():
    """Show statistics about the built matrix."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)
    matrix_path = data_dir / "matrix.json"

    # Raw stats
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.jsonl"))
        raw_count = 0
        for f in raw_files:
            with open(f) as fh:
                raw_count += sum(1 for line in fh if line.strip())
        console.print(f"Raw issues: {raw_count} across {len(raw_files)} repos")
    else:
        console.print("[dim]No raw data.[/dim]")

    # Extracted stats
    extracted_dir = data_dir / "extracted"
    if extracted_dir.exists():
        ext_files = list(extracted_dir.glob("*.jsonl"))
        ext_count = 0
        for f in ext_files:
            with open(f) as fh:
                ext_count += sum(1 for line in fh if line.strip())
        console.print(f"Extracted issues: {ext_count} across {len(ext_files)} repos")
    else:
        console.print("[dim]No extracted data.[/dim]")

    # Matrix stats
    if matrix_path.exists():
        matrix = CompatibilityMatrix.load(matrix_path)
        console.print(f"Matrix entries: {matrix.entry_count} version pairs")
    else:
        console.print("[dim]Matrix not built yet.[/dim]")


@matrix_app.command()
def heatmap(
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output HTML file path. Defaults to data/matrix/heatmap.html.",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open",
        help="Open the heatmap in browser after generating.",
    ),
):
    """Generate an interactive heatmap visualization of the compatibility matrix."""
    from octoscout.config import Config
    from octoscout.matrix.aggregator import CompatibilityMatrix
    from octoscout.matrix.visualizer import generate_heatmap_html

    config = Config.load()
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"

    if not matrix_path.exists():
        console.print("[red]Matrix not built yet. Run 'octoscout matrix build' first.[/red]")
        raise typer.Exit(1)

    matrix = CompatibilityMatrix.load(matrix_path)
    output_path = Path(output) if output else Path(config.matrix_data_dir) / "heatmap.html"
    generate_heatmap_html(matrix, output_path)

    console.print(f"[bold green]Heatmap generated:[/bold green] {output_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(str(output_path.resolve()))


@matrix_app.command()
def download(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing matrix.json if present.",
    ),
):
    """Download pre-built compatibility matrix from GitHub Releases."""
    from octoscout.config import Config
    from octoscout.matrix.downloader import DownloadError

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)

    async def _run():
        from octoscout.matrix.downloader import download_matrix
        return await download_matrix(
            data_dir=data_dir,
            token=config.github_token or None,
            force=force,
        )

    try:
        output = asyncio.run(_run())
        console.print(f"[bold green]Saved to:[/bold green] {output}")
    except DownloadError as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)


@matrix_app.command(name="update-check")
def update_check():
    """Check if a newer pre-built matrix is available."""
    from octoscout.config import Config

    config = Config.load()
    data_dir = Path(config.matrix_data_dir)

    async def _run():
        from octoscout.matrix.downloader import check_update
        return await check_update(data_dir, token=config.github_token or None)

    result = asyncio.run(_run())
    if result is None:
        console.print("[bold green]Matrix is up to date.[/bold green]")
    else:
        console.print(
            f"[bold yellow]Update available![/bold yellow] "
            f"Release {result['tag']} (published: {result['published_at']})"
        )
        local = result.get("local_built_at")
        if local:
            console.print(f"  Local matrix built: {local}")
        else:
            console.print("  No local matrix found.")
        console.print("  Run [bold]octoscout matrix download --force[/bold] to update.")


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
    direct: bool = True,
):
    """Run the full diagnosis pipeline."""
    from octoscout.agent.core import DiagnosisAgent

    # Hint if matrix data is not available
    matrix_path = Path(config.matrix_data_dir) / "matrix.json"
    if not matrix_path.exists():
        console.print(
            "[dim]Tip: Run [bold]octoscout matrix download[/bold] to get "
            "pre-built compatibility data (35,000+ version pairs).[/dim]\n"
        )

    # Hint if GitHub token is not set
    if not config.github_token:
        console.print(
            "[dim]Tip: Set GITHUB_TOKEN for higher API rate limits "
            "(5,000/hr vs 60/hr).[/dim]\n"
        )

    agent = DiagnosisAgent(config=config, verbose=verbose, direct=direct)
    result = await agent.diagnose(
        traceback_text=traceback_text,
        auto_env=auto_env,
        extra_repos=repos,
    )

    # Display result — LLM summary already contains Related Issues and Fix sections,
    # so we only print the summary to avoid duplication.
    if result.summary:
        console.print()
        console.print(Markdown(result.summary))

    # --- Community features: offer to draft issue or suggest reply ---
    if sys.stdin.isatty():
        await _offer_community_actions(result, traceback_text, config, auto_env)


async def _offer_community_actions(result, traceback_text: str, config, auto_env: bool):
    """Offer to draft an issue or suggest a reply after diagnosis."""
    from rich.prompt import Confirm

    from octoscout.search.github_client import GitHubClient

    provider = config.get_provider()

    # Check for open related issues we could reply to
    open_issues = [i for i in result.related_issues if i.state == "open"]

    if open_issues:
        console.print()
        console.print(
            f"[bold cyan]Found {len(open_issues)} open issue(s) with similar problems.[/bold cyan]"
        )
        for issue in open_issues[:3]:
            console.print(f"  [green]open[/green] {issue.url}")
            console.print(f"        {issue.title}")

        if Confirm.ask("\nDraft a reply to share your solution?", default=False):
            from octoscout.community.reply_suggester import ReplySuggester

            suggester = ReplySuggester(provider)
            target = open_issues[0]
            with Status("[bold cyan]Drafting reply...", console=console):
                draft = await suggester.draft_reply(target, result)

            console.print()
            console.print(Panel(
                f"[bold]Reply to:[/bold] {draft.issue_url}\n"
                f"[bold]Issue:[/bold] {draft.issue_title}\n\n"
                f"{draft.comment_body}",
                title="[cyan]Draft Reply[/cyan]",
                style="cyan",
            ))

            # Offer to post
            if config.github_token and Confirm.ask(
                "\n[bold yellow]Post this reply to GitHub?[/bold yellow]", default=False
            ):
                client = GitHubClient(token=config.github_token)
                try:
                    with Status("[bold cyan]Posting reply...", console=console):
                        resp = await client.post_comment(
                            target.repo, target.number, draft.comment_body
                        )
                    comment_url = resp.get("html_url", draft.issue_url)
                    console.print(f"\n[bold green]Reply posted![/bold green] {comment_url}")
                except Exception as e:
                    console.print(f"\n[red]Failed to post: {e}[/red]")
                    console.print(
                        f"[dim]You can manually post at: {draft.issue_url}[/dim]"
                    )
                finally:
                    await client.close()
            else:
                console.print(
                    f"\n[dim]You can manually post at: {draft.issue_url}[/dim]"
                )

    # If no solution was found, offer to draft a new issue
    has_solution = result.suggested_fix or (
        result.summary and any(
            kw in result.summary.lower()
            for kw in ["fix", "solution", "workaround", "upgrade", "downgrade", "replace"]
        )
    )

    if not has_solution:
        console.print()
        if Confirm.ask("No clear solution found. Draft a GitHub issue to report this?", default=False):
            from octoscout.community.issue_drafter import IssueDrafter

            env = None
            if auto_env:
                from octoscout.diagnosis.env_snapshot import collect_env_snapshot
                env = collect_env_snapshot()

            drafter = IssueDrafter(provider)
            with Status("[bold cyan]Drafting issue...", console=console):
                draft = await drafter.draft(result, traceback_text, env)

            console.print()
            console.print(Panel(
                f"[bold]Repository:[/bold] {draft.repo}\n"
                f"[bold]Title:[/bold] {draft.title}\n\n"
                f"{draft.body}",
                title="[cyan]Draft Issue[/cyan]",
                style="cyan",
            ))

            # Offer to create
            if config.github_token and Confirm.ask(
                "\n[bold yellow]Create this issue on GitHub?[/bold yellow]", default=False
            ):
                client = GitHubClient(token=config.github_token)
                try:
                    with Status("[bold cyan]Creating issue...", console=console):
                        resp = await client.create_issue(
                            draft.repo, draft.title, draft.body, draft.labels
                        )
                    issue_url = resp.get("html_url", "")
                    console.print(f"\n[bold green]Issue created![/bold green] {issue_url}")
                except Exception as e:
                    console.print(f"\n[red]Failed to create issue: {e}[/red]")
                    console.print(
                        f"[dim]Create manually at: https://github.com/{draft.repo}/issues/new[/dim]"
                    )
                finally:
                    await client.close()
            else:
                console.print(
                    f"\n[dim]Create this issue at: https://github.com/{draft.repo}/issues/new[/dim]"
                )

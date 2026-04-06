"""Bulk GitHub issue crawler for the Compatibility Matrix pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from octoscout.matrix.comment_scorer import format_scored_comments, score_comments
from octoscout.matrix.models import CrawlConfig, RawIssue
from octoscout.search.github_client import GitHubClient
from octoscout.search.version_filter import _VERSION_RE

_console = Console()

# Patterns that indicate an issue likely contains useful compatibility info
_ERROR_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)|Error:|Exception:|FAILED|CUDA error",
    re.IGNORECASE,
)


@dataclass
class CrawlStats:
    """Statistics from a single repo crawl."""

    repo: str
    total_fetched: int = 0
    passed_filter: int = 0
    already_existed: int = 0
    pages_fetched: int = 0


def _passes_prefilter(title: str, body: str, comments: str) -> bool:
    """Check if an issue likely contains version/compatibility info.

    An issue passes if it mentions a version number OR contains an error pattern.
    This cuts volume by ~60% before any LLM calls.
    """
    text = f"{title}\n{body}\n{comments}"
    if _VERSION_RE.search(text):
        return True
    if _ERROR_PATTERNS.search(text):
        return True
    return False


def _repo_slug(repo: str) -> str:
    """Convert 'owner/name' to 'owner_name' for filenames."""
    return repo.replace("/", "_")


class MatrixCrawler:
    """Bulk-crawls GitHub issues and saves them as JSONL for later extraction."""

    def __init__(
        self,
        github_client: GitHubClient,
        output_dir: Path,
        fetch_comments: bool = False,
    ):
        self._github = github_client
        self._output_dir = output_dir
        self._raw_dir = output_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._fetch_comments = fetch_comments

    async def crawl_repo(self, config: CrawlConfig) -> CrawlStats:
        """Crawl a single repository's issues."""
        stats = CrawlStats(repo=config.repo)
        slug = _repo_slug(config.repo)
        output_path = self._raw_dir / f"{slug}.jsonl"

        # Load existing issue numbers for resume support
        existing_numbers: set[int] = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            existing_numbers.add(d["number"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            stats.already_existed = len(existing_numbers)

        labels_str = ",".join(config.labels) if config.labels else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task(
                f"Crawling {config.repo}",
                total=config.max_pages,
            )

            with open(output_path, "a", encoding="utf-8") as f:
                page = 1
                while page <= config.max_pages:
                    issues, has_next = await self._github.list_issues(
                        repo=config.repo,
                        state=config.state,
                        labels=labels_str,
                        per_page=100,
                        page=page,
                    )

                    if not issues:
                        break

                    stats.pages_fetched += 1

                    for item in issues:
                        # GitHub Issues API also returns PRs — skip them
                        if "pull_request" in item:
                            continue

                        number = item["number"]
                        stats.total_fetched += 1

                        if number in existing_numbers:
                            continue

                        title = item.get("title", "")
                        body = item.get("body", "") or ""

                        # Optionally fetch comments (slow — adds 1 API call per issue)
                        comments_text = ""
                        if self._fetch_comments:
                            try:
                                comments = await self._github.get_issue_comments(
                                    config.repo, number, per_page=5
                                )
                                comments_text = "\n---\n".join(
                                    c.get("body", "") for c in comments if c.get("body")
                                )
                            except Exception:
                                pass

                        # Apply keyword filter if configured
                        if config.keywords:
                            combined = f"{title}\n{body}\n{comments_text}".lower()
                            if not any(kw.lower() in combined for kw in config.keywords):
                                continue

                        # Apply pre-filter
                        if not _passes_prefilter(title, body, comments_text):
                            continue

                        stats.passed_filter += 1

                        # Extract comment count and reactions from issue metadata
                        comment_count = item.get("comments", 0)
                        reactions = item.get("reactions", {})
                        issue_reactions = reactions.get("total_count", 0) if isinstance(reactions, dict) else 0

                        raw_issue = RawIssue(
                            number=number,
                            repo=config.repo,
                            title=title,
                            body=body,
                            state=item.get("state", "closed"),
                            created_at=item.get("created_at", ""),
                            updated_at=item.get("updated_at", ""),
                            labels=[
                                lbl["name"] for lbl in item.get("labels", [])
                                if isinstance(lbl, dict)
                            ],
                            comments_text=comments_text,
                            comment_count=comment_count,
                            issue_reactions=issue_reactions,
                        )

                        f.write(json.dumps(raw_issue.to_dict(), ensure_ascii=False) + "\n")
                        existing_numbers.add(number)

                    progress.update(task, advance=1)

                    if not has_next:
                        break
                    page += 1

        return stats

    async def crawl_all(self, configs: list[CrawlConfig]) -> dict[str, CrawlStats]:
        """Crawl multiple repositories sequentially."""
        results: dict[str, CrawlStats] = {}
        for config in configs:
            _console.print(f"\n[bold]Crawling {config.repo}...[/bold]")
            stats = await self.crawl_repo(config)
            results[config.repo] = stats
            _console.print(
                f"  Fetched {stats.total_fetched} issues, "
                f"{stats.passed_filter} passed filter, "
                f"{stats.already_existed} already existed"
            )
        return results

    async def enrich_comments(
        self,
        repo_slug: str | None = None,
        min_comments: int = 1,
        top_k: int = 8,
    ) -> dict[str, int]:
        """Selectively fetch and score comments for high-value issues.

        Only fetches comments for issues that:
        - Have >= min_comments comments (from the issue metadata)
        - Don't already have comments_text populated

        Updates the JSONL file in-place with scored top-k comments.

        Args:
            repo_slug: Specific repo slug, or None to process all.
            min_comments: Minimum comment count to bother fetching.
            top_k: Number of top-scored comments to keep per issue.

        Returns:
            Dict of repo_slug -> number of issues enriched.
        """
        if repo_slug:
            slugs = [repo_slug]
        else:
            slugs = [p.stem for p in self._raw_dir.glob("*.jsonl")]

        results: dict[str, int] = {}

        for slug in slugs:
            path = self._raw_dir / f"{slug}.jsonl"
            if not path.exists():
                continue

            # Load all issues
            issues: list[RawIssue] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            issues.append(RawIssue.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            pass

            # Filter to issues worth enriching:
            # - Not yet enriched (comments_enriched == False)
            # - Has enough comments to be worth fetching
            not_enriched = [i for i in issues if not i.comments_enriched]
            to_enrich = [
                i for i in not_enriched
                if i.comment_count >= min_comments
            ]
            # Mark low-comment issues as enriched so we don't recheck them
            skipped_low = 0
            for i in not_enriched:
                if i.comment_count < min_comments:
                    i.comments_enriched = True
                    skipped_low += 1

            if not to_enrich:
                _console.print(
                    f"[dim]{slug}: no issues to enrich "
                    f"({skipped_low} marked done with <{min_comments} comments)[/dim]"
                )
                results[slug] = 0
                # Still rewrite to save the comments_enriched flags
                if skipped_low > 0:
                    with open(path, "w", encoding="utf-8") as f:
                        for issue in issues:
                            f.write(json.dumps(issue.to_dict(), ensure_ascii=False) + "\n")
                continue

            _console.print(
                f"  {len(to_enrich)} issues to enrich "
                f"({skipped_low} marked done with <{min_comments} comments)"
            )

            enriched_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=_console,
            ) as progress:
                task = progress.add_task(
                    f"Enriching {slug}",
                    total=len(to_enrich),
                )

                for issue in to_enrich:
                    try:
                        raw_comments = await self._github.get_issue_comments_with_reactions(
                            issue.repo, issue.number, max_pages=3,
                        )

                        if raw_comments:
                            scored = score_comments(raw_comments, top_k=top_k)
                            text = format_scored_comments(scored)
                            if text:
                                issue.comments_text = text
                                enriched_count += 1

                        # Mark as enriched regardless — don't retry next time
                        issue.comments_enriched = True
                    except Exception:
                        pass  # Non-fatal; leave comments_enriched=False to retry later

                    progress.update(task, advance=1)

            # Rewrite the JSONL with enriched data
            with open(path, "w", encoding="utf-8") as f:
                for issue in issues:
                    f.write(json.dumps(issue.to_dict(), ensure_ascii=False) + "\n")

            results[slug] = enriched_count
            _console.print(f"  {slug}: enriched {enriched_count}/{len(to_enrich)} issues")

        return results

    async def patch_metadata(self, repo_slug: str | None = None) -> dict[str, int]:
        """Batch-update comment_count and issue_reactions for existing issues.

        Uses list_issues API (100 per page) instead of per-issue API calls.
        250 issues = 3 API calls instead of 250.
        """
        if repo_slug:
            slugs = [repo_slug]
        else:
            slugs = [p.stem for p in self._raw_dir.glob("*.jsonl")]

        results: dict[str, int] = {}

        for slug in slugs:
            path = self._raw_dir / f"{slug}.jsonl"
            if not path.exists():
                continue

            # Load existing issues
            issues: list[RawIssue] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            issues.append(RawIssue.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            pass

            if not issues:
                results[slug] = 0
                continue

            # Build lookup by issue number
            issue_map: dict[int, RawIssue] = {i.number: i for i in issues}
            repo = issues[0].repo

            # Fetch metadata in bulk via list_issues (100 per page)
            _console.print(f"  Fetching metadata for {slug}...")
            patched = 0
            page = 1
            max_pages = (len(issues) // 100) + 5  # a few extra pages for safety

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=_console,
            ) as progress:
                task = progress.add_task(f"Patching {slug}", total=len(issues))

                while page <= max_pages:
                    try:
                        api_issues, has_next = await self._github.list_issues(
                            repo=repo, state="closed", per_page=100, page=page,
                        )
                    except Exception:
                        break

                    if not api_issues:
                        break

                    for item in api_issues:
                        if "pull_request" in item:
                            continue
                        number = item.get("number", 0)
                        if number not in issue_map:
                            continue

                        issue = issue_map[number]
                        old_count = issue.comment_count
                        issue.comment_count = item.get("comments", 0)
                        reactions = item.get("reactions", {})
                        issue.issue_reactions = (
                            reactions.get("total_count", 0)
                            if isinstance(reactions, dict) else 0
                        )

                        # Reset comments_enriched if comment_count was wrong
                        if old_count == 0 and issue.comment_count > 0 and not issue.comments_text:
                            issue.comments_enriched = False

                        patched += 1
                        progress.update(task, advance=1)

                    if not has_next:
                        break
                    page += 1

            # Count unlocked
            reset_count = sum(
                1 for i in issues
                if not i.comments_enriched and i.comment_count > 0 and not i.comments_text
            )

            # Rewrite
            with open(path, "w", encoding="utf-8") as f:
                for issue in issues:
                    f.write(json.dumps(issue.to_dict(), ensure_ascii=False) + "\n")

            results[slug] = patched
            msg = f"  {slug}: patched {patched}/{len(issues)} issues"
            if reset_count:
                msg += f" ({reset_count} unlocked for re-enrichment)"
            _console.print(msg)

        return results


# ---------------------------------------------------------------------------
# Default crawl configurations per the project plan
# ---------------------------------------------------------------------------

DEFAULT_CRAWL_CONFIGS: list[CrawlConfig] = [
    # Closed issues counts as of 2026-03 (pages = ceil(count / 100))
    CrawlConfig(
        repo="huggingface/transformers",  # ~17,700 closed
        labels=["bug"],
        max_pages=180,
    ),
    CrawlConfig(
        repo="vllm-project/vllm",  # ~12,600 closed
        max_pages=130,
    ),
    CrawlConfig(
        repo="huggingface/peft",  # ~1,350 closed
        max_pages=15,
    ),
    CrawlConfig(
        repo="microsoft/DeepSpeed",  # ~5,000 closed
        labels=["bug"],
        max_pages=50,
    ),
    CrawlConfig(
        repo="QwenLM/Qwen2.5",  # ~1,500 closed
        max_pages=15,
    ),
    CrawlConfig(
        repo="pytorch/pytorch",  # ~41,500 closed (keyword-filtered)
        keywords=["CUDA", "compatibility", "version mismatch", "breaking change"],
        max_pages=50,
    ),
    CrawlConfig(
        repo="Dao-AILab/flash-attention",  # ~740 closed
        max_pages=10,
    ),
    CrawlConfig(
        repo="huggingface/trl",  # ~1,800 closed
        max_pages=20,
    ),
    CrawlConfig(
        repo="hiyouga/LLaMA-Factory",  # ~7,500 closed
        max_pages=80,
    ),
    CrawlConfig(
        repo="QwenLM/Qwen3-VL",  # ~1,500 closed
        max_pages=20,
    ),
]

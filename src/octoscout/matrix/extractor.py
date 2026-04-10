"""LLM-driven structured extraction from raw GitHub issues."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from octoscout.matrix.models import ExtractedIssueInfo, RawIssue
from octoscout.models import Message, Role
from octoscout.prompts import load_prompt
from octoscout.providers.base import LLMProvider

_console = Console()

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = load_prompt("extraction_system")
EXTRACTION_USER_TEMPLATE = load_prompt("extraction_user")


@dataclass
class ExtractStats:
    """Statistics from extraction of a single repo."""

    repo: str
    total: int = 0
    extracted: int = 0
    failed: int = 0
    skipped: int = 0


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def parse_llm_json(text: str) -> dict | None:
    """Try to parse JSON from LLM output using multiple strategies."""
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from fenced code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Repair truncated JSON — try closing open braces/brackets
    if start != -1:
        fragment = text[start:]
        repaired = _repair_truncated_json(fragment)
        if repaired:
            return repaired

    return None


def _repair_truncated_json(fragment: str) -> dict | None:
    """Attempt to repair a truncated JSON object by closing open structures."""
    # Strip trailing comma and whitespace
    s = fragment.rstrip()
    if s.endswith(","):
        s = s[:-1]

    # Count open braces/brackets
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    # Close them
    s += "]" * max(open_brackets, 0)
    s += "}" * max(open_braces, 0)

    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try stripping the last incomplete key-value pair
    # e.g. `..."foo": "bar`, `..."count": 12` (missing closing quote or comma)
    last_comma = s.rfind(",")
    if last_comma > 0:
        trimmed = s[:last_comma]
        trimmed += "]" * max(trimmed.count("[") - trimmed.count("]"), 0)
        trimmed += "}" * max(trimmed.count("{") - trimmed.count("}"), 0)
        try:
            result = json.loads(trimmed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "_")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MatrixExtractor:
    """Extracts structured compatibility info from raw issues using LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        input_dir: Path,
        output_dir: Path,
        concurrency: int = 5,
        log_errors: bool = False,
    ):
        self._provider = provider
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._concurrency = concurrency
        self._log_errors = log_errors
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_repo(self, repo_slug: str) -> ExtractStats:
        """Extract all raw issues for a repo slug (e.g. 'huggingface_transformers')."""
        raw_path = self._input_dir / "raw" / f"{repo_slug}.jsonl"
        extracted_path = self._output_dir / f"{repo_slug}.jsonl"

        if not raw_path.exists():
            _console.print(f"[yellow]No raw data for {repo_slug}[/yellow]")
            return ExtractStats(repo=repo_slug)

        # Load raw issues
        raw_issues: list[RawIssue] = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_issues.append(RawIssue.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Load already-extracted issue IDs for resume
        existing_ids: set[str] = set()
        if extracted_path.exists():
            with open(extracted_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            existing_ids.add(d.get("issue_id", ""))
                        except json.JSONDecodeError:
                            pass

        stats = ExtractStats(repo=repo_slug, total=len(raw_issues))
        to_extract = [
            r for r in raw_issues
            if f"{r.repo}#{r.number}" not in existing_ids
        ]
        stats.skipped = len(raw_issues) - len(to_extract)

        if not to_extract:
            _console.print(f"[dim]All {stats.total} issues already extracted.[/dim]")
            return stats

        sem = asyncio.Semaphore(self._concurrency)

        async def _extract_with_sem(raw: RawIssue) -> ExtractedIssueInfo | None:
            async with sem:
                return await self.extract_issue(raw)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task(
                f"Extracting {repo_slug}",
                total=len(to_extract),
            )

            # Process in batches to write results incrementally
            batch_size = self._concurrency * 2
            with open(extracted_path, "a", encoding="utf-8") as f:
                for i in range(0, len(to_extract), batch_size):
                    batch = to_extract[i:i + batch_size]
                    results = await asyncio.gather(
                        *[_extract_with_sem(r) for r in batch],
                        return_exceptions=True,
                    )

                    for idx, result in enumerate(results):
                        if isinstance(result, Exception):
                            stats.failed += 1
                            if self._log_errors:
                                issue = batch[idx] if idx < len(batch) else None
                                label = f"{issue.repo}#{issue.number}" if issue else "?"
                                _console.print(f"[red]Exception for {label}: {result}[/red]")
                        elif result is not None:
                            stats.extracted += 1
                            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                        else:
                            stats.failed += 1

                    progress.update(task, advance=len(batch))
                    f.flush()

        return stats

    async def extract_issue(self, raw: RawIssue) -> ExtractedIssueInfo | None:
        """Extract structured info from a single raw issue via LLM."""
        # Truncate body and comments to keep prompt reasonable
        body = (raw.body or "")[:3000]
        comments = (raw.comments_text or "")[:4000]

        prompt = EXTRACTION_USER_TEMPLATE.format(
            repo=raw.repo,
            number=raw.number,
            title=raw.title,
            body=body,
            comments=comments,
        )

        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = await self._provider.chat(
                    [Message(role=Role.USER, content=prompt)],
                    system=EXTRACTION_SYSTEM_PROMPT,
                )
                break
            except Exception as e:
                err_str = str(e)
                is_retryable = "overloaded" in err_str.lower() or "529" in err_str or "rate" in err_str.lower() or "connection" in err_str.lower()
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    await asyncio.sleep(wait)
                    continue
                if self._log_errors:
                    _console.print(f"[red]LLM error for {raw.repo}#{raw.number}: {e}[/red]")
                return None

        if response is None:
            return None

        parsed = parse_llm_json(response.text or "")
        if parsed is None:
            if self._log_errors:
                preview = (response.text or "")[:200]
                _console.print(
                    f"[yellow]JSON parse failed for {raw.repo}#{raw.number}: {preview}[/yellow]"
                )
            return None

        # Normalize common LLM mistakes
        parsed = _normalize_parsed(parsed)

        issue_id = f"{raw.repo}#{raw.number}"

        return ExtractedIssueInfo(
            issue_id=issue_id,
            title=raw.title,
            reported_versions=parsed.get("reported_versions", {}),
            python_version=parsed.get("python_version"),
            cuda_version=parsed.get("cuda_version"),
            problem_type=parsed.get("problem_type", "other"),
            error_type=parsed.get("error_type"),
            error_message_summary=parsed.get("error_message_summary", ""),
            has_solution=bool(parsed.get("has_solution", False)),
            solution_type=parsed.get("solution_type", "none"),
            solution_detail=parsed.get("solution_detail"),
            fix_version=parsed.get("fix_version"),
            affected_version_range=parsed.get("affected_version_range"),
            related_issues=parsed.get("related_issues", []),
        )

    async def extract_all(self, repo_slugs: list[str] | None = None) -> dict[str, ExtractStats]:
        """Extract all repos. If repo_slugs is None, discover from raw/ dir."""
        if repo_slugs is None:
            raw_dir = self._input_dir / "raw"
            if not raw_dir.exists():
                return {}
            repo_slugs = [
                p.stem for p in raw_dir.glob("*.jsonl")
            ]

        results: dict[str, ExtractStats] = {}
        for slug in repo_slugs:
            _console.print(f"\n[bold]Extracting {slug}...[/bold]")
            stats = await self.extract_repo(slug)
            results[slug] = stats
            _console.print(
                f"  Total: {stats.total}, Extracted: {stats.extracted}, "
                f"Failed: {stats.failed}, Skipped: {stats.skipped}"
            )
        return results


_VALID_PROBLEM_TYPES = {"crash", "wrong_output", "performance", "install", "other"}
_VALID_SOLUTION_TYPES = {"version_change", "code_fix", "config_change", "workaround", "none"}


def _normalize_parsed(parsed: dict) -> dict:
    """Fix common LLM output mistakes: wrong key names, invalid enum values."""
    # Fix wrong key names (e.g. "released_versions" → "reported_versions")
    if "reported_versions" not in parsed:
        for wrong_key in ("released_versions", "versions", "package_versions"):
            if wrong_key in parsed and isinstance(parsed[wrong_key], dict):
                parsed["reported_versions"] = parsed.pop(wrong_key)
                break

    # Fix invalid problem_type values
    pt = parsed.get("problem_type", "other")
    if pt not in _VALID_PROBLEM_TYPES:
        # Try to map common mistakes
        pt_lower = pt.lower()
        if "wrong" in pt_lower or "output" in pt_lower or "unexpected" in pt_lower:
            parsed["problem_type"] = "wrong_output"
        elif "crash" in pt_lower or "error" in pt_lower or "exception" in pt_lower:
            parsed["problem_type"] = "crash"
        elif "install" in pt_lower or "build" in pt_lower:
            parsed["problem_type"] = "install"
        elif "perf" in pt_lower or "slow" in pt_lower or "memory" in pt_lower:
            parsed["problem_type"] = "performance"
        else:
            parsed["problem_type"] = "other"

    # Fix invalid solution_type values
    st = parsed.get("solution_type", "none")
    if st not in _VALID_SOLUTION_TYPES:
        st_lower = st.lower()
        if "version" in st_lower or "upgrade" in st_lower or "downgrade" in st_lower:
            parsed["solution_type"] = "version_change"
        elif "code" in st_lower or "api" in st_lower:
            parsed["solution_type"] = "code_fix"
        elif "config" in st_lower or "env" in st_lower:
            parsed["solution_type"] = "config_change"
        elif "workaround" in st_lower or "hack" in st_lower:
            parsed["solution_type"] = "workaround"
        else:
            parsed["solution_type"] = "none"

    return parsed

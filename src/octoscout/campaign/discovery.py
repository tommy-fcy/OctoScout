"""Discover hot open issues from covered repos for campaign diagnosis."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

from octoscout.campaign.models import CampaignIssue, append_jsonl
from octoscout.search.github_client import GitHubClient
from octoscout.search.version_filter import _VERSION_RE

# --- Prefilter patterns (reused from matrix crawler) ---

_ERROR_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)|Error:|Exception:|FAILED|CUDA error",
    re.IGNORECASE,
)

_EXCLUDE_LABELS = {"enhancement", "feature request", "feature", "question", "documentation", "docs"}

# --- Env category classification ---

_GPU_PATTERNS = re.compile(
    r"CUDA error|cuda\.is_available|torch\.cuda|nccl|flash_attn|triton"
    r"|\.half\(\)|\.to\(['\"]cuda|RuntimeError.*CUDA|CudaError|GPU"
    r"|torch\.distributed|\.bfloat16",
    re.IGNORECASE,
)

_MODEL_PATTERNS = re.compile(
    r"from_pretrained|AutoModel|AutoTokenizer|AutoProcessor|pipeline\(",
)

# --- Traceback extraction ---

_TB_HEADER_RE = re.compile(r"Traceback \(most recent call last\):")
_EXCEPTION_LINE_RE = re.compile(r"^[A-Z]\w*(\.\w+)*Error:", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py|bash|shell|text|console|)?\s*\n(.*?)```", re.DOTALL)

# --- Issue reference extraction ---

_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)"
)
_ISSUE_REF_RE = re.compile(
    r"(?:^|[\s(])([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)#(\d+)"
)
_SHORT_REF_RE = re.compile(
    r"(?:^|[\s(])#(\d+)(?=[\s),.]|$)"
)


def _passes_prefilter(title: str, body: str) -> bool:
    """Check if an issue likely contains version/compatibility info."""
    text = f"{title}\n{body}"
    return bool(_VERSION_RE.search(text) or _ERROR_PATTERNS.search(text))


def classify_env_category(text: str) -> str:
    """Classify an issue's environment requirements from its text."""
    if _GPU_PATTERNS.search(text):
        return "gpu_required"
    if _MODEL_PATTERNS.search(text):
        return "model_download"
    return "cpu_only"


def extract_traceback(text: str) -> str:
    """Extract the first Python traceback from issue text.

    Handles tracebacks inside markdown code fences and raw text.
    Preserves raw content (including rank prefixes) — cleaning is done
    downstream by the traceback parser.
    """
    # First try inside code fences
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1)
        tb_match = _TB_HEADER_RE.search(block)
        if tb_match:
            tb_text = block[tb_match.start():]
            return tb_text.strip()

    # Fallback: search raw text
    tb_match = _TB_HEADER_RE.search(text)
    if tb_match:
        remaining = text[tb_match.start():]
        lines = remaining.split("\n")
        tb_lines = []
        for line in lines:
            tb_lines.append(line)
            if _EXCEPTION_LINE_RE.match(line) and len(tb_lines) > 2:
                break
        return "\n".join(tb_lines).strip()

    return ""


def extract_code_snippet(text: str) -> str | None:
    """Extract a runnable Python code snippet from issue text.

    Looks for code blocks with import statements, excludes pip commands and logs.
    """
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1).strip()
        # Skip installation commands
        if block.startswith(("pip ", "pip3 ", "conda ", "$ pip")):
            continue
        # Skip obvious output logs
        if block.startswith(("Traceback", ">>>")):
            continue
        # Must have at least one import or function call to be runnable
        if re.search(r"^(?:import |from \w+ import )", block, re.MULTILINE):
            return block

    return None


def extract_issue_references(text: str, current_repo: str) -> list[str]:
    """Extract cross-issue references from text.

    Returns list of full references like 'owner/repo#123'.
    """
    refs = set()

    # Full URLs: https://github.com/owner/repo/issues/123
    for m in _ISSUE_URL_RE.finditer(text):
        refs.add(f"{m.group(1)}#{m.group(2)}")

    # Explicit refs: owner/repo#123
    for m in _ISSUE_REF_RE.finditer(text):
        refs.add(f"{m.group(1)}#{m.group(2)}")

    # Short refs: #123 (assume current repo)
    for m in _SHORT_REF_RE.finditer(text):
        refs.add(f"{current_repo}#{m.group(1)}")

    return sorted(refs)


def compute_discovery_score(
    comment_count: int,
    created_at: str,
    has_traceback: bool,
    max_age_days: int = 90,
) -> float:
    """Score an issue by engagement signals and diagnosability.

    Weights: recency(0.40) + comments(0.35) + has_traceback(0.25).
    Reactions intentionally excluded — high reactions may mean already solved.
    """
    # Recency: linear decay over max_age_days
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).days
        recency = max(0.0, 1.0 - age_days / max_age_days)
    except (ValueError, TypeError):
        recency = 0.5

    # Comments: log scale
    comment_score = min(1.0, math.log(comment_count + 1) / math.log(50))

    # Traceback bonus
    tb_score = 1.0 if has_traceback else 0.0

    return 0.40 * recency + 0.35 * comment_score + 0.25 * tb_score


async def discover_open_issues(
    client: GitHubClient,
    repos: list[str],
    campaign_dir: Path,
    max_pages: int = 5,
    min_comments: int = 2,
    max_age_days: int = 90,
) -> list[CampaignIssue]:
    """Crawl open issues, filter, rank, and save to discovered.jsonl.

    For each issue, also fetches comments to extract cross-issue references.
    """
    now = datetime.now(timezone.utc).isoformat()
    output_path = campaign_dir / "discovered.jsonl"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Load existing to avoid duplicates on resume
    existing_keys = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    import json
                    d = json.loads(line)
                    existing_keys.add((d["repo"], d["number"]))

    all_issues: list[CampaignIssue] = []

    for repo in repos:
        for page in range(1, max_pages + 1):
            raw_issues, has_next = await client.list_issues(
                repo, state="open", per_page=100, page=page,
            )

            for raw in raw_issues:
                # Skip pull requests (GitHub API includes them in issues endpoint)
                if "pull_request" in raw:
                    continue

                number = raw["number"]
                if (repo, number) in existing_keys:
                    continue

                title = raw.get("title", "")
                body = raw.get("body", "") or ""
                comment_count = raw.get("comments", 0)
                labels = [lb["name"] for lb in raw.get("labels", [])]

                # Filter: exclude non-bug labels
                if _EXCLUDE_LABELS & {lb.lower() for lb in labels}:
                    continue

                # Filter: minimum comments
                if comment_count < min_comments:
                    continue

                # Filter: body too short
                if len(body) < 100:
                    continue

                # Prefilter: must have version or error pattern
                if not _passes_prefilter(title, body):
                    continue

                # Age filter
                created_at = raw.get("created_at", "")
                try:
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - created).days
                    if age_days > max_age_days:
                        continue
                except (ValueError, TypeError):
                    pass

                # Fetch comments for cross-issue reference extraction
                comments_text = ""
                try:
                    comments = await client.get_issue_comments(repo, number, per_page=30)
                    comments_text = "\n".join(c.get("body", "") for c in comments)
                except Exception:
                    pass

                full_text = f"{body}\n{comments_text}"

                # Extract fields
                traceback = extract_traceback(full_text)
                code_snippet = extract_code_snippet(body)
                env_cat = classify_env_category(full_text)
                refs = extract_issue_references(full_text, repo)

                score = compute_discovery_score(
                    comment_count, created_at, bool(traceback), max_age_days,
                )

                issue = CampaignIssue(
                    repo=repo,
                    number=number,
                    title=title,
                    body=body,
                    url=raw.get("html_url", f"https://github.com/{repo}/issues/{number}"),
                    labels=labels,
                    comment_count=comment_count,
                    created_at=created_at,
                    updated_at=raw.get("updated_at", ""),
                    extracted_traceback=traceback,
                    extracted_code_snippet=code_snippet,
                    has_traceback=bool(traceback),
                    env_category=env_cat,
                    referenced_issues=refs,
                    discovery_score=score,
                    discovered_at=now,
                )

                append_jsonl(output_path, issue)
                existing_keys.add((repo, number))
                all_issues.append(issue)

            if not has_next:
                break

    # Sort by score descending
    all_issues.sort(key=lambda x: x.discovery_score, reverse=True)
    return all_issues

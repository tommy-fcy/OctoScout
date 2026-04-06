"""Tests for the matrix crawler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from octoscout.matrix.crawler import (
    MatrixCrawler,
    _passes_prefilter,
    _repo_slug,
)
from octoscout.matrix.models import CrawlConfig, RawIssue


# ---------------------------------------------------------------------------
# Pre-filter tests
# ---------------------------------------------------------------------------


def test_prefilter_passes_with_version():
    assert _passes_prefilter("Bug", "I'm using transformers==4.55.0", "")


def test_prefilter_passes_with_traceback():
    assert _passes_prefilter("Error", "Traceback (most recent call last):", "")


def test_prefilter_passes_with_error_pattern():
    assert _passes_prefilter("Issue", "CUDA error: out of memory", "")


def test_prefilter_rejects_no_signals():
    assert not _passes_prefilter("Feature request", "Please add dark mode", "")


def test_prefilter_checks_comments():
    assert _passes_prefilter("Bug", "something broke", "I'm on v4.55.0")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_raw_issue_round_trip():
    raw = RawIssue(
        number=42,
        repo="owner/name",
        title="Test issue",
        body="Some body text",
        state="closed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
        labels=["bug"],
        comments_text="A comment",
    )
    d = raw.to_dict()
    restored = RawIssue.from_dict(d)
    assert restored.number == 42
    assert restored.repo == "owner/name"
    assert restored.labels == ["bug"]
    assert restored.comments_text == "A comment"


def test_repo_slug():
    assert _repo_slug("huggingface/transformers") == "huggingface_transformers"


# ---------------------------------------------------------------------------
# Crawler tests (mocked GitHub API)
# ---------------------------------------------------------------------------


def _make_issue(number: int, title: str = "Bug", body: str = "v4.55.0 error") -> dict:
    """Create a mock GitHub issue API response."""
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": "closed",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "labels": [{"name": "bug"}],
        "html_url": f"https://github.com/owner/repo/issues/{number}",
    }


@pytest.mark.asyncio
async def test_crawler_basic_flow():
    """Test that the crawler fetches, filters, and saves issues."""
    mock_client = AsyncMock()

    # Page 1: 2 issues, page 2: empty (stop)
    mock_client.list_issues = AsyncMock(side_effect=[
        ([_make_issue(1, body="transformers==4.55 crash"), _make_issue(2, body="No version info here, just a question")], True),
        ([], False),
    ])
    mock_client.get_issue_comments = AsyncMock(return_value=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = MatrixCrawler(mock_client, Path(tmpdir))
        config = CrawlConfig(repo="owner/repo", max_pages=5)
        stats = await crawler.crawl_repo(config)

        assert stats.total_fetched == 2
        assert stats.passed_filter == 1  # Only issue #1 has a version
        assert stats.pages_fetched >= 1

        # Verify JSONL output
        output_path = Path(tmpdir) / "raw" / "owner_repo.jsonl"
        assert output_path.exists()
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["number"] == 1


@pytest.mark.asyncio
async def test_crawler_resume_skips_existing():
    """Test that the crawler skips already-crawled issues."""
    mock_client = AsyncMock()
    mock_client.list_issues = AsyncMock(return_value=(
        [_make_issue(1, body="v1.0 error"), _make_issue(2, body="v2.0 error")],
        False,
    ))
    mock_client.get_issue_comments = AsyncMock(return_value=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir(parents=True)
        # Pre-populate with issue #1
        output_path = raw_dir / "owner_repo.jsonl"
        with open(output_path, "w") as f:
            f.write(json.dumps({"number": 1, "repo": "owner/repo", "title": "old", "body": ""}) + "\n")

        crawler = MatrixCrawler(mock_client, Path(tmpdir))
        config = CrawlConfig(repo="owner/repo", max_pages=1)
        stats = await crawler.crawl_repo(config)

        assert stats.already_existed == 1
        # Issue #2 should be newly added
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2  # 1 existing + 1 new


@pytest.mark.asyncio
async def test_crawler_skips_pull_requests():
    """Issues API returns PRs too — crawler should skip them."""
    mock_client = AsyncMock()
    pr_item = _make_issue(10, body="v1.0 error")
    pr_item["pull_request"] = {"url": "..."}

    mock_client.list_issues = AsyncMock(return_value=(
        [pr_item, _make_issue(11, body="v1.0 error")],
        False,
    ))
    mock_client.get_issue_comments = AsyncMock(return_value=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = MatrixCrawler(mock_client, Path(tmpdir))
        config = CrawlConfig(repo="owner/repo", max_pages=1)
        stats = await crawler.crawl_repo(config)

        assert stats.total_fetched == 1  # PR skipped
        assert stats.passed_filter == 1

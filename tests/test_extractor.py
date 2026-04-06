"""Tests for the matrix extractor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from octoscout.matrix.extractor import (
    MatrixExtractor,
    parse_llm_json,
)
from octoscout.matrix.models import ExtractedIssueInfo, RawIssue
from octoscout.models import AgentResponse


# ---------------------------------------------------------------------------
# parse_llm_json tests
# ---------------------------------------------------------------------------


def test_parse_raw_json():
    text = '{"reported_versions": {"torch": "2.3.0"}, "problem_type": "crash"}'
    result = parse_llm_json(text)
    assert result is not None
    assert result["problem_type"] == "crash"


def test_parse_fenced_json():
    text = 'Here is the result:\n```json\n{"has_solution": true}\n```\n'
    result = parse_llm_json(text)
    assert result is not None
    assert result["has_solution"] is True


def test_parse_json_in_prose():
    text = 'The extracted info is: {"error_type": "TypeError"} as shown above.'
    result = parse_llm_json(text)
    assert result is not None
    assert result["error_type"] == "TypeError"


def test_parse_malformed_returns_none():
    assert parse_llm_json("This is not JSON at all") is None
    assert parse_llm_json("") is None
    assert parse_llm_json("{{broken}") is None


# ---------------------------------------------------------------------------
# ExtractedIssueInfo round-trip
# ---------------------------------------------------------------------------


def test_extracted_issue_round_trip():
    info = ExtractedIssueInfo(
        issue_id="owner/repo#42",
        title="Test issue",
        reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
        python_version="3.12.0",
        cuda_version="12.1",
        problem_type="crash",
        error_type="TypeError",
        error_message_summary="unexpected keyword argument",
        has_solution=True,
        solution_type="version_change",
        solution_detail="downgrade to 4.52.3",
        fix_version="4.56.0",
        affected_version_range=">=4.53,<4.56",
        related_issues=["owner/repo#40"],
    )
    d = info.to_dict()
    restored = ExtractedIssueInfo.from_dict(d)
    assert restored.issue_id == "owner/repo#42"
    assert restored.reported_versions == {"transformers": "4.55.0", "torch": "2.3.0"}
    assert restored.has_solution is True
    assert restored.affected_version_range == ">=4.53,<4.56"


def test_extracted_issue_from_partial_dict():
    """Handles missing optional fields gracefully."""
    d = {"issue_id": "repo#1", "title": "Minimal"}
    info = ExtractedIssueInfo.from_dict(d)
    assert info.issue_id == "repo#1"
    assert info.reported_versions == {}
    assert info.python_version is None
    assert info.has_solution is False
    assert info.solution_type == "none"


# ---------------------------------------------------------------------------
# MatrixExtractor tests (mocked LLM)
# ---------------------------------------------------------------------------


def _make_raw_issue(number: int) -> RawIssue:
    return RawIssue(
        number=number,
        repo="owner/repo",
        title=f"Issue #{number}",
        body="Using transformers==4.55.0 and torch==2.3.0, got TypeError",
        state="closed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
        labels=["bug"],
        comments_text="Fixed by downgrading to 4.52.3",
    )


_MOCK_LLM_RESPONSE = json.dumps({
    "reported_versions": {"transformers": "4.55.0", "torch": "2.3.0"},
    "python_version": "3.12.0",
    "cuda_version": None,
    "problem_type": "crash",
    "error_type": "TypeError",
    "error_message_summary": "unexpected keyword argument",
    "has_solution": True,
    "solution_type": "version_change",
    "solution_detail": "downgrade transformers to 4.52.3",
    "fix_version": None,
    "affected_version_range": None,
    "related_issues": [],
})


@pytest.mark.asyncio
async def test_extract_issue_success():
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(text=_MOCK_LLM_RESPONSE))

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = MatrixExtractor(
            provider=mock_provider,
            input_dir=Path(tmpdir),
            output_dir=Path(tmpdir) / "extracted",
        )
        raw = _make_raw_issue(1)
        result = await extractor.extract_issue(raw)

        assert result is not None
        assert result.issue_id == "owner/repo#1"
        assert result.reported_versions == {"transformers": "4.55.0", "torch": "2.3.0"}
        assert result.problem_type == "crash"
        assert result.has_solution is True


@pytest.mark.asyncio
async def test_extract_issue_bad_llm_output():
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(text="I don't know"))

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = MatrixExtractor(
            provider=mock_provider,
            input_dir=Path(tmpdir),
            output_dir=Path(tmpdir) / "extracted",
        )
        raw = _make_raw_issue(1)
        result = await extractor.extract_issue(raw)
        assert result is None


@pytest.mark.asyncio
async def test_extract_repo_with_resume():
    """Already-extracted issues should be skipped."""
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(text=_MOCK_LLM_RESPONSE))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir)
        raw_dir = input_dir / "raw"
        raw_dir.mkdir(parents=True)
        extracted_dir = input_dir / "extracted"
        extracted_dir.mkdir(parents=True)

        # Write 2 raw issues
        with open(raw_dir / "owner_repo.jsonl", "w") as f:
            f.write(json.dumps(_make_raw_issue(1).to_dict()) + "\n")
            f.write(json.dumps(_make_raw_issue(2).to_dict()) + "\n")

        # Pre-populate issue #1 as already extracted
        with open(extracted_dir / "owner_repo.jsonl", "w") as f:
            f.write(json.dumps({"issue_id": "owner/repo#1", "title": "old"}) + "\n")

        extractor = MatrixExtractor(
            provider=mock_provider,
            input_dir=input_dir,
            output_dir=extracted_dir,
        )
        stats = await extractor.extract_repo("owner_repo")

        assert stats.total == 2
        assert stats.skipped == 1
        assert stats.extracted == 1  # Only #2 was extracted
        assert mock_provider.chat.call_count == 1  # Only called once for #2


@pytest.mark.asyncio
async def test_extract_issue_llm_exception():
    """LLM call failure should return None, not crash."""
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(side_effect=Exception("API error"))

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = MatrixExtractor(
            provider=mock_provider,
            input_dir=Path(tmpdir),
            output_dir=Path(tmpdir) / "extracted",
        )
        raw = _make_raw_issue(1)
        result = await extractor.extract_issue(raw)
        assert result is None

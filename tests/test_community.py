"""Tests for community features: issue drafter and reply suggester."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from octoscout.community.issue_drafter import IssueDrafter
from octoscout.community.models import DraftIssue, DraftReply
from octoscout.community.reply_suggester import ReplySuggester
from octoscout.models import (
    AgentResponse,
    DiagnosisResult,
    EnvSnapshot,
    GitHubIssueRef,
    ProblemType,
    TriageResult,
)


def _make_diagnosis(
    has_solution: bool = True,
    open_issues: int = 1,
    closed_issues: int = 1,
) -> DiagnosisResult:
    """Create a test diagnosis result."""
    issues = []
    for i in range(open_issues):
        issues.append(GitHubIssueRef(
            repo="huggingface/transformers",
            number=100 + i,
            title=f"Open issue #{100 + i}",
            url=f"https://github.com/huggingface/transformers/issues/{100 + i}",
            state="open",
        ))
    for i in range(closed_issues):
        issues.append(GitHubIssueRef(
            repo="huggingface/transformers",
            number=200 + i,
            title=f"Closed issue #{200 + i}",
            url=f"https://github.com/huggingface/transformers/issues/{200 + i}",
            state="closed",
        ))

    return DiagnosisResult(
        triage=TriageResult.UPSTREAM_ISSUE,
        problem_type=ProblemType.API_SIGNATURE_CHANGE,
        summary="Trainer no longer accepts tokenizer param. Use processing_class instead.",
        suggested_fix="Replace tokenizer=tokenizer with processing_class=tokenizer" if has_solution else None,
        confidence=0.9,
        related_issues=issues,
    )


def _make_env() -> EnvSnapshot:
    return EnvSnapshot(
        python_version="3.12.0",
        os_info="Linux",
        installed_packages={"transformers": "5.0.0", "torch": "2.3.0"},
    )


# ---------------------------------------------------------------------------
# Reply Suggester
# ---------------------------------------------------------------------------


def test_find_replyable_issues_filters_open():
    diagnosis = _make_diagnosis(open_issues=2, closed_issues=3)
    suggester = ReplySuggester(AsyncMock())
    candidates = suggester.find_replyable_issues(diagnosis)
    assert len(candidates) == 2
    assert all(c.state == "open" for c in candidates)


def test_find_replyable_issues_empty_when_no_issues():
    diagnosis = _make_diagnosis(open_issues=0, closed_issues=0)
    diagnosis.related_issues = []
    suggester = ReplySuggester(AsyncMock())
    candidates = suggester.find_replyable_issues(diagnosis)
    assert candidates == []


@pytest.mark.asyncio
async def test_draft_reply():
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(
        text="I encountered the same issue. The fix is to use processing_class=tokenizer instead."
    ))

    suggester = ReplySuggester(mock_provider)
    issue = GitHubIssueRef(
        repo="huggingface/transformers",
        number=100,
        title="TypeError with Trainer",
        url="https://github.com/huggingface/transformers/issues/100",
        state="open",
    )
    diagnosis = _make_diagnosis()

    draft = await suggester.draft_reply(issue, diagnosis)

    assert isinstance(draft, DraftReply)
    assert "processing_class" in draft.comment_body
    assert draft.issue_url == issue.url


# ---------------------------------------------------------------------------
# Issue Drafter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_issue():
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(
        text="TITLE: TypeError in Trainer.__init__ with transformers 5.0\n\n"
             "BODY:\n## Environment\nPython 3.12, transformers 5.0\n\n"
             "## Description\nTrainer no longer accepts tokenizer param.\n"
    ))

    drafter = IssueDrafter(mock_provider)
    diagnosis = _make_diagnosis(has_solution=False)
    env = _make_env()

    draft = await drafter.draft(diagnosis, "Traceback...", env)

    assert isinstance(draft, DraftIssue)
    assert "TypeError" in draft.title
    assert draft.repo == "huggingface/transformers"
    assert "bug" in draft.labels


@pytest.mark.asyncio
async def test_draft_issue_infers_repo():
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=AgentResponse(
        text="TITLE: Bug\n\nBODY:\nSome body"
    ))

    drafter = IssueDrafter(mock_provider)
    diagnosis = _make_diagnosis()
    # related_issues all point to huggingface/transformers

    draft = await drafter.draft(diagnosis, "Traceback...")
    assert draft.repo == "huggingface/transformers"


def test_draft_issue_parse_fallback():
    """When LLM doesn't follow TITLE:/BODY: format, still produces a draft."""
    draft = IssueDrafter._parse_draft(
        "This is just plain text about a bug",
        "some/repo",
    )
    assert draft.title  # Should extract something
    assert draft.repo == "some/repo"

"""Data models for community interaction features."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DraftIssue:
    """A drafted GitHub issue ready for the user to review and submit."""

    title: str
    body: str
    repo: str  # Target repo, e.g. "huggingface/transformers"
    labels: list[str] = field(default_factory=list)


@dataclass
class DraftReply:
    """A drafted reply to an existing GitHub issue."""

    issue_url: str
    issue_title: str
    comment_body: str

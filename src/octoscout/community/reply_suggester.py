"""Suggest replies to open GitHub issues when the user's problem is solved."""

from __future__ import annotations

from octoscout.community.models import DraftReply
from octoscout.models import DiagnosisResult, GitHubIssueRef, Message, Role
from octoscout.prompts import load_prompt
from octoscout.providers.base import LLMProvider

_SYSTEM_PROMPT = load_prompt("reply_system")
_REPLY_TEMPLATE = load_prompt("reply_template")


class ReplySuggester:
    """Finds replyable issues and drafts community replies."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def find_replyable_issues(
        self, diagnosis: DiagnosisResult,
    ) -> list[GitHubIssueRef]:
        """Find open issues from the diagnosis that could benefit from a reply.

        Criteria:
        - Issue must be open
        - Diagnosis must have a solution (summary contains fix info)
        """
        if not diagnosis.related_issues:
            return []

        # Only suggest replying to open issues
        candidates = [
            issue for issue in diagnosis.related_issues
            if issue.state == "open"
        ]

        return candidates

    async def draft_reply(
        self,
        issue: GitHubIssueRef,
        diagnosis: DiagnosisResult,
    ) -> DraftReply:
        """Draft a reply to an open issue based on the diagnosis.

        Args:
            issue: The open issue to reply to.
            diagnosis: The completed diagnosis with solution info.

        Returns:
            A DraftReply with the issue URL and comment body.
        """
        prompt = _REPLY_TEMPLATE.format(
            issue_title=issue.title,
            issue_url=issue.url,
            issue_state=issue.state,
            summary=diagnosis.summary[:2000],
        )

        response = await self._provider.chat(
            [Message(role=Role.USER, content=prompt)],
            system=_SYSTEM_PROMPT,
        )

        comment_body = (response.text or "").strip()

        return DraftReply(
            issue_url=issue.url,
            issue_title=issue.title,
            comment_body=comment_body,
        )

"""Auto-draft high-quality GitHub issues from diagnosis results."""

from __future__ import annotations

from octoscout.community.models import DraftIssue
from octoscout.models import DiagnosisResult, EnvSnapshot, Message, Role
from octoscout.prompts import load_prompt
from octoscout.providers.base import LLMProvider

_SYSTEM_PROMPT = load_prompt("issue_draft_system")
_DRAFT_TEMPLATE = load_prompt("issue_draft_template")


class IssueDrafter:
    """Generates draft GitHub issues from diagnosis results."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def draft(
        self,
        diagnosis: DiagnosisResult,
        traceback_text: str,
        env: EnvSnapshot | None = None,
        target_repo: str | None = None,
    ) -> DraftIssue:
        """Draft a GitHub issue from a diagnosis result.

        Args:
            diagnosis: The completed diagnosis.
            traceback_text: The original traceback text.
            env: Environment snapshot (optional).
            target_repo: Target repo. If None, inferred from related issues.

        Returns:
            A DraftIssue ready for the user to review.
        """
        repo = target_repo or self._infer_repo(diagnosis)
        env_info = env.format_for_llm() if env else "Not available"

        related_str = "None found"
        if diagnosis.related_issues:
            lines = []
            for issue in diagnosis.related_issues[:5]:
                lines.append(f"- {issue.url} — {issue.title} ({issue.state})")
            related_str = "\n".join(lines)

        prompt = _DRAFT_TEMPLATE.format(
            repo=repo,
            summary=diagnosis.summary[:2000],
            env_info=env_info,
            traceback=traceback_text[:3000],
            related_issues=related_str,
        )

        response = await self._provider.chat(
            [Message(role=Role.USER, content=prompt)],
            system=_SYSTEM_PROMPT,
        )

        return self._parse_draft(response.text or "", repo)

    @staticmethod
    def _infer_repo(diagnosis: DiagnosisResult) -> str:
        """Infer the target repository from the diagnosis."""
        if diagnosis.related_issues:
            # Use the most common repo from related issues
            repo_counts: dict[str, int] = {}
            for issue in diagnosis.related_issues:
                repo_counts[issue.repo] = repo_counts.get(issue.repo, 0) + 1
            return max(repo_counts, key=repo_counts.get)  # type: ignore[arg-type]
        return "unknown/repo"

    @staticmethod
    def _parse_draft(text: str, repo: str) -> DraftIssue:
        """Parse the LLM output into a DraftIssue."""
        title = ""
        body = text

        # Try to extract TITLE: and BODY: sections
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("TITLE:"):
                title = line.split(":", 1)[1].strip()
                # Everything after TITLE line (skipping BODY: marker) is the body
                rest = "\n".join(lines[i + 1:])
                if rest.strip().upper().startswith("BODY:"):
                    body = rest.split("\n", 1)[1] if "\n" in rest else ""
                else:
                    body = rest
                break

        if not title:
            # Fallback: use first non-empty line as title
            for line in lines:
                line = line.strip().strip("#").strip()
                if line:
                    title = line[:100]
                    break

        return DraftIssue(
            title=title or "Bug report",
            body=body.strip(),
            repo=repo,
            labels=["bug"],
        )

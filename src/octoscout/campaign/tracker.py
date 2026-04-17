"""Track the effect of campaign replies on GitHub issues."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from octoscout.campaign.models import (
    ReplyRecord,
    TrackingSnapshot,
    append_jsonl,
)
from octoscout.search.github_client import GitHubClient

_POSITIVE_RE = re.compile(
    r"\bthank|works|solved|fixed|helped|great|awesome|resolved|perfect\b",
    re.IGNORECASE,
)


async def track_single_issue(
    reply: ReplyRecord,
    client: GitHubClient,
) -> TrackingSnapshot:
    """Check a replied issue for state changes, reactions, and responses."""
    now = datetime.now(timezone.utc).isoformat()

    try:
        issue_data = await client.get_issue(reply.repo, reply.number)
        comments = await client.get_issue_comments(reply.repo, reply.number, per_page=30)
    except Exception:
        return TrackingSnapshot(
            repo=reply.repo,
            number=reply.number,
            checked_at=now,
            issue_state="unknown",
        )

    # Find our comment and count new comments since
    our_comment_idx = -1
    reply_reactions: dict[str, int] = {}
    for i, comment in enumerate(comments):
        body = comment.get("body", "")
        if "OctoScout" in body and "AI-assisted ML compatibility" in body:
            our_comment_idx = i
            # Get reactions on our comment
            reactions = comment.get("reactions", {})
            for key in ["+1", "-1", "heart", "hooray", "laugh", "confused", "rocket", "eyes"]:
                count = reactions.get(key, 0)
                if count > 0:
                    reply_reactions[key] = count
            break

    new_comments = 0
    has_positive = False
    if our_comment_idx >= 0:
        new_comments = len(comments) - our_comment_idx - 1
        # Check if any subsequent comments are positive
        for comment in comments[our_comment_idx + 1:]:
            body = comment.get("body", "")
            if _POSITIVE_RE.search(body):
                has_positive = True
                break

    return TrackingSnapshot(
        repo=reply.repo,
        number=reply.number,
        checked_at=now,
        issue_state=issue_data.get("state", "unknown"),
        comment_count=issue_data.get("comments", 0),
        reaction_count=sum(
            issue_data.get("reactions", {}).get(k, 0)
            for k in ["+1", "-1", "heart", "hooray", "laugh", "confused", "rocket", "eyes"]
        ),
        reply_reactions=reply_reactions,
        new_comments_since_reply=new_comments,
        has_positive_response=has_positive,
    )


async def track_all_replied(
    replies: list[ReplyRecord],
    client: GitHubClient,
    campaign_dir: Path,
) -> list[TrackingSnapshot]:
    """Track all posted replies."""
    posted = [r for r in replies if r.posted]
    results = []

    for reply in posted:
        snapshot = await track_single_issue(reply, client)
        append_jsonl(campaign_dir / "tracking.jsonl", snapshot)
        results.append(snapshot)

    return results

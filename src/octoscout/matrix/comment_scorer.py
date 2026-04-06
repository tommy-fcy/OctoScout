"""Score and filter GitHub issue comments by value for compatibility extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredComment:
    """A GitHub comment with a computed value score."""

    author: str
    body: str
    score: int
    reactions_total: int
    thumbs_up: int


# Short comments that are pure noise — match after lowercase + strip punctuation
_ME_TOO_PHRASES = {
    "same problem here",
    "same problem",
    "same issue",
    "same issue here",
    "same here",
    "me too",
    "+1",
    "i have the same issue",
    "i have the same problem",
}

# Keywords that indicate technical value
_SOLUTION_KEYWORDS = [
    "fix", "fixed", "solved", "solution", "workaround", "resolved",
    "the issue is", "root cause", "this is because", "the problem is",
]

_TECHNICAL_KEYWORDS = [
    "version", "upgrade", "downgrade", "pip install", "padding",
    "float16", "float32", "bfloat16", "batch", "cuda", "torch",
    "transformers", "config", "parameter", "dtype",
]


def _normalize_for_matching(text: str) -> str:
    """Normalize text for 'me too' matching."""
    return text.strip().lower().rstrip("!.。？?").strip()


def _is_auto_reply(body: str) -> bool:
    """Detect auto-reply messages."""
    short = body.strip()
    if len(short) > 50:
        return False
    return "自动回复" in short or "auto-reply" in short.lower() or "auto reply" in short.lower()


def _is_bot(author: str) -> bool:
    return "[bot]" in author.lower()


def score_comments(
    comments: list[dict],
    top_k: int = 8,
) -> list[ScoredComment]:
    """Score and filter GitHub issue comments, returning the top-k most valuable.

    Scoring priorities:
    - Reactions (thumbs_up * 3, heart * 2, total_count)
    - Content length (substantial > 200 chars, has code > 500 chars)
    - Solution keywords (+3)
    - Technical keywords (+1)

    Filters out:
    - Bot comments
    - Auto-reply messages
    - Pure "me too" / "+1" comments (exact match only, to avoid false positives)

    Args:
        comments: Raw comment dicts from GitHub API (must include reactions).
        top_k: Number of top comments to return.

    Returns:
        List of ScoredComment, sorted by score descending, length <= top_k.
    """
    scored: list[ScoredComment] = []

    for c in comments:
        author = c.get("user", {}).get("login", "unknown")
        body = c.get("body", "")
        reactions = c.get("reactions", {})
        thumbs = reactions.get("+1", 0)
        heart = reactions.get("heart", 0)
        total = reactions.get("total_count", 0)

        # Hard filters — skip entirely
        if _is_bot(author):
            continue
        if _is_auto_reply(body):
            continue
        if _normalize_for_matching(body) in _ME_TOO_PHRASES:
            continue

        # Score
        score = 0

        # Reactions
        score += thumbs * 3
        score += heart * 2
        score += total

        # Content substance
        if len(body) > 200:
            score += 2
        if len(body) > 500:
            score += 2

        # Solution indicators
        body_lower = body.lower()
        if any(kw in body_lower for kw in _SOLUTION_KEYWORDS):
            score += 3

        # Technical content
        if any(kw in body_lower for kw in _TECHNICAL_KEYWORDS):
            score += 1

        # Code blocks (``` or indented code)
        if "```" in body:
            score += 2

        scored.append(ScoredComment(
            author=author,
            body=body,
            score=score,
            reactions_total=total,
            thumbs_up=thumbs,
        ))

    # Sort by score descending, then by length descending (prefer more content at same score)
    scored.sort(key=lambda c: (-c.score, -len(c.body)))
    return scored[:top_k]


def format_scored_comments(comments: list[ScoredComment]) -> str:
    """Format scored comments into a single text block for LLM extraction."""
    if not comments:
        return ""
    parts = []
    for c in comments:
        parts.append(f"@{c.author} [reactions: {c.reactions_total}]:\n{c.body}")
    return "\n---\n".join(parts)

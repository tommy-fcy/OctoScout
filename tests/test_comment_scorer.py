"""Tests for comment scoring and filtering."""

from octoscout.matrix.comment_scorer import (
    ScoredComment,
    format_scored_comments,
    score_comments,
)


def _make_comment(
    author: str = "user",
    body: str = "A comment",
    thumbs_up: int = 0,
    heart: int = 0,
    total: int = 0,
) -> dict:
    return {
        "user": {"login": author},
        "body": body,
        "reactions": {"+1": thumbs_up, "heart": heart, "total_count": total or thumbs_up + heart},
    }


def test_filters_bot_comments():
    comments = [
        _make_comment("github-actions[bot]", "Auto-closed due to inactivity"),
        _make_comment("user1", "This is a real comment with some details about the version"),
    ]
    scored = score_comments(comments, top_k=5)
    assert len(scored) == 1
    assert scored[0].author == "user1"


def test_filters_auto_reply():
    comments = [
        _make_comment("hlchen23", "您的来件已收到，谢谢！（自动回复）"),
        _make_comment("user1", "Here is a fix: use padding_side='left'"),
    ]
    scored = score_comments(comments, top_k=5)
    assert len(scored) == 1
    assert scored[0].author == "user1"


def test_filters_me_too():
    comments = [
        _make_comment("user1", "Same problem here!"),
        _make_comment("user2", "same issue"),
        _make_comment("user3", "+1"),
        _make_comment("user4", "me too"),
        _make_comment("user5", "I found the fix: downgrade transformers to 4.52"),
    ]
    scored = score_comments(comments, top_k=5)
    assert len(scored) == 1
    assert scored[0].author == "user5"


def test_does_not_filter_short_but_useful():
    """Short comments with technical content should NOT be filtered."""
    comments = [
        _make_comment("user1", "Same problem here!"),
        _make_comment("user2", "Fixed by setting padding_side='left'"),
    ]
    scored = score_comments(comments, top_k=5)
    # user2's comment is short but NOT a "me too" phrase, should be kept
    assert any(c.author == "user2" for c in scored)


def test_reactions_boost_score():
    comments = [
        _make_comment("user1", "Long detailed comment with fix " * 10, thumbs_up=0),
        _make_comment("user2", "Short but popular fix for the version issue", thumbs_up=10, total=15),
    ]
    scored = score_comments(comments, top_k=5)
    # user2 should rank higher due to reactions
    assert scored[0].author == "user2"


def test_solution_keywords_boost():
    comments = [
        _make_comment("user1", "I also see this error with my setup using batch processing"),
        _make_comment("user2", "I solved this by downgrading transformers to 4.52.3"),
    ]
    scored = score_comments(comments, top_k=5)
    assert scored[0].author == "user2"


def test_code_blocks_boost():
    comments = [
        _make_comment("user1", "Something is wrong"),
        _make_comment("user2", "Here is the fix:\n```python\nprocessor.padding_side = 'left'\n```\nThis works!"),
    ]
    scored = score_comments(comments, top_k=5)
    assert scored[0].author == "user2"


def test_top_k_limits_output():
    comments = [_make_comment(f"user{i}", f"Comment number {i} with some text") for i in range(20)]
    scored = score_comments(comments, top_k=3)
    assert len(scored) == 3


def test_format_scored_comments():
    comments = [
        ScoredComment(author="user1", body="Fix: use left padding", score=10, reactions_total=5, thumbs_up=3),
        ScoredComment(author="user2", body="Confirmed working", score=5, reactions_total=2, thumbs_up=1),
    ]
    text = format_scored_comments(comments)
    assert "@user1" in text
    assert "@user2" in text
    assert "---" in text
    assert "reactions: 5" in text


def test_format_empty():
    assert format_scored_comments([]) == ""


def test_preserves_non_english_content():
    """Chinese/mixed language comments should not be incorrectly filtered."""
    comments = [
        _make_comment("user1", "大概率是因为generate代码的原因：processor的问题，可以参照下面这段代码修复"),
    ]
    scored = score_comments(comments, top_k=5)
    assert len(scored) == 1

"""Tests for GitHub client (unit tests with no real API calls)."""

from octoscout.search.github_client import GitHubClient


def test_parse_search_results():
    """Test that search result parsing works correctly."""
    data = {
        "total_count": 2,
        "items": [
            {
                "number": 123,
                "title": "TypeError with trust_remote_code",
                "html_url": "https://github.com/huggingface/transformers/issues/123",
                "state": "open",
                "body": "I got a TypeError when using trust_remote_code=True",
                "repository_url": "https://api.github.com/repos/huggingface/transformers",
            },
            {
                "number": 456,
                "title": "Another issue",
                "html_url": "https://github.com/huggingface/transformers/issues/456",
                "state": "closed",
                "body": None,
                "repository_url": "https://api.github.com/repos/huggingface/transformers",
            },
        ],
    }

    results = GitHubClient._parse_search_results(data, None)
    assert len(results) == 2
    assert results[0].number == 123
    assert results[0].state == "open"
    assert results[0].repo == "huggingface/transformers"
    assert results[1].snippet == ""  # body was None


def test_search_result_snippet_truncation():
    """Test that long bodies are truncated in snippets."""
    data = {
        "items": [
            {
                "number": 1,
                "title": "Test",
                "html_url": "https://github.com/test/test/issues/1",
                "state": "open",
                "body": "x" * 500,
                "repository_url": "https://api.github.com/repos/test/test",
            },
        ],
    }

    results = GitHubClient._parse_search_results(data, None)
    assert len(results[0].snippet) == 300

"""GitHub API client with authentication, rate-limit handling, and caching."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from octoscout.models import GitHubIssueRef

_GITHUB_API = "https://api.github.com"
_SEARCH_RATE_LIMIT_WINDOW = 60  # seconds
_SEARCH_RATE_LIMIT_MAX = 30  # requests per window


@dataclass
class RateLimitState:
    """Track API rate limit usage."""

    remaining: int = 5000
    reset_at: float = 0.0
    search_timestamps: list[float] = field(default_factory=list)


class GitHubClient:
    """Async GitHub API client with caching and rate-limit awareness."""

    def __init__(self, token: str | None = None):
        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=_GITHUB_API,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        self._rate = RateLimitState()
        self._cache: dict[str, dict] = {}

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_issues(
        self,
        query: str,
        repo: str | None = None,
        state: str | None = None,
        per_page: int = 10,
    ) -> list[GitHubIssueRef]:
        """Search GitHub issues using the Search API.

        Args:
            query: Search query text.
            repo: Optional repo filter (e.g. "huggingface/transformers").
            state: Filter by state ("open", "closed").
            per_page: Max results to return.
        """
        q_parts = [query]
        if repo:
            q_parts.append(f"repo:{repo}")
        if state:
            q_parts.append(f"state:{state}")
        q = " ".join(q_parts)

        await self._wait_for_search_rate_limit()

        cache_key = f"search:{q}:{per_page}"
        if cache_key in self._cache:
            return self._parse_search_results(self._cache[cache_key], repo)

        resp = await self._client.get(
            "/search/issues",
            params={"q": q, "per_page": per_page, "sort": "relevance"},
        )
        self._update_rate_limit(resp)
        self._record_search_call()
        resp.raise_for_status()

        data = resp.json()
        self._cache[cache_key] = data
        return self._parse_search_results(data, repo)

    async def get_issue(self, repo: str, number: int) -> dict:
        """Get full issue details including body."""
        cache_key = f"issue:{repo}:{number}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        resp = await self._client.get(f"/repos/{repo}/issues/{number}")
        self._update_rate_limit(resp)
        resp.raise_for_status()

        data = resp.json()
        self._cache[cache_key] = data
        return data

    async def get_issue_comments(
        self, repo: str, number: int, per_page: int = 10
    ) -> list[dict]:
        """Get comments on an issue."""
        cache_key = f"comments:{repo}:{number}:{per_page}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        resp = await self._client.get(
            f"/repos/{repo}/issues/{number}/comments",
            params={"per_page": per_page},
        )
        self._update_rate_limit(resp)
        resp.raise_for_status()

        data = resp.json()
        self._cache[cache_key] = data
        return data

    async def get_issue_comments_with_reactions(
        self, repo: str, number: int, max_pages: int = 3,
    ) -> list[dict]:
        """Get all comments with reaction data (for comment scoring).

        Uses the reactions preview header to include reaction counts per comment.
        Paginates up to max_pages (30 comments per page).
        """
        all_comments: list[dict] = []
        for page in range(1, max_pages + 1):
            resp = await self._client.get(
                f"/repos/{repo}/issues/{number}/comments",
                params={"per_page": 30, "page": page},
                headers={"Accept": "application/vnd.github.squirrel-girl-preview+json"},
            )
            self._update_rate_limit(resp)
            resp.raise_for_status()

            data = resp.json()
            if not data:
                break
            all_comments.extend(data)

            link_header = resp.headers.get("Link", "")
            if 'rel="next"' not in link_header:
                break

        return all_comments

    async def list_issues(
        self,
        repo: str,
        state: str = "closed",
        labels: str | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> tuple[list[dict], bool]:
        """List issues using the Issues API (for bulk crawling).

        Unlike search_issues(), this uses the general rate limit pool (5000/hr)
        and has no result cap.

        Returns:
            Tuple of (raw issue dicts, has_next_page).
        """
        import asyncio

        # Respect general rate limit
        if self._rate.remaining < 10:
            wait = max(0, self._rate.reset_at - time.time())
            if wait > 0:
                await asyncio.sleep(wait + 1)

        params: dict[str, str | int] = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        if labels:
            params["labels"] = labels

        resp = await self._client.get(f"/repos/{repo}/issues", params=params)
        self._update_rate_limit(resp)
        resp.raise_for_status()

        data = resp.json()

        # Check for next page via Link header
        link_header = resp.headers.get("Link", "")
        has_next = 'rel="next"' in link_header

        return data, has_next

    async def create_issue(
        self, repo: str, title: str, body: str, labels: list[str] | None = None,
    ) -> dict:
        """Create a new issue in a repository. Requires a token with repo scope."""
        payload: dict = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels

        resp = await self._client.post(f"/repos/{repo}/issues", json=payload)
        self._update_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    async def post_comment(self, repo: str, number: int, body: str) -> dict:
        """Post a comment on an existing issue. Requires a token with repo scope."""
        resp = await self._client.post(
            f"/repos/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        self._update_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    @property
    def rate_limit_remaining(self) -> int:
        return self._rate.remaining

    # ------------------------------------------------------------------
    # Rate limit helpers
    # ------------------------------------------------------------------

    def _update_rate_limit(self, resp: httpx.Response):
        """Update rate limit state from response headers."""
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._rate.remaining = int(remaining)
        if reset is not None:
            self._rate.reset_at = float(reset)

    def _record_search_call(self):
        self._rate.search_timestamps.append(time.time())

    async def _wait_for_search_rate_limit(self):
        """Respect the 30 req/min search rate limit."""
        now = time.time()
        cutoff = now - _SEARCH_RATE_LIMIT_WINDOW
        self._rate.search_timestamps = [
            t for t in self._rate.search_timestamps if t > cutoff
        ]

        if len(self._rate.search_timestamps) >= _SEARCH_RATE_LIMIT_MAX:
            wait_until = self._rate.search_timestamps[0] + _SEARCH_RATE_LIMIT_WINDOW
            delay = wait_until - now
            if delay > 0:
                import asyncio
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_results(data: dict, default_repo: str | None) -> list[GitHubIssueRef]:
        results: list[GitHubIssueRef] = []
        for item in data.get("items", []):
            # Extract repo from URL: https://api.github.com/repos/owner/name/issues/123
            repo = default_repo or ""
            if "repository_url" in item:
                repo = item["repository_url"].replace(f"{_GITHUB_API}/repos/", "")

            results.append(
                GitHubIssueRef(
                    repo=repo,
                    number=item["number"],
                    title=item["title"],
                    url=item["html_url"],
                    state=item["state"],
                    snippet=item.get("body", "")[:300] if item.get("body") else "",
                )
            )
        return results

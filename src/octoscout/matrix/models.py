"""Data models for the Compatibility Matrix pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Crawl configuration
# ---------------------------------------------------------------------------


@dataclass
class CrawlConfig:
    """Configuration for crawling a single repository."""

    repo: str  # "owner/name"
    labels: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_pages: int = 100
    state: str = "closed"


# ---------------------------------------------------------------------------
# Raw issue (before LLM extraction)
# ---------------------------------------------------------------------------


@dataclass
class RawIssue:
    """A GitHub issue fetched by the crawler, before LLM extraction."""

    number: int
    repo: str
    title: str
    body: str
    state: str
    created_at: str
    updated_at: str
    labels: list[str] = field(default_factory=list)
    comments_text: str = ""
    comments_enriched: bool = False  # True after enrich attempted (even if no useful comments)
    comment_count: int = 0
    issue_reactions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "repo": self.repo,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "labels": self.labels,
            "comments_text": self.comments_text,
            "comments_enriched": self.comments_enriched,
            "comment_count": self.comment_count,
            "issue_reactions": self.issue_reactions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawIssue:
        return cls(
            number=d["number"],
            repo=d["repo"],
            title=d["title"],
            body=d.get("body", ""),
            state=d.get("state", "closed"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            labels=d.get("labels", []),
            comments_text=d.get("comments_text", ""),
            comments_enriched=d.get("comments_enriched", False),
            comment_count=d.get("comment_count", 0),
            issue_reactions=d.get("issue_reactions", 0),
        )


# ---------------------------------------------------------------------------
# Extracted issue info (after LLM extraction)
# ---------------------------------------------------------------------------


@dataclass
class ExtractedIssueInfo:
    """Structured compatibility info extracted from a GitHub issue by LLM."""

    issue_id: str  # "owner/repo#number"
    title: str

    # Version information
    reported_versions: dict[str, str] = field(default_factory=dict)
    python_version: str | None = None
    cuda_version: str | None = None

    # Problem classification
    problem_type: str = "other"  # crash | wrong_output | performance | install | other
    error_type: str | None = None  # TypeError, AttributeError, ...
    error_message_summary: str = ""

    # Solution information
    has_solution: bool = False
    solution_type: str = "none"  # version_change | code_fix | config_change | workaround | none
    solution_detail: str | None = None
    fix_version: str | None = None

    # Scope
    affected_version_range: str | None = None  # e.g. ">= 4.53, < 4.56"
    related_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "reported_versions": self.reported_versions,
            "python_version": self.python_version,
            "cuda_version": self.cuda_version,
            "problem_type": self.problem_type,
            "error_type": self.error_type,
            "error_message_summary": self.error_message_summary,
            "has_solution": self.has_solution,
            "solution_type": self.solution_type,
            "solution_detail": self.solution_detail,
            "fix_version": self.fix_version,
            "affected_version_range": self.affected_version_range,
            "related_issues": self.related_issues,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractedIssueInfo:
        return cls(
            issue_id=d.get("issue_id", ""),
            title=d.get("title", ""),
            reported_versions=d.get("reported_versions", {}),
            python_version=d.get("python_version"),
            cuda_version=d.get("cuda_version"),
            problem_type=d.get("problem_type", "other"),
            error_type=d.get("error_type"),
            error_message_summary=d.get("error_message_summary", ""),
            has_solution=d.get("has_solution", False),
            solution_type=d.get("solution_type", "none"),
            solution_detail=d.get("solution_detail"),
            fix_version=d.get("fix_version"),
            affected_version_range=d.get("affected_version_range"),
            related_issues=d.get("related_issues", []),
        )


# ---------------------------------------------------------------------------
# Aggregated matrix types
# ---------------------------------------------------------------------------


@dataclass
class KnownProblem:
    """A single known problem for a version pair."""

    summary: str
    severity: str  # high | medium | low
    solution: str
    source_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "severity": self.severity,
            "solution": self.solution,
            "source_issues": self.source_issues,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnownProblem:
        return cls(
            summary=d.get("summary", ""),
            severity=d.get("severity", "medium"),
            solution=d.get("solution", ""),
            source_issues=d.get("source_issues", []),
        )


@dataclass
class CompatibilityEntry:
    """Aggregated compatibility data for a version pair."""

    score: float  # 0.0 = high risk, 1.0 = no known issues
    issue_count: int
    known_problems: list[KnownProblem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "issue_count": self.issue_count,
            "known_problems": [p.to_dict() for p in self.known_problems],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompatibilityEntry:
        return cls(
            score=d.get("score", 1.0),
            issue_count=d.get("issue_count", 0),
            known_problems=[
                KnownProblem.from_dict(p) for p in d.get("known_problems", [])
            ],
        )


@dataclass
class CompatibilityWarning:
    """A compatibility warning returned by matrix.check()."""

    packages: dict[str, str]  # {"transformers": "4.55.0", "torch": "2.3.0"}
    score: float
    problems: list[KnownProblem] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class PairResult:
    """Result of querying a specific version pair."""

    score: float
    issue_count: int
    problems: list[KnownProblem] = field(default_factory=list)

"""Data models for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ExpectedIssueRef:
    repo: str
    number: int


@dataclass
class EvalCase:
    """A single evaluation test case loaded from YAML."""

    id: str
    category: str
    difficulty: str
    source: str
    description: str

    # Input
    traceback: str
    env_packages: dict[str, str] = field(default_factory=dict)

    # Expected outputs
    expected_triage: str = ""
    expected_problem_type: str = ""
    root_cause_keywords: list[str] = field(default_factory=list)
    fix_must_contain: list[str] = field(default_factory=list)
    fix_must_not_contain: list[str] = field(default_factory=list)
    valid_issue_refs: list[ExpectedIssueRef] = field(default_factory=list)
    confidence_min: float = 0.5

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> EvalCase:
        """Parse a YAML dict into an EvalCase."""
        inp = data.get("input", {})
        exp = data.get("expected", {})

        issue_refs = []
        for ref in exp.get("valid_issue_refs", []):
            if isinstance(ref, dict):
                issue_refs.append(ExpectedIssueRef(
                    repo=ref.get("repo", ""),
                    number=ref.get("number", 0),
                ))

        return cls(
            id=data.get("id", "unknown"),
            category=data.get("category", ""),
            difficulty=data.get("difficulty", ""),
            source=data.get("source", ""),
            description=data.get("description", ""),
            traceback=inp.get("traceback", ""),
            env_packages=inp.get("env_packages", {}),
            expected_triage=exp.get("triage", ""),
            expected_problem_type=exp.get("problem_type", ""),
            root_cause_keywords=exp.get("root_cause_keywords", []),
            fix_must_contain=exp.get("fix_must_contain", []),
            fix_must_not_contain=exp.get("fix_must_not_contain", []),
            valid_issue_refs=issue_refs,
            confidence_min=exp.get("confidence_min", 0.5),
        )


class ScoreVerdict(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: str
    score: float  # 0.0 - 1.0
    verdict: ScoreVerdict
    details: str = ""


@dataclass
class CaseResult:
    """Complete evaluation result for a single test case."""

    case_id: str
    category: str
    difficulty: str
    scores: list[DimensionScore] = field(default_factory=list)
    weighted_score: float = 0.0
    latency_seconds: float = 0.0
    tool_calls_count: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.weighted_score >= 0.6 and self.error is None


@dataclass
class EvalReport:
    """Aggregated evaluation report across all cases."""

    model: str
    total_cases: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.weighted_score for r in self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    def by_category(self) -> dict[str, list[CaseResult]]:
        groups: dict[str, list[CaseResult]] = {}
        for r in self.results:
            groups.setdefault(r.category, []).append(r)
        return groups

    def by_dimension(self) -> dict[str, list[float]]:
        dims: dict[str, list[float]] = {}
        for r in self.results:
            for s in r.scores:
                dims.setdefault(s.dimension, []).append(s.score)
        return dims

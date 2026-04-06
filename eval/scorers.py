"""Scoring functions for each evaluation dimension."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from octoscout.models import DiagnosisResult

from eval.models import DimensionScore, EvalCase, ScoreVerdict


def _verdict(score: float) -> ScoreVerdict:
    if score >= 0.8:
        return ScoreVerdict.PASS
    if score >= 0.4:
        return ScoreVerdict.PARTIAL
    return ScoreVerdict.FAIL


# ---------------------------------------------------------------------------
# 1. Triage accuracy (exact match)
# ---------------------------------------------------------------------------

def score_triage(result: DiagnosisResult, case: EvalCase) -> DimensionScore:
    """Check if the triage classification matches expected."""
    if not case.expected_triage:
        return DimensionScore(
            dimension="triage", score=1.0,
            verdict=ScoreVerdict.PASS, details="No expected triage specified",
        )

    actual = result.triage.value
    expected = case.expected_triage

    if actual == expected:
        return DimensionScore(
            dimension="triage", score=1.0,
            verdict=ScoreVerdict.PASS,
            details=f"Correct: {actual}",
        )

    # Partial credit: ambiguous is close to either
    if actual == "ambiguous" or expected == "ambiguous":
        return DimensionScore(
            dimension="triage", score=0.5,
            verdict=ScoreVerdict.PARTIAL,
            details=f"Expected {expected}, got {actual} (partial credit for ambiguous)",
        )

    return DimensionScore(
        dimension="triage", score=0.0,
        verdict=ScoreVerdict.FAIL,
        details=f"Expected {expected}, got {actual}",
    )


# ---------------------------------------------------------------------------
# 2. Root cause keywords
# ---------------------------------------------------------------------------

def score_root_cause(result: DiagnosisResult, case: EvalCase) -> DimensionScore:
    """Check if the diagnosis summary contains expected root cause keywords."""
    if not case.root_cause_keywords:
        return DimensionScore(
            dimension="root_cause", score=1.0,
            verdict=ScoreVerdict.PASS, details="No keywords to check",
        )

    text = (result.summary + " " + result.details).lower()
    found = [kw for kw in case.root_cause_keywords if kw.lower() in text]
    total = len(case.root_cause_keywords)

    score = len(found) / total
    missing = [kw for kw in case.root_cause_keywords if kw.lower() not in text]

    return DimensionScore(
        dimension="root_cause", score=score,
        verdict=_verdict(score),
        details=f"Found {len(found)}/{total} keywords. Missing: {missing}" if missing else "All keywords found",
    )


# ---------------------------------------------------------------------------
# 3. Fix correctness
# ---------------------------------------------------------------------------

def score_fix(result: DiagnosisResult, case: EvalCase) -> DimensionScore:
    """Check if fix suggestions contain required terms and avoid forbidden ones."""
    # Combine all text that might contain fix info
    text = " ".join(filter(None, [
        result.summary,
        result.details,
        result.suggested_fix,
    ])).lower()

    if not case.fix_must_contain and not case.fix_must_not_contain:
        return DimensionScore(
            dimension="fix", score=1.0,
            verdict=ScoreVerdict.PASS, details="No fix constraints specified",
        )

    score = 1.0
    issues: list[str] = []

    # Must-contain checks
    if case.fix_must_contain:
        found = sum(1 for kw in case.fix_must_contain if kw.lower() in text)
        contain_ratio = found / len(case.fix_must_contain)
        if contain_ratio < 1.0:
            missing = [kw for kw in case.fix_must_contain if kw.lower() not in text]
            issues.append(f"Missing required fix terms: {missing}")
            score *= contain_ratio

    # Must-not-contain checks
    if case.fix_must_not_contain:
        violations = [kw for kw in case.fix_must_not_contain if kw.lower() in text]
        if violations:
            issues.append(f"Contains forbidden terms: {violations}")
            score *= 0.5

    return DimensionScore(
        dimension="fix", score=score,
        verdict=_verdict(score),
        details="; ".join(issues) if issues else "All fix constraints satisfied",
    )


# ---------------------------------------------------------------------------
# 4. Citation accuracy (GitHub issue refs)
# ---------------------------------------------------------------------------

def score_citations(result: DiagnosisResult, case: EvalCase) -> DimensionScore:
    """Check if cited GitHub issues match expected references."""
    if not case.valid_issue_refs:
        return DimensionScore(
            dimension="citations", score=1.0,
            verdict=ScoreVerdict.PASS, details="No expected issue refs",
        )

    # Collect cited issue numbers from result
    cited_issues: set[tuple[str, int]] = set()

    # From structured related_issues
    for issue in result.related_issues:
        cited_issues.add((issue.repo.lower(), issue.number))

    # Also scan summary text for patterns like #43733 or repo#43733
    text = result.summary + " " + result.details
    for ref in case.valid_issue_refs:
        # Match patterns: #43733, huggingface/transformers#43733, /pull/43733, /issues/43733
        patterns = [
            rf"#{ref.number}\b",
            rf"/(?:pull|issues)/{ref.number}\b",
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                cited_issues.add((ref.repo.lower(), ref.number))

    # Score
    found = 0
    for ref in case.valid_issue_refs:
        if (ref.repo.lower(), ref.number) in cited_issues:
            found += 1

    score = found / len(case.valid_issue_refs)

    return DimensionScore(
        dimension="citations", score=score,
        verdict=_verdict(score),
        details=f"Found {found}/{len(case.valid_issue_refs)} expected issue references",
    )


# ---------------------------------------------------------------------------
# 5. Hallucination detection
# ---------------------------------------------------------------------------

def score_hallucination(result: DiagnosisResult, _case: EvalCase) -> DimensionScore:
    """Basic hallucination check: look for fabricated-looking version numbers or issue refs.

    This is a heuristic check. For more rigorous checking, use the async
    version that validates GitHub URLs.
    """
    text = result.summary + " " + result.details
    issues_found: list[str] = []

    # Check for suspiciously round version numbers that might be hallucinated
    # e.g., "fixed in v10.0.0" — not inherently wrong, but flag for review
    suspicious_versions = re.findall(r"v?(\d+\.0\.0)", text)
    if len(suspicious_versions) > 3:
        issues_found.append(f"Many round version numbers (possible fabrication): {suspicious_versions[:5]}")

    # If no issues detected, score is 1.0 (no hallucination)
    if not issues_found:
        return DimensionScore(
            dimension="hallucination", score=1.0,
            verdict=ScoreVerdict.PASS,
            details="No obvious hallucination detected",
        )

    return DimensionScore(
        dimension="hallucination", score=0.5,
        verdict=ScoreVerdict.PARTIAL,
        details="; ".join(issues_found),
    )


# ---------------------------------------------------------------------------
# 6. Confidence calibration
# ---------------------------------------------------------------------------

def score_confidence(result: DiagnosisResult, case: EvalCase) -> DimensionScore:
    """Check if reported confidence meets the minimum threshold."""
    if result.confidence >= case.confidence_min:
        return DimensionScore(
            dimension="confidence", score=1.0,
            verdict=ScoreVerdict.PASS,
            details=f"Confidence {result.confidence:.2f} >= {case.confidence_min:.2f}",
        )

    # Partial credit for being close
    ratio = result.confidence / case.confidence_min if case.confidence_min > 0 else 0.0
    return DimensionScore(
        dimension="confidence", score=ratio,
        verdict=_verdict(ratio),
        details=f"Confidence {result.confidence:.2f} < {case.confidence_min:.2f}",
    )


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

# Default weights for each dimension
DEFAULT_WEIGHTS: dict[str, float] = {
    "triage": 0.20,
    "root_cause": 0.25,
    "fix": 0.25,
    "citations": 0.10,
    "hallucination": 0.10,
    "confidence": 0.10,
}

ALL_SCORERS = [
    score_triage,
    score_root_cause,
    score_fix,
    score_citations,
    score_hallucination,
    score_confidence,
]


def score_case(
    result: DiagnosisResult,
    case: EvalCase,
    weights: dict[str, float] | None = None,
) -> tuple[list[DimensionScore], float]:
    """Run all scorers and return (scores, weighted_total)."""
    w = weights or DEFAULT_WEIGHTS
    scores: list[DimensionScore] = []

    for scorer in ALL_SCORERS:
        dim_score = scorer(result, case)
        scores.append(dim_score)

    weighted = sum(
        s.score * w.get(s.dimension, 0.0)
        for s in scores
    )

    return scores, weighted

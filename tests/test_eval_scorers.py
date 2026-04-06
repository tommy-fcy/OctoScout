"""Tests for the eval scoring functions."""

from octoscout.models import DiagnosisResult, GitHubIssueRef, ProblemType, TriageResult

from eval.models import DimensionScore, EvalCase, ExpectedIssueRef, ScoreVerdict
from eval.scorers import (
    score_case,
    score_citations,
    score_fix,
    score_root_cause,
    score_triage,
)


def _make_result(**overrides) -> DiagnosisResult:
    defaults = dict(
        triage=TriageResult.UPSTREAM_ISSUE,
        problem_type=ProblemType.API_SIGNATURE_CHANGE,
        summary="Use processing_class=tokenizer instead of tokenizer=tokenizer. "
                "The tokenizer argument was removed in transformers v5.0.0.",
        details="See PR #43733 in huggingface/transformers.",
        suggested_fix="Replace tokenizer=tokenizer with processing_class=tokenizer",
        confidence=0.85,
        related_issues=[
            GitHubIssueRef(
                repo="huggingface/transformers",
                number=43733,
                title="Fix trainer tokenizer",
                url="https://github.com/huggingface/transformers/pull/43733",
                state="closed",
            )
        ],
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


def _make_case(**overrides) -> EvalCase:
    defaults = dict(
        id="test_case",
        category="api_signature_change",
        difficulty="medium",
        source="test",
        description="test case",
        traceback="TypeError: ...",
        expected_triage="upstream_issue",
        expected_problem_type="api_signature_change",
        root_cause_keywords=["processing_class", "tokenizer", "removed"],
        fix_must_contain=["processing_class"],
        fix_must_not_contain=[],
        valid_issue_refs=[ExpectedIssueRef(repo="huggingface/transformers", number=43733)],
        confidence_min=0.7,
    )
    defaults.update(overrides)
    return EvalCase(**defaults)


# ---- Triage ----

def test_triage_exact_match():
    result = _make_result()
    case = _make_case()
    score = score_triage(result, case)
    assert score.score == 1.0
    assert score.verdict == ScoreVerdict.PASS


def test_triage_mismatch():
    result = _make_result(triage=TriageResult.LOCAL_ISSUE)
    case = _make_case(expected_triage="upstream_issue")
    score = score_triage(result, case)
    assert score.score == 0.0
    assert score.verdict == ScoreVerdict.FAIL


def test_triage_ambiguous_partial():
    result = _make_result(triage=TriageResult.AMBIGUOUS)
    case = _make_case(expected_triage="upstream_issue")
    score = score_triage(result, case)
    assert score.score == 0.5
    assert score.verdict == ScoreVerdict.PARTIAL


# ---- Root Cause Keywords ----

def test_root_cause_all_found():
    result = _make_result()
    case = _make_case()
    score = score_root_cause(result, case)
    assert score.score == 1.0


def test_root_cause_partial():
    result = _make_result(summary="The tokenizer argument was changed.", details="")
    case = _make_case(root_cause_keywords=["tokenizer", "processing_class", "v5"])
    score = score_root_cause(result, case)
    assert 0.0 < score.score < 1.0  # Found "tokenizer" but not all


# ---- Fix ----

def test_fix_all_satisfied():
    result = _make_result()
    case = _make_case()
    score = score_fix(result, case)
    assert score.score == 1.0


def test_fix_missing_required():
    result = _make_result(summary="You have a bug.", suggested_fix="Check your code.")
    case = _make_case(fix_must_contain=["processing_class"])
    score = score_fix(result, case)
    assert score.score < 1.0


# ---- Citations ----

def test_citations_found():
    result = _make_result()
    case = _make_case()
    score = score_citations(result, case)
    assert score.score == 1.0


def test_citations_in_text():
    # Issue ref found in text even without structured related_issues
    result = _make_result(
        related_issues=[],
        summary="See PR #43733 in huggingface/transformers for details.",
    )
    case = _make_case()
    score = score_citations(result, case)
    assert score.score == 1.0


# ---- Composite ----

def test_score_case_perfect():
    result = _make_result()
    case = _make_case()
    scores, weighted = score_case(result, case)
    assert len(scores) == 6  # All 6 dimensions
    assert weighted > 0.8

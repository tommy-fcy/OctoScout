"""Triage logic: classify errors as local vs upstream using heuristic rules."""

from __future__ import annotations

from dataclasses import dataclass

from octoscout.models import EnvSnapshot, ParsedTraceback, ProblemType, TriageResult


@dataclass
class TriageDecision:
    result: TriageResult
    problem_type: ProblemType
    confidence: float  # 0.0 - 1.0
    reasoning: str


# Exception types strongly suggesting user code bug
_USER_CODE_EXCEPTIONS = {"NameError", "SyntaxError", "IndentationError", "UnboundLocalError"}

# Exception types suggesting upstream / API change
_UPSTREAM_EXCEPTIONS = {"TypeError", "AttributeError", "ImportError", "ModuleNotFoundError"}


def triage(
    tb: ParsedTraceback,
    env: EnvSnapshot | None = None,
) -> TriageDecision:
    """Classify an error as local issue, upstream issue, or ambiguous.

    Uses heuristic rules (no LLM). This is the fast first-pass triage;
    ambiguous cases can be escalated to LLM-based analysis.
    """
    reasons: list[str] = []
    upstream_score = 0.0
    local_score = 0.0

    # --- Rule 1: Exception type ---
    exc_base = tb.exception_type.split(".")[-1]

    if exc_base in _USER_CODE_EXCEPTIONS:
        local_score += 0.4
        reasons.append(f"{exc_base} is typically a user code error")
    elif exc_base in _UPSTREAM_EXCEPTIONS:
        upstream_score += 0.3
        reasons.append(f"{exc_base} can indicate an API/version change")

    # --- Rule 2: Where the exception occurred ---
    if tb.is_user_code:
        local_score += 0.3
        reasons.append("Exception occurred in user code")
    else:
        upstream_score += 0.3
        reasons.append("Exception occurred in library code")

    # --- Rule 3: Cross-library interaction ---
    if len(tb.involved_packages) >= 2:
        upstream_score += 0.2
        reasons.append(
            f"Call stack crosses {len(tb.involved_packages)} packages: "
            f"{', '.join(sorted(tb.involved_packages))}"
        )

    # --- Rule 4: TypeError with "unexpected keyword argument" pattern ---
    if exc_base == "TypeError" and "unexpected keyword argument" in tb.exception_message:
        upstream_score += 0.3
        reasons.append("'unexpected keyword argument' strongly suggests API signature change")

    # --- Rule 5: ImportError / ModuleNotFoundError ---
    if exc_base in ("ImportError", "ModuleNotFoundError"):
        if not tb.is_user_code:
            upstream_score += 0.2
            reasons.append("Import failure in library code suggests broken dependency")

    # --- Decide ---
    diff = upstream_score - local_score
    if diff > 0.2:
        result = TriageResult.UPSTREAM_ISSUE
        confidence = min(0.5 + diff, 0.95)
        problem_type = _infer_problem_type(tb)
    elif diff < -0.2:
        result = TriageResult.LOCAL_ISSUE
        confidence = min(0.5 - diff, 0.95)
        problem_type = ProblemType.USER_CODE_BUG
    else:
        result = TriageResult.AMBIGUOUS
        confidence = 0.3
        problem_type = _infer_problem_type(tb)

    return TriageDecision(
        result=result,
        problem_type=problem_type,
        confidence=confidence,
        reasoning="; ".join(reasons),
    )


def _infer_problem_type(tb: ParsedTraceback) -> ProblemType:
    """Infer the problem type from exception details."""
    exc = tb.exception_type.split(".")[-1]
    msg = tb.exception_message.lower()

    if exc == "TypeError" and "unexpected keyword argument" in msg:
        return ProblemType.API_SIGNATURE_CHANGE
    if exc == "TypeError" and ("argument" in msg or "positional" in msg):
        return ProblemType.API_SIGNATURE_CHANGE
    if exc == "AttributeError":
        return ProblemType.API_SIGNATURE_CHANGE
    if exc in ("ImportError", "ModuleNotFoundError"):
        return ProblemType.IMPORT_ERROR
    if "version" in msg or "cuda" in msg or "mismatch" in msg:
        return ProblemType.VERSION_MISMATCH

    return ProblemType.UNKNOWN

"""Tests for the matrix aggregator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from octoscout.matrix.aggregator import (
    CompatibilityMatrix,
    _is_valid_version,
    _make_pair_key,
    _normalize_package_name,
)
from octoscout.matrix.models import (
    CompatibilityEntry,
    ExtractedIssueInfo,
    KnownProblem,
)
from octoscout.models import EnvSnapshot


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_normalize_package_name():
    assert _normalize_package_name("huggingface_hub") == "huggingface-hub"
    assert _normalize_package_name("Transformers") == "transformers"
    assert _normalize_package_name("flash_attn") == "flash-attn"


def test_is_valid_version():
    assert _is_valid_version("4.55.0")
    assert _is_valid_version("2.3")
    assert not _is_valid_version("unknown")
    assert not _is_valid_version("main")
    assert not _is_valid_version(">=4.53")
    assert not _is_valid_version("")
    assert not _is_valid_version("nightly")


# ---------------------------------------------------------------------------
# Pair key tests
# ---------------------------------------------------------------------------


def test_pair_key_alphabetical():
    key = _make_pair_key("torch", "2.3.0", "transformers", "4.55.0")
    # Now uses minor versions
    assert key == "torch==2.3+transformers==4.55"


def test_pair_key_sorted():
    """Keys should be the same regardless of argument order."""
    k1 = _make_pair_key("transformers", "4.55.0", "torch", "2.3.0")
    k2 = _make_pair_key("torch", "2.3.0", "transformers", "4.55.0")
    assert k1 == k2


def test_pair_key_normalizes_names():
    k = _make_pair_key("huggingface_hub", "1.0", "Flash_Attn", "2.5")
    assert "huggingface-hub" in k
    assert "flash-attn" in k


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


def _write_extracted(tmpdir: Path, issues: list[ExtractedIssueInfo]) -> Path:
    """Write extracted issues to a JSONL file and return the directory."""
    extracted_dir = tmpdir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with open(extracted_dir / "test_repo.jsonl", "w") as f:
        for issue in issues:
            f.write(json.dumps(issue.to_dict()) + "\n")
    return extracted_dir


def test_scoring_single_crash():
    """A single crash issue should reduce the score by 0.15."""
    issues = [
        ExtractedIssueInfo(
            issue_id="repo#1",
            title="Crash",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
            error_message_summary="segfault",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_dir = _write_extracted(tmpdir, issues)
        output_path = tmpdir / "matrix.json"

        matrix = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)

        # Query uses minor version (aggregator collapses to minor)
        result = matrix.query_pair("transformers", "4.55", "torch", "2.3")
        assert result is not None
        assert abs(result.score - 0.85) < 0.01
        assert result.issue_count == 1


def test_scoring_multiple_issues():
    """Multiple crash issues should reduce the score cumulatively."""
    issues = [
        ExtractedIssueInfo(
            issue_id="repo#1",
            title="Crash 1",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
        ),
        ExtractedIssueInfo(
            issue_id="repo#2",
            title="Crash 2",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
        ),
        ExtractedIssueInfo(
            issue_id="repo#3",
            title="Performance issue",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="performance",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_dir = _write_extracted(tmpdir, issues)
        output_path = tmpdir / "matrix.json"

        matrix = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)

        result = matrix.query_pair("transformers", "4.55", "torch", "2.3")
        assert result is not None
        # 1.0 - (0.15 + 0.15 + 0.05) = 0.65
        assert abs(result.score - 0.65) < 0.01
        assert result.issue_count == 3


def test_query_missing_pair():
    """Querying a non-existent pair returns None."""
    matrix = CompatibilityMatrix()
    result = matrix.query_pair("foo", "1.0", "bar", "2.0")
    assert result is None


# ---------------------------------------------------------------------------
# check() tests
# ---------------------------------------------------------------------------


def test_check_returns_warnings():
    """check() should return warnings for low-score pairs."""
    issues = [
        ExtractedIssueInfo(
            issue_id="repo#1",
            title="Crash",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
            error_message_summary="segfault",
            has_solution=True,
            solution_detail="downgrade transformers to 4.52.3",
        ),
        # Add enough crash issues to bring score below 0.7
        ExtractedIssueInfo(
            issue_id="repo#2",
            title="Crash 2",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
        ),
        ExtractedIssueInfo(
            issue_id="repo#3",
            title="Crash 3",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_dir = _write_extracted(tmpdir, issues)
        output_path = tmpdir / "matrix.json"

        matrix = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)

        env = EnvSnapshot(
            installed_packages={"transformers": "4.55.0", "torch": "2.3.0"},
        )
        warnings = matrix.check(env)
        assert len(warnings) >= 1
        assert warnings[0].packages == {"transformers": "4.55.0", "torch": "2.3.0"}


def test_check_no_warnings_for_safe_pair():
    """check() should return no warnings for high-score pairs."""
    issues = [
        ExtractedIssueInfo(
            issue_id="repo#1",
            title="Minor perf issue",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="performance",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_dir = _write_extracted(tmpdir, issues)
        output_path = tmpdir / "matrix.json"

        matrix = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)

        env = EnvSnapshot(
            installed_packages={"transformers": "4.55.0", "torch": "2.3.0"},
        )
        warnings = matrix.check(env)
        assert len(warnings) == 0  # score = 0.95, above threshold


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip():
    """Matrix should survive save/load cycle."""
    issues = [
        ExtractedIssueInfo(
            issue_id="repo#1",
            title="Crash",
            reported_versions={"transformers": "4.55.0", "torch": "2.3.0"},
            problem_type="crash",
            error_message_summary="segfault",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_dir = _write_extracted(tmpdir, issues)
        output_path = tmpdir / "matrix.json"

        original = CompatibilityMatrix.build_from_extracted(extracted_dir, output_path)
        loaded = CompatibilityMatrix.load(output_path)

        # Both should give the same query result
        r1 = original.query_pair("transformers", "4.55", "torch", "2.3")
        r2 = loaded.query_pair("transformers", "4.55", "torch", "2.3")

        assert r1 is not None
        assert r2 is not None
        assert abs(r1.score - r2.score) < 0.001
        assert r1.issue_count == r2.issue_count


def test_entry_count():
    matrix = CompatibilityMatrix(entries={
        "a==1+b==2": CompatibilityEntry(score=0.5, issue_count=3),
        "c==1+d==2": CompatibilityEntry(score=0.8, issue_count=1),
    })
    assert matrix.entry_count == 2

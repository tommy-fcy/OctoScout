"""Tests for triage logic."""

from pathlib import Path

from octoscout.diagnosis.traceback_parser import parse_traceback
from octoscout.diagnosis.triage import triage
from octoscout.models import TriageResult

FIXTURES = Path(__file__).parent / "fixtures"


def test_triage_upstream_type_error():
    text = (FIXTURES / "transformers_type_error.txt").read_text()
    tb = parse_traceback(text)
    result = triage(tb)

    assert result.result == TriageResult.UPSTREAM_ISSUE
    assert result.confidence > 0.5


def test_triage_upstream_import_error():
    text = (FIXTURES / "vllm_cuda_mismatch.txt").read_text()
    tb = parse_traceback(text)
    result = triage(tb)

    assert result.result == TriageResult.UPSTREAM_ISSUE


def test_triage_local_name_error():
    text = (FIXTURES / "user_name_error.txt").read_text()
    tb = parse_traceback(text)
    result = triage(tb)

    assert result.result == TriageResult.LOCAL_ISSUE
    assert result.confidence > 0.5

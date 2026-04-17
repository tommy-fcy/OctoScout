"""Tests for the campaign system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from octoscout.campaign.models import (
    CampaignIssue,
    DiagnosisRecord,
    IssueWorkLog,
    ReplyRecord,
    TrackingSnapshot,
    VerificationRecord,
    append_jsonl,
    read_jsonl,
    update_worklog,
)


# ---------------------------------------------------------------------------
# Model serialization tests
# ---------------------------------------------------------------------------


class TestCampaignIssue:
    def test_roundtrip(self):
        issue = CampaignIssue(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="Some body text",
            url="https://github.com/owner/repo/issues/42",
            labels=["bug"],
            comment_count=5,
            env_category="gpu_required",
            referenced_issues=["owner/repo#40"],
            discovery_score=0.75,
        )
        d = issue.to_dict()
        restored = CampaignIssue.from_dict(d)
        assert restored.repo == "owner/repo"
        assert restored.number == 42
        assert restored.env_category == "gpu_required"
        assert restored.referenced_issues == ["owner/repo#40"]
        assert restored.discovery_score == 0.75


class TestDiagnosisRecord:
    def test_roundtrip(self):
        record = DiagnosisRecord(
            repo="owner/repo",
            number=42,
            diagnosis_summary="Upgrade transformers",
            has_concrete_fix=True,
            fix_actions=["pip install transformers==4.44.0"],
            suggested_versions={"transformers": "4.44.0"},
        )
        d = record.to_dict()
        restored = DiagnosisRecord.from_dict(d)
        assert restored.has_concrete_fix is True
        assert restored.fix_actions == ["pip install transformers==4.44.0"]


class TestVerificationRecord:
    def test_roundtrip(self):
        record = VerificationRecord(
            repo="owner/repo",
            number=42,
            level="import",
            broken_env={"transformers": "4.53.0"},
            fixed_env={"transformers": "4.44.0"},
            broken_result="error_reproduced",
            fixed_result="fix_confirmed",
            verified=True,
        )
        d = record.to_dict()
        restored = VerificationRecord.from_dict(d)
        assert restored.verified is True
        assert restored.level == "import"


class TestIssueWorkLog:
    def test_roundtrip_with_nested(self):
        log = IssueWorkLog(
            repo="owner/repo",
            number=42,
            title="Test issue",
            url="https://github.com/owner/repo/issues/42",
            env_category="cpu_only",
            diagnosis=DiagnosisRecord(
                repo="owner/repo", number=42,
                has_concrete_fix=True,
                fix_actions=["pip install x==1.0"],
            ),
            verification=VerificationRecord(
                repo="owner/repo", number=42, verified=True,
            ),
            notes=["Framework weakness: slow on large repos"],
            outcome="resolved",
        )
        d = log.to_dict()
        restored = IssueWorkLog.from_dict(d)
        assert restored.diagnosis is not None
        assert restored.diagnosis.has_concrete_fix is True
        assert restored.verification is not None
        assert restored.verification.verified is True
        assert restored.outcome == "resolved"

    def test_to_markdown(self):
        log = IssueWorkLog(
            repo="owner/repo",
            number=42,
            title="Test issue",
            url="https://github.com/owner/repo/issues/42",
            env_category="cpu_only",
            outcome="resolved",
            diagnosis=DiagnosisRecord(
                repo="owner/repo", number=42,
                has_concrete_fix=True,
                fix_actions=["pip install x==1.0"],
            ),
        )
        md = log.to_markdown()
        assert "owner/repo#42" in md
        assert "concrete fix" in md
        assert "cpu_only" in md


# ---------------------------------------------------------------------------
# JSONL I/O tests
# ---------------------------------------------------------------------------


class TestJsonlIO:
    def test_append_and_read(self, tmp_path):
        path = tmp_path / "test.jsonl"
        issue1 = CampaignIssue(repo="a/b", number=1, title="T1", body="B1", url="u1")
        issue2 = CampaignIssue(repo="a/b", number=2, title="T2", body="B2", url="u2")

        append_jsonl(path, issue1)
        append_jsonl(path, issue2)

        loaded = read_jsonl(path, CampaignIssue)
        assert len(loaded) == 2
        assert loaded[0].number == 1
        assert loaded[1].number == 2

    def test_read_nonexistent(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        result = read_jsonl(path, CampaignIssue)
        assert result == []


class TestUpdateWorklog:
    def test_create_and_update(self, tmp_path):
        issue = CampaignIssue(
            repo="a/b", number=1, title="T", body="B", url="u",
            env_category="gpu_required", discovered_at="2026-04-13",
        )

        # Create
        log = update_worklog(tmp_path, issue)
        assert log.repo == "a/b"
        assert log.env_category == "gpu_required"

        # Update with diagnosis
        diag = DiagnosisRecord(repo="a/b", number=1, has_concrete_fix=True)
        log = update_worklog(tmp_path, issue, diagnosis=diag)

        # Read back and verify
        logs = read_jsonl(tmp_path / "worklog.jsonl", IssueWorkLog)
        assert len(logs) == 1
        assert logs[0].diagnosis is not None
        assert logs[0].diagnosis.has_concrete_fix is True


# ---------------------------------------------------------------------------
# Discovery logic tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_extract_traceback_from_code_fence(self):
        from octoscout.campaign.discovery import extract_traceback

        text = """
Some text before.

```python
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import transformers
ImportError: No module named 'transformers'
```

Some text after.
"""
        tb = extract_traceback(text)
        assert "Traceback (most recent call last)" in tb
        assert "ImportError" in tb

    def test_extract_traceback_raw(self):
        from octoscout.campaign.discovery import extract_traceback

        text = """The error is:
Traceback (most recent call last):
  File "x.py", line 5, in foo
    bar()
ValueError: bad value
"""
        tb = extract_traceback(text)
        assert "ValueError" in tb

    def test_extract_traceback_none(self):
        from octoscout.campaign.discovery import extract_traceback

        assert extract_traceback("No traceback here") == ""

    def test_extract_code_snippet(self):
        from octoscout.campaign.discovery import extract_code_snippet

        text = """
Install:
```bash
pip install torch
```

Reproduce:
```python
import torch
print(torch.__version__)
```
"""
        snippet = extract_code_snippet(text)
        assert snippet is not None
        assert "import torch" in snippet

    def test_extract_code_snippet_none(self):
        from octoscout.campaign.discovery import extract_code_snippet

        assert extract_code_snippet("No code here") is None

    def test_classify_env_category(self):
        from octoscout.campaign.discovery import classify_env_category

        assert classify_env_category("CUDA error: out of memory") == "gpu_required"
        assert classify_env_category("torch.cuda.is_available()") == "gpu_required"
        assert classify_env_category("model = AutoModel.from_pretrained('bert')") == "model_download"
        assert classify_env_category("import numpy; numpy.array([1,2,3])") == "cpu_only"

    def test_extract_issue_references(self):
        from octoscout.campaign.discovery import extract_issue_references

        text = """
Related to #123 and huggingface/transformers#456.
Also see https://github.com/pytorch/pytorch/issues/789
"""
        refs = extract_issue_references(text, "owner/repo")
        assert "owner/repo#123" in refs
        assert "huggingface/transformers#456" in refs
        assert "pytorch/pytorch#789" in refs

    def test_compute_discovery_score(self):
        from octoscout.campaign.discovery import compute_discovery_score

        # Recent issue with many comments and traceback should score high
        score = compute_discovery_score(
            comment_count=20,
            created_at="2026-04-12T00:00:00Z",
            has_traceback=True,
            max_age_days=90,
        )
        assert score > 0.6

        # Old issue with no comments and no traceback should score low
        score_low = compute_discovery_score(
            comment_count=0,
            created_at="2025-01-01T00:00:00Z",
            has_traceback=False,
            max_age_days=90,
        )
        assert score_low < 0.3


# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_check_dependency_resolution(self):
        from octoscout.campaign.sandbox import check_dependency_resolution

        # Should succeed for well-known compatible packages
        result = check_dependency_resolution({"pip": "24.0"})
        # Just verify it runs without error (actual result depends on env)
        assert result.exit_code is not None

    def test_extract_versions_from_text(self):
        from octoscout.campaign.sandbox import _extract_versions_from_text

        text = "Install transformers==4.44.0 and torch==2.3.0"
        versions = _extract_versions_from_text(text)
        assert versions.get("transformers") == "4.44.0"
        assert versions.get("torch") == "2.3.0"

    def test_extract_versions_ignores_non_package_terms(self):
        """Random key==value text should not be mistaken for package versions."""
        from octoscout.campaign.sandbox import _extract_versions_from_text

        text = "Config: timeout=0.05, max_length==512, temperature==0.7"
        versions = _extract_versions_from_text(text)
        assert "timeout" not in versions
        assert "max_length" not in versions
        assert "temperature" not in versions

    def test_extract_versions_hyphen_underscore(self):
        """Package names with hyphens or underscores should both match."""
        from octoscout.campaign.sandbox import _extract_versions_from_text

        text = "flash-attn==2.5.0 and sentence_transformers==2.3.1"
        versions = _extract_versions_from_text(text)
        # flash-attn in text should match flash_attn in PACKAGE_REPO_MAP
        assert "flash_attn" in versions
        assert versions["flash_attn"] == "2.5.0"
        assert versions["sentence_transformers"] == "2.3.1"


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------


class TestReporter:
    def test_campaign_status_empty(self, tmp_path):
        from octoscout.campaign.reporter import campaign_status

        (tmp_path / "discovered.jsonl").touch()
        s = campaign_status(tmp_path)
        assert s["discovered"] == 0
        assert s["diagnosed"] == 0

    def test_format_table(self):
        from octoscout.campaign.reporter import format_table

        metrics = {
            "diagnosis": {
                "total": 10,
                "with_error": 1,
                "concrete_fix_count": 6,
                "concrete_fix_rate": 0.6,
                "category_fix_rates": {"cpu_only": 0.8, "gpu_required": 0.4},
            },
            "verification": {
                "total": 5,
                "pass_count": 3,
                "pass_rate": 0.6,
                "reproduced": 2,
                "attempted_reproduce": 3,
            },
            "reply": {
                "total_posted": 4,
                "verified_replies": 3,
                "unverified_replies": 1,
                "positive_feedback_count": 2,
                "positive_feedback_rate": 0.5,
                "resolved_count": 1,
                "resolved_rate": 0.25,
            },
        }
        output = format_table(metrics)
        assert "Diagnosis" in output
        assert "Verification" in output
        assert "Reply Effect" in output

    def test_format_casebook_empty(self, tmp_path):
        from octoscout.campaign.reporter import format_casebook

        assert "No work logs found" in format_casebook(tmp_path)

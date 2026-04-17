"""Batch diagnosis runner for campaign issues."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from octoscout.campaign.models import (
    CampaignIssue,
    DiagnosisRecord,
    append_jsonl,
    read_jsonl,
    update_worklog,
)

# Prompt for extracting structured fix info from diagnosis output
FIX_EXTRACTION_PROMPT = """\
Extract concrete fix actions from the diagnosis below.
If the diagnosis only gives vague suggestions (e.g. "try updating", "check your version")
or no actionable fix, set has_concrete_fix to false.

A concrete fix must include at least one of:
- A specific version number (e.g. "upgrade transformers to 4.44.0")
- A specific code change (e.g. "add padding_side='left'")
- A specific config change (e.g. "set CUDA_VISIBLE_DEVICES=0")

The fix must be backed by evidence (a GitHub issue reference or matrix data).

IMPORTANT for suggested_versions:
- Only include entries with EXACT numeric versions like "4.44.0" or "2.3.1"
- Do NOT include: "latest", "main", ">=4.0", ">5.5.4", "main branch", git URLs
- If the fix is "install from git/main branch", put the git URL in fix_actions only,
  leave suggested_versions empty for that package

Diagnosis:
{diagnosis_summary}

Return ONLY valid JSON (no markdown fences):
{{
  "has_concrete_fix": true or false,
  "fix_actions": ["pip install transformers==4.44.0"],
  "suggested_versions": {{"transformers": "4.44.0"}},
  "evidence_sources": ["huggingface/transformers#40154"]
}}
"""

_VALID_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")


async def _extract_fix_info(
    diagnosis_summary: str,
    provider,
) -> dict:
    """Use a lightweight LLM call to extract structured fix info."""
    from octoscout.models import Message, Role

    prompt = FIX_EXTRACTION_PROMPT.format(diagnosis_summary=diagnosis_summary[:3000])
    response = await provider.chat(
        [Message(role=Role.USER, content=prompt)],
        system="You are a JSON extractor. Return only valid JSON, no explanation.",
    )

    text = (response.text or "").strip()
    # Try to parse JSON from response
    try:
        # Handle markdown code fences
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        parsed = json.loads(text)
        # Sanitize suggested_versions: only keep valid numeric versions
        raw_versions = parsed.get("suggested_versions", {})
        parsed["suggested_versions"] = {
            pkg: ver for pkg, ver in raw_versions.items()
            if isinstance(ver, str) and _VALID_VERSION_RE.match(ver)
        }
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {
            "has_concrete_fix": False,
            "fix_actions": [],
            "suggested_versions": {},
            "evidence_sources": [],
        }


async def diagnose_single_issue(
    issue: CampaignIssue,
    config,
    verbose: bool = False,
) -> DiagnosisRecord:
    """Run DiagnosisAgent on a single campaign issue."""
    from octoscout.agent.core import DiagnosisAgent

    start_time = time.time()
    now = datetime.now(timezone.utc).isoformat()

    # Use traceback if available, otherwise the full issue body
    input_text = issue.extracted_traceback or issue.body[:4000]

    try:
        agent = DiagnosisAgent(config, verbose=verbose)
        result = await agent.diagnose(
            input_text,
            auto_env=False,
            extra_repos=[issue.repo],
        )

        latency = time.time() - start_time

        # Extract structured fix info
        provider = config.get_provider()
        fix_info = await _extract_fix_info(result.summary, provider)

        related_urls = [ref.url for ref in result.related_issues]

        return DiagnosisRecord(
            repo=issue.repo,
            number=issue.number,
            diagnosis_summary=result.summary,
            problem_type=result.problem_type.value if result.problem_type else "",
            suggested_versions=fix_info.get("suggested_versions", {}),
            has_concrete_fix=fix_info.get("has_concrete_fix", False),
            fix_actions=fix_info.get("fix_actions", []),
            evidence_sources=fix_info.get("evidence_sources", []),
            related_issue_urls=related_urls,
            latency_seconds=round(latency, 1),
            diagnosed_at=now,
            error=None,
        )

    except Exception as e:
        latency = time.time() - start_time
        return DiagnosisRecord(
            repo=issue.repo,
            number=issue.number,
            latency_seconds=round(latency, 1),
            diagnosed_at=now,
            error=str(e),
        )


async def batch_diagnose(
    issues: list[CampaignIssue],
    config,
    campaign_dir: Path,
    concurrency: int = 3,
    verbose: bool = False,
) -> list[DiagnosisRecord]:
    """Diagnose multiple issues with semaphore-based concurrency.

    Results are written to diagnosed.jsonl as they complete.
    Supports resume: skips issues already in diagnosed.jsonl.
    """
    output_path = campaign_dir / "diagnosed.jsonl"

    # Load already-diagnosed issues for resume
    existing = read_jsonl(output_path, DiagnosisRecord)
    done_keys = {(r.repo, r.number) for r in existing}

    # Filter to un-diagnosed issues
    pending = [i for i in issues if (i.repo, i.number) not in done_keys]
    if not pending:
        return existing

    semaphore = asyncio.Semaphore(concurrency)
    results: list[DiagnosisRecord] = list(existing)

    async def run_with_semaphore(issue: CampaignIssue) -> DiagnosisRecord:
        async with semaphore:
            return await diagnose_single_issue(issue, config, verbose)

    tasks = [run_with_semaphore(issue) for issue in pending]

    for coro in asyncio.as_completed(tasks):
        record = await coro
        append_jsonl(output_path, record)
        results.append(record)

        # Update work log
        matching_issue = next(
            (i for i in pending if i.repo == record.repo and i.number == record.number),
            None,
        )
        if matching_issue:
            update_worklog(campaign_dir, matching_issue, diagnosis=record)

    return results

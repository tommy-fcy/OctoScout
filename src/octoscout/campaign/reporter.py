"""Generate campaign reports and metrics."""

from __future__ import annotations

from pathlib import Path

from octoscout.campaign.models import (
    DiagnosisRecord,
    IssueWorkLog,
    ReplyRecord,
    TrackingSnapshot,
    VerificationRecord,
    read_jsonl,
)


def compute_metrics(campaign_dir: Path) -> dict:
    """Compute all campaign metrics from JSONL files."""
    diagnosed = read_jsonl(campaign_dir / "diagnosed.jsonl", DiagnosisRecord)
    verified = read_jsonl(campaign_dir / "verified.jsonl", VerificationRecord)
    replied = read_jsonl(campaign_dir / "replied.jsonl", ReplyRecord)
    tracked = read_jsonl(campaign_dir / "tracking.jsonl", TrackingSnapshot)

    total_diagnosed = len(diagnosed)
    concrete_fix = [d for d in diagnosed if d.has_concrete_fix]
    with_error = [d for d in diagnosed if d.error]

    # By env_category (need worklogs for this)
    worklogs = read_jsonl(campaign_dir / "worklog.jsonl", IssueWorkLog)
    cat_map = {(w.repo, w.number): w.env_category for w in worklogs}
    category_fix_rates = {}
    for cat in ["cpu_only", "gpu_required", "model_download"]:
        cat_diagnosed = [d for d in diagnosed if cat_map.get((d.repo, d.number)) == cat]
        cat_concrete = [d for d in cat_diagnosed if d.has_concrete_fix]
        if cat_diagnosed:
            category_fix_rates[cat] = len(cat_concrete) / len(cat_diagnosed)
        else:
            category_fix_rates[cat] = 0.0

    # Verification metrics
    total_verified = len(verified)
    verified_pass = [v for v in verified if v.verified]
    reproduced = [v for v in verified if v.broken_result == "error_reproduced"]
    attempted_reproduce = [v for v in verified if v.level == "reproduce"]

    # Reply metrics
    posted = [r for r in replied if r.posted]
    verified_replies = [r for r in posted if r.verified]
    unverified_replies = [r for r in posted if not r.verified]

    # Tracking metrics
    # Group tracking by issue (latest snapshot)
    latest_tracking: dict[tuple[str, int], TrackingSnapshot] = {}
    for t in tracked:
        key = (t.repo, t.number)
        if key not in latest_tracking or t.checked_at > latest_tracking[key].checked_at:
            latest_tracking[key] = t

    positive_replies = [t for t in latest_tracking.values() if t.has_positive_response]
    closed_after = [t for t in latest_tracking.values() if t.issue_state == "closed"]

    return {
        "diagnosis": {
            "total": total_diagnosed,
            "with_error": len(with_error),
            "concrete_fix_count": len(concrete_fix),
            "concrete_fix_rate": len(concrete_fix) / total_diagnosed if total_diagnosed else 0,
            "category_fix_rates": category_fix_rates,
        },
        "verification": {
            "total": total_verified,
            "pass_count": len(verified_pass),
            "pass_rate": len(verified_pass) / total_verified if total_verified else 0,
            "reproduced": len(reproduced),
            "attempted_reproduce": len(attempted_reproduce),
            "reproduce_rate": len(reproduced) / len(attempted_reproduce) if attempted_reproduce else 0,
        },
        "reply": {
            "total_posted": len(posted),
            "verified_replies": len(verified_replies),
            "unverified_replies": len(unverified_replies),
            "positive_feedback_count": len(positive_replies),
            "positive_feedback_rate": len(positive_replies) / len(posted) if posted else 0,
            "resolved_count": len(closed_after),
            "resolved_rate": len(closed_after) / len(posted) if posted else 0,
        },
    }


def format_table(metrics: dict) -> str:
    """Format metrics as a terminal-friendly table."""
    lines = []

    d = metrics["diagnosis"]
    lines.append("=== Diagnosis ===")
    lines.append(f"  Total diagnosed:    {d['total']}")
    lines.append(f"  Errors:             {d['with_error']}")
    lines.append(f"  Concrete fix:       {d['concrete_fix_count']} ({d['concrete_fix_rate']:.0%})")
    for cat, rate in d["category_fix_rates"].items():
        lines.append(f"    {cat}: {rate:.0%}")

    v = metrics["verification"]
    lines.append("\n=== Verification ===")
    lines.append(f"  Total verified:     {v['total']}")
    lines.append(f"  Pass:               {v['pass_count']} ({v['pass_rate']:.0%})")
    lines.append(f"  Reproduced:         {v['reproduced']}/{v['attempted_reproduce']}")

    r = metrics["reply"]
    lines.append("\n=== Reply Effect ===")
    lines.append(f"  Total posted:       {r['total_posted']}")
    lines.append(f"  Positive feedback:  {r['positive_feedback_count']} ({r['positive_feedback_rate']:.0%})")
    lines.append(f"  Issues resolved:    {r['resolved_count']} ({r['resolved_rate']:.0%})")

    return "\n".join(lines)


def format_markdown(metrics: dict) -> str:
    """Format metrics as markdown for README."""
    d = metrics["diagnosis"]
    v = metrics["verification"]
    r = metrics["reply"]

    lines = [
        "## Campaign Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Issues diagnosed | {d['total']} |",
        f"| Concrete fix rate | {d['concrete_fix_rate']:.0%} |",
        f"| Verification pass rate | {v['pass_rate']:.0%} |",
        f"| Replies posted | {r['total_posted']} |",
        f"| Positive feedback rate | {r['positive_feedback_rate']:.0%} |",
        f"| Issues resolved after reply | {r['resolved_rate']:.0%} |",
    ]
    return "\n".join(lines)


def format_casebook(campaign_dir: Path) -> str:
    """Format per-issue work logs as a casebook."""
    worklogs = read_jsonl(campaign_dir / "worklog.jsonl", IssueWorkLog)
    if not worklogs:
        return "No work logs found."

    lines = ["# Campaign Casebook", ""]
    for log in worklogs:
        lines.append(log.to_markdown())
        lines.append("")

    return "\n".join(lines)


def campaign_status(campaign_dir: Path) -> dict:
    """Get counts for each campaign phase."""
    from octoscout.campaign.models import CampaignIssue

    discovered = read_jsonl(campaign_dir / "discovered.jsonl", CampaignIssue)
    diagnosed = read_jsonl(campaign_dir / "diagnosed.jsonl", DiagnosisRecord)
    verified = read_jsonl(campaign_dir / "verified.jsonl", VerificationRecord)
    replied = read_jsonl(campaign_dir / "replied.jsonl", ReplyRecord)
    tracked = read_jsonl(campaign_dir / "tracking.jsonl", TrackingSnapshot)

    concrete = [d for d in diagnosed if d.has_concrete_fix]
    posted = [r for r in replied if r.posted]

    return {
        "discovered": len(discovered),
        "diagnosed": len(diagnosed),
        "has_concrete_fix": len(concrete),
        "verified": len(verified),
        "verified_pass": len([v for v in verified if v.verified]),
        "replied": len(replied),
        "posted": len(posted),
        "tracked": len(tracked),
    }

"""Data models for the campaign system."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CampaignIssue:
    """An open issue discovered during a campaign."""

    repo: str
    number: int
    title: str
    body: str
    url: str
    labels: list[str] = field(default_factory=list)
    comment_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    # Derived fields
    extracted_traceback: str = ""
    extracted_code_snippet: str | None = None
    has_traceback: bool = False
    env_category: str = "cpu_only"  # "cpu_only" | "gpu_required" | "model_download"
    referenced_issues: list[str] = field(default_factory=list)
    discovery_score: float = 0.0
    discovered_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CampaignIssue:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DiagnosisRecord:
    """Result of diagnosing a campaign issue."""

    repo: str
    number: int
    diagnosis_summary: str = ""
    problem_type: str = ""
    suggested_versions: dict[str, str] = field(default_factory=dict)
    has_concrete_fix: bool = False
    fix_actions: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    related_issue_urls: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    diagnosed_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DiagnosisRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VerificationRecord:
    """Result of sandbox verification for a diagnosed issue."""

    repo: str
    number: int
    verified_at: str = ""
    level: str = "quick"  # "quick" | "import" | "reproduce"
    broken_env: dict[str, str] = field(default_factory=dict)
    fixed_env: dict[str, str] | None = None
    broken_result: str = ""  # "error_reproduced" | "no_error" | "install_failed" | "timeout" | "skip_no_gpu"
    fixed_result: str | None = None  # "fix_confirmed" | "still_broken" | "install_failed" | "timeout"
    reproduction_log: str = ""
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VerificationRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReplyRecord:
    """Record of a reply posted (or drafted) for a campaign issue."""

    repo: str
    number: int
    comment_body: str = ""
    comment_url: str | None = None
    posted: bool = False
    posted_at: str | None = None
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReplyRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrackingSnapshot:
    """A point-in-time snapshot of a replied issue's status."""

    repo: str
    number: int
    checked_at: str = ""
    issue_state: str = "open"
    comment_count: int = 0
    reaction_count: int = 0
    reply_reactions: dict[str, int] = field(default_factory=dict)
    new_comments_since_reply: int = 0
    has_positive_response: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrackingSnapshot:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class IssueWorkLog:
    """Complete work history for a single issue across all campaign phases."""

    repo: str
    number: int
    title: str = ""
    url: str = ""
    discovered_at: str = ""
    env_category: str = "cpu_only"
    diagnosis: DiagnosisRecord | None = None
    verification: VerificationRecord | None = None
    reply: ReplyRecord | None = None
    tracking_history: list[TrackingSnapshot] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    outcome: str = "pending"  # "resolved" | "incorrect" | "no_response" | "pending"

    def to_dict(self) -> dict:
        d = {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "discovered_at": self.discovered_at,
            "env_category": self.env_category,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "reply": self.reply.to_dict() if self.reply else None,
            "tracking_history": [t.to_dict() for t in self.tracking_history],
            "notes": self.notes,
            "outcome": self.outcome,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IssueWorkLog:
        diag = DiagnosisRecord.from_dict(d["diagnosis"]) if d.get("diagnosis") else None
        verif = VerificationRecord.from_dict(d["verification"]) if d.get("verification") else None
        reply = ReplyRecord.from_dict(d["reply"]) if d.get("reply") else None
        tracking = [TrackingSnapshot.from_dict(t) for t in d.get("tracking_history", [])]
        return cls(
            repo=d["repo"],
            number=d["number"],
            title=d.get("title", ""),
            url=d.get("url", ""),
            discovered_at=d.get("discovered_at", ""),
            env_category=d.get("env_category", "cpu_only"),
            diagnosis=diag,
            verification=verif,
            reply=reply,
            tracking_history=tracking,
            notes=d.get("notes", []),
            outcome=d.get("outcome", "pending"),
        )

    def to_markdown(self) -> str:
        """Render this work log as a markdown case study."""
        lines = [
            f"### [{self.repo}#{self.number}]({self.url})",
            f"**{self.title}**",
            f"- Category: `{self.env_category}` | Outcome: `{self.outcome}`",
        ]
        if self.diagnosis:
            status = "concrete fix" if self.diagnosis.has_concrete_fix else "no concrete fix"
            lines.append(f"- Diagnosis: {status}")
            if self.diagnosis.fix_actions:
                lines.append(f"  - Fix: `{self.diagnosis.fix_actions[0]}`")
        if self.verification:
            lines.append(f"- Verification ({self.verification.level}): {'PASS' if self.verification.verified else 'FAIL'}")
        if self.reply:
            status = "posted" if self.reply.posted else "drafted"
            lines.append(f"- Reply: {status}")
        if self.notes:
            for note in self.notes:
                lines.append(f"- Note: {note}")
        return "\n".join(lines)


# --- JSONL I/O helpers ---

def append_jsonl(path: Path, record) -> None:
    """Append a record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls):
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(cls.from_dict(json.loads(line)))
    return records


def update_worklog(campaign_dir: Path, issue: CampaignIssue, **kwargs) -> IssueWorkLog:
    """Update or create a work log entry for an issue.

    Reads existing worklog, finds or creates entry for issue, updates fields,
    writes back.
    """
    worklog_path = campaign_dir / "worklog.jsonl"
    logs = read_jsonl(worklog_path, IssueWorkLog)

    # Find existing entry
    entry = None
    for log in logs:
        if log.repo == issue.repo and log.number == issue.number:
            entry = log
            break

    if entry is None:
        entry = IssueWorkLog(
            repo=issue.repo,
            number=issue.number,
            title=issue.title,
            url=issue.url,
            discovered_at=issue.discovered_at,
            env_category=issue.env_category,
        )
        logs.append(entry)

    # Apply updates
    for k, v in kwargs.items():
        if hasattr(entry, k):
            setattr(entry, k, v)

    # Write all back
    worklog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(worklog_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

    return entry

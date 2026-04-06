"""Core data models shared across OctoScout modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# LLM Provider models
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema (used by both Claude and OpenAI)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Environment / Diagnosis models
# ---------------------------------------------------------------------------


@dataclass
class EnvSnapshot:
    python_version: str | None = None
    os_info: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_model: str | None = None
    installed_packages: dict[str, str] = field(default_factory=dict)
    declared_dependencies: dict[str, str] = field(default_factory=dict)

    def format_for_llm(self) -> str:
        """Format as a concise string suitable for LLM context."""
        lines = ["## Environment Snapshot"]
        if self.python_version:
            lines.append(f"Python: {self.python_version}")
        if self.os_info:
            lines.append(f"OS: {self.os_info}")
        if self.cuda_version:
            lines.append(f"CUDA: {self.cuda_version}")
        if self.cudnn_version:
            lines.append(f"cuDNN: {self.cudnn_version}")
        if self.gpu_model:
            lines.append(f"GPU: {self.gpu_model}")
        if self.installed_packages:
            lines.append("\nInstalled packages:")
            for pkg, ver in sorted(self.installed_packages.items()):
                lines.append(f"  {pkg}=={ver}")
        return "\n".join(lines)


@dataclass
class StackFrame:
    file: str
    line: int
    function: str
    code: str | None = None
    package: str | None = None


@dataclass
class ParsedTraceback:
    exception_type: str
    exception_message: str
    frames: list[StackFrame] = field(default_factory=list)
    root_package: str | None = None
    is_user_code: bool = False
    involved_packages: set[str] = field(default_factory=set)

    def format_for_llm(self) -> str:
        lines = ["## Parsed Traceback"]
        lines.append(f"Exception: {self.exception_type}: {self.exception_message}")
        if self.root_package:
            lines.append(f"Root package: {self.root_package}")
        lines.append(f"User code: {self.is_user_code}")
        if self.involved_packages:
            lines.append(f"Involved packages: {', '.join(sorted(self.involved_packages))}")
        if self.frames:
            lines.append("\nCall stack (innermost last):")
            for f in self.frames:
                pkg_tag = f" [{f.package}]" if f.package else ""
                lines.append(f"  {f.file}:{f.line} in {f.function}{pkg_tag}")
                if f.code:
                    lines.append(f"    {f.code}")
        return "\n".join(lines)


class TriageResult(str, Enum):
    LOCAL_ISSUE = "local_issue"
    UPSTREAM_ISSUE = "upstream_issue"
    AMBIGUOUS = "ambiguous"


class ProblemType(str, Enum):
    API_SIGNATURE_CHANGE = "api_signature_change"
    IMPORT_ERROR = "import_error"
    VERSION_MISMATCH = "version_mismatch"
    USER_CODE_BUG = "user_code_bug"
    UNKNOWN = "unknown"


@dataclass
class DiagnosisResult:
    triage: TriageResult
    problem_type: ProblemType
    summary: str
    details: str = ""
    suggested_fix: str | None = None
    confidence: float = 0.0  # 0.0 - 1.0
    related_issues: list[GitHubIssueRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GitHub / Search models
# ---------------------------------------------------------------------------


@dataclass
class GitHubIssueRef:
    repo: str  # e.g. "huggingface/transformers"
    number: int
    title: str
    url: str
    state: str = "open"
    relevance_score: float = 0.0
    snippet: str = ""

"""Aggregate extracted issues into a queryable compatibility matrix."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from octoscout.matrix.models import (
    CompatibilityEntry,
    CompatibilityWarning,
    ExtractedIssueInfo,
    KnownProblem,
    PairResult,
)
from octoscout.models import EnvSnapshot
from octoscout.search.realtime import PACKAGE_REPO_MAP

# Severity weights: higher = more impact on score
_SEVERITY_WEIGHTS: dict[str, float] = {
    "crash": 0.15,
    "install": 0.12,
    "wrong_output": 0.10,
    "performance": 0.05,
    "other": 0.03,
}

# Score threshold below which a pair triggers a warning
_WARNING_THRESHOLD = 0.7


# Valid version: starts with digit, contains dots, no words like "unknown", "latest", "main"
_VALID_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:\.\d+)?$")

# Invalid version tokens to reject
_INVALID_VERSIONS = {
    "unknown", "none", "latest", "main", "master", "source", "nightly", "git",
    "git-latest", "tip of tree", "not specified",
}


def _normalize_package_name(name: str) -> str:
    """Normalize package name: lowercase, underscores to hyphens."""
    return name.lower().replace("_", "-")


def _is_valid_version(ver: str) -> bool:
    """Check if a version string is a real semver-like version."""
    if not ver:
        return False
    # Strip leading 'v'
    v = ver.lstrip("v").strip()
    if v.lower() in _INVALID_VERSIONS:
        return False
    # Reject versions with operators (>=, <=, etc.)
    if any(c in v for c in (">=", "<=", "!=", "~=", ">", "<", "^")):
        return False
    return bool(_VALID_VERSION_RE.match(v))


def _clean_version(ver: str) -> str:
    """Clean a version string: strip 'v' prefix, whitespace."""
    return ver.lstrip("v").strip()


def _to_minor(ver: str) -> str:
    """Collapse version to major.minor (e.g. '4.55.1' -> '4.55')."""
    parts = ver.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return ver


def _make_pair_key(pkg_a: str, ver_a: str, pkg_b: str, ver_b: str) -> str:
    """Create a canonical pair key (alphabetically sorted), using minor versions."""
    pkg_a = _normalize_package_name(pkg_a)
    pkg_b = _normalize_package_name(pkg_b)
    ver_a = _to_minor(_clean_version(ver_a))
    ver_b = _to_minor(_clean_version(ver_b))
    a = f"{pkg_a}=={ver_a}"
    b = f"{pkg_b}=={ver_b}"
    if a > b:
        a, b = b, a
    return f"{a}+{b}"


# Build _REPO_TO_PACKAGE by inverting PACKAGE_REPO_MAP (first package wins per repo)
_REPO_TO_PACKAGE: dict[str, str] = {}
for _pkg, _repo in PACKAGE_REPO_MAP.items():
    if _repo not in _REPO_TO_PACKAGE:
        _REPO_TO_PACKAGE[_repo] = _pkg
# Add extra repos not in PACKAGE_REPO_MAP
_REPO_TO_PACKAGE.setdefault("QwenLM/Qwen2.5", "qwen")
_REPO_TO_PACKAGE.setdefault("QwenLM/Qwen3-VL", "qwen-vl-utils")


def _infer_package_from_issue_id(issue_id: str) -> str | None:
    """Infer the primary package name from an issue ID like 'owner/repo#123'."""
    if "#" not in issue_id:
        return None
    repo = issue_id.split("#")[0]
    return _REPO_TO_PACKAGE.get(repo)


def _problem_severity(problem_type: str) -> str:
    """Map problem_type to severity level."""
    if problem_type in ("crash", "install"):
        return "high"
    if problem_type in ("wrong_output",):
        return "medium"
    return "low"


class CompatibilityMatrix:
    """Queryable compatibility matrix built from extracted GitHub issues."""

    def __init__(
        self,
        entries: dict[str, CompatibilityEntry] | None = None,
        single_pkg_issues: list[dict] | None = None,
    ):
        self._entries: dict[str, CompatibilityEntry] = entries or {}
        # Issues with 0 or 1 reported version — can't form pairs but still searchable
        self._single_pkg_issues: list[dict] = single_pkg_issues or []
        # Build package index for fast single-package lookups
        self._pkg_index: dict[str, list[dict]] = defaultdict(list)
        for si in self._single_pkg_issues:
            pkg = si.get("package")
            if pkg:
                self._pkg_index[_normalize_package_name(pkg)].append(si)

    @classmethod
    def build_from_extracted(
        cls,
        extracted_dir: Path | list[Path],
        output_path: Path,
    ) -> CompatibilityMatrix:
        """Build the matrix from all extracted JSONL files.

        Args:
            extracted_dir: A single directory or a list of directories to read
                extracted JSONL files from. When multiple directories are given,
                their data is merged (later dirs can supplement earlier ones).
            output_path: Where to save the built matrix.json.
        """
        # Normalize to list
        dirs = [extracted_dir] if isinstance(extracted_dir, Path) else extracted_dir

        # Read all extracted issues from all directories
        all_issues: list[ExtractedIssueInfo] = []
        seen_ids: set[str] = set()
        for d in dirs:
            if not d.exists():
                continue
            for jsonl_file in d.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                issue = ExtractedIssueInfo.from_dict(json.loads(line))
                                # Deduplicate by issue_id (later dirs override earlier)
                                if issue.issue_id in seen_ids:
                                    continue
                                seen_ids.add(issue.issue_id)
                                all_issues.append(issue)
                            except (json.JSONDecodeError, KeyError):
                                pass

        # Group issues by version pair (only valid versions)
        pair_issues: dict[str, list[ExtractedIssueInfo]] = defaultdict(list)
        single_pkg_issues: list[dict] = []

        for issue in all_issues:
            # Filter to only valid version entries
            valid_versions = {
                _normalize_package_name(pkg): _clean_version(ver)
                for pkg, ver in issue.reported_versions.items()
                if ver and _is_valid_version(ver)
            }

            # Collect issues with 0-1 versions into the single-package index
            if len(valid_versions) < 2:
                if issue.error_message_summary or issue.title:
                    pkg_name = list(valid_versions.keys())[0] if valid_versions else None
                    pkg_ver = list(valid_versions.values())[0] if valid_versions else None

                    # For 0-version issues, infer package from issue_id repo name
                    if pkg_name is None and issue.issue_id:
                        pkg_name = _infer_package_from_issue_id(issue.issue_id)

                    single_pkg_issues.append({
                        "issue_id": issue.issue_id,
                        "title": issue.title,
                        "package": pkg_name,
                        "version": _to_minor(pkg_ver) if pkg_ver else None,
                        "problem_type": issue.problem_type,
                        "severity": _problem_severity(issue.problem_type),
                        "summary": (issue.error_message_summary or issue.title)[:150],
                        "solution": (issue.solution_detail or "")[:150],
                        "has_solution": issue.has_solution,
                    })
                continue

            # Create pairs from all reported package versions
            pkg_names = list(valid_versions.keys())
            for pkg_a, pkg_b in combinations(pkg_names, 2):
                key = _make_pair_key(pkg_a, valid_versions[pkg_a], pkg_b, valid_versions[pkg_b])
                pair_issues[key].append(issue)

            # Also pair each package with python_version if available
            if issue.python_version and _is_valid_version(issue.python_version):
                py_ver = _clean_version(issue.python_version)
                for pkg in pkg_names:
                    key = _make_pair_key(
                        pkg, valid_versions[pkg], "python", py_ver
                    )
                    pair_issues[key].append(issue)

            # Pair with cuda_version if available
            if issue.cuda_version and _is_valid_version(issue.cuda_version):
                cuda_ver = _clean_version(issue.cuda_version)
                for pkg in pkg_names:
                    key = _make_pair_key(
                        pkg, valid_versions[pkg], "cuda", cuda_ver
                    )
                    pair_issues[key].append(issue)

        # Build entries
        entries: dict[str, CompatibilityEntry] = {}
        for key, issues in pair_issues.items():
            total_weight = sum(
                _SEVERITY_WEIGHTS.get(i.problem_type, 0.03) for i in issues
            )
            score = max(0.0, 1.0 - total_weight)

            problems: list[KnownProblem] = []
            for issue in issues:
                problems.append(
                    KnownProblem(
                        summary=issue.error_message_summary or issue.title,
                        severity=_problem_severity(issue.problem_type),
                        solution=issue.solution_detail or "",
                        source_issues=[issue.issue_id],
                    )
                )

            entries[key] = CompatibilityEntry(
                score=score,
                issue_count=len(issues),
                known_problems=problems,
            )

        matrix = cls(entries=entries, single_pkg_issues=single_pkg_issues)
        matrix.save(output_path)
        return matrix

    def save(self, path: Path) -> None:
        """Save matrix to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.1",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "single_pkg_issues": self._single_pkg_issues,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> CompatibilityMatrix:
        """Load matrix from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = {
            k: CompatibilityEntry.from_dict(v)
            for k, v in data.get("entries", {}).items()
        }
        return cls(
            entries=entries,
            single_pkg_issues=data.get("single_pkg_issues", []),
        )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def check(self, env: EnvSnapshot) -> list[CompatibilityWarning]:
        """Given an environment snapshot, return all known compatibility risks."""
        warnings: list[CompatibilityWarning] = []

        # Filter to packages we track
        tracked = {
            pkg: ver
            for pkg, ver in env.installed_packages.items()
            if pkg in PACKAGE_REPO_MAP
        }

        # Check all pairs of tracked packages
        for (pkg_a, ver_a), (pkg_b, ver_b) in combinations(tracked.items(), 2):
            key = _make_pair_key(pkg_a, ver_a, pkg_b, ver_b)
            entry = self._entries.get(key)
            if entry and entry.score < _WARNING_THRESHOLD:
                warnings.append(
                    CompatibilityWarning(
                        packages={pkg_a: ver_a, pkg_b: ver_b},
                        score=entry.score,
                        problems=entry.known_problems,
                        recommendation=self._make_recommendation(entry),
                    )
                )

        # Check packages against python version
        if env.python_version:
            for pkg, ver in tracked.items():
                key = _make_pair_key(pkg, ver, "python", env.python_version)
                entry = self._entries.get(key)
                if entry and entry.score < _WARNING_THRESHOLD:
                    warnings.append(
                        CompatibilityWarning(
                            packages={pkg: ver, "python": env.python_version},
                            score=entry.score,
                            problems=entry.known_problems,
                            recommendation=self._make_recommendation(entry),
                        )
                    )

        # Check packages against CUDA version
        if env.cuda_version:
            for pkg, ver in tracked.items():
                key = _make_pair_key(pkg, ver, "cuda", env.cuda_version)
                entry = self._entries.get(key)
                if entry and entry.score < _WARNING_THRESHOLD:
                    warnings.append(
                        CompatibilityWarning(
                            packages={pkg: ver, "cuda": env.cuda_version},
                            score=entry.score,
                            problems=entry.known_problems,
                            recommendation=self._make_recommendation(entry),
                        )
                    )

        # Check single-package issues for installed packages
        for pkg, ver in tracked.items():
            issues = self.query_package(pkg, ver)
            high_sev = [i for i in issues if i.get("severity") == "high"]
            if high_sev:
                # Convert single-pkg issues into a warning
                problems = [
                    KnownProblem(
                        summary=i.get("summary", ""),
                        severity=i.get("severity", "low"),
                        solution=i.get("solution", ""),
                        source_issues=[i["issue_id"]] if i.get("issue_id") else [],
                    )
                    for i in high_sev[:5]
                ]
                warnings.append(
                    CompatibilityWarning(
                        packages={pkg: ver},
                        score=max(0.0, 1.0 - 0.15 * len(high_sev)),
                        problems=problems,
                        recommendation=problems[0].solution if problems[0].solution else
                            f"{len(high_sev)} known high-severity issue(s) for {pkg}=={ver}.",
                    )
                )

        # Sort by score ascending (most risky first)
        warnings.sort(key=lambda w: w.score)
        return warnings

    def query_pair(
        self, pkg_a: str, ver_a: str, pkg_b: str, ver_b: str
    ) -> PairResult | None:
        """Query a specific version pair."""
        key = _make_pair_key(pkg_a, ver_a, pkg_b, ver_b)
        entry = self._entries.get(key)
        if entry is None:
            return None
        return PairResult(
            score=entry.score,
            issue_count=entry.issue_count,
            problems=entry.known_problems,
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def top_risks(
        self, packages: list[str] | None = None, n: int = 10,
    ) -> list[tuple[str, CompatibilityEntry]]:
        """Return the top N riskiest version pairs, optionally filtered to given packages.

        Args:
            packages: If provided, only return pairs involving at least one of these packages.
            n: Maximum number of results.

        Returns:
            List of (pair_key, entry) sorted by score ascending (riskiest first).
        """
        candidates = []
        pkg_set = {_normalize_package_name(p) for p in packages} if packages else None

        for key, entry in self._entries.items():
            if pkg_set:
                # Check if any package in the key matches
                parts = key.split("+")
                key_pkgs = set()
                for part in parts:
                    if "==" in part:
                        key_pkgs.add(part.split("==")[0])
                if not key_pkgs & pkg_set:
                    continue
            candidates.append((key, entry))

        candidates.sort(key=lambda x: x[1].score)
        return candidates[:n]

    def query_package(
        self, pkg: str, ver: str | None = None, limit: int = 20,
    ) -> list[dict]:
        """Query single-package issues by package name, optionally filtered by version."""
        pkg_norm = _normalize_package_name(pkg)
        results = self._pkg_index.get(pkg_norm, [])
        if ver:
            minor = _to_minor(_clean_version(ver))
            results = [r for r in results if r.get("version") == minor]
        return results[:limit]

    def search_issues(self, keyword: str, limit: int = 20) -> list[dict]:
        """Search single-package issues by keyword in summary and solution."""
        kw = keyword.lower()
        results = []
        for si in self._single_pkg_issues:
            text = f"{si.get('summary', '')} {si.get('solution', '')} {si.get('title', '')}".lower()
            if kw in text:
                results.append(si)
                if len(results) >= limit:
                    break
        return results

    @staticmethod
    def _make_recommendation(entry: CompatibilityEntry) -> str:
        """Generate a human-readable recommendation from an entry."""
        solutions = [
            p.solution for p in entry.known_problems if p.solution
        ]
        if solutions:
            return solutions[0]
        if entry.score < 0.3:
            return "This version combination has significant known issues. Consider changing versions."
        return "Known compatibility issues exist. Check the details."

"""L1 Version-aware filtering: filter and rank search results by version relevance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from octoscout.models import EnvSnapshot, GitHubIssueRef

# Match version patterns like 4.55.0, v4.55, ==4.55.0, >=4.53
_VERSION_RE = re.compile(
    r"(?:(?:==|>=|<=|!=|~=|>|<|v)\s*)?(\d+\.\d+(?:\.\d+)?)"
)


@dataclass
class VersionContext:
    """Version mentions extracted from an issue."""

    versions: dict[str, list[str]]  # package_name -> list of version strings
    raw_mentions: list[str]  # all version strings found


def extract_versions_from_text(text: str) -> list[str]:
    """Extract version-like strings from text."""
    return _VERSION_RE.findall(text)


def filter_by_version(
    issues: list[GitHubIssueRef],
    env: EnvSnapshot,
    target_packages: set[str] | None = None,
) -> list[GitHubIssueRef]:
    """Filter issues by version relevance using L1 heuristics.

    L1 filtering uses simple version string matching:
    - Issues mentioning the user's exact version get a relevance boost
    - Issues mentioning a completely different major version get filtered
    - Issues with no version info are kept (neutral)

    Returns issues sorted by relevance_score (descending).
    """
    if not env.installed_packages:
        return issues

    scored: list[GitHubIssueRef] = []
    for issue in issues:
        text = f"{issue.title} {issue.snippet}"
        mentioned_versions = extract_versions_from_text(text)

        score = _compute_version_score(
            mentioned_versions, env.installed_packages, target_packages
        )
        issue.relevance_score = score
        scored.append(issue)

    # Sort by relevance (keep issues with score >= -0.3)
    scored = [i for i in scored if i.relevance_score >= -0.3]
    scored.sort(key=lambda i: i.relevance_score, reverse=True)
    return scored


def _compute_version_score(
    mentioned_versions: list[str],
    installed: dict[str, str],
    target_packages: set[str] | None,
) -> float:
    """Score an issue based on version mentions vs installed versions.

    Scoring:
    - Exact match on a target package version: +0.5
    - Same major.minor: +0.3
    - Same major: +0.1
    - Different major: -0.3
    - No version mentions: 0.0 (neutral)
    """
    if not mentioned_versions:
        return 0.0

    best_score = 0.0
    packages_to_check = target_packages or set(installed.keys())

    for pkg in packages_to_check:
        if pkg not in installed:
            continue
        installed_ver = installed[pkg]
        installed_parts = _parse_version(installed_ver)
        if not installed_parts:
            continue

        for ver_str in mentioned_versions:
            ver_parts = _parse_version(ver_str)
            if not ver_parts:
                continue

            if installed_parts == ver_parts:
                best_score = max(best_score, 0.5)
            elif installed_parts[:2] == ver_parts[:2]:
                best_score = max(best_score, 0.3)
            elif installed_parts[0] == ver_parts[0]:
                best_score = max(best_score, 0.1)
            else:
                best_score = max(best_score, -0.3)

    return best_score


def _parse_version(ver_str: str) -> tuple[int, ...] | None:
    """Parse a version string into a tuple of integers."""
    try:
        parts = ver_str.strip().split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# L2 Version Semantics: range-aware filtering
# ---------------------------------------------------------------------------


def parse_version_range(range_str: str):
    """Parse a version range string like '>=4.53,<4.56' into a SpecifierSet.

    Returns a packaging.specifiers.SpecifierSet or None if parsing fails.
    """
    if not range_str:
        return None
    try:
        from packaging.specifiers import SpecifierSet
        return SpecifierSet(range_str)
    except Exception:
        return None


def filter_by_version_range(
    issues: list[GitHubIssueRef],
    env: EnvSnapshot,
    extracted_data: dict[str, object] | None = None,
) -> list[GitHubIssueRef]:
    """L2 version filtering: use extracted affected_version_range for precise matching.

    When extracted data is available (keyed by "repo#number"), checks whether the
    user's installed version falls within the affected range. Falls back to L1
    scoring when no extracted data is available.

    Args:
        issues: Issues to filter/rank.
        env: User's environment snapshot.
        extracted_data: Dict mapping "owner/repo#number" to ExtractedIssueInfo objects.

    Returns:
        Issues sorted by relevance_score (descending).
    """
    if not extracted_data:
        return filter_by_version(issues, env)

    try:
        from packaging.version import Version
    except ImportError:
        return filter_by_version(issues, env)

    scored: list[GitHubIssueRef] = []

    for issue in issues:
        key = f"{issue.repo}#{issue.number}"
        extracted = extracted_data.get(key)

        if extracted is None:
            # No extracted data — fall back to L1
            text = f"{issue.title} {issue.snippet}"
            mentioned = extract_versions_from_text(text)
            issue.relevance_score = _compute_version_score(
                mentioned, env.installed_packages, None
            )
            scored.append(issue)
            continue

        # Use extracted version range for precise matching
        range_str = getattr(extracted, "affected_version_range", None)
        reported_versions = getattr(extracted, "reported_versions", {}) or {}
        spec = parse_version_range(range_str) if range_str else None

        best_score = 0.0
        matched = False

        if spec and reported_versions:
            # Check if user's installed version falls in the affected range
            for pkg, _ in reported_versions.items():
                user_ver = env.installed_packages.get(pkg)
                if not user_ver:
                    continue
                try:
                    if Version(user_ver) in spec:
                        best_score = max(best_score, 0.7)
                        matched = True
                    else:
                        best_score = max(best_score, -0.3)
                except Exception:
                    pass

        if not matched and reported_versions:
            # Fall back to exact version comparison from extracted data
            for pkg, ver in reported_versions.items():
                user_ver = env.installed_packages.get(pkg)
                if not user_ver:
                    continue
                user_parts = _parse_version(user_ver)
                issue_parts = _parse_version(ver)
                if user_parts and issue_parts:
                    if user_parts == issue_parts:
                        best_score = max(best_score, 0.5)
                    elif user_parts[:2] == issue_parts[:2]:
                        best_score = max(best_score, 0.3)
                    elif user_parts[0] == issue_parts[0]:
                        best_score = max(best_score, 0.1)

        issue.relevance_score = best_score
        scored.append(issue)

    scored = [i for i in scored if i.relevance_score >= -0.3]
    scored.sort(key=lambda i: i.relevance_score, reverse=True)
    return scored

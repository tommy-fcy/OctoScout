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

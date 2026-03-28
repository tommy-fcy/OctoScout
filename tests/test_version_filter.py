"""Tests for version filtering."""

from octoscout.models import EnvSnapshot, GitHubIssueRef
from octoscout.search.version_filter import extract_versions_from_text, filter_by_version


def test_extract_versions():
    text = "This bug was introduced in transformers==4.55.0 and fixed in v4.56.1"
    versions = extract_versions_from_text(text)
    assert "4.55.0" in versions
    assert "4.56.1" in versions


def test_filter_boosts_matching_version():
    env = EnvSnapshot(installed_packages={"transformers": "4.55.0"})
    issues = [
        GitHubIssueRef(
            repo="huggingface/transformers", number=1, title="Bug in 4.55.0",
            url="https://github.com/huggingface/transformers/issues/1",
            snippet="This affects transformers==4.55.0",
        ),
        GitHubIssueRef(
            repo="huggingface/transformers", number=2, title="Bug in 3.0.0",
            url="https://github.com/huggingface/transformers/issues/2",
            snippet="This affects transformers==3.0.0",
        ),
    ]

    filtered = filter_by_version(issues, env, target_packages={"transformers"})

    # Issue 1 should rank higher (exact version match)
    assert len(filtered) >= 1
    assert filtered[0].number == 1


def test_filter_keeps_no_version_issues():
    env = EnvSnapshot(installed_packages={"transformers": "4.55.0"})
    issues = [
        GitHubIssueRef(
            repo="huggingface/transformers", number=1, title="Some general bug",
            url="https://github.com/huggingface/transformers/issues/1",
            snippet="No version mentioned here",
        ),
    ]

    filtered = filter_by_version(issues, env)
    assert len(filtered) == 1  # kept with neutral score

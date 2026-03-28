"""Tests for environment snapshot collection."""

from octoscout.diagnosis.env_snapshot import collect_env_snapshot


def test_collect_env_snapshot_returns_python_version():
    snap = collect_env_snapshot()
    assert snap.python_version is not None
    # Should be a valid version string like "3.10.12"
    parts = snap.python_version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_collect_env_snapshot_returns_os_info():
    snap = collect_env_snapshot()
    assert snap.os_info is not None
    assert len(snap.os_info) > 0


def test_collect_env_snapshot_returns_packages():
    snap = collect_env_snapshot()
    # pip itself should always be installed
    assert len(snap.installed_packages) > 0


def test_format_for_llm():
    snap = collect_env_snapshot()
    text = snap.format_for_llm()
    assert "Environment Snapshot" in text
    assert "Python:" in text

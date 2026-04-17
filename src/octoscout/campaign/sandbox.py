"""Sandbox environment creation and verification for campaign diagnoses."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from octoscout.campaign.models import (
    CampaignIssue,
    DiagnosisRecord,
    VerificationRecord,
    append_jsonl,
    update_worklog,
)


@dataclass
class SandboxEnv:
    """A temporary isolated Python environment."""

    path: Path
    python: str
    pip: str


@dataclass
class RunResult:
    """Result of running a command in a sandbox."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str


def _find_uv() -> str | None:
    """Find the uv executable path."""
    # Check common install locations
    candidates = [
        "uv",
        str(Path.home() / ".local" / "bin" / "uv"),
        str(Path.home() / ".local" / "bin" / "uv.exe"),
        str(Path.home() / ".cargo" / "bin" / "uv"),
    ]
    for candidate in candidates:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            continue
    return None


_UV_PATH: str | None = None


def _get_uv() -> str | None:
    """Get the uv executable path (cached)."""
    global _UV_PATH
    if _UV_PATH is None:
        _UV_PATH = _find_uv() or ""
    return _UV_PATH or None


def _has_uv() -> bool:
    """Check if uv is available."""
    return _get_uv() is not None


def create_sandbox(base_dir: Path, name: str) -> SandboxEnv:
    """Create an isolated venv. Uses uv if available for speed."""
    venv_path = base_dir / name
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    uv = _get_uv()
    if uv:
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(venv_path)],
            capture_output=True, timeout=30, check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True, timeout=60, check=True,
        )

    # Determine python/pip paths (cross-platform)
    if sys.platform == "win32":
        python = str(venv_path / "Scripts" / "python.exe")
        pip = str(venv_path / "Scripts" / "pip.exe")
    else:
        python = str(venv_path / "bin" / "python")
        pip = str(venv_path / "bin" / "pip")

    return SandboxEnv(path=venv_path, python=python, pip=pip)


_PYPI_MIRROR = os.environ.get("OCTOSCOUT_PYPI_MIRROR", "https://mirrors.cloud.tencent.com/pypi/simple/")


def install_packages(
    sandbox: SandboxEnv,
    packages: dict[str, str],
    timeout: int = 300,
) -> RunResult:
    """Install specific package versions in the sandbox."""
    if not packages:
        return RunResult(success=True, exit_code=0, stdout="", stderr="No packages to install")

    reqs = [f"{pkg}=={ver}" for pkg, ver in packages.items()]

    uv = _get_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", sandbox.python,
               "--index-url", _PYPI_MIRROR] + reqs
    else:
        cmd = [sandbox.pip, "install", "-i", _PYPI_MIRROR] + reqs

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return RunResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout[-2000:] if proc.stdout else "",
            stderr=proc.stderr[-2000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(success=False, exit_code=-1, stdout="", stderr="Install timed out")


def check_imports(sandbox: SandboxEnv, packages: list[str], timeout: int = 60) -> RunResult:
    """Verify packages can be imported in the sandbox."""
    import_stmts = "; ".join(f"import {pkg.replace('-', '_')}" for pkg in packages)
    script = f'{import_stmts}; print("OK")'

    try:
        proc = subprocess.run(
            [sandbox.python, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return RunResult(
            success=proc.returncode == 0 and "OK" in proc.stdout,
            exit_code=proc.returncode,
            stdout=proc.stdout[-1000:] if proc.stdout else "",
            stderr=proc.stderr[-1000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(success=False, exit_code=-1, stdout="", stderr="Import check timed out")


def run_script(sandbox: SandboxEnv, script: str, timeout: int = 60) -> RunResult:
    """Run a Python script in the sandbox."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sandbox.python, script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return RunResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout[-2000:] if proc.stdout else "",
            stderr=proc.stderr[-2000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(success=False, exit_code=-1, stdout="", stderr="Script timed out")
    finally:
        Path(script_path).unlink(missing_ok=True)


def check_dependency_resolution(packages: dict[str, str], timeout: int = 30) -> RunResult:
    """Quick check: can pip resolve these versions together (dry-run)?"""
    if not packages:
        return RunResult(success=True, exit_code=0, stdout="", stderr="No packages")

    reqs = [f"{pkg}=={ver}" for pkg, ver in packages.items()]

    uv = _get_uv()
    if uv:
        cmd = [uv, "pip", "install", "--dry-run", "--python", sys.executable] + reqs
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run"] + reqs

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return RunResult(
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout[-1000:] if proc.stdout else "",
            stderr=proc.stderr[-1000:] if proc.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(success=False, exit_code=-1, stdout="", stderr="Dependency check timed out")


def cleanup_sandbox(sandbox: SandboxEnv) -> None:
    """Remove the sandbox directory."""
    if sandbox.path.exists():
        shutil.rmtree(sandbox.path, ignore_errors=True)


def _has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10,
        )
        return "True" in proc.stdout
    except Exception:
        return False


def _extract_versions_from_text(text: str) -> dict[str, str]:
    """Extract package==version pairs from diagnosis/issue text.

    Only recognizes known ML ecosystem packages to avoid false matches
    on config values like `timeout=0.05` or `max_length==512`.
    """
    import re

    from octoscout.search.realtime import PACKAGE_REPO_MAP

    known_packages = set(PACKAGE_REPO_MAP.keys())
    # Also recognize common aliases (hyphen/underscore variants)
    known_packages |= {p.replace("_", "-") for p in known_packages}
    known_packages |= {p.replace("-", "_") for p in known_packages}

    pattern = re.compile(r"([a-zA-Z][\w.-]*?)(?:==|\s+version:?\s*|>=)(\d+\.\d+(?:\.\d+)?)")
    result = {}
    for pkg, ver in pattern.findall(text):
        normalized = pkg.lower().replace("-", "_")
        if normalized in known_packages or pkg.lower() in known_packages:
            # Store using the canonical package name from PACKAGE_REPO_MAP if possible
            canonical = normalized if normalized in PACKAGE_REPO_MAP else pkg.lower()
            result[canonical] = ver
    return result


async def verify_diagnosis(
    record: DiagnosisRecord,
    issue: CampaignIssue,
    campaign_dir: Path,
    level: str = "quick",
    timeout: int = 300,
) -> VerificationRecord:
    """Verify a diagnosis at the specified level.

    Levels:
      quick: dependency resolution check only (no venv created)
      import: create venv, install, verify imports
      reproduce: create venv, install broken -> run script -> install fixed -> run script
    """
    now = datetime.now(timezone.utc).isoformat()
    sandbox_dir = campaign_dir / "sandboxes"

    # Determine package versions
    broken_env = _extract_versions_from_text(issue.body)
    fixed_env = dict(record.suggested_versions) if record.suggested_versions else None

    # GPU check
    if issue.env_category == "gpu_required" and not _has_gpu():
        result = VerificationRecord(
            repo=issue.repo,
            number=issue.number,
            verified_at=now,
            level=level,
            broken_env=broken_env,
            fixed_env=fixed_env,
            broken_result="skip_no_gpu",
            reproduction_log="Skipped: issue requires GPU but none available",
            verified=False,
        )
        _save_verification(result, issue, campaign_dir)
        return result

    if level == "quick":
        return await _verify_quick(record, issue, campaign_dir, broken_env, fixed_env, now)
    elif level == "import":
        return await _verify_import(record, issue, campaign_dir, sandbox_dir, broken_env, fixed_env, now, timeout)
    else:  # reproduce
        return await _verify_reproduce(record, issue, campaign_dir, sandbox_dir, broken_env, fixed_env, now, timeout)


async def _verify_quick(record, issue, campaign_dir, broken_env, fixed_env, now):
    """Quick verification: dependency resolution check."""
    log_parts = []

    if fixed_env:
        result = check_dependency_resolution(fixed_env)
        log_parts.append(f"Fixed env resolution: {'OK' if result.success else 'CONFLICT'}")
        if result.stderr:
            log_parts.append(result.stderr[:500])

        return _make_record(
            issue, now, "quick", broken_env, fixed_env,
            broken_result="not_tested",
            fixed_result="fix_confirmed" if result.success else "install_failed",
            log="\n".join(log_parts),
            verified=result.success,
            campaign_dir=campaign_dir,
        )

    # No pinned versions — check if fix_actions has git URLs (cannot dry-run)
    git_actions = [a for a in record.fix_actions if "git+" in a or "github.com" in a]
    if git_actions:
        return _make_record(
            issue, now, "quick", broken_env, fixed_env,
            broken_result="not_tested", fixed_result="skip_git_install",
            log=f"Fix requires git install: {git_actions[0]}. Use --level import to verify.",
            verified=False, campaign_dir=campaign_dir,
        )

    return _make_record(
        issue, now, "quick", broken_env, fixed_env,
        broken_result="not_tested", fixed_result=None,
        log="No suggested versions to verify",
        verified=False, campaign_dir=campaign_dir,
    )


async def _verify_import(record, issue, campaign_dir, sandbox_dir, broken_env, fixed_env, now, timeout):
    """Import-level verification: create venv, install, verify imports."""
    sandbox = None
    log_parts = []

    try:
        sandbox = create_sandbox(sandbox_dir, f"{issue.repo.replace('/', '_')}_{issue.number}")
        packages_to_check = list((fixed_env or broken_env).keys())

        if fixed_env:
            install_result = install_packages(sandbox, fixed_env, timeout=timeout)
            log_parts.append(f"Install fixed env: {'OK' if install_result.success else 'FAILED'}")
            if not install_result.success:
                log_parts.append(install_result.stderr[:500])
                return _make_record(
                    issue, now, "import", broken_env, fixed_env,
                    broken_result="not_tested", fixed_result="install_failed",
                    log="\n".join(log_parts), verified=False, campaign_dir=campaign_dir,
                )

            import_result = check_imports(sandbox, packages_to_check)
            log_parts.append(f"Import check: {'OK' if import_result.success else 'FAILED'}")
            if import_result.stderr:
                log_parts.append(import_result.stderr[:500])

            return _make_record(
                issue, now, "import", broken_env, fixed_env,
                broken_result="not_tested",
                fixed_result="fix_confirmed" if import_result.success else "still_broken",
                log="\n".join(log_parts),
                verified=import_result.success,
                campaign_dir=campaign_dir,
            )

        return _make_record(
            issue, now, "import", broken_env, fixed_env,
            broken_result="not_tested", fixed_result=None,
            log="No suggested versions to verify",
            verified=False, campaign_dir=campaign_dir,
        )

    except Exception as e:
        return _make_record(
            issue, now, "import", broken_env, fixed_env,
            broken_result="not_tested", fixed_result=None,
            log=f"Error: {e}", verified=False, campaign_dir=campaign_dir,
        )
    finally:
        if sandbox:
            cleanup_sandbox(sandbox)


async def _verify_reproduce(record, issue, campaign_dir, sandbox_dir, broken_env, fixed_env, now, timeout):
    """Full reproduction: install broken, run script, install fixed, run script."""
    sandbox = None
    log_parts = []
    code = issue.extracted_code_snippet

    if not code:
        return _make_record(
            issue, now, "reproduce", broken_env, fixed_env,
            broken_result="not_tested", fixed_result=None,
            log="No code snippet available for reproduction",
            verified=False, campaign_dir=campaign_dir,
        )

    try:
        sandbox = create_sandbox(sandbox_dir, f"{issue.repo.replace('/', '_')}_{issue.number}")

        # Phase 1: Install broken env and reproduce error
        broken_result_status = "no_error"
        if broken_env:
            install_r = install_packages(sandbox, broken_env, timeout=timeout)
            log_parts.append(f"Install broken env: {'OK' if install_r.success else 'FAILED'}")
            if not install_r.success:
                broken_result_status = "install_failed"
                log_parts.append(install_r.stderr[:500])
            else:
                run_r = run_script(sandbox, code, timeout=60)
                if not run_r.success:
                    broken_result_status = "error_reproduced"
                    log_parts.append(f"Broken env: error reproduced (exit={run_r.exit_code})")
                    log_parts.append(run_r.stderr[:500])
                else:
                    log_parts.append("Broken env: no error (issue may not be reproducible)")

        # Phase 2: Install fixed env and verify fix
        cleanup_sandbox(sandbox)
        sandbox = create_sandbox(sandbox_dir, f"{issue.repo.replace('/', '_')}_{issue.number}_fixed")

        fixed_result_status = None
        if fixed_env:
            install_r = install_packages(sandbox, fixed_env, timeout=timeout)
            log_parts.append(f"Install fixed env: {'OK' if install_r.success else 'FAILED'}")
            if not install_r.success:
                fixed_result_status = "install_failed"
                log_parts.append(install_r.stderr[:500])
            else:
                run_r = run_script(sandbox, code, timeout=60)
                if run_r.success:
                    fixed_result_status = "fix_confirmed"
                    log_parts.append("Fixed env: script ran successfully")
                else:
                    fixed_result_status = "still_broken"
                    log_parts.append(f"Fixed env: still broken (exit={run_r.exit_code})")
                    log_parts.append(run_r.stderr[:500])

        verified = (
            broken_result_status == "error_reproduced"
            and fixed_result_status == "fix_confirmed"
        )

        return _make_record(
            issue, now, "reproduce", broken_env, fixed_env,
            broken_result=broken_result_status,
            fixed_result=fixed_result_status,
            log="\n".join(log_parts),
            verified=verified,
            campaign_dir=campaign_dir,
        )

    except Exception as e:
        return _make_record(
            issue, now, "reproduce", broken_env, fixed_env,
            broken_result="not_tested", fixed_result=None,
            log=f"Error: {e}", verified=False, campaign_dir=campaign_dir,
        )
    finally:
        if sandbox:
            cleanup_sandbox(sandbox)


def _make_record(issue, now, level, broken_env, fixed_env,
                 broken_result, fixed_result, log, verified, campaign_dir):
    """Helper to create a VerificationRecord and persist it."""
    rec = VerificationRecord(
        repo=issue.repo,
        number=issue.number,
        verified_at=now,
        level=level,
        broken_env=broken_env,
        fixed_env=fixed_env,
        broken_result=broken_result,
        fixed_result=fixed_result,
        reproduction_log=log,
        verified=verified,
    )
    _save_verification(rec, issue, campaign_dir)
    return rec


def _save_verification(record, issue, campaign_dir):
    """Save verification record and update work log."""
    append_jsonl(campaign_dir / "verified.jsonl", record)
    update_worklog(campaign_dir, issue, verification=record)

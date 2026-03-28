"""Parse Python tracebacks into structured representations."""

from __future__ import annotations

import re
import site
import sys
from pathlib import PurePosixPath, PureWindowsPath

from octoscout.models import ParsedTraceback, StackFrame

# Regex patterns for Python traceback parsing
_TB_HEADER = re.compile(r"Traceback \(most recent call last\):")
_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>.+?)", line (?P<line>\d+), in (?P<func>.+)$'
)
_CODE_RE = re.compile(r"^\s{4,}\S")  # indented code line following a frame
_EXCEPTION_RE = re.compile(r"^(?P<type>[\w.]+):\s*(?P<msg>.+)$")
_EXCEPTION_ONLY_RE = re.compile(r"^(?P<type>[\w.]+)\s*$")

# Known site-packages patterns
_SITE_PACKAGES_RE = re.compile(r"[/\\]site-packages[/\\](?P<pkg>[^/\\]+)")
_STDLIB_PATHS = {str(p) for p in (getattr(sys, "stdlib_module_names", set()))}


def parse_traceback(text: str) -> ParsedTraceback:
    """Parse a Python traceback string into a structured ParsedTraceback."""
    lines = text.strip().splitlines()

    frames: list[StackFrame] = []
    exception_type = ""
    exception_message = ""

    i = 0
    # Find the start of the traceback
    while i < len(lines):
        if _TB_HEADER.match(lines[i].strip()):
            i += 1
            break
        i += 1

    # Parse frames
    while i < len(lines):
        line = lines[i]
        frame_match = _FRAME_RE.match(line)
        if frame_match:
            file_path = frame_match.group("file")
            line_no = int(frame_match.group("line"))
            func = frame_match.group("func")

            code = None
            if i + 1 < len(lines) and _CODE_RE.match(lines[i + 1]):
                code = lines[i + 1].strip()
                i += 1

            package = _extract_package_from_path(file_path)
            frames.append(StackFrame(
                file=file_path,
                line=line_no,
                function=func,
                code=code,
                package=package,
            ))
            i += 1
            continue

        # Check for exception line
        exc_match = _EXCEPTION_RE.match(line.strip())
        if exc_match:
            exception_type = exc_match.group("type")
            exception_message = exc_match.group("msg")
            break
        exc_only = _EXCEPTION_ONLY_RE.match(line.strip())
        if exc_only and not _FRAME_RE.match(line):
            exception_type = exc_only.group("type")
            exception_message = ""
            break

        i += 1

    # Derive metadata
    involved_packages: set[str] = set()
    for f in frames:
        if f.package:
            involved_packages.add(f.package)

    root_package = frames[-1].package if frames else None
    is_user_code = _is_user_code(frames[-1].file) if frames else True

    return ParsedTraceback(
        exception_type=exception_type,
        exception_message=exception_message,
        frames=frames,
        root_package=root_package,
        is_user_code=is_user_code,
        involved_packages=involved_packages,
    )


def _extract_package_from_path(file_path: str) -> str | None:
    """Extract the top-level package name from a file path."""
    m = _SITE_PACKAGES_RE.search(file_path)
    if m:
        pkg = m.group("pkg")
        # Normalize: strip .py extension, strip version suffixes from dist-info
        pkg = re.sub(r"\.py$", "", pkg)
        pkg = re.sub(r"-\d+.*$", "", pkg)
        return pkg.replace("-", "_").lower()
    return None


def _is_user_code(file_path: str) -> bool:
    """Heuristic: is this file user code (not in site-packages or stdlib)?"""
    if "site-packages" in file_path:
        return False
    # Check against known site paths
    site_paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    for sp in site_paths:
        if file_path.startswith(sp):
            return False
    return True

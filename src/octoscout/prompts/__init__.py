"""Prompt management: load prompt templates from .md files.

Prompts are stored as Markdown files in this directory. This makes them
easy to edit, version-control, and A/B test without touching Python code.

Usage:
    from octoscout.prompts import load_prompt
    system = load_prompt("diagnosis_system")
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent
_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .md extension).

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    if name in _cache:
        return _cache[name]

    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Available prompts: {', '.join(p.stem for p in _PROMPT_DIR.glob('*.md'))}"
        )

    text = path.read_text(encoding="utf-8").strip()
    _cache[name] = text
    return text

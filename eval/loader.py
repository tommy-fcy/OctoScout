"""Load eval cases from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from eval.models import EvalCase

CASES_DIR = Path(__file__).parent / "cases"


def load_all_cases(category: str | None = None) -> list[EvalCase]:
    """Load all eval cases, optionally filtered by category subdirectory."""
    cases: list[EvalCase] = []

    if category:
        search_dir = CASES_DIR / category
        if not search_dir.is_dir():
            raise FileNotFoundError(f"Category directory not found: {search_dir}")
        yaml_files = sorted(search_dir.glob("*.yaml"))
    else:
        yaml_files = sorted(CASES_DIR.rglob("*.yaml"))

    for path in yaml_files:
        case = load_case(path)
        if case is not None:
            cases.append(case)

    return cases


def load_case(path: Path) -> EvalCase | None:
    """Load a single eval case from a YAML file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        return EvalCase.from_yaml(data)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None

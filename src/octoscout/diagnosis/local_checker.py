"""Local diagnostic checks: API signature verification, version analysis."""

from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import dataclass


@dataclass
class SignatureCheckResult:
    matches: bool
    function_path: str
    expected_params: list[str]
    unexpected_args: list[str]
    missing_args: list[str]
    message: str


def check_api_signature(
    function_path: str,
    called_kwargs: list[str] | None = None,
) -> SignatureCheckResult | None:
    """Inspect a function's signature and compare with the called arguments.

    Args:
        function_path: Dotted path like "transformers.AutoModel.from_pretrained"
        called_kwargs: Keyword argument names that were used in the call

    Returns:
        SignatureCheckResult or None if the function cannot be imported.
    """
    called_kwargs = called_kwargs or []

    try:
        obj = _import_dotted_path(function_path)
    except Exception:
        return None

    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None

    param_names = list(sig.parameters.keys())
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if has_var_keyword:
        # Function accepts **kwargs — any keyword arg is technically valid
        return SignatureCheckResult(
            matches=True,
            function_path=function_path,
            expected_params=param_names,
            unexpected_args=[],
            missing_args=[],
            message=f"{function_path} accepts **kwargs, so any keyword argument is valid.",
        )

    unexpected = [k for k in called_kwargs if k not in param_names]
    matches = len(unexpected) == 0

    return SignatureCheckResult(
        matches=matches,
        function_path=function_path,
        expected_params=param_names,
        unexpected_args=unexpected,
        missing_args=[],
        message=(
            f"{function_path} signature matches the call."
            if matches
            else f"{function_path} does not accept: {', '.join(unexpected)}. "
            f"Valid params: {', '.join(param_names)}"
        ),
    )


def extract_function_and_arg_from_typeerror(message: str) -> tuple[str | None, str | None]:
    """Extract function name and unexpected argument from a TypeError message.

    Example:
        "Qwen2VLForConditionalGeneration.__init__() got an unexpected keyword argument 'trust_remote_code'"
        -> ("Qwen2VLForConditionalGeneration.__init__", "trust_remote_code")
    """
    m = re.match(
        r"(\S+)\(\) got an unexpected keyword argument '(\w+)'",
        message.strip(),
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


def _import_dotted_path(path: str):
    """Import and return the object at a dotted path like 'pkg.mod.Class.method'."""
    import warnings

    parts = path.split(".")
    obj = None
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = importlib.import_module(module_path)
            for attr_name in parts[i:]:
                obj = getattr(obj, attr_name)
            return obj
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Cannot import '{path}'")

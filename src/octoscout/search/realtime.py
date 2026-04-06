"""Realtime GitHub search strategy: build search queries from traceback context."""

from __future__ import annotations

from dataclasses import dataclass

from octoscout.models import EnvSnapshot, GitHubIssueRef, ParsedTraceback

# Mapping of common PyPI package names to GitHub repos
PACKAGE_REPO_MAP: dict[str, str] = {
    "transformers": "huggingface/transformers",
    "torch": "pytorch/pytorch",
    "torchvision": "pytorch/vision",
    "torchaudio": "pytorch/audio",
    "vllm": "vllm-project/vllm",
    "peft": "huggingface/peft",
    "accelerate": "huggingface/accelerate",
    "datasets": "huggingface/datasets",
    "tokenizers": "huggingface/tokenizers",
    "safetensors": "huggingface/safetensors",
    "deepspeed": "microsoft/DeepSpeed",
    "bitsandbytes": "bitsandbytes-foundation/bitsandbytes",
    "auto_gptq": "AutoGPTQ/AutoGPTQ",
    "flash_attn": "Dao-AILab/flash-attention",
    "xformers": "facebookresearch/xformers",
    "triton": "triton-lang/triton",
    "numpy": "numpy/numpy",
    "scipy": "scipy/scipy",
    "pandas": "pandas-dev/pandas",
    "scikit_learn": "scikit-learn/scikit-learn",
    "sklearn": "scikit-learn/scikit-learn",
    "tensorflow": "tensorflow/tensorflow",
    "keras": "keras-team/keras",
    "jax": "jax-ml/jax",
    "flax": "google/flax",
    "diffusers": "huggingface/diffusers",
    "trl": "huggingface/trl",
    "llamafactory": "hiyouga/LLaMA-Factory",
    "llmtuner": "hiyouga/LLaMA-Factory",
    "sentence_transformers": "UKPLab/sentence-transformers",
    "langchain": "langchain-ai/langchain",
    "llamacpp": "ggml-org/llama.cpp",
    "gguf": "ggml-org/llama.cpp",
    "qwen_vl_utils": "QwenLM/Qwen2.5-VL",
}


@dataclass
class SearchQuery:
    """A single search query to execute against GitHub."""

    query: str
    repo: str | None = None
    rationale: str = ""


def build_search_queries(
    tb: ParsedTraceback,
    env: EnvSnapshot | None = None,
    extra_repos: list[str] | None = None,
) -> list[SearchQuery]:
    """Build a set of GitHub search queries from a parsed traceback.

    Generates multiple query strategies:
    1. Exact error message search in the root package's repo
    2. Exception type + key terms in related repos
    3. Version-specific queries if env info is available
    """
    queries: list[SearchQuery] = []

    # Determine target repos
    target_repos: list[str] = []
    if tb.root_package and tb.root_package in PACKAGE_REPO_MAP:
        target_repos.append(PACKAGE_REPO_MAP[tb.root_package])
    for pkg in tb.involved_packages:
        if pkg in PACKAGE_REPO_MAP:
            repo = PACKAGE_REPO_MAP[pkg]
            if repo not in target_repos:
                target_repos.append(repo)
    if extra_repos:
        for r in extra_repos:
            if r not in target_repos:
                target_repos.append(r)

    # Strategy 1: Exact error message in root repo
    error_snippet = _truncate_error_message(tb.exception_message, max_len=80)
    if error_snippet:
        for repo in target_repos[:2]:
            queries.append(SearchQuery(
                query=f'"{error_snippet}"',
                repo=repo,
                rationale=f"Exact error message search in {repo}",
            ))

    # Strategy 2: Exception type + key class/function names
    key_terms = _extract_key_terms(tb)
    if key_terms:
        term_str = " ".join(key_terms[:4])
        for repo in target_repos[:3]:
            queries.append(SearchQuery(
                query=f"{tb.exception_type.split('.')[-1]} {term_str}",
                repo=repo,
                rationale=f"Exception type + key terms in {repo}",
            ))

    # Strategy 3: Version-specific search
    if env and env.installed_packages:
        for pkg in tb.involved_packages:
            if pkg in env.installed_packages and pkg in PACKAGE_REPO_MAP:
                version = env.installed_packages[pkg]
                repo = PACKAGE_REPO_MAP[pkg]
                queries.append(SearchQuery(
                    query=f'"{version}" {tb.exception_type.split(".")[-1]}',
                    repo=repo,
                    rationale=f"Version-specific search for {pkg}=={version}",
                ))

    # Deduplicate
    seen = set()
    unique: list[SearchQuery] = []
    for q in queries:
        key = (q.query, q.repo)
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique


def infer_repo(package_name: str) -> str | None:
    """Infer the GitHub repo for a Python package name."""
    name = package_name.lower().strip()
    # Try both dash and underscore forms
    return (
        PACKAGE_REPO_MAP.get(name)
        or PACKAGE_REPO_MAP.get(name.replace("-", "_"))
        or PACKAGE_REPO_MAP.get(name.replace("_", "-"))
    )


def _truncate_error_message(msg: str, max_len: int = 80) -> str:
    """Truncate error message to a reasonable search length."""
    msg = msg.strip()
    if len(msg) <= max_len:
        return msg
    # Try to break at a word boundary
    truncated = msg[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space]
    return truncated


def _extract_key_terms(tb: ParsedTraceback) -> list[str]:
    """Extract key class/function names from the traceback."""
    terms: list[str] = []

    # From exception message: look for CamelCase class names
    import re

    camel_case = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-z][a-zA-Z0-9]*)+\b", tb.exception_message)
    terms.extend(camel_case[:3])

    # From innermost frame function name
    if tb.frames:
        inner = tb.frames[-1]
        if inner.function not in ("<module>", "<lambda>"):
            terms.append(inner.function)

    # From exception message: quoted strings
    quoted = re.findall(r"'(\w+)'", tb.exception_message)
    terms.extend(quoted[:2])

    return terms


# ---------------------------------------------------------------------------
# Result merging (realtime + local index)
# ---------------------------------------------------------------------------


def merge_search_results(
    realtime: list[GitHubIssueRef],
    local: list[GitHubIssueRef],
    max_results: int = 15,
) -> list[GitHubIssueRef]:
    """Merge and deduplicate results from realtime search and local index.

    For duplicates (same repo + number), keeps the higher relevance_score.
    Returns up to max_results sorted by score descending.
    """
    seen: dict[tuple[str, int], GitHubIssueRef] = {}

    for issue in realtime + local:
        key = (issue.repo, issue.number)
        existing = seen.get(key)
        if existing is None or issue.relevance_score > existing.relevance_score:
            seen[key] = issue

    merged = sorted(seen.values(), key=lambda i: i.relevance_score, reverse=True)
    return merged[:max_results]

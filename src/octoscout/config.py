"""Configuration management for OctoScout.

Priority (lowest → highest):
    dataclass defaults < config file (~/.octoscout/config.yaml) < environment variables < CLI args
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Config file location
CONFIG_FILE = Path.home() / ".octoscout" / "config.yaml"

# Mapping: Config field name → environment variable name
# Only variables that are explicitly set in the environment will override lower-priority values.
_ENV_MAP: dict[str, str] = {
    "llm_provider":        "OCTOSCOUT_LLM_PROVIDER",
    "claude_model":        "OCTOSCOUT_CLAUDE_MODEL",
    "openai_model":        "OCTOSCOUT_OPENAI_MODEL",
    "anthropic_api_key":   "ANTHROPIC_API_KEY",
    "anthropic_auth_token": "ANTHROPIC_AUTH_TOKEN",
    "anthropic_base_url":  "ANTHROPIC_BASE_URL",
    "openai_api_key":      "OPENAI_API_KEY",
    "github_token":        "GITHUB_TOKEN",
    "max_search_calls":    "OCTOSCOUT_MAX_SEARCH_CALLS",
    "extraction_model_claude": "OCTOSCOUT_EXTRACTION_MODEL_CLAUDE",
    "extraction_model_openai": "OCTOSCOUT_EXTRACTION_MODEL_OPENAI",
    "matrix_data_dir":     "OCTOSCOUT_MATRIX_DATA_DIR",
    "extract_concurrency": "OCTOSCOUT_EXTRACT_CONCURRENCY",
}


@dataclass
class Config:
    """Global configuration for OctoScout."""

    # LLM settings
    llm_provider: str = "claude"
    claude_model: str = "claude-opus-4-6"
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_auth_token: str = ""
    anthropic_base_url: str = ""
    openai_api_key: str = ""

    # GitHub settings
    github_token: str = ""
    max_search_calls: int = 10

    # Matrix pipeline settings
    extraction_model_claude: str = "claude-sonnet-4-6"
    extraction_model_openai: str = "gpt-4o-mini"
    matrix_data_dir: str = "data/matrix"
    extract_concurrency: int = 5

    # Search settings
    default_repos: list[str] = field(default_factory=lambda: [
        "huggingface/transformers",
        "vllm-project/vllm",
        "huggingface/peft",
        "microsoft/DeepSpeed",
        "pytorch/pytorch",
    ])

    # ---------------------------------------------------------------------------
    # Public constructors
    # ---------------------------------------------------------------------------

    @classmethod
    def load(cls, cli_overrides: dict[str, Any] | None = None) -> Config:
        """Load configuration with the full priority chain.

        Layers applied in order (each layer only overrides if the value is
        explicitly present — never with an empty/None fallback):

            1. dataclass defaults
            2. ~/.octoscout/config.yaml  (if the file exists)
            3. environment variables     (only if the variable is actually set)
            4. cli_overrides dict        (only non-None values)
        """
        config = cls()                          # Layer 1: dataclass defaults

        cls._apply(config, cls._load_yaml())    # Layer 2: config file
        cls._apply(config, cls._load_env())     # Layer 3: env vars
        if cli_overrides:                       # Layer 4: CLI
            cls._apply(config, cli_overrides)

        return config

    @classmethod
    def from_env(cls) -> Config:
        """Backward-compatible alias for load()."""
        return cls.load()

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _apply(config: Config, values: dict[str, Any]) -> None:
        """Merge *values* into *config*, skipping None and unknown keys."""
        for key, value in values.items():
            if value is None or not hasattr(config, key):
                continue
            # Coerce type to match the existing field (env vars arrive as str)
            current = getattr(config, key)
            if isinstance(current, int) and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    continue
            setattr(config, key, value)

    @classmethod
    def _load_yaml(cls) -> dict[str, Any]:
        """Read ~/.octoscout/config.yaml.  Returns {} when absent or unreadable."""
        if not CONFIG_FILE.exists():
            return {}
        try:
            import yaml  # optional dependency — graceful fallback if missing
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Keep only keys that map to known Config fields, drop None values
            return {k: v for k, v in data.items() if v is not None}
        except Exception:
            return {}

    @classmethod
    def _load_env(cls) -> dict[str, Any]:
        """Read environment variables — only includes vars that are *actually set*."""
        result: dict[str, Any] = {}
        for field_name, env_var in _ENV_MAP.items():
            value = os.environ.get(env_var)   # returns None when not set
            if value is not None:
                result[field_name] = value
        return result

    # ---------------------------------------------------------------------------
    # Provider factory
    # ---------------------------------------------------------------------------

    def get_provider(self):
        """Instantiate the configured LLM provider."""
        if self.llm_provider == "claude":
            from octoscout.providers.claude import ClaudeProvider
            return ClaudeProvider(
                model=self.claude_model,
                api_key=self.anthropic_api_key or None,
                auth_token=self.anthropic_auth_token or None,
                base_url=self.anthropic_base_url or None,
            )
        if self.llm_provider == "openai":
            from octoscout.providers.openai import OpenAIProvider
            return OpenAIProvider(
                model=self.openai_model,
                api_key=self.openai_api_key or None,
            )
        raise ValueError(f"Unknown LLM provider: {self.llm_provider!r}")

    def get_extraction_provider(self):
        """Instantiate a cheap LLM provider for bulk matrix extraction."""
        if self.llm_provider == "claude":
            from octoscout.providers.claude import ClaudeProvider
            return ClaudeProvider(
                model=self.extraction_model_claude,
                api_key=self.anthropic_api_key or None,
                auth_token=self.anthropic_auth_token or None,
                base_url=self.anthropic_base_url or None,
            )
        if self.llm_provider == "openai":
            from octoscout.providers.openai import OpenAIProvider
            return OpenAIProvider(
                model=self.extraction_model_openai,
                api_key=self.openai_api_key or None,
            )
        raise ValueError(f"Unknown LLM provider: {self.llm_provider!r}")

"""Configuration management for OctoScout."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Global configuration loaded from environment / config file."""

    # LLM settings
    llm_provider: str = "claude"  # "claude" or "openai"
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # GitHub settings
    github_token: str = ""
    max_search_calls: int = 10  # per diagnosis session

    # Search settings
    default_repos: list[str] = field(default_factory=lambda: [
        "huggingface/transformers",
        "vllm-project/vllm",
        "huggingface/peft",
        "microsoft/DeepSpeed",
        "pytorch/pytorch",
    ])

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        return cls(
            llm_provider=os.environ.get("OCTOSCOUT_LLM_PROVIDER", "claude"),
            claude_model=os.environ.get("OCTOSCOUT_CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            openai_model=os.environ.get("OCTOSCOUT_OPENAI_MODEL", "gpt-4o"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
        )

    def get_provider(self):
        """Create the configured LLM provider instance."""
        if self.llm_provider == "claude":
            from octoscout.providers.claude import ClaudeProvider

            return ClaudeProvider(
                model=self.claude_model,
                api_key=self.anthropic_api_key or None,
            )
        elif self.llm_provider == "openai":
            from octoscout.providers.openai import OpenAIProvider

            return OpenAIProvider(
                model=self.openai_model,
                api_key=self.openai_api_key or None,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

"""Tests for octoscout.config — ConfigError and key validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from octoscout.config import Config, ConfigError


class TestConfigError:
    """Test that ConfigError is raised when API keys are missing."""

    def _clean_env(self):
        """Return a dict of env overrides that clear all API key env vars."""
        return {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "OPENAI_API_KEY": "",
        }

    def test_claude_provider_no_key_raises(self):
        """get_provider() should raise ConfigError when no Anthropic key is set."""
        config = Config(
            llm_provider="claude",
            anthropic_api_key="",
            anthropic_auth_token="",
        )
        with patch.dict(os.environ, self._clean_env(), clear=False):
            # Also remove the keys entirely if present
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            with pytest.raises(ConfigError, match="Anthropic API key not configured"):
                config.get_provider()

    def test_openai_provider_no_key_raises(self):
        """get_provider() should raise ConfigError when no OpenAI key is set."""
        config = Config(
            llm_provider="openai",
            openai_api_key="",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigError, match="OpenAI API key not configured"):
                config.get_provider()

    def test_unknown_provider_raises(self):
        """get_provider() should raise ConfigError for unknown providers."""
        config = Config(llm_provider="unknown-llm")
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            config.get_provider()

    def test_claude_with_api_key_works(self):
        """get_provider() should succeed when ANTHROPIC_API_KEY is set."""
        config = Config(
            llm_provider="claude",
            anthropic_api_key="sk-ant-test123",
        )
        provider = config.get_provider()
        assert provider is not None

    def test_claude_with_auth_token_works(self):
        """get_provider() should succeed when anthropic_auth_token is set."""
        config = Config(
            llm_provider="claude",
            anthropic_auth_token="test-auth-token",
        )
        provider = config.get_provider()
        assert provider is not None

    def test_claude_with_env_key_works(self):
        """get_provider() should succeed when env var ANTHROPIC_API_KEY is set."""
        config = Config(
            llm_provider="claude",
            anthropic_api_key="",
            anthropic_auth_token="",
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env123"}, clear=False):
            provider = config.get_provider()
            assert provider is not None

    def test_openai_with_api_key_works(self):
        """get_provider() should succeed when openai_api_key is set."""
        config = Config(
            llm_provider="openai",
            openai_api_key="sk-test123",
        )
        provider = config.get_provider()
        assert provider is not None


class TestExtractionProviderValidation:
    """Test that get_extraction_provider() also validates keys."""

    def test_extraction_claude_no_key_raises(self):
        config = Config(
            llm_provider="claude",
            anthropic_api_key="",
            anthropic_auth_token="",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            with pytest.raises(ConfigError, match="Anthropic API key not configured"):
                config.get_extraction_provider()

    def test_extraction_openai_no_key_raises(self):
        config = Config(
            llm_provider="openai",
            openai_api_key="",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigError, match="OpenAI API key not configured"):
                config.get_extraction_provider()

    def test_extraction_claude_with_key_works(self):
        config = Config(
            llm_provider="claude",
            anthropic_api_key="sk-ant-test",
        )
        provider = config.get_extraction_provider()
        assert provider is not None


class TestConfigErrorMessage:
    """Verify error messages contain helpful guidance."""

    def test_claude_error_includes_env_hint(self):
        config = Config(llm_provider="claude", anthropic_api_key="", anthropic_auth_token="")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            try:
                config.get_provider()
                pytest.fail("Expected ConfigError")
            except ConfigError as e:
                msg = str(e)
                assert "export ANTHROPIC_API_KEY" in msg
                assert "config.yaml" in msg
                assert "console.anthropic.com" in msg

    def test_openai_error_includes_env_hint(self):
        config = Config(llm_provider="openai", openai_api_key="")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                config.get_provider()
                pytest.fail("Expected ConfigError")
            except ConfigError as e:
                msg = str(e)
                assert "export OPENAI_API_KEY" in msg
                assert "config.yaml" in msg
                assert "platform.openai.com" in msg

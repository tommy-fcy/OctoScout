"""Tests for octoscout.mcp.server — tool registration, handlers, resources, prompts."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octoscout.mcp.server import mcp


# ---------------------------------------------------------------------------
# Helper: extract text from call_tool / read_resource results
# ---------------------------------------------------------------------------

def _tool_text(result) -> str:
    """Extract text from mcp.call_tool() result: (content_list, metadata)."""
    content_list = result[0]
    return content_list[0].text


def _resource_text(result) -> str:
    """Extract text from mcp.read_resource() result."""
    item = result[0]
    return item.text if hasattr(item, "text") else str(item)


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------


class TestToolRegistration:
    async def test_expected_tools_registered(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "octoscout_diagnose",
            "octoscout_search_issues",
            "octoscout_get_issue_detail",
            "octoscout_check_compatibility",
            "octoscout_check_api_signature",
        }
        for name in expected:
            assert name in tool_names, f"Tool '{name}' not registered"

    async def test_tool_count(self):
        tools = await mcp.list_tools()
        assert len(tools) == 5


# ---------------------------------------------------------------------------
# Resource registration tests
# ---------------------------------------------------------------------------


class TestResourceRegistration:
    async def test_resources_registered(self):
        resources = await mcp.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "octoscout://matrix/stats" in uris
        assert "octoscout://packages" in uris


# ---------------------------------------------------------------------------
# Prompt registration tests
# ---------------------------------------------------------------------------


class TestPromptRegistration:
    async def test_prompts_registered(self):
        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert "diagnose_error" in prompt_names
        assert "check_environment" in prompt_names


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestOctoscoutSearchIssues:
    async def test_empty_query_returns_error(self):
        result = await mcp.call_tool("octoscout_search_issues", {"query": ""})
        assert "Error" in _tool_text(result)

    async def test_search_returns_results(self):
        from octoscout.models import GitHubIssueRef

        mock_issues = [
            GitHubIssueRef(
                repo="huggingface/transformers",
                number=12345,
                title="TypeError in Trainer with tokenizer",
                url="https://github.com/huggingface/transformers/issues/12345",
                state="closed",
                snippet="Some snippet text",
            ),
        ]

        # Patch at the source module (lazy imports inside the function)
        with patch("octoscout.config.Config.load") as mock_load, \
             patch("octoscout.search.github_client.GitHubClient") as MockClient:
            mock_config = MagicMock()
            mock_config.github_token = "fake-token"
            mock_load.return_value = mock_config

            instance = AsyncMock()
            MockClient.return_value = instance
            instance.search_issues = AsyncMock(return_value=mock_issues)
            instance.close = AsyncMock()

            result = await mcp.call_tool(
                "octoscout_search_issues",
                {"query": "TypeError Trainer", "repo": "huggingface/transformers"},
            )

        text = _tool_text(result)
        assert "Found 1 issues" in text
        assert "TypeError in Trainer" in text
        assert "12345" in text


class TestOctoscoutGetIssueDetail:
    async def test_returns_formatted_issue(self):
        mock_issue = {
            "title": "Test Issue",
            "state": "open",
            "created_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/1",
            "body": "This is the issue body.",
        }
        mock_comments = [
            {
                "user": {"login": "testuser"},
                "created_at": "2026-01-02T00:00:00Z",
                "body": "This is a comment.",
            }
        ]

        with patch("octoscout.config.Config.load") as mock_load, \
             patch("octoscout.search.github_client.GitHubClient") as MockClient:
            mock_config = MagicMock()
            mock_config.github_token = ""
            mock_load.return_value = mock_config

            instance = AsyncMock()
            MockClient.return_value = instance
            instance.get_issue = AsyncMock(return_value=mock_issue)
            instance.get_issue_comments = AsyncMock(return_value=mock_comments)
            instance.close = AsyncMock()

            result = await mcp.call_tool(
                "octoscout_get_issue_detail",
                {"repo": "test/repo", "issue_number": 1},
            )

        text = _tool_text(result)
        assert "# Test Issue" in text
        assert "State: open" in text
        assert "This is the issue body" in text
        assert "@testuser" in text
        assert "This is a comment" in text


class TestOctoscoutCheckApiSignature:
    async def test_known_function(self):
        result = await mcp.call_tool(
            "octoscout_check_api_signature",
            {"function_path": "json.dumps", "kwargs": "obj,indent"},
        )
        assert "json.dumps" in _tool_text(result)

    async def test_nonexistent_function(self):
        result = await mcp.call_tool(
            "octoscout_check_api_signature",
            {"function_path": "nonexistent.module.func"},
        )
        assert "Could not import" in _tool_text(result)

    async def test_no_kwargs(self):
        result = await mcp.call_tool(
            "octoscout_check_api_signature",
            {"function_path": "json.dumps"},
        )
        assert "json.dumps" in _tool_text(result)


class TestOctoscoutCheckCompatibility:
    async def test_matrix_not_found(self):
        with patch("octoscout.config.Config.load") as mock_load:
            mock_config = MagicMock()
            mock_config.matrix_data_dir = "/nonexistent/path"
            mock_load.return_value = mock_config

            result = await mcp.call_tool(
                "octoscout_check_compatibility",
                {"packages": "torch==2.3.0,transformers==4.55.0"},
            )

        assert "not available" in _tool_text(result)

    async def test_too_few_packages(self, tmp_path):
        matrix_data = {
            "version": "1.1",
            "built_at": "2026-01-01T00:00:00+00:00",
            "entry_count": 0,
            "entries": {},
            "single_pkg_issues": [],
        }
        (tmp_path / "matrix.json").write_text(json.dumps(matrix_data))

        with patch("octoscout.config.Config.load") as mock_load:
            mock_config = MagicMock()
            mock_config.matrix_data_dir = str(tmp_path)
            mock_load.return_value = mock_config

            result = await mcp.call_tool(
                "octoscout_check_compatibility",
                {"packages": "torch==2.3.0"},
            )

        assert "at least 2" in _tool_text(result)


class TestOctoscoutDiagnose:
    async def test_empty_traceback(self):
        result = await mcp.call_tool("octoscout_diagnose", {"traceback": ""})
        assert "Error" in _tool_text(result)

    async def test_missing_api_key(self):
        import os

        from octoscout.config import ConfigError

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

            with patch("octoscout.config.Config.load") as mock_load:
                mock_load.side_effect = ConfigError("Anthropic API key not configured.")

                result = await mcp.call_tool(
                    "octoscout_diagnose",
                    {"traceback": "TypeError: test error"},
                )

        assert "Configuration error" in _tool_text(result)


# ---------------------------------------------------------------------------
# Resource handler tests
# ---------------------------------------------------------------------------


class TestResources:
    async def test_packages_resource(self):
        result = await mcp.read_resource("octoscout://packages")
        text = _resource_text(result)
        assert "transformers" in text
        assert "huggingface/transformers" in text

    async def test_matrix_stats_resource_no_matrix(self):
        with patch("octoscout.config.Config.load") as mock_load:
            mock_config = MagicMock()
            mock_config.matrix_data_dir = "/nonexistent/path"
            mock_load.return_value = mock_config

            result = await mcp.read_resource("octoscout://matrix/stats")
            text = _resource_text(result)
            assert "not built" in text.lower()


# ---------------------------------------------------------------------------
# Prompt handler tests
# ---------------------------------------------------------------------------


class TestPrompts:
    async def test_diagnose_error_prompt(self):
        result = await mcp.get_prompt(
            "diagnose_error",
            arguments={"traceback": "TypeError: unexpected kwarg"},
        )
        assert len(result.messages) > 0
        text = result.messages[0].content.text
        assert "TypeError: unexpected kwarg" in text
        assert "octoscout_diagnose" in text

    async def test_check_environment_prompt(self):
        result = await mcp.get_prompt("check_environment")
        assert len(result.messages) > 0
        text = result.messages[0].content.text
        assert "octoscout_check_compatibility" in text

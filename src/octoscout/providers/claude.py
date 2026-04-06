"""Claude API provider implementation."""

from __future__ import annotations

import asyncio
import os

import anthropic

from octoscout.models import AgentResponse, Message, Role, ToolCall, ToolDefinition

_MAX_RETRIES = 3
_RETRYABLE_KEYWORDS = {"overloaded", "529", "rate", "connection", "timeout", "503"}


class ClaudeProvider:
    """LLM Provider backed by the Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens

        resolved_auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None

        if resolved_auth_token:
            self._client = anthropic.AsyncAnthropic(
                auth_token=resolved_auth_token,
                base_url=resolved_base_url,
            )
        else:
            self._client = anthropic.AsyncAnthropic(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
            )

    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system: str = "",
    ) -> AgentResponse:
        claude_messages = self._convert_messages(messages)
        claude_tools = self._convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "tools": claude_tools,
        }
        if system:
            kwargs["system"] = system

        response = await self._call_with_retry(**kwargs)
        return self._parse_response(response)

    async def chat(
        self,
        messages: list[Message],
        system: str = "",
    ) -> AgentResponse:
        claude_messages = self._convert_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
        }
        if system:
            kwargs["system"] = system

        response = await self._call_with_retry(**kwargs)
        return self._parse_response(response)

    async def _call_with_retry(self, **kwargs):
        """Call the API with exponential backoff retry on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._client.messages.create(**kwargs)
            except Exception as e:
                err = str(e).lower()
                is_retryable = any(kw in err for kw in _RETRYABLE_KEYWORDS)
                if is_retryable and attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert internal messages to Claude API format."""
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # system is passed separately

            if msg.role == Role.TOOL:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal tool definitions to Claude API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.to_json_schema(),
            }
            for t in tools
        ]

    @staticmethod
    def _parse_response(response) -> AgentResponse:
        """Parse Claude API response into AgentResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return AgentResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

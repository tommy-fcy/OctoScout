"""Claude API provider implementation."""

from __future__ import annotations

import json
import os

import anthropic

from octoscout.models import AgentResponse, Message, Role, ToolCall, ToolDefinition


class ClaudeProvider:
    """LLM Provider backed by the Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
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

        response = await self._client.messages.create(**kwargs)
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

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

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

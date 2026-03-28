"""OpenAI API provider implementation."""

from __future__ import annotations

import json
import os

import openai

from octoscout.models import AgentResponse, Message, Role, ToolCall, ToolDefinition


class OpenAIProvider:
    """LLM Provider backed by the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system: str = "",
    ) -> AgentResponse:
        oai_messages = self._convert_messages(messages, system)
        oai_tools = self._convert_tools(tools)

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=oai_messages,
            tools=oai_tools,
        )
        return self._parse_response(response)

    async def chat(
        self,
        messages: list[Message],
        system: str = "",
    ) -> AgentResponse:
        oai_messages = self._convert_messages(messages, system)

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=oai_messages,
        )
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message], system: str = "") -> list[dict]:
        """Convert internal messages to OpenAI API format."""
        result: list[dict] = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.content})
            elif msg.role == Role.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                oai_msg: dict = {"role": "assistant"}
                if msg.content:
                    oai_msg["content"] = msg.content
                oai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append(oai_msg)
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal tool definitions to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.to_json_schema(),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _parse_response(response) -> AgentResponse:
        """Parse OpenAI API response into AgentResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return AgentResponse(
            text=message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            if response.usage
            else {},
        )

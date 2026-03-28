"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from octoscout.models import AgentResponse, Message, ToolDefinition


class LLMProvider(ABC):
    """Unified interface for LLM providers that support tool use."""

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system: str = "",
    ) -> AgentResponse:
        """Send a conversation with tool definitions and get a response.

        The provider is responsible for converting internal ToolDefinition
        format to the provider-specific format (Claude tool_use / OpenAI
        function calling) and normalizing the response back to AgentResponse.
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system: str = "",
    ) -> AgentResponse:
        """Simple chat without tools."""
        ...

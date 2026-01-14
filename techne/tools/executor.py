"""Tool executor for managing tool invocations."""

from __future__ import annotations

import asyncio

from techne.config import TagConfig, ToolConfig
from techne.rollout.parser import TagParser
from techne.tools.base import Tool, ToolResult, ToolResultStatus


class ToolExecutor:
    """Executes tools based on tag detection in generated text.

    The executor:
    1. Parses text to find tool invocations using TagParser
    2. Routes content to appropriate tools
    3. Manages concurrent execution with rate limiting
    4. Formats tool responses for model consumption
    """

    def __init__(
        self,
        tags: TagConfig,
        tools: dict[str, Tool] | None = None,
        config: ToolConfig | None = None,
        parser: TagParser | None = None,
    ):
        """Initialize tool executor.

        Args:
            tags: Tag configuration for parsing
            tools: Dict mapping tool names to Tool instances
            config: Tool configuration (timeouts, limits)
            parser: Optional shared TagParser instance
        """
        from techne.rollout.parser import TagParser

        self._tags = tags
        self._tools: dict[str, Tool] = tools or {}
        self._config = config
        self._parser = parser or TagParser(tags)
        self._semaphore = asyncio.Semaphore(config.concurrent_limit if config else 10)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the executor.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call.

        Args:
            text: Generated text to check

        Returns:
            True if tool call tags are detected
        """
        return self._parser.has_tool_call(text)

    def extract_tool_calls(self, text: str) -> list[tuple[str, int, int]]:
        """Extract all tool calls from text.

        Args:
            text: Generated text to parse

        Returns:
            List of (content, start_pos, end_pos) tuples
        """
        calls = self._parser.parse_tool_calls(text)
        return [(c.content, c.start_pos, c.end_pos) for c in calls]

    async def execute_tool(self, tool_name: str, content: str) -> ToolResult:
        """Execute a specific tool.

        Args:
            tool_name: Name of tool to execute
            content: Content to pass to tool

        Returns:
            ToolResult from execution
        """
        if tool_name not in self._tools:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        tool = self._tools[tool_name]

        # Validate content
        is_valid, error = await tool.validate(content)
        if not is_valid:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=error or "Validation failed",
            )

        # Execute with concurrency limit
        async with self._semaphore:
            return await tool.execute(content)

    async def execute_all(self, text: str, tool_name: str = "code") -> list[ToolResult]:
        """Execute all tool calls found in text.

        Args:
            text: Text containing tool calls
            tool_name: Name of tool to use for all calls

        Returns:
            List of ToolResults in order of appearance
        """
        calls = self.extract_tool_calls(text)
        if not calls:
            return []

        # Execute concurrently
        tasks = [self.execute_tool(tool_name, content) for content, _, _ in calls]
        return await asyncio.gather(*tasks)

    def format_response(self, text: str, results: list[ToolResult], tool_name: str = "code") -> str:
        """Format text with tool responses inserted.

        Args:
            text: Original text with tool calls
            results: Results from execute_all
            tool_name: Name of tool used

        Returns:
            Text with tool responses inserted after each tool call
        """
        calls = self._parser.parse_tool_calls(text)
        if len(calls) != len(results):
            raise ValueError(f"Mismatch: {len(calls)} calls vs {len(results)} results")

        # Build output with responses inserted
        # Use TagParser's logic implicitly or rebuild string manually
        # Since we have results and positions, manual rebuild using parser info is safest

        parts = []
        last_end = 0

        for call, result in zip(calls, results):
            # Add text before tool call + tool call itself
            parts.append(text[last_end : call.end_pos])

            # Add response
            response_text = result.format_for_model()
            parts.append(self._tags.wrap_response(response_text))

            last_end = call.end_pos

        # Add remaining text
        parts.append(text[last_end:])

        return "".join(parts)

    async def process_generation(
        self, text: str, tool_name: str = "code"
    ) -> tuple[str, list[ToolResult]]:
        """Process generated text: detect tool calls, execute, and format response.

        Args:
            text: Generated text potentially containing tool calls
            tool_name: Default tool to use

        Returns:
            Tuple of (formatted_text_with_responses, list_of_results)
        """
        if not self.has_tool_call(text):
            return text, []

        results = await self.execute_all(text, tool_name)
        formatted = self.format_response(text, results, tool_name)
        return formatted, results

    async def cleanup(self) -> None:
        """Cleanup all registered tools."""
        for tool in self._tools.values():
            await tool.cleanup()

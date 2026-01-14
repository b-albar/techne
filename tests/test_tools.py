"""Tests for tool system."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from techne.tools.base import Tool, ToolResult, ToolResultStatus
from techne.tools.sandbox import MockSandboxTool
from techne.tools.executor import ToolExecutor
from techne.config import TagConfig


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            status=ToolResultStatus.SUCCESS,
            output="42",
        )
        assert result.is_success
        assert result.format_for_model() == "42"

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error="SyntaxError: invalid syntax",
        )
        assert not result.is_success
        assert "SyntaxError" in result.format_for_model()

    def test_timeout_result(self):
        """Test timeout result."""
        result = ToolResult(
            status=ToolResultStatus.TIMEOUT,
            output="",
            error="Execution timed out after 30s",
        )
        assert not result.is_success


class TestMockSandboxTool:
    """Tests for MockSandboxTool."""

    @pytest.mark.asyncio
    async def test_mock_execution(self):
        """Test mock tool execution."""
        tool = MockSandboxTool()
        result = await tool.execute("print('hello')")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_predefined_responses(self):
        """Test with predefined responses."""
        tool = MockSandboxTool(responses={"1+1": "2"})
        result = await tool.execute("1+1")
        assert result.output == "2"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_has_tool_call(self):
        """Test tool call detection."""
        tags = TagConfig()
        executor = ToolExecutor(tags)

        assert executor.has_tool_call("Here is code: <code>print('hi')</code>")
        assert not executor.has_tool_call("Just plain text")

    def test_extract_tool_calls(self):
        """Test extracting tool calls."""
        tags = TagConfig()
        executor = ToolExecutor(tags)

        text = "First <code>x=1</code> then <code>x=2</code>"
        calls = executor.extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0][0] == "x=1"
        assert calls[1][0] == "x=2"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a single tool."""
        tags = TagConfig()
        executor = ToolExecutor(tags)
        executor.register_tool(MockSandboxTool())

        result = await executor.execute_tool("code", "print('test')")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test executing unknown tool."""
        tags = TagConfig()
        executor = ToolExecutor(tags)

        result = await executor.execute_tool("unknown", "content")
        assert result.status == ToolResultStatus.ERROR
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_format_response(self):
        """Test formatting response with tool results."""
        tags = TagConfig()
        executor = ToolExecutor(tags)
        executor.register_tool(MockSandboxTool(responses={"x=1": "Done"}))

        text = "Let me run: <code>x=1</code>"
        results = await executor.execute_all(text)
        formatted = executor.format_response(text, results)

        assert "<interpreter>" in formatted
        assert "</interpreter>" in formatted

    @pytest.mark.asyncio
    async def test_process_generation(self):
        """Test full generation processing."""
        tags = TagConfig()
        executor = ToolExecutor(tags)
        executor.register_tool(MockSandboxTool())

        text = "Solving: <code>2+2</code>"
        formatted, results = await executor.process_generation(text)

        assert len(results) == 1
        assert "<interpreter>" in formatted

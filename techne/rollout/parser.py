"""Tag parsing utilities for tool call detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from techne.config import TagConfig


@dataclass
class ParsedToolCall:
    """Parsed tool call from generated text.

    Attributes:
        content: The tool call content (e.g., code)
        start_pos: Start position in original text
        end_pos: End position in original text
        has_response: Whether a response already exists
        response_content: The response content if present
        response_end_pos: End position of response if present
    """

    content: str
    start_pos: int
    end_pos: int
    has_response: bool = False
    response_content: str | None = None
    response_end_pos: int | None = None


class TagParser:
    """Parser for detecting and extracting tool calls from generated text.

    Supports configurable tags for different use cases:
    - ReTool style: <code>...</code> / <interpreter>...</interpreter>
    - OpenAI style: <tool_call>...</tool_call> / <tool_response>...</tool_response>
    """

    def __init__(self, tags: TagConfig):
        """Initialize parser with tag configuration.

        Args:
            tags: TagConfig defining tool and response tags
        """
        self._tags = tags
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for tag detection."""
        # Escape special regex characters in tags
        ts = re.escape(self._tags.tool_start)
        te = re.escape(self._tags.tool_end)
        rs = re.escape(self._tags.response_start)
        rse = re.escape(self._tags.response_end)

        # Pattern for tool call (captures content)
        self._tool_pattern = re.compile(rf"({ts})(.*?)({te})", re.DOTALL)

        # Pattern for response (captures content)
        self._response_pattern = re.compile(rf"({rs})(.*?)({rse})", re.DOTALL)

        # Pattern for tool call followed by optional response
        self._full_pattern = re.compile(
            rf"({ts})(.*?)({te})(?:\s*({rs})(.*?)({rse}))?",
            re.DOTALL,
        )

        # Pattern for detecting incomplete tool call (started but not ended)
        self._incomplete_tool = re.compile(rf"{ts}(?!.*{te})", re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a complete tool call.

        Args:
            text: Text to check

        Returns:
            True if a complete tool call is present
        """
        return bool(self._tool_pattern.search(text))

    def has_incomplete_tool_call(self, text: str) -> bool:
        """Check if text has an incomplete tool call (started but not ended).

        Useful for determining if generation should continue.

        Args:
            text: Text to check

        Returns:
            True if tool call is started but not completed
        """
        return bool(self._incomplete_tool.search(text))

    def has_response(self, text: str) -> bool:
        """Check if text contains a tool response.

        Args:
            text: Text to check

        Returns:
            True if response tags are present
        """
        return bool(self._response_pattern.search(text))

    def parse_tool_calls(self, text: str) -> list[ParsedToolCall]:
        """Parse all tool calls from text.

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        results = []

        for match in self._full_pattern.finditer(text):
            tool_content = match.group(2).strip()

            # Check if response is present
            has_response = match.group(4) is not None
            response_content = match.group(5).strip() if has_response else None
            response_end = match.end() if has_response else None

            results.append(
                ParsedToolCall(
                    content=tool_content,
                    start_pos=match.start(),
                    end_pos=match.start(3) + len(match.group(3)),  # End of tool_end tag
                    has_response=has_response,
                    response_content=response_content,
                    response_end_pos=response_end,
                )
            )

        return results

    def extract_last_tool_call(self, text: str) -> ParsedToolCall | None:
        """Extract the last tool call from text.

        Useful for processing the most recent tool invocation.

        Args:
            text: Text to parse

        Returns:
            Last ParsedToolCall or None if no tool calls found
        """
        calls = self.parse_tool_calls(text)
        return calls[-1] if calls else None

    def split_at_tool_call(self, text: str) -> tuple[str, str | None, str]:
        """Split text at the first tool call.

        Returns:
            Tuple of (before_tool, tool_content, after_tool)
            If no tool call, returns (text, None, "")
        """
        match = self._tool_pattern.search(text)
        if not match:
            return text, None, ""

        before = text[: match.start()]
        content = match.group(2).strip()
        after = text[match.end() :]

        return before, content, after

    def insert_response(self, text: str, response: str, position: int | None = None) -> str:
        """Insert a tool response after a tool call.

        Args:
            text: Text containing tool call
            response: Response content to insert
            position: Position after which to insert (default: after last tool call)

        Returns:
            Text with response inserted
        """
        if position is None:
            # Find last tool call
            matches = list(self._tool_pattern.finditer(text))
            if not matches:
                return text
            position = matches[-1].end()

        wrapped = self._tags.wrap_response(response)
        return text[:position] + wrapped + text[position:]

    def get_response_mask_ranges(self, text: str) -> list[tuple[int, int]]:
        """Get character ranges that should be masked in loss computation.

        This marks interpreter/response sections that come from external tools
        and should not be included in the training loss.

        Args:
            text: Full conversation text

        Returns:
            List of (start, end) tuples for masking
        """
        ranges = []
        for match in self._response_pattern.finditer(text):
            ranges.append((match.start(), match.end()))
        return ranges

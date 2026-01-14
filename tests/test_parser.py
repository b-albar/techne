"""Tests for tag parser."""

import pytest
from techne.config import TagConfig
from techne.rollout.parser import TagParser, ParsedToolCall


class TestTagParser:
    """Tests for TagParser."""

    @pytest.fixture
    def parser(self):
        """Create default parser."""
        return TagParser(TagConfig())

    @pytest.fixture
    def custom_parser(self):
        """Create parser with custom tags."""
        return TagParser(
            TagConfig(
                tool_start="<tool_call>",
                tool_end="</tool_call>",
                response_start="<tool_response>",
                response_end="</tool_response>",
            )
        )

    def test_has_tool_call(self, parser):
        """Test tool call detection."""
        assert parser.has_tool_call("Here: <code>x=1</code>")
        assert not parser.has_tool_call("No code here")

    def test_has_incomplete_tool_call(self, parser):
        """Test incomplete tool call detection."""
        assert parser.has_incomplete_tool_call("Starting <code>x=1")
        assert not parser.has_incomplete_tool_call("Complete <code>x=1</code>")

    def test_parse_tool_calls(self, parser):
        """Test parsing tool calls."""
        text = "First <code>x=1</code> then <code>y=2</code>"
        calls = parser.parse_tool_calls(text)

        assert len(calls) == 2
        assert calls[0].content == "x=1"
        assert calls[1].content == "y=2"
        assert not calls[0].has_response

    def test_parse_with_response(self, parser):
        """Test parsing tool calls with responses."""
        text = "<code>x=1</code><interpreter>Done</interpreter>"
        calls = parser.parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0].has_response
        assert calls[0].response_content == "Done"

    def test_extract_last_tool_call(self, parser):
        """Test extracting last tool call."""
        text = "<code>first</code> text <code>second</code>"
        last = parser.extract_last_tool_call(text)

        assert last is not None
        assert last.content == "second"

    def test_split_at_tool_call(self, parser):
        """Test splitting at tool call."""
        text = "Before <code>content</code> after"
        before, content, after = parser.split_at_tool_call(text)

        assert before == "Before "
        assert content == "content"
        assert after == " after"

    def test_insert_response(self, parser):
        """Test inserting response."""
        text = "Code: <code>x=1</code>"
        result = parser.insert_response(text, "Success")

        assert "<interpreter>" in result
        assert "Success" in result
        assert result.index("</code>") < result.index("<interpreter>")

    def test_get_response_mask_ranges(self, parser):
        """Test getting mask ranges."""
        text = "Start <interpreter>masked</interpreter> end"
        ranges = parser.get_response_mask_ranges(text)

        assert len(ranges) == 1
        start, end = ranges[0]
        assert text[start:end] == "<interpreter>masked</interpreter>"

    def test_custom_tags(self, custom_parser):
        """Test with custom tags."""
        text = "<tool_call>code here</tool_call>"
        assert custom_parser.has_tool_call(text)

        calls = custom_parser.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].content == "code here"

    def test_multiline_content(self, parser):
        """Test parsing multiline code blocks."""
        text = """<code>
def hello():
    print("world")
</code>"""
        calls = parser.parse_tool_calls(text)

        assert len(calls) == 1
        assert "def hello():" in calls[0].content

    def test_no_tool_calls(self, parser):
        """Test parsing text without tool calls."""
        text = "Just regular text without any code"
        assert parser.parse_tool_calls(text) == []
        assert parser.extract_last_tool_call(text) is None

        before, content, after = parser.split_at_tool_call(text)
        assert before == text
        assert content is None
        assert after == ""

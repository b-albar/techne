"""Abstract base classes for tool system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolResultStatus(str, Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result from tool execution.

    Attributes:
        status: Execution status (success, error, timeout)
        output: Output from the tool (stdout for code, result for other tools)
        error: Error message if execution failed
        metadata: Additional metadata (execution time, etc.)
    """

    status: ToolResultStatus
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    def format_for_model(self) -> str:
        """Format result for inclusion in model context."""
        if self.is_success:
            return self.output
        else:
            error_msg = self.error or "Unknown error"
            return f"Error: {error_msg}\n{self.output}" if self.output else f"Error: {error_msg}"


class Tool(ABC):
    """Abstract base class for tools.

    Tools are external capabilities that the model can invoke during reasoning.
    Examples include code interpreters, search engines, calculators, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this tool."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the tool."""
        return f"{self.name} tool"

    @abstractmethod
    async def execute(self, content: str) -> ToolResult:
        """Execute the tool with the given content.

        Args:
            content: The content extracted from tool tags (e.g., code to execute)

        Returns:
            ToolResult with status, output, and optional error
        """
        ...

    async def validate(self, content: str) -> tuple[bool, str | None]:
        """Validate content before execution.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None

    async def cleanup(self) -> None:
        """Cleanup resources after execution."""
        pass

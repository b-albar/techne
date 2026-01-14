"""Microsandbox code execution tool."""

from __future__ import annotations

import asyncio

from techne.config import ToolConfig
from techne.tools.base import Tool, ToolResult, ToolResultStatus


class MicrosandboxTool(Tool):
    """Code sandbox tool using Microsandbox.

    Microsandbox provides secure, isolated microVM environments for executing
    untrusted code with VM-level isolation and fast startup (<200ms).

    Requires microsandbox server running (default: http://127.0.0.1:5555).
    Install SDK: pip install microsandbox
    """

    def __init__(
        self,
        config: ToolConfig | None = None,
        server_url: str = "http://127.0.0.1:5555",
        api_key: str | None = None,
        language: str = "python",
        timeout: float = 30.0,
    ):
        """Initialize Microsandbox tool.

        Args:
            config: ToolConfig with sandbox settings
            server_url: Microsandbox server URL
            api_key: API key if authentication is enabled
            language: Default language for code execution
            timeout: Execution timeout in seconds
        """
        self._server_url = config.sandbox_url if config and config.sandbox_url else server_url
        self._api_key = api_key
        self._language = language
        self._timeout = config.sandbox_timeout if config else timeout
        self._sandbox = None

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return f"Execute {self._language} code in isolated Microsandbox environment"

    async def _get_sandbox(self):
        """Get or create sandbox instance."""
        if self._sandbox is None:
            try:
                from microsandbox import Sandbox
            except ImportError as e:
                raise ImportError(
                    "microsandbox package not installed. Install with: pip install microsandbox"
                ) from e

            self._sandbox = await Sandbox.create(
                name="techne-sandbox",
                image="python:3.11-slim"
                if self._language == "python"
                else f"{self._language}:latest",
            )
        return self._sandbox

    async def execute(self, content: str) -> ToolResult:
        """Execute code in Microsandbox.

        Args:
            content: Code to execute

        Returns:
            ToolResult with execution output or error
        """
        try:
            sandbox = await self._get_sandbox()

            # Execute with timeout
            result = await asyncio.wait_for(
                sandbox.run(self._language, content),
                timeout=self._timeout,
            )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=result.stdout or "",
                error=result.stderr if result.stderr else None,
                metadata={"exit_code": result.exit_code},
            )

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                output="",
                error=f"Execution timed out after {self._timeout}s",
            )
        except ImportError as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Execution failed: {e!s}",
            )

    async def cleanup(self) -> None:
        """Cleanup sandbox resources."""
        if self._sandbox is not None:
            try:
                await self._sandbox.stop()
            except Exception:
                pass
            self._sandbox = None


class MockSandboxTool(Tool):
    """Mock sandbox for testing without actual code execution."""

    def __init__(self, responses: dict[str, str] | None = None):
        """Initialize mock sandbox.

        Args:
            responses: Optional dict mapping code -> output for testing
        """
        self._responses = responses or {}

    @property
    def name(self) -> str:
        return "code"

    async def execute(self, content: str) -> ToolResult:
        """Return mock response or echo input."""
        if content in self._responses:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=self._responses[content],
            )
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=f"[Mock execution of {len(content)} chars]",
        )

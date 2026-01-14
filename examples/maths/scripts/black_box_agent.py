"""Example black-box agent implementation for consistency testing.

This demonstrates how to wrap a vLLM server API to work with Techne's RL training.

Purpose: Compare white-box vs black-box approaches using the SAME MODEL
to verify tokenization consistency. The black-box uses the same policy model
but accessed via API (re-tokenization path) to validate the fix.
"""

from __future__ import annotations

import os
from typing import Any

from techne.config import TagConfig


class MathToolAgent:
    """Example black-box agent using an external API for math problems.

    This agent:
    1. Uses an external LLM API (e.g., OpenAI, local vLLM server)
    2. Executes Python code for tool calls
    3. Returns trajectories with optional token_ids
    """

    def __init__(
        self,
        model_name: str,
        tags: TagConfig,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the black-box agent.

        Args:
            model_name: Model identifier for the API
            tags: Tag configuration for tool calls
            api_base: Optional API base URL (for local servers)
            api_key: Optional API key
        """
        self.model_name = model_name
        self.tags = tags
        self.api_base = api_base
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # For this example, we'll use OpenAI-compatible API
        # Could also use Anthropic, local vLLM, etc.
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    async def generate_trajectory(self, prompt: str, **kwargs: Any) -> tuple[str, list[int] | None]:
        """Generate a complete trajectory using the external agent.

        Args:
            prompt: The math problem to solve
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (generated_text, optional_token_ids)
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt()

        # Generate with tool use capability
        full_trajectory = ""
        max_turns = kwargs.get("max_new_tokens", 4096) // 1024  # Rough estimate

        for turn in range(max_turns):
            # Call the API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + full_trajectory},
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_new_tokens", 1024),
                stop=[self.tags.tool_end],  # Stop after tool call
            )

            generated_text = response.choices[0].message.content or ""
            full_trajectory += generated_text

            # Check if there's a tool call
            if self.tags.tool_start in generated_text:
                # Execute tool and append response
                full_trajectory += await self._execute_tool(generated_text)
            else:
                # No tool call, we're done
                break

        # Note: We don't have access to token_ids from the API
        # Return None to trigger fallback re-tokenization during training
        return (full_trajectory, None)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for tool use."""
        return f"""You are a mathematical problem solver with access to Python code execution.

When you need to perform calculations, use the following format:
{self.tags.tool_start}
# Your Python code here
result = ...
print(result)
{self.tags.tool_end}

The system will execute your code and return the result in:
{self.tags.response_start}output{self.tags.response_end}

Use this capability to solve complex math problems step-by-step."""

    async def _execute_tool(self, text: str) -> str:
        """Extract and execute Python code from tool call.

        Args:
            text: Text containing tool call

        Returns:
            Formatted tool response
        """
        # Extract code from tags
        start_idx = text.find(self.tags.tool_start) + len(self.tags.tool_start)
        end_idx = text.find(self.tags.tool_end)

        if start_idx == -1 or end_idx == -1:
            return ""

        code = text[start_idx:end_idx].strip()

        # Execute code (in a sandbox in production!)
        try:
            # Simple execution for demo - use microsandbox in production
            import contextlib
            import io

            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                exec(code, {"__builtins__": __builtins__})

            result = output_buffer.getvalue().strip()
        except Exception as e:
            result = f"Error: {str(e)}"

        # Format response
        return self.tags.wrap_response(result)

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update agent weights (if supported).

        For external APIs, this is typically not supported.
        For local models, you could push to a local vLLM server.
        """
        # Not implemented for external APIs
        pass

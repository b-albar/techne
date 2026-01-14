"""Abstract base class for rollout backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    """Output from a generation request.

    Attributes:
        text: Generated text
        token_ids: Token IDs of generated text
        prompt_tokens: Number of prompt tokens
        generated_tokens: Number of generated tokens
        finish_reason: Reason for stopping (length, stop, tool_call)
        metadata: Additional backend-specific metadata
    """

    text: str
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    finish_reason: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling (0 to disable)
        stop_strings: Strings that stop generation
        include_stop_str_in_output: Whether to include stop string in output
    """

    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    stop_strings: list[str] = field(default_factory=list)
    include_stop_str_in_output: bool = True


class RolloutBackend(ABC):
    """Abstract base class for inference backends.

    Backends provide text generation capabilities using different engines
    (vLLM, SGLang, etc.) with support for weight updates during RL training.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is ready for generation."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions for prompts.

        Args:
            prompts: List of prompt strings
            config: Generation configuration

        Returns:
            List of GenerationOutput objects
        """
        ...

    @abstractmethod
    async def generate_with_stop_on_tags(
        self,
        prompts: list[str],
        stop_tags: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate with automatic stopping on specified tags.

        Used for multi-turn generation where we stop on tool call end tags.

        Args:
            prompts: List of prompt strings
            stop_tags: Tags that should stop generation
            config: Generation configuration

        Returns:
            List of GenerationOutput objects
        """
        ...

    @abstractmethod
    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights during RL training.

        Args:
            state_dict: New model state dict (may be partial for LoRA)
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the backend server/engine."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend server/engine."""
        ...

    async def __aenter__(self) -> RolloutBackend:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

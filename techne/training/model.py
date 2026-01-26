"""Abstract model interfaces for training and inference.

This module provides abstract base classes that define the contract
for training and inference models. Implementations (HuggingFace, etc.)
should be placed in techne.integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceModel(ABC):
    """Abstract base class for inference models.

    Defines the interface for models used in:
    - Evaluation
    - Data generation
    - Teacher models in distillation
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device the model is on."""
        ...

    @property
    @abstractmethod
    def config(self) -> Any:
        """Return the model configuration."""
        ...

    @abstractmethod
    def generate(self, input_ids: Any, **kwargs) -> Any:
        """Generate tokens from input IDs.

        Args:
            input_ids: Input token IDs tensor
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        ...

    @abstractmethod
    def generate_with_cache(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_token_ids: list[int] | None = None,
        continue_from_cache: bool = False,
        keep_cache: bool = False,
    ) -> tuple[list[int], list[float]]:
        """Generate tokens using KV cache for efficiency.

        Supports multi-turn generation by preserving cache between calls.

        Args:
            prompt_ids: Prompt token IDs to process before generating
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            stop_token_ids: Token IDs that stop generation
            continue_from_cache: If True, don't clear cache at start
            keep_cache: If True, don't clear cache at end

        Returns:
            Tuple of (generated_ids, log_probs)
        """
        ...

    @abstractmethod
    def prefill_cache(self, token_ids: list[int]) -> None:
        """Process tokens and add to KV cache without generating.

        Use this to add context (e.g., tool outputs) between generation turns.

        Args:
            token_ids: Token IDs to process and cache
        """
        ...

    @abstractmethod
    def clear_kv_cache(self) -> None:
        """Clear the KV cache."""
        ...

    @abstractmethod
    def compute_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        """Compute log probabilities for completion tokens.

        Args:
            prompt_ids: Prompt token IDs
            completion_ids: Completion token IDs

        Returns:
            Log probabilities for each completion token
        """
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return the tokenizer associated with this model."""
        ...

    @abstractmethod
    def eval(self) -> InferenceModel:
        """Set model to evaluation mode."""
        ...


class TrainingModel(InferenceModel):
    """Abstract base class for training models.

    Extends InferenceModel with training-specific methods:
    - Forward pass with gradient support
    - Parameter access for optimizers
    - State dict management for checkpointing
    """

    @abstractmethod
    def forward(self, input_ids: Any, **kwargs) -> Any:
        """Standard forward pass.

        Args:
            input_ids: Input token IDs tensor
            **kwargs: Additional forward parameters

        Returns:
            Model outputs (typically including logits)
        """
        ...

    @abstractmethod
    def __call__(self, input_ids: Any, **kwargs) -> Any:
        """Make the model callable."""
        ...

    @abstractmethod
    def train(self, mode: bool = True) -> TrainingModel:
        """Set model to training mode.

        Args:
            mode: If True, set to training mode; if False, evaluation mode

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def parameters(self) -> Any:
        """Return model parameters for optimizer."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Return the model state dictionary for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state dictionary into the model.

        Args:
            state_dict: State dictionary to load
        """
        ...

    @abstractmethod
    def save_pretrained(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Directory path to save the model
        """
        ...

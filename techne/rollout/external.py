"""External agent interface for black-box rollouts."""

from __future__ import annotations

from typing import Any, Protocol


class ExternalAgent(Protocol):
    """Protocol for external/black-box agents.

    Implement this protocol to use your own agent logic with Techne's
    training loop. The agent is responsible for generating the full
    response text given a prompt.
    """

    async def generate_trajectory(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, list[int] | None]:
        """Generate a complete trajectory for a given prompt.

        Args:
            prompt: The input prompt/query
            **kwargs: Additional arguments passed from rollout config

        Returns:
            Tuple of (text, optional_token_ids):
            - text: The complete generated text (including any tool calls/responses)
            - optional_token_ids: Token IDs if available, None otherwise.
              Providing token_ids avoids decode-encode inconsistency during training.
        """
        ...

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights (optional).

        If your agent uses a model that can be updated during training,
        implement this method to receive weight updates from the policy.

        Args:
            state_dict: New model state dict (may be partial for LoRA)
        """
        ...

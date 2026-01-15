from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A single atomic unit of interaction in a trajectory.

    Represents a message, an action, a thought, or an observation.
    """

    role: Literal["system", "user", "assistant", "environment", "tool"] = Field(
        description="The source of this step."
    )
    content: str = Field(description="The textual content of the step.")

    # Structured Data (Optional)
    tool_calls: list[dict] | None = Field(
        default=None, description="Structured tool calls if this is an action step."
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call this step responds to."
    )

    # Training & RL Metadata
    trainable: bool = Field(
        default=False, description="Whether this step should be trained on (masking)."
    )
    reward: float | None = Field(
        default=None, description="Immediate reward received after this step."
    )
    value: float | None = Field(
        default=None, description="Estimated value of this state (for critics)."
    )

    # Off-policy / Logprob data
    log_probs: list[float] | None = Field(
        default=None, description="Log probabilities of the generated tokens."
    )
    token_ids: list[int] | None = Field(
        default=None, description="Token IDs corresponding to the content."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata (e.g. timestamps, tags)."
    )


class Trajectory(BaseModel):
    """A sequence of steps representing a full episode of interaction."""

    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Episode-level metadata (e.g. source, id, metrics)."
    )
    # Episode-level global reward (optional cache)
    reward: float | None = Field(default=None, description="Total reward for the trajectory.")

    def add_step(self, step: Step):
        self.steps.append(step)

    @property
    def total_reward(self) -> float:
        """Sum of all rewards in the trajectory."""
        return sum(s.reward for s in self.steps if s.reward is not None)

    def to_sft_format(self, user_role: str = "user", assistant_role: str = "assistant") -> str:
        """Simple concatenation for SFT (can be customized)."""
        return "".join(s.content for s in self.steps)

    def get_all_token_ids(self) -> list[int]:
        """Flatten token IDs from all steps."""
        ids = []
        for s in self.steps:
            if s.token_ids:
                ids.extend(s.token_ids)
        return ids

    def to_text(self) -> str:
        """Get the full text content of the trajectory."""
        return "".join(s.content for s in self.steps)


class TrainingSample(BaseModel):
    """A single processed sample ready for the model training loop.

    This abstracts away the details of how the context was constructed (e.g. windowing).
    """

    input_ids: list[int] = Field(description="Full token sequence for this sample.")
    labels: list[int] = Field(
        description="Labels for loss calculation (usually input_ids with masked parts)."
    )

    # RL specific fields
    rewards: list[float] | None = Field(default=None, description="Rewards aligned with input_ids.")
    values: list[float] | None = Field(
        default=None, description="Value estimates aligned with input_ids."
    )
    advantages: list[float] | None = Field(
        default=None, description="Advantages aligned with input_ids."
    )
    log_probs: list[float] | None = Field(default=None, description="Old log probs for PPO.")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Trace back to original trajectory/step."
    )

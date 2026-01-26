from pathlib import Path
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

    def to_training_sample(
        self,
        label_mask_value: int = -100,
        tokenizer: Any | None = None,
    ) -> "TrainingSample":
        """Convert trajectory to a flat, token-aligned training sample.

        Args:
            label_mask_value: Value to use for masked (non-trainable) positions in labels.
            tokenizer: Optional tokenizer for fallback if step.token_ids is None.

        Returns:
            TrainingSample with flattened token-level data.

        Raises:
            ValueError: If steps don't have token_ids and no tokenizer provided.
        """
        input_ids: list[int] = []
        labels: list[int] = []
        rewards: list[float] = []
        values: list[float] = []
        log_probs: list[float] = []

        has_rewards = False
        has_values = False
        has_log_probs = False

        for step in self.steps:
            if step.token_ids is not None:
                tokens = step.token_ids
            elif tokenizer is not None:
                tokens = tokenizer.encode(step.content, add_special_tokens=False)
            else:
                raise ValueError(
                    "Step must have token_ids populated or tokenizer must be provided"
                )

            n_tokens = len(tokens)
            input_ids.extend(tokens)

            # Labels: mask non-trainable steps
            if step.trainable:
                labels.extend(tokens)
            else:
                labels.extend([label_mask_value] * n_tokens)

            # Rewards: assign to last token of the step (common convention)
            if step.reward is not None:
                has_rewards = True
                rewards.extend([0.0] * (n_tokens - 1) + [step.reward])
            else:
                rewards.extend([0.0] * n_tokens)

            # Values: broadcast step value across all tokens
            if step.value is not None:
                has_values = True
                values.extend([step.value] * n_tokens)
            else:
                values.extend([0.0] * n_tokens)

            # Log probs: must match token count
            if step.log_probs is not None:
                has_log_probs = True
                if len(step.log_probs) != n_tokens:
                    raise ValueError(
                        f"Step log_probs length ({len(step.log_probs)}) != "
                        f"token_ids length ({n_tokens})"
                    )
                log_probs.extend(step.log_probs)
            else:
                log_probs.extend([0.0] * n_tokens)

        return TrainingSample(
            input_ids=input_ids,
            labels=labels,
            rewards=rewards if has_rewards else None,
            values=values if has_values else None,
            advantages=None,  # Computed separately (e.g., GAE)
            log_probs=log_probs if has_log_probs else None,
            metadata={"trajectory_metadata": self.metadata},
        )


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


def save_trajectories(
    trajectories: list[Trajectory],
    path: str | Path,
    format: Literal["jsonl", "huggingface"] = "jsonl",
):
    """Save trajectories to disk.

    Args:
        trajectories: List of trajectories to save.
        path: Output path. For jsonl, a file path. For huggingface, a directory.
        format: "jsonl" saves as JSONL file, "huggingface" saves as HF dataset.
    """
    from datasets import Dataset

    path = Path(path)

    if format == "jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")
    else:
        path.mkdir(parents=True, exist_ok=True)
        prompts = [
            [{"role": s.role, "content": s.content} for s in t.steps]
            for t in trajectories
        ]
        ds = Dataset.from_dict({"prompt": prompts})
        ds.save_to_disk(str(path))

        # Also save trajectories.jsonl for full fidelity
        with open(path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file or HuggingFace dataset directory."""
    path = Path(path)

    # If path is a directory, look for trajectories.jsonl inside
    if path.is_dir():
        jsonl_path = path / "trajectories.jsonl"
        if jsonl_path.exists():
            path = jsonl_path
        else:
            raise FileNotFoundError(f"No trajectories.jsonl found in {path}")

    trajectories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(Trajectory.model_validate_json(line))
    return trajectories

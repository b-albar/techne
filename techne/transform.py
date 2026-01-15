from abc import ABC, abstractmethod
from typing import Any

from techne.data import TrainingSample, Trajectory


class TrajectoryTransform(ABC):
    """Abstract base class for converting trajectories into training samples.

    This is where 'Memory Mechanisms' and formatting logic live.
    It takes raw interaction logs and produces what the model actually sees/learns from.
    """

    @abstractmethod
    def process(self, trajectories: list[Trajectory], tokenizer: Any) -> list[TrainingSample]:
        """Convert a list of trajectories into a batch of training samples."""
        pass


class FullHistoryTransform(TrajectoryTransform):
    """Standard transformer behavior: Use full history for every step."""

    def process(self, trajectories: list[Trajectory], tokenizer: Any) -> list[TrainingSample]:
        samples = []
        for traj in trajectories:
            # RL Alignment: Use pre-tokenized IDs to avoid re-tokenization issues
            token_ids = traj.get_all_token_ids()

            # Separate Prompt and Completion
            # Everything before the first trainable step is 'prompt'
            # Everything from the first trainable step onwards is 'completion'
            prompt_ids = []
            completion_ids = []
            found_trainable = False

            for step in traj.steps:
                ids = step.token_ids or []
                if step.trainable:
                    found_trainable = True

                if not found_trainable:
                    prompt_ids.extend(ids)
                else:
                    completion_ids.extend(ids)

            # Mask prompt in labels
            labels = [-100] * len(prompt_ids) + completion_ids

            # Metadata for high-level trainers (like GRPOTrainer)
            metadata = {
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True)
                if prompt_ids
                else "",
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True)
                if completion_ids
                else "",
            }

            # Propagate reward
            reward_list = [traj.reward] * len(token_ids) if traj.reward is not None else None

            samples.append(
                TrainingSample(
                    input_ids=token_ids, labels=labels, rewards=reward_list, metadata=metadata
                )
            )
        return samples

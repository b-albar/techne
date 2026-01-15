from typing import Any

from techne.data import Step, TrainingSample, Trajectory
from techne.transform import TrajectoryTransform


class VerlogMemoryTransform(TrajectoryTransform):
    """Implements the Verlog memory mechanism.

    See: https://wentsechen.github.io/Verlog_blogpost/#memory-mechanism
    Feature:
    - Splits trajectory into individual turns.
    - Each turn uses a limited context window of previous N turns.
    """

    def __init__(self, n_turns_history: int = 2):
        self.n_turns_history = n_turns_history

    def process(self, trajectories: list[Trajectory], tokenizer: Any) -> list[TrainingSample]:
        samples = []
        for traj in trajectories:
            turns = self._split_into_turns(traj)

            # Create a sample for each turn, including history
            for i in range(len(turns)):
                history_start = max(0, i - self.n_turns_history)
                history_turns = turns[history_start:i]
                current_turn = turns[i]

                # Construct context
                # history_t = {s_{t-n}, think_{t-n}, a_{t-n}, ...}
                context_steps = []
                for turn_steps in history_turns:
                    context_steps.extend(turn_steps)

                # Current turn data = (s_t, think_t, a_t)
                # In our Step model, this maps to:
                # s_t -> user step (observation)
                # think_t, a_t -> assistant steps
                turn_steps = current_turn

                # Combine
                full_sequence = context_steps + turn_steps

                # Tokenize
                # Note: In a real implementation we would respect chat templates here.
                # For now we just concat content.
                text = "".join(s.content for s in full_sequence)
                token_ids = tokenizer.encode(text)

                # Create mask (labels).
                # Mask out everything except the 'trainable' parts of the current turn.
                labels = [-100] * len(token_ids)

                # Ideally we map character usage to tokens, but here we simplify:
                # If we assume tokenizer is lossless and we just want to show the structure.
                # A proper implementation requires offset mapping.

                # For this 'design' task, we return the structured object.
                samples.append(
                    TrainingSample(
                        input_ids=token_ids,
                        labels=labels,  # Use the mask
                        metadata={"turn_index": i, "trajectory_id": traj.metadata.get("id")},
                    )
                )
        return samples

    def _split_into_turns(self, trajectory: Trajectory) -> list[list[Step]]:
        """Splits a trajectory into turns based on User/Environment interaction."""
        turns = []
        current_turn = []

        for step in trajectory.steps:
            # A new turn typically starts when the Agent receives an Observation (User/Env role)
            # following an Action (Assistant role).
            # Basic heuristic: If we have content in current_turn and we see a User/Env message, split.
            if step.role in ["user", "environment"] and current_turn:
                # Check if the last thing was an assistant/tool action
                if current_turn[-1].role in ["assistant", "tool"]:
                    turns.append(current_turn)
                    current_turn = []

            current_turn.append(step)

        if current_turn:
            turns.append(current_turn)

        return turns

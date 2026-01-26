"""Verlog memory mechanism for trajectory truncation.

See: https://wentsechen.github.io/Verlog_blogpost/#memory-mechanism
"""

from techne.data import Step, Trajectory


class VerlogMemory:
    """Implements the Verlog memory mechanism.

    Truncates trajectories to a sliding window of the last N turns,
    reducing context length while preserving recent history.
    """

    def __init__(self, n_turns_history: int = 2):
        self.n_turns_history = n_turns_history

    def truncate(self, trajectory: Trajectory, current_turn: int | None = None) -> Trajectory:
        """Truncate trajectory to keep only recent turns.

        Args:
            trajectory: Full trajectory to truncate.
            current_turn: If provided, truncate up to this turn index.
                         If None, keeps the last n_turns_history turns.

        Returns:
            New Trajectory with only the relevant turns.
        """
        turns = self._split_into_turns(trajectory)

        if not turns:
            return Trajectory(metadata=trajectory.metadata.copy())

        if current_turn is not None:
            # Keep history up to current turn
            end_idx = min(current_turn + 1, len(turns))
            start_idx = max(0, end_idx - self.n_turns_history - 1)
            selected_turns = turns[start_idx:end_idx]
        else:
            # Keep last n turns
            start_idx = max(0, len(turns) - self.n_turns_history)
            selected_turns = turns[start_idx:]

        # Flatten turns back into steps
        steps = []
        for turn in selected_turns:
            steps.extend(turn)

        return Trajectory(steps=steps, metadata=trajectory.metadata.copy(), reward=trajectory.reward)

    def expand(self, trajectory: Trajectory) -> list[Trajectory]:
        """Expand trajectory into per-turn training trajectories.

        Creates one trajectory per turn, each with its own sliding window
        of history. Used for training where each turn is a separate sample.

        Args:
            trajectory: Full trajectory to expand.

        Returns:
            List of trajectories, one per turn, each truncated to its window.
        """
        turns = self._split_into_turns(trajectory)
        trajectories = []

        for i in range(len(turns)):
            # Get history window for this turn
            history_start = max(0, i - self.n_turns_history)
            selected_turns = turns[history_start : i + 1]

            # Flatten to steps
            steps = []
            for turn in selected_turns:
                steps.extend(turn)

            trajectories.append(
                Trajectory(
                    steps=steps,
                    metadata={**trajectory.metadata, "turn_index": i, "total_turns": len(turns)},
                    reward=trajectory.reward,
                )
            )

        return trajectories

    def _split_into_turns(self, trajectory: Trajectory) -> list[list[Step]]:
        """Split trajectory into turns based on user/environment boundaries."""
        turns: list[list[Step]] = []
        current_turn: list[Step] = []

        for step in trajectory.steps:
            # New turn starts when user/environment follows assistant/tool
            if step.role in ["user", "environment"] and current_turn:
                if current_turn[-1].role in ["assistant", "tool"]:
                    turns.append(current_turn)
                    current_turn = []

            current_turn.append(step)

        if current_turn:
            turns.append(current_turn)

        return turns

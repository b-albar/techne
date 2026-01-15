from typing import Any, Protocol, runtime_checkable

from techne.data import Trajectory


@runtime_checkable
class Agent(Protocol):
    """Interface for agents that can collect trajectories and accept weight updates.

    This interface dissociates rollout generation from training.
    - On-policy: The agent uses the same model/weights that are currently being trained.
    - Off-policy: The agent is an external entity (e.g., a teacher model or static dataset).
    """

    async def collect_trajectories(self, prompts: list[str]) -> list[Trajectory]:
        """Collect trajectories for the given prompts.

        The agent manages the execution loop, including tool calls and context
        modifications. Each step in the returned Trajectory should capture the
        model's output and any external context changes.
        """
        ...

    async def update_model(self, state_dict: dict[str, Any]) -> None:
        """Update the agent's model weights.

        Used in an on-policy loop to sync trainer weights back to the agent
        after a training step.
        """
        ...

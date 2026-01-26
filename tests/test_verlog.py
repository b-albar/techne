from techne.data import Step, Trajectory
from techne.memory.verlog import VerlogMemory


def test_verlog_truncate():
    """Test basic trajectory truncation."""
    traj = Trajectory(metadata={"id": "test"})

    # Create 5 turns
    for i in range(5):
        traj.add_step(Step(role="user", content=f"Observation {i} "))
        traj.add_step(Step(role="assistant", content=f"Action {i} ", trainable=True))

    memory = VerlogMemory(n_turns_history=2)

    # Truncate to last 2 turns
    truncated = memory.truncate(traj)

    # Should have 2 turns = 4 steps (user + assistant each)
    assert len(truncated.steps) == 4
    assert "Observation 3" in truncated.steps[0].content
    assert "Observation 4" in truncated.steps[2].content


def test_verlog_expand():
    """Test expanding trajectory into per-turn samples."""
    traj = Trajectory(metadata={"id": "test"})

    # Create 5 turns
    for i in range(5):
        traj.add_step(Step(role="user", content=f"Observation {i} "))
        traj.add_step(Step(role="assistant", content=f"Action {i} ", trainable=True))

    memory = VerlogMemory(n_turns_history=2)
    expanded = memory.expand(traj)

    # Should get 5 trajectories (one per turn)
    assert len(expanded) == 5

    # First turn has no history
    assert len(expanded[0].steps) == 2

    # Third turn has 2 turns of history + current = 3 turns = 6 steps
    assert len(expanded[2].steps) == 6

    # Fifth turn also has 2 turns history + current = 6 steps
    assert len(expanded[4].steps) == 6

    # Metadata should include turn info
    assert expanded[2].metadata["turn_index"] == 2
    assert expanded[2].metadata["total_turns"] == 5


def test_verlog_empty_trajectory():
    """Test handling of empty trajectory."""
    traj = Trajectory(metadata={"id": "empty"})
    memory = VerlogMemory(n_turns_history=2)

    truncated = memory.truncate(traj)
    assert len(truncated.steps) == 0

    expanded = memory.expand(traj)
    assert len(expanded) == 0


if __name__ == "__main__":
    test_verlog_truncate()
    test_verlog_expand()
    test_verlog_empty_trajectory()
    print("All Verlog tests passed.")

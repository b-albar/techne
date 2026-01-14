import pytest

from techne.config import TagConfig
from techne.rollout.orchestrator import BlackBoxOrchestrator
from techne.rollout.parser import TagParser


class MockExternalAgent:
    """Mock external agent for testing."""

    async def generate_trajectory(self, prompt, **kwargs):
        # Return a canned response with tool usage
        text = f"Thinking about {prompt}. <code>print('hello')</code><interpreter>hello</interpreter>Done."
        return (text, None)  # No token_ids for basic mock


@pytest.mark.asyncio
async def test_black_box_orchestrator():
    """Test standard flow of BlackBoxOrchestrator."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = MockExternalAgent()

    orchestrator = BlackBoxOrchestrator(agent, parser)

    trajectory = await orchestrator.rollout_single("Question")

    assert trajectory.initial_prompt == "Question"
    assert len(trajectory.turns) == 1
    assert "print('hello')" in trajectory.final_response

    # Verify masking
    turn = trajectory.turns[0]
    assert len(turn.masked_ranges) == 1
    # Check that mask range covers the interpreter tag
    start, end = turn.masked_ranges[0]
    # The parser returns range of inner content depending on implementation
    # Actually TagParser.get_response_mask_ranges returns the FULL tag range including <interpreter>
    # Let's verify what the parser does via the slice
    assert (
        "<interpreter>hello</interpreter>" in turn.generation.text[start:end]
        or "hello" in turn.generation.text[start:end]
    )


@pytest.mark.asyncio
async def test_black_box_batch():
    """Test batch execution."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = MockExternalAgent()
    orchestrator = BlackBoxOrchestrator(agent, parser)

    trajectories = await orchestrator.rollout_batch(["Q1", "Q2"])
    assert len(trajectories) == 2
    assert trajectories[0].initial_prompt == "Q1"
    assert trajectories[1].initial_prompt == "Q2"

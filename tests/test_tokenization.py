"""Tests for tokenization consistency in rollout orchestrators.

This test suite verifies that turn-by-turn token concatenation matches
the actual generated tokens, preventing decode-encode inconsistency during RL training.
"""

from dataclasses import dataclass

import pytest

from techne.config import RolloutConfig, TagConfig
from techne.rollout.backends.base import GenerationOutput
from techne.rollout.orchestrator import BlackBoxOrchestrator, RolloutOrchestrator, Trajectory
from techne.rollout.parser import TagParser


class MockBackend:
    """Mock backend that tracks token_ids for testing."""

    def __init__(self):
        self.is_ready = True
        self.name = "mock"
        self._call_count = 0

    async def generate_with_stop_on_tags(self, prompts, stop_tags, config):
        """Generate with known token_ids."""
        self._call_count += 1

        # Simulate different responses based on call count
        if self._call_count == 1:
            # First turn: generate a tool call
            text = "Let me calculate. <code>2+2</code>"
            # Simulate token_ids: [1, 2, 3, 4, 5] for "Let me calculate. <code>2+2</code>"
            token_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108]
        else:
            # Second turn: final response
            text = " The answer is 4."
            # Simulate token_ids: [6, 7, 8] for " The answer is 4."
            token_ids = [200, 201, 202, 203, 204]

        return [
            GenerationOutput(
                text=text,
                token_ids=token_ids,
                generated_tokens=len(token_ids),
                finish_reason="stop",
            )
        ]

    async def start(self):
        pass

    async def stop(self):
        pass

    async def update_weights(self, state_dict):
        pass


class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(self, tags):
        self._tags = tags

    async def execute_all(self, text):
        """Return mock execution results."""

        @dataclass
        class MockResult:
            def format_for_model(self):
                return "4"

        return [MockResult()]

    def format_response(self, text, results):
        """Format with interpreter tags."""
        return text + "<interpreter>4</interpreter>"


class MockExternalAgentWithTokens:
    """Mock external agent that provides token_ids."""

    async def generate_trajectory(self, prompt, **kwargs):
        text = f"Analyzing {prompt}. <code>result</code><interpreter>done</interpreter>Complete."
        # Provide token_ids to avoid re-tokenization
        token_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        return (text, token_ids)


class MockExternalAgentWithoutTokens:
    """Mock external agent that doesn't provide token_ids."""

    async def generate_trajectory(self, prompt, **kwargs):
        text = f"Analyzing {prompt}. <code>result</code><interpreter>done</interpreter>Complete."
        # Old-style return (backwards compatibility)
        return (text, None)


@pytest.mark.asyncio
async def test_whitebox_preserves_token_ids():
    """Test that RolloutOrchestrator preserves actual generated token_ids."""
    tags = TagConfig()
    parser = TagParser(tags)
    backend = MockBackend()
    executor = MockToolExecutor(tags)

    rollout_config = RolloutConfig(max_turns=2)

    orchestrator = RolloutOrchestrator(
        backend=backend,
        tool_executor=executor,
        parser=parser,
        rollout_config=rollout_config,
        tags=tags,
    )

    trajectory = await orchestrator.rollout_single("Calculate 2+2")

    # Verify token_ids are stored in turns
    assert len(trajectory.turns) == 2
    assert trajectory.turns[0].token_ids == [100, 101, 102, 103, 104, 105, 106, 107, 108]
    assert trajectory.turns[1].token_ids == [200, 201, 202, 203, 204]

    # Verify all_token_ids concatenates correctly
    expected_all_tokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 200, 201, 202, 203, 204]
    assert trajectory.all_token_ids == expected_all_tokens

    # Verify total_tokens matches
    assert trajectory.total_tokens == len(expected_all_tokens)


@pytest.mark.asyncio
async def test_whitebox_empty_trajectory():
    """Test that empty trajectory handles token_ids gracefully."""
    trajectory = Trajectory(initial_prompt="test")

    assert trajectory.all_token_ids == []
    assert trajectory.total_tokens == 0


@pytest.mark.asyncio
async def test_blackbox_with_token_ids():
    """Test BlackBoxOrchestrator when agent provides token_ids."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = MockExternalAgentWithTokens()

    orchestrator = BlackBoxOrchestrator(agent, parser)
    trajectory = await orchestrator.rollout_single("Test prompt")

    # Verify token_ids are stored
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].token_ids == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert trajectory.all_token_ids == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert trajectory.total_tokens == 9


@pytest.mark.asyncio
async def test_blackbox_without_token_ids():
    """Test BlackBoxOrchestrator when agent doesn't provide token_ids."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = MockExternalAgentWithoutTokens()

    orchestrator = BlackBoxOrchestrator(agent, parser)
    trajectory = await orchestrator.rollout_single("Test prompt")

    # Verify token_ids are empty (will need re-tokenization during training)
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].token_ids == []
    assert trajectory.all_token_ids == []
    # Falls back to approximation
    assert trajectory.total_tokens > 0


@pytest.mark.asyncio
async def test_no_decode_encode_mismatch():
    """
    Test that demonstrates the fix: token_ids from turns should equal
    the actual generated tokens, NOT re-tokenized concatenated text.

    This is the critical test that would fail with the old approach.
    """
    tags = TagConfig()
    parser = TagParser(tags)
    backend = MockBackend()
    executor = MockToolExecutor(tags)

    rollout_config = RolloutConfig(max_turns=2)

    orchestrator = RolloutOrchestrator(
        backend=backend,
        tool_executor=executor,
        parser=parser,
        rollout_config=rollout_config,
        tags=tags,
    )

    trajectory = await orchestrator.rollout_single("Test")

    # The key assertion: concatenated token_ids should match generated tokens
    # This is what preserves the policy model distribution
    actual_generated_tokens = []
    for turn in trajectory.turns:
        if turn.generation:
            actual_generated_tokens.extend(turn.generation.token_ids)

    # This should be identical (no decode-encode mismatch)
    assert trajectory.all_token_ids == actual_generated_tokens

    # In the OLD approach, we would re-tokenize full_text:
    # retokenized = tokenizer(trajectory.full_text)["input_ids"]
    # assert retokenized != trajectory.all_token_ids  # This would be the mismatch!

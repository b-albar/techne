"""
Additional tests for BlackBoxOrchestrator edge cases and validation.
"""

import pytest

from techne.config import TagConfig
from techne.rollout.orchestrator import BlackBoxOrchestrator
from techne.rollout.parser import TagParser


class EmptyResponseAgent:
    """Agent that returns empty responses."""

    async def generate_trajectory(self, prompt, **kwargs):
        return ""


class NoneResponseAgent:
    """Agent that returns None."""

    async def generate_trajectory(self, prompt, **kwargs):
        return None


class WhitespaceResponseAgent:
    """Agent that returns only whitespace."""

    async def generate_trajectory(self, prompt, **kwargs):
        return "   \n\t  "


@pytest.mark.asyncio
async def test_empty_response_validation():
    """Test that empty responses are properly rejected."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = EmptyResponseAgent()
    orchestrator = BlackBoxOrchestrator(agent, parser)

    with pytest.raises(ValueError, match="empty response"):
        await orchestrator.rollout_single("Test prompt")


@pytest.mark.asyncio
async def test_none_response_validation():
    """Test that None responses are properly rejected."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = NoneResponseAgent()
    orchestrator = BlackBoxOrchestrator(agent, parser)

    with pytest.raises(ValueError, match="empty response"):
        await orchestrator.rollout_single("Test prompt")


@pytest.mark.asyncio
async def test_whitespace_response_validation():
    """Test that whitespace-only responses are properly rejected."""
    tags = TagConfig()
    parser = TagParser(tags)
    agent = WhitespaceResponseAgent()
    orchestrator = BlackBoxOrchestrator(agent, parser)

    with pytest.raises(ValueError, match="empty response"):
        await orchestrator.rollout_single("Test prompt")

"""Multi-turn rollout orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from techne.config import RolloutConfig, TagConfig
from techne.rollout.backends.base import GenerationConfig, GenerationOutput, RolloutBackend
from techne.rollout.parser import TagParser
from techne.tools.executor import ToolExecutor


@dataclass
class Turn:
    """A single turn in a multi-turn conversation.

    Attributes:
        prompt: The prompt/input for this turn
        generation: Model's generated response
        tool_calls: Tool calls extracted from generation
        tool_responses: Responses from tool execution
        masked_ranges: Character ranges to mask in loss computation
        token_ids: Actual generated token IDs for this turn
        masked_token_indices: Token indices to mask in loss computation
    """

    prompt: str
    generation: GenerationOutput | None = None
    tool_calls: list[str] = field(default_factory=list)
    tool_responses: list[str] = field(default_factory=list)
    masked_ranges: list[tuple[int, int]] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    masked_token_indices: list[int] = field(default_factory=list)


@dataclass
class Trajectory:
    """A complete multi-turn trajectory.

    Attributes:
        initial_prompt: The starting prompt
        turns: List of conversation turns
        final_response: Final response after all tool calls
        reward: Reward for this trajectory (set after evaluation)
        metadata: Additional metadata
    """

    initial_prompt: str
    turns: list[Turn] = field(default_factory=list)
    final_response: str = ""
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get full conversation text."""
        parts = [self.initial_prompt]
        for turn in self.turns:
            if turn.generation:
                parts.append(turn.generation.text)
        return "".join(parts)

    @property
    def all_token_ids(self) -> list[int]:
        """Get concatenated token IDs from all turns.

        This preserves the actual generated tokens without decode-encode inconsistency.
        Prefer this over re-tokenizing full_text for RL training.
        """
        token_ids = []
        for turn in self.turns:
            token_ids.extend(turn.token_ids)
        return token_ids

    @property
    def total_tokens(self) -> int:
        """Get total generated tokens."""
        return (
            len(self.all_token_ids)
            if self.all_token_ids
            else sum(turn.generation.generated_tokens for turn in self.turns if turn.generation)
        )

    @property
    def num_tool_calls(self) -> int:
        """Get total number of tool calls."""
        return sum(len(turn.tool_calls) for turn in self.turns)


class RolloutOrchestrator:
    """Orchestrates multi-turn rollouts with tool integration.

    The orchestrator:
    1. Generates model responses using the backend
    2. Detects tool calls in generated text
    3. Executes tools and formats responses
    4. Continues generation until completion or max turns
    5. Tracks masked ranges for loss computation
    """

    def __init__(
        self,
        backend: RolloutBackend,
        tool_executor: ToolExecutor,
        parser: TagParser,
        rollout_config: RolloutConfig,
        tags: TagConfig,
    ):
        """Initialize orchestrator.

        Args:
            backend: Rollout backend for generation
            tool_executor: Tool executor for handling tool calls
            parser: Tag parser for detecting tool calls
            rollout_config: Rollout configuration
            tags: Tag configuration
        """
        self._backend = backend
        self._executor = tool_executor
        self._parser = parser
        self._config = rollout_config
        self._tags = tags

    async def rollout_single(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> Trajectory:
        """Execute a single multi-turn rollout.

        Args:
            prompt: Initial prompt
            generation_config: Generation configuration

        Returns:
            Complete trajectory with all turns
        """
        config = generation_config or GenerationConfig(
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )

        trajectory = Trajectory(initial_prompt=prompt)
        current_prompt = prompt
        stop_tags = [self._tags.tool_end]

        for turn_idx in range(self._config.max_turns):
            # Generate with stopping on tool end tag
            outputs = await self._backend.generate_with_stop_on_tags(
                [current_prompt],
                stop_tags=stop_tags,
                config=config,
            )
            output = outputs[0]

            turn = Turn(prompt=current_prompt, generation=output)

            # Store the actual generated token_ids from this generation
            turn.token_ids = output.token_ids.copy()

            # Check for tool calls
            if self._parser.has_tool_call(output.text):
                # Extract and execute tool calls
                parsed_calls = self._parser.parse_tool_calls(output.text)
                turn.tool_calls = [call.content for call in parsed_calls]

                # Execute tools
                results = await self._executor.execute_all(output.text)
                turn.tool_responses = [r.format_for_model() for r in results]

                # Format response with tool outputs
                formatted_text = self._executor.format_response(output.text, results)

                # Track masked ranges (interpreter responses)
                turn.masked_ranges = self._parser.get_response_mask_ranges(formatted_text)

                # Prepare next prompt
                current_prompt = current_prompt + formatted_text

            trajectory.turns.append(turn)

            # Check if we should stop
            if not self._parser.has_tool_call(output.text):
                # No tool call means we're done
                trajectory.final_response = output.text
                break

            if output.finish_reason == "length":
                # Hit length limit
                break

        return trajectory

    async def rollout_batch(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[Trajectory]:
        """Execute batch of multi-turn rollouts.

        Args:
            prompts: List of initial prompts
            generation_config: Generation configuration

        Returns:
            List of complete trajectories
        """
        # Run rollouts concurrently
        tasks = [self.rollout_single(prompt, generation_config) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def get_token_mask(self, trajectory: Trajectory, tokenizer) -> list[bool]:
        """Get token-level mask for loss computation.

        Tokens corresponding to interpreter responses should be masked out.

        Args:
            trajectory: Completed trajectory
            tokenizer: Tokenizer for converting text to tokens

        Returns:
            List of booleans (True = include in loss, False = mask out)
        """
        full_text = trajectory.full_text

        # Tokenize full text
        tokens = tokenizer(full_text, return_offsets_mapping=True)
        offsets = tokens["offset_mapping"]

        # Collect all masked character ranges
        all_masked_ranges = []
        for turn in trajectory.turns:
            all_masked_ranges.extend(turn.masked_ranges)

        # Create token mask
        mask = []
        for start, end in offsets:
            # Check if this token overlaps with any masked range
            is_masked = any(
                start < mask_end and end > mask_start for mask_start, mask_end in all_masked_ranges
            )
            mask.append(not is_masked)

        return mask


def create_orchestrator(
    backend: RolloutBackend,
    tags: TagConfig,
    rollout_config: RolloutConfig,
    tool_executor: ToolExecutor | None = None,
) -> RolloutOrchestrator:
    """Factory function to create a configured orchestrator.

    Args:
        backend: Rollout backend
        tags: Tag configuration
        rollout_config: Rollout configuration
        tool_executor: Optional tool executor (created if not provided)

    Returns:
        Configured RolloutOrchestrator
    """
    parser = TagParser(tags)

    if tool_executor is None:
        from techne.tools.sandbox import MicrosandboxTool

        tool_executor = ToolExecutor(tags)
        tool_executor.register_tool(MicrosandboxTool())

    return RolloutOrchestrator(
        backend=backend,
        tool_executor=tool_executor,
        parser=parser,
        rollout_config=rollout_config,
        tags=tags,
    )


class BlackBoxOrchestrator:
    """Orchestrator for external/black-box agents.

    Delegates generation to an ExternalAgent and uses TagParser to
    identify tool outputs for masking.
    """

    def __init__(
        self,
        agent: Any,  # ExternalAgent protocol - using Any to avoid runtime Protocol issues
        parser: TagParser,
    ):
        """Initialize black-box orchestrator.

        Args:
            agent: Implementation of ExternalAgent protocol
            parser: TagParser for detecting tool outputs
        """
        self._agent = agent
        self._parser = parser

    async def rollout_single(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> Trajectory:
        """Execute a single rollout using external agent.

        Args:
            prompt: Initial prompt
            generation_config: Config passed to agent via kwargs

        Returns:
            Trajectory with single turn containing full generation
        """
        # Call external agent
        # We pass generation config attributes as kwargs if available
        kwargs = {}
        if generation_config:
            kwargs = {
                "max_new_tokens": generation_config.max_new_tokens,
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
            }

        result = await self._agent.generate_trajectory(prompt, **kwargs)

        # Handle both old (str) and new (tuple) return formats for backwards compatibility
        if isinstance(result, tuple):
            full_text, optional_token_ids = result
        else:
            full_text = result
            optional_token_ids = None

        # Validate response
        if not full_text or not full_text.strip():
            raise ValueError(f"External agent returned empty response for prompt: {prompt[:50]}...")

        # Create trajectory object
        trajectory = Trajectory(initial_prompt=prompt)

        # Analyze the result with parser to find tool calls/responses
        # For training, we treat this as a single "turn" containing the full execution trace
        # but we need to identify masked ranges (tool outputs)

        masked_ranges = self._parser.get_response_mask_ranges(full_text)

        # Create a single Turn representing the whole execution
        # We don't have separate generation/tool_response objects easily
        # so we put everything into generation output for simplicity
        generation_output = GenerationOutput(
            text=full_text,
            token_ids=optional_token_ids or [],  # Use provided token_ids if available
            generated_tokens=len(optional_token_ids)
            if optional_token_ids
            else len(full_text.split()),
            finish_reason="stop",
        )

        turn = Turn(
            prompt=prompt,
            generation=generation_output,
            masked_ranges=masked_ranges,
            token_ids=optional_token_ids or [],  # Store token_ids if available
            # We don't populate tool_calls/responses separately as they are baked into text
        )

        trajectory.turns.append(turn)
        trajectory.final_response = full_text

        return trajectory

    async def rollout_batch(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[Trajectory]:
        """Execute batch of rollouts.

        Args:
            prompts: List of initial prompts
            generation_config: Generation configuration

        Returns:
            List of complete trajectories
        """
        tasks = [self.rollout_single(prompt, generation_config) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def get_token_mask(self, trajectory: Trajectory, tokenizer) -> list[bool]:
        """Get token-level mask for loss computation.

        Args:
            trajectory: Completed trajectory
            tokenizer: Tokenizer for converting text to tokens

        Returns:
            List of booleans (True = include in loss, False = mask out)
        """
        if not trajectory.turns:
            return []

        if not trajectory.turns[0].generation:
            return []

        # Use the generation text (not full_text which includes initial_prompt)
        full_text = trajectory.turns[0].generation.text

        # Tokenize
        tokens = tokenizer(full_text, return_offsets_mapping=True)
        offsets = tokens["offset_mapping"]

        # Get masked ranges
        masked_ranges = trajectory.turns[0].masked_ranges

        # Create mask
        mask = []
        for start, end in offsets:
            is_masked = any(
                start < mask_end and end > mask_start for mask_start, mask_end in masked_ranges
            )
            mask.append(not is_masked)

        return mask

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update agent weights if supported.

        Args:
            state_dict: New model state dict
        """
        # Check if agent has update_weights method (duck typing)
        if hasattr(self._agent, "update_weights") and callable(self._agent.update_weights):
            await self._agent.update_weights(state_dict)

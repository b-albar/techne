"""SGLang backend for rollout generation."""

from __future__ import annotations

from typing import Any

from techne.config import ModelConfig, RolloutConfig
from techne.rollout.backends.base import GenerationConfig, GenerationOutput, RolloutBackend


class SGLangBackend(RolloutBackend):
    """SGLang-based rollout backend.

    Provides high-performance inference with:
    - RadixAttention for efficient multi-turn caching
    - Optimized for agentic/multi-turn workflows
    - Tensor parallelism for multi-GPU
    - Native multi-turn conversation support
    """

    def __init__(
        self,
        model_config: ModelConfig,
        rollout_config: RolloutConfig,
    ):
        """Initialize SGLang backend.

        Args:
            model_config: Model configuration
            rollout_config: Rollout configuration
        """
        self._model_config = model_config
        self._rollout_config = rollout_config
        self._runtime = None
        self._ready = False

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        """Start SGLang runtime."""
        try:
            import sglang as sgl
        except ImportError as e:
            raise ImportError(
                "SGLang not installed. Install with: pip install techne[sglang]"
            ) from e

        # Configure and start runtime
        self._runtime = sgl.Runtime(
            model_path=self._model_config.name_or_path,
            tp_size=self._rollout_config.tensor_parallel_size,
            dtype=self._model_config.torch_dtype,
            trust_remote_code=self._model_config.trust_remote_code,
            mem_fraction_static=self._rollout_config.gpu_memory_utilization,
        )

        # Set default backend
        sgl.set_default_backend(self._runtime)
        self._ready = True

    async def stop(self) -> None:
        """Stop SGLang runtime."""
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None
            self._ready = False

    async def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions using SGLang.

        Args:
            prompts: List of prompt strings
            config: Generation configuration

        Returns:
            List of GenerationOutput objects
        """
        if not self._ready:
            raise RuntimeError("Backend not started. Call start() first.")

        config = config or GenerationConfig()
        return await self._generate_internal(prompts, config)

    async def generate_with_stop_on_tags(
        self,
        prompts: list[str],
        stop_tags: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate with stopping on specified tags.

        Args:
            prompts: List of prompt strings
            stop_tags: Tags that trigger stop
            config: Generation configuration

        Returns:
            List of GenerationOutput objects
        """
        config = config or GenerationConfig()
        config.stop_strings = list(set(config.stop_strings + stop_tags))
        return await self.generate(prompts, config)

    async def _generate_internal(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        """Internal generation implementation using SGLang."""
        import sglang as sgl

        # Build generation function
        @sgl.function
        def generate_fn(s, prompt: str):
            s += prompt
            s += sgl.gen(
                "response",
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_strings if config.stop_strings else None,
            )

        # Run batch generation
        states = generate_fn.run_batch(
            [{"prompt": p} for p in prompts],
            progress_bar=False,
        )

        # Convert to GenerationOutput
        results = []
        for state in states:
            text = state["response"]
            results.append(
                GenerationOutput(
                    text=text,
                    token_ids=[],  # SGLang doesn't expose token IDs in simple API
                    generated_tokens=len(text.split()),  # Approximate
                    finish_reason="stop"
                    if any(s in text for s in config.stop_strings)
                    else "length",
                )
            )

        return results

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights (for LoRA during RL).

        Args:
            state_dict: New state dict with updated weights
        """
        if not self._ready:
            raise RuntimeError("Backend not started.")

        # SGLang supports weight updates through its LoRA mechanisms
        # This is a simplified placeholder - real implementation would
        # use SGLang's weight update API
        if self._model_config.lora.enabled:
            # SGLang's update mechanism
            self._runtime.update_weights(state_dict)

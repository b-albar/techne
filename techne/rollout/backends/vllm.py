"""vLLM backend for rollout generation."""

from __future__ import annotations

from typing import Any

from techne.config import ModelConfig, RolloutConfig
from techne.rollout.backends.base import GenerationConfig, GenerationOutput, RolloutBackend


class VLLMBackend(RolloutBackend):
    """vLLM-based rollout backend.

    Provides high-performance inference with:
    - PagedAttention for efficient memory usage
    - Tensor parallelism for multi-GPU
    - LoRA adapter support
    - Async generation API
    """

    def __init__(
        self,
        model_config: ModelConfig,
        rollout_config: RolloutConfig,
    ):
        """Initialize vLLM backend.

        Args:
            model_config: Model configuration
            rollout_config: Rollout configuration
        """
        self._model_config = model_config
        self._rollout_config = rollout_config
        self._engine = None
        self._tokenizer = None
        self._ready = False

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        """Start vLLM engine."""
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
        except ImportError as e:
            raise ImportError("vLLM not installed. Install with: pip install techne[vllm]") from e

        # Configure engine
        engine_args = AsyncEngineArgs(
            model=self._model_config.name_or_path,
            dtype=self._model_config.torch_dtype,
            trust_remote_code=self._model_config.trust_remote_code,
            tensor_parallel_size=self._rollout_config.tensor_parallel_size,
            gpu_memory_utilization=self._rollout_config.gpu_memory_utilization,
            enable_lora=self._model_config.lora.enabled,
            max_lora_rank=self._model_config.lora.r if self._model_config.lora.enabled else None,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._tokenizer = await self._engine.get_tokenizer()
        self._ready = True

    async def stop(self) -> None:
        """Stop vLLM engine."""
        if self._engine is not None:
            # vLLM doesn't have explicit shutdown, just cleanup
            self._engine = None
            self._tokenizer = None
            self._ready = False

    async def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions using vLLM.

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
        """Internal generation implementation."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            stop=config.stop_strings if config.stop_strings else None,
            include_stop_str_in_output=config.include_stop_str_in_output,
        )

        # Generate for all prompts
        request_ids = []

        for i, prompt in enumerate(prompts):
            request_id = f"req_{i}"
            request_ids.append(request_id)
            await self._engine.add_request(request_id, prompt, sampling_params)

        # Collect results
        outputs_map: dict[str, GenerationOutput] = {}

        async for request_output in self._engine.generate(None):
            req_id = request_output.request_id
            if request_output.finished:
                output = request_output.outputs[0]
                outputs_map[req_id] = GenerationOutput(
                    text=output.text,
                    token_ids=list(output.token_ids),
                    prompt_tokens=len(request_output.prompt_token_ids),
                    generated_tokens=len(output.token_ids),
                    finish_reason=output.finish_reason or "unknown",
                )

            if len(outputs_map) == len(prompts):
                break

        # Return in order
        return [outputs_map[f"req_{i}"] for i in range(len(prompts))]

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights (for LoRA during RL).

        Args:
            state_dict: New state dict with updated weights
        """
        if not self._ready:
            raise RuntimeError("Backend not started.")

        # vLLM supports LoRA adapter updates
        if self._model_config.lora.enabled:
            # Convert state dict to LoRA adapter format if needed
            # This is a simplified version - real implementation would
            # use vLLM's LoRA loading mechanisms
            await self._engine.add_lora(
                lora_name="techne_lora",
                lora_path=None,  # Would be the path or state dict
            )

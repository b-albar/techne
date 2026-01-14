"""Hugging Face backend for rollout generation."""

from __future__ import annotations

import asyncio
from typing import Any

from techne.config import ModelConfig, RolloutConfig
from techne.rollout.backends.base import GenerationConfig, GenerationOutput, RolloutBackend


class HuggingFaceBackend(RolloutBackend):
    """Hugging Face transformers-based rollout backend.

    Standard PyTorch inference using transformers library.
    Useful for:
    - Debugging and development
    - Small models or CPU inference
    - Custom model architectures not supported by vLLM/SGLang
    """

    def __init__(
        self,
        model_config: ModelConfig,
        rollout_config: RolloutConfig,
    ):
        """Initialize Hugging Face backend.

        Args:
            model_config: Model configuration
            rollout_config: Rollout configuration
        """
        self._model_config = model_config
        self._rollout_config = rollout_config
        self._model = None
        self._tokenizer = None
        self._ready = False

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        """Start backend (load model)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers not installed. Install with: pip install techne[hf]"
            ) from e

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.name_or_path,
            trust_remote_code=self._model_config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self._model_config.trust_remote_code,
            "torch_dtype": getattr(torch, self._model_config.torch_dtype)
            if isinstance(self._model_config.torch_dtype, str)
            else self._model_config.torch_dtype,
            "device_map": "auto",
        }

        # Add attention implementation if specified
        if self._model_config.attn_implementation:
            model_kwargs["attn_implementation"] = self._model_config.attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_config.name_or_path,
            **model_kwargs,
        )

        # Apply LoRA if enabled
        if self._model_config.lora.enabled:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType(self._model_config.lora.task_type),
                inference_mode=True,
                r=self._model_config.lora.r,
                lora_alpha=self._model_config.lora.alpha,
                lora_dropout=self._model_config.lora.dropout,
                target_modules=self._model_config.lora.target_modules,
            )
            self._model = get_peft_model(self._model, peft_config)

        self._ready = True

    async def stop(self) -> None:
        """Stop backend (unload model)."""
        import gc
        import torch

        self._model = None
        self._tokenizer = None
        self._ready = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions using transformers."""
        if not self._ready:
            raise RuntimeError("Backend not started. Call start() first.")

        config = config or GenerationConfig()

        # Run generation in thread pool to avoid blocking asyncio loop
        return await asyncio.to_thread(self._generate_sync, prompts, config)

    def _generate_sync(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        """Synchronous generation implementation."""
        import torch

        results = []

        # Batch generation could be optimized, but for simplicity we iterate or support simple batching
        # Transformers generate supports batching if inputs are padded

        # Tokenize
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(
            self._model.device
        )

        input_len = inputs.input_ids.shape[1]

        # Generate
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        if config.stop_strings:
            # Simple stop string handling requires custom stopping criteria or post-processing
            # We'll rely on post-processing for simplicity in this basic backend
            # or could use StopStringCriteria if available in transformers version
            pass

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Process outputs
        # We need to extract just the new tokens, but batch_decode gives full text usually
        # Actually it depends. We want full text or just response?
        # Base class implies we return the generated text mainly, but usually standard is full text or response.
        # Let's align with other backends: usually returning the *response* part is useful if we want to append.
        # But `generate` usually returns the full completion or suffix.
        # Let's check `GenerationOutput` doc: "Generated text".
        # If we look at `sglang.py`, it does `s += sgl.gen("response"...)` and returns `state["response"]`.
        # So we should return only the NEWLY generated text.

        for i, full_output in enumerate(outputs):
            # Slice off input tokens
            generated_ids = full_output[input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Handle stop strings (naive post-processing)
            finish_reason = "length"
            if config.stop_strings:
                for stop_str in config.stop_strings:
                    if stop_str in text:
                        split_idx = text.find(stop_str)
                        if config.include_stop_str_in_output:
                            text = text[: split_idx + len(stop_str)]
                        else:
                            text = text[:split_idx]
                        finish_reason = "stop"
                        break

            results.append(
                GenerationOutput(
                    text=text,
                    token_ids=generated_ids.tolist(),
                    prompt_tokens=input_len,
                    generated_tokens=len(generated_ids),
                    finish_reason=finish_reason,
                )
            )

        return results

    async def generate_with_stop_on_tags(
        self,
        prompts: list[str],
        stop_tags: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationOutput]:
        """Generate with stopping on specified tags."""
        config = config or GenerationConfig()
        config.stop_strings = list(set(config.stop_strings + stop_tags))
        return await self.generate(prompts, config)

    async def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights (for LoRA during RL)."""
        if not self._ready:
            raise RuntimeError("Backend not started.")

        # Determine if we're updating LoRA or full model
        # state_dict keys usually have 'base_model.model...' or 'lora_A...'.
        # Assuming standard Peft loading:

        # Use load_state_dict. strict=False is safer for partial updates (LoRA)
        self._model.load_state_dict(state_dict, strict=False)

"""HuggingFace implementations of Techne model interfaces.

This module provides:
- HuggingFaceInferenceModel: For inference, evaluation, and teacher models
- HuggingFaceTrainingModel: For training with gradient support
- create_teacher_model: Factory for distillation teacher models
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from techne.training.model import InferenceModel, TrainingModel


class HuggingFaceInferenceModel(InferenceModel):
    """HuggingFace implementation of InferenceModel.

    Used for:
    - Evaluation on a single GPU
    - Teacher model in distillation
    - Data generation

    Supports KV cache for efficient autoregressive inference.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device or str(next(model.parameters()).device)
        self._config = model.config

        # KV cache for inference
        self._kv_cache: tuple | None = None
        self._cache_seq_len: int = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> HuggingFaceInferenceModel:
        """Load a model from HuggingFace."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, device=device)

    @property
    def device(self) -> str:
        return self._device

    @property
    def config(self) -> Any:
        return self._config

    def generate(self, input_ids: torch.Tensor = None, **kwargs) -> Any:
        """Generate tokens."""
        return self._model.generate(input_ids=input_ids, **kwargs)

    def clear_kv_cache(self):
        """Clear the KV cache."""
        self._kv_cache = None
        self._cache_seq_len = 0

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Forward pass with KV cache support.

        Args:
            input_ids: Input token IDs [batch, seq] or [batch, 1] for cached inference
            use_cache: Whether to use/update the KV cache

        Returns:
            logits: Output logits [batch, seq, vocab]
            past_key_values: Updated KV cache (if use_cache=True)
        """
        with torch.no_grad():
            outputs = self._model(
                input_ids,
                past_key_values=self._kv_cache if use_cache else None,
                use_cache=use_cache,
            )

        if use_cache:
            self._kv_cache = outputs.past_key_values
            self._cache_seq_len += input_ids.shape[1]

        return outputs.logits, outputs.past_key_values if use_cache else None

    def generate_with_cache(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_token_ids: list[int] | None = None,
        continue_from_cache: bool = False,
        keep_cache: bool = False,
    ) -> tuple[list[int], list[float]]:
        """Generate tokens using KV cache for efficiency.

        Supports multi-turn generation by preserving cache between calls.

        Args:
            prompt_ids: Prompt token IDs to process before generating
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            stop_token_ids: Token IDs that stop generation
            continue_from_cache: If True, don't clear cache at start (for multi-turn)
            keep_cache: If True, don't clear cache at end (for multi-turn)

        Returns:
            generated_ids: List of generated token IDs (excluding prompt)
            log_probs: Log probabilities for each generated token
        """
        if not continue_from_cache:
            self.clear_kv_cache()

        if stop_token_ids is None:
            stop_token_ids = []
        if self._tokenizer and self._tokenizer.eos_token_id is not None:
            stop_token_ids = list(stop_token_ids) + [self._tokenizer.eos_token_id]

        input_ids = torch.tensor([prompt_ids], device=self._device)
        generated_ids: list[int] = []
        log_probs: list[float] = []

        # Process prompt (prefill)
        logits, _ = self.forward_with_cache(input_ids, use_cache=True)
        next_logits = logits[:, -1, :]  # [batch, vocab]

        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature > 0:
                scaled_logits = next_logits / temperature
            else:
                scaled_logits = next_logits

            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    scaled_logits, min(top_k, scaled_logits.size(-1))
                )
                scaled_logits = torch.full_like(scaled_logits, float("-inf"))
                scaled_logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample or argmax
            probs = F.softmax(scaled_logits, dim=-1)
            if temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = scaled_logits.argmax(dim=-1, keepdim=True)

            next_token_id = next_token.item()

            # Compute log probability
            log_prob = F.log_softmax(next_logits, dim=-1)
            token_log_prob = log_prob[0, next_token_id].item()

            generated_ids.append(next_token_id)
            log_probs.append(token_log_prob)

            # Check stop condition
            if next_token_id in stop_token_ids:
                break

            # Forward with cache (single token)
            logits, _ = self.forward_with_cache(next_token, use_cache=True)
            next_logits = logits[:, -1, :]

        if not keep_cache:
            self.clear_kv_cache()

        return generated_ids, log_probs

    def prefill_cache(self, token_ids: list[int]) -> None:
        """Process tokens and add to KV cache without generating.

        Use this to add context (e.g., tool outputs) between generation turns.

        Args:
            token_ids: Token IDs to process and cache
        """
        if not token_ids:
            return
        input_ids = torch.tensor([token_ids], device=self._device)
        self.forward_with_cache(input_ids, use_cache=True)

    def compute_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        """Compute log probabilities for completion tokens."""
        if not completion_ids:
            return []

        input_ids = torch.tensor([prompt_ids + completion_ids], device=self._device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits

        start_idx = len(prompt_ids) - 1
        end_idx = start_idx + len(completion_ids)

        completion_logits = logits[0, start_idx:end_idx]
        log_probs = F.log_softmax(completion_logits, dim=-1)

        completion_tensor = torch.tensor(completion_ids, device=self._device)
        token_logprobs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze(1)

        return token_logprobs.tolist()

    def get_tokenizer(self) -> PreTrainedTokenizer | None:
        return self._tokenizer

    def eval(self) -> HuggingFaceInferenceModel:
        """Set model to eval mode."""
        self._model.eval()
        return self


class HuggingFaceTrainingModel(HuggingFaceInferenceModel, TrainingModel):
    """HuggingFace implementation of TrainingModel.

    Extends HuggingFaceInferenceModel with training-specific functionality:
    - Forward pass with gradient support
    - Parameter access for optimizers
    - State dict management for checkpointing
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> HuggingFaceTrainingModel:
        """Load a model from HuggingFace."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, device=device)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Standard forward pass."""
        return self._model(input_ids, **kwargs)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Make the wrapper callable like the original model."""
        return self.forward(input_ids, **kwargs)

    def train(self, mode: bool = True) -> HuggingFaceTrainingModel:
        """Set model to train mode."""
        self._model.train(mode)
        return self

    def parameters(self):
        """Get model parameters (for optimizer)."""
        return self._model.parameters()

    def state_dict(self) -> dict:
        """Get model state dict."""
        return self._model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load model state dict."""
        self._model.load_state_dict(state_dict)

    def save_pretrained(self, path: str) -> None:
        """Save model to disk."""
        self._model.save_pretrained(path)


# =============================================================================
# Factory Functions
# =============================================================================


def create_teacher_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HuggingFaceInferenceModel:
    """Create a teacher model for distillation.

    Args:
        model_name: HuggingFace model name/path
        device: Device to load model on
        dtype: Model dtype (torch.bfloat16 or torch.float16)

    Returns:
        HuggingFaceInferenceModel ready for inference
    """
    model = HuggingFaceInferenceModel.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    return model

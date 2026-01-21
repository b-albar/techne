"""Model abstraction for inference.

This module provides:
- InferenceModel: Protocol for any model that can do inference.
- LocalModel: Synchronous wrapper around a HuggingFace model.
- create_teacher_model: Factory to create models for distillation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class InferenceModel(Protocol):
    """Protocol for any model that can perform inference."""

    @property
    def device(self) -> str:
        """Device the model is on."""
        ...

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Standard forward pass, returns ModelOutput with .logits."""
        ...

    def generate(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Generate tokens, returns sequences."""
        ...

    def compute_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        """Compute log probabilities for completion tokens."""
        ...


# =============================================================================
# Local Model (Synchronous HuggingFace)
# =============================================================================


class LocalModel:
    """Wrapper around a local HuggingFace model.

    This is used for:
    - Training (where we need gradients)
    - Evaluation on a single GPU
    - Teacher model in distillation
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._device = device or str(next(model.parameters()).device)

        # Expose config for compatibility
        self.config = model.config

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LocalModel:
        """Load a model from HuggingFace."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            **kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, device=device)

    @property
    def device(self) -> str:
        return self._device

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Standard forward pass."""
        return self.model(input_ids, **kwargs)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Make the wrapper callable like the original model."""
        return self.forward(input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor = None, **kwargs) -> Any:
        """Generate tokens."""
        return self.model.generate(input_ids=input_ids, **kwargs)

    def compute_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        """Compute log probabilities for completion tokens."""
        if not completion_ids:
            return []

        input_ids = torch.tensor([prompt_ids + completion_ids], device=self._device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        start_idx = len(prompt_ids) - 1
        end_idx = start_idx + len(completion_ids)

        completion_logits = logits[0, start_idx:end_idx]
        log_probs = F.log_softmax(completion_logits, dim=-1)

        completion_tensor = torch.tensor(completion_ids, device=self._device)
        token_logprobs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze(1)

        return token_logprobs.tolist()

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """Set model to train mode."""
        self.model.train(mode)
        return self

    def parameters(self):
        """Get model parameters (for optimizer)."""
        return self.model.parameters()

    def get_tokenizer(self):
        return self.tokenizer

    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)

    def save_pretrained(self, path: str):
        """Save model to disk."""
        self.model.save_pretrained(path)


# =============================================================================
# Factory Functions
# =============================================================================


def create_teacher_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> InferenceModel:
    """Create a teacher model for distillation.

    Args:
        model_name: HuggingFace model name/path
        device: Device to load model on
        dtype: Model dtype (torch.bfloat16 or torch.float16)

    Returns:
        LocalModel ready for inference
    """
    model = LocalModel.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    return model

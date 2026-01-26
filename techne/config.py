"""Configuration system for Techne training."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from techne.training.model import InferenceModel, TrainingModel


# =============================================================================
# Utilities
# =============================================================================


def parse_dtype(v: Any) -> torch.dtype | str:
    """Parse dtype from string or return as-is if already a torch.dtype."""
    if isinstance(v, torch.dtype):
        return v
    if isinstance(v, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(v, v)
    return v


# =============================================================================
# Enums
# =============================================================================


class TrainingAlgorithm(str, Enum):
    """Training algorithm to use."""

    # On-policy
    PPO = "ppo"
    GRPO = "grpo"
    GSPO = "gspo"
    DISTILL = "distill"

    # Off-policy / Offline
    OFFLINE_RL = "offline_rl"
    DISTILL_OFFLINE = "distill_offline"
    SFT = "sft"
    DFT = "dft"


class DistillationMode(str, Enum):
    """Distillation loss mode."""

    FORWARD_KL = "forward_kl"  # KL(teacher || student) - mode-covering
    REVERSE_KL = "reverse_kl"  # KL(student || teacher) - mode-seeking
    AKD = "akd"  # Adaptive KL Divergence - weighted bidirectional KL
    SLIM = "slim"  # Sparse Logit Infused Modeling (top-k)


class AlignerType(str, Enum):
    """Tokenizer alignment strategy for distillation."""

    AUTO = "auto"  # Auto-detect based on tokenizer identity
    DIRECT = "direct"  # Direct alignment (same tokenizer)
    GOLD = "gold"  # GOLD approach (decode & re-tokenize)


class DistributedBackend(str, Enum):
    """Distributed training backend."""

    NONE = "none"
    FSDP = "fsdp"
    DDP = "ddp"


# =============================================================================
# Model Configuration
# =============================================================================


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    enabled: bool = True
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


class InferenceConfig(BaseModel):
    """Configuration for inference models and generation.

    Combines model loading settings with generation parameters.
    Use create_inference_model() or create_training_model() to instantiate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Model settings
    name_or_path: str
    backend: Literal["huggingface", "vllm"] = "huggingface"
    device: str = "cuda"
    dtype: Any = torch.bfloat16
    trust_remote_code: bool = True

    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20

    # HuggingFace options
    attn_implementation: str | None = "sdpa"

    # vLLM options
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    @field_validator("dtype", mode="before")
    @classmethod
    def _parse_dtype(cls, v):
        return parse_dtype(v)

    def create_inference_model(self) -> InferenceModel:
        """Create an InferenceModel from this config."""
        if self.backend == "huggingface":
            from techne.integrations.huggingface import HuggingFaceInferenceModel

            return HuggingFaceInferenceModel.from_pretrained(
                self.name_or_path,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                **(
                    {"attn_implementation": self.attn_implementation}
                    if self.attn_implementation
                    else {}
                ),
            )
        elif self.backend == "vllm":
            raise NotImplementedError("vLLM backend not yet implemented")
        raise ValueError(f"Unknown backend: {self.backend}")

    def create_training_model(self) -> TrainingModel:
        """Create a TrainingModel from this config."""
        if self.backend == "huggingface":
            from techne.integrations.huggingface import HuggingFaceTrainingModel

            return HuggingFaceTrainingModel.from_pretrained(
                self.name_or_path,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                **(
                    {"attn_implementation": self.attn_implementation}
                    if self.attn_implementation
                    else {}
                ),
            )
        elif self.backend == "vllm":
            raise NotImplementedError("vLLM backend does not support training")
        raise ValueError(f"Unknown backend: {self.backend}")

    def create_model_factory(self, for_training: bool = False):
        """Create a callable that returns a model instance."""
        return self.create_training_model if for_training else self.create_inference_model


class ModelConfig(BaseModel):
    """Configuration for the main model being trained.

    Separate from InferenceConfig because training has additional options
    like LoRA that don't apply to inference workers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name_or_path: str
    dtype: Any = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    @field_validator("dtype", mode="before")
    @classmethod
    def _parse_dtype(cls, v):
        return parse_dtype(v)

    def to_inference_config(self, device: str = "cuda", **overrides) -> InferenceConfig:
        """Convert to InferenceConfig, with optional overrides."""
        return InferenceConfig(
            name_or_path=self.name_or_path,
            dtype=self.dtype,
            device=device,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
            **overrides,
        )


# =============================================================================
# Training Configuration
# =============================================================================


class TrainingConfig(BaseModel):
    """Training hyperparameters and settings."""

    model_config = ConfigDict(extra="allow")

    # Algorithm
    algorithm: TrainingAlgorithm = TrainingAlgorithm.SFT

    # Optimization
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096

    # Schedule
    max_steps: int = -1
    num_train_epochs: float = 3.0

    # Logging
    report_to: str = "none"

    # Distillation
    teacher: InferenceConfig | None = None
    kl_weight: float = 0.5
    distillation_mode: DistillationMode = DistillationMode.FORWARD_KL
    distillation_temperature: float = 2.0
    aligner_type: AlignerType = AlignerType.AUTO
    slim_top_k: int = 100

    # Distributed
    distributed_backend: DistributedBackend = DistributedBackend.NONE
    num_training_workers: int = 1

    # RL / On-policy
    inference: InferenceConfig | None = None
    num_inference_workers: int = 1
    num_generations: int = 4
    num_teacher_workers: int = 1
    clip_eps: float = 0.2
    clip_range_ratio: list[float] | None = None
    kl_coef: float = 0.1
    ppo_epochs: int = 1
    ppo_batch_size: int | None = None
    sync_weights: bool = True
    sync_weights_interval: int = 10


# =============================================================================
# Main Configuration
# =============================================================================


class TechneConfig(BaseModel):
    """Root configuration for Techne.

    Example YAML:
        model:
          name_or_path: Qwen/Qwen3-0.6B
          dtype: bfloat16

        training:
          algorithm: sft
          learning_rate: 1e-5
          inference:
            name_or_path: Qwen/Qwen3-0.6B
            temperature: 0.7

        max_turns: 10
        output_dir: ./output
    """

    model_config = ConfigDict(extra="allow")

    # Required
    model: ModelConfig

    # Optional with defaults
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tags: dict[str, str] = Field(default_factory=dict)

    # Agent settings
    max_turns: int = 5

    # Output
    output_dir: str = "./output"
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> TechneConfig:
        """Load config from YAML file."""
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def get_inference_config(self, device: str = "cuda") -> InferenceConfig:
        """Get inference config for workers.

        Returns training.inference if set, otherwise derives from model config.
        """
        if self.training.inference is not None:
            return self.training.inference
        return self.model.to_inference_config(device=device)

    def get_teacher_config(self) -> InferenceConfig | None:
        """Get teacher model config for distillation."""
        return self.training.teacher

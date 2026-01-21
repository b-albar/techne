"""Configuration system for Techne training."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainingAlgorithm(str, Enum):
    # On-policy algorithms
    PPO = "ppo"
    GRPO = "grpo"
    GSPO = "gspo"
    DISTILL = "distill"  # On-policy distillation (student generates, teacher scores)

    # Off-policy / Offline algorithms
    OFFLINE_RL = "offline_rl"  # Off-policy RL on pre-collected trajectories
    DISTILL_OFFLINE = "distill_offline"  # Standard distillation (train on teacher outputs)
    SFT = "sft"
    DFT = "dft"


class DistillationMode(str, Enum):
    FORWARD_KL = "forward_kl"
    SLIM = "slim"


class InferenceBackend(str, Enum):
    HF = "hf"  # HuggingFace transformers
    VLLM = "vllm"  # vLLM (faster, requires vllm installed)


class DistributedBackend(str, Enum):
    NONE = "none"  # Single GPU
    FSDP = "fsdp"  # Fully Sharded Data Parallel
    DDP = "ddp"  # Distributed Data Parallel


class LoRAConfig(BaseModel):
    enabled: bool = Field(default=True)
    r: int = Field(default=64)
    alpha: int = Field(default=128)
    dropout: float = Field(default=0.05)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = Field(default="none")


class ModelConfig(BaseModel):
    name_or_path: str
    dtype: str = Field(default="bfloat16")
    attn_implementation: str = Field(default="flash_attention_2")
    trust_remote_code: bool = Field(default=True)
    compile: bool = Field(default=False)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.SFT)
    learning_rate: float = Field(default=1e-6)
    batch_size: int = Field(default=8)
    gradient_accumulation_steps: int = Field(default=4)
    max_steps: int = Field(default=-1)
    num_train_epochs: float = Field(default=3.0)
    warmup_ratio: float = Field(default=0.1)
    weight_decay: float = Field(default=0.01)
    max_grad_norm: float = Field(default=1.0)
    sync_weights: bool = Field(default=True)
    max_seq_length: int = Field(default=4096)
    report_to: str = Field(default="none")
    # For distillation: teacher model path and KL weight
    teacher_model: str | None = Field(default=None)
    kl_weight: float = Field(default=0.5)  # Weight for KL loss vs SFT loss in distillation
    distillation_mode: DistillationMode = Field(default=DistillationMode.FORWARD_KL)
    slim_top_k: int | None = Field(default=None)
    # Distributed training config
    num_training_workers: int = Field(default=1)  # Number of training workers (GPUs)
    distributed_backend: DistributedBackend = Field(default=DistributedBackend.NONE)
    # Async RL config (for on-policy: GRPO/PPO/GSPO/DISTILL)
    num_inference_workers: int = Field(default=1)
    num_generations: int = Field(default=4)  # Completions per prompt
    inference_backend: InferenceBackend = Field(default=InferenceBackend.HF)
    clip_eps: float = Field(default=0.2)  # PPO/GRPO clipping epsilon
    kl_coef: float = Field(default=0.1)  # KL penalty coefficient
    sync_weights_interval: int = Field(default=10)  # Steps between weight syncs to workers
    # On-policy distillation: separate teacher workers
    num_teacher_workers: int = Field(default=1)  # For DISTILL: workers computing teacher logprobs


class RolloutConfig(BaseModel):
    max_turns: int = Field(default=5)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)


class TechneConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    tags: dict[str, str] = Field(default_factory=dict)

    output_dir: str = Field(default="./output")
    seed: int = Field(default=42)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=100)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TechneConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

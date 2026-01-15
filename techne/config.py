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
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class TrainingConfig(BaseModel):
    algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.SFT)
    learning_rate: float = Field(default=1e-6)
    batch_size: int = Field(default=8)
    gradient_accumulation_steps: int = Field(default=4)
    max_steps: int = Field(default=1000)
    sync_weights: bool = Field(default=True)
    max_seq_length: int = Field(default=4096)
    report_to: str = Field(default="none")
    # For distillation: teacher model path and KL weight
    teacher_model: str | None = Field(default=None)
    kl_weight: float = Field(default=0.5)  # Weight for KL loss vs SFT loss in distillation


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

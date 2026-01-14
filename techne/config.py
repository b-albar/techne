"""Configuration system for Techne using Pydantic dataclasses."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RolloutBackendType(str, Enum):
    """Supported rollout backends."""

    VLLM = "vllm"
    SGLANG = "sglang"
    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"


class TrainingAlgorithm(str, Enum):
    """Supported RL training algorithms."""

    PPO = "ppo"
    GRPO = "grpo"
    GSPO = "gspo"


class TagConfig(BaseModel):
    """Configuration for customizable tool calling tags.

    These tags define how the model indicates tool invocation and how
    tool responses are formatted in the conversation.

    Example configurations:
        - ReTool style: <code>...</code> and <interpreter>...</interpreter>
        - OpenAI style: <tool_call>...</tool_call> and <tool_response>...</tool_response>
    """

    tool_start: str = Field(default="<code>", description="Tag marking start of tool call")
    tool_end: str = Field(default="</code>", description="Tag marking end of tool call")
    response_start: str = Field(
        default="<interpreter>", description="Tag marking start of tool response"
    )
    response_end: str = Field(
        default="</interpreter>", description="Tag marking end of tool response"
    )

    def get_tool_pattern(self) -> str:
        """Return regex pattern for matching tool calls."""
        import re

        start = re.escape(self.tool_start)
        end = re.escape(self.tool_end)
        return rf"{start}(.*?){end}"

    def wrap_tool_call(self, content: str) -> str:
        """Wrap content in tool call tags."""
        return f"{self.tool_start}{content}{self.tool_end}"

    def wrap_response(self, content: str) -> str:
        """Wrap content in response tags."""
        return f"{self.response_start}{content}{self.response_end}"


class ToolConfig(BaseModel):
    """Configuration for tool execution."""

    sandbox_url: str | None = Field(
        default=None, description="URL for code sandbox (SandboxFusion compatible)"
    )
    sandbox_timeout: float = Field(default=30.0, description="Timeout for sandbox execution")
    max_retries: int = Field(default=3, description="Max retries for tool execution")
    concurrent_limit: int = Field(default=10, description="Max concurrent tool executions")


class LoRAConfig(BaseModel):
    """Configuration for LoRA training."""

    enabled: bool = Field(default=True, description="Whether to use LoRA")
    r: int = Field(default=64, description="LoRA rank")
    alpha: int = Field(default=128, description="LoRA alpha")
    dropout: float = Field(default=0.05, description="LoRA dropout")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Modules to apply LoRA to",
    )
    bias: str = Field(default="none", description="LoRA bias setting")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for PEFT")


class ModelConfig(BaseModel):
    """Configuration for the model."""

    name_or_path: str = Field(description="Model name or path")
    torch_dtype: str = Field(default="bfloat16", description="Model dtype")
    attn_implementation: str = Field(
        default="flash_attention_2", description="Attention implementation"
    )
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")


class RolloutConfig(BaseModel):
    """Configuration for rollout generation."""

    backend: RolloutBackendType = Field(
        default=RolloutBackendType.VLLM, description="Inference backend"
    )
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    max_new_tokens: int = Field(default=4096, description="Max new tokens per generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism for inference")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory fraction")


class TrainingConfig(BaseModel):
    """Configuration for training."""

    algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.GRPO, description="RL algorithm")
    loss_type: str = Field(default="nll", description="Loss type for SFT (e.g. 'nll', 'dft')")
    learning_rate: float = Field(default=1e-6, description="Learning rate")
    batch_size: int = Field(default=8, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation")
    max_steps: int = Field(default=1000, description="Maximum training steps")
    num_train_epochs: int | None = Field(
        default=None, description="Number of training epochs (overrides max_steps if set)"
    )
    warmup_ratio: float = Field(default=0.1, description="Warmup ratio")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm")

    # RL-specific
    kl_coef: float = Field(default=0.0, description="KL penalty coefficient")
    gamma: float = Field(default=1.0, description="Discount factor")
    num_rollouts_per_prompt: int = Field(default=4, description="Rollouts per prompt for GRPO")

    # Distributed (DeepSpeed & FSDP)
    deepspeed_config: str | None = Field(default=None, description="DeepSpeed config path")

    # FSDP 2.0 Support
    # Usage: fsdp="full_shard auto_wrap"
    fsdp: str | list[str] | None = Field(
        default=None, description="FSDP sharding strategy (e.g. 'full_shard auto_wrap')"
    )
    fsdp_config: dict[str, Any] | None = Field(
        default=None, description="FSDP configuration dictionary"
    )


class TechneConfig(BaseModel):
    """Main configuration for Techne framework."""

    model: ModelConfig
    tags: TagConfig = Field(default_factory=TagConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output_dir: str = Field(default="./output", description="Output directory")
    seed: int = Field(default=42, description="Random seed")
    logging_steps: int = Field(default=10, description="Logging frequency")
    save_steps: int = Field(default=100, description="Checkpoint frequency")
    weight_sync_interval: int = Field(
        default=10,
        description="Steps between syncing policy weights to rollout backend (0 to disable)",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TechneConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

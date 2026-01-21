"""Simple unified trainer wrapper for Techne.

Training types:
- SFT/DFT: Supervised/Direct fine-tuning
- GRPO/PPO/GSPO/DISTILL: Async on-policy RL with Ray
- DISTILL_OFFLINE: Offline distillation
"""

from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from techne.config import TechneConfig, TrainingAlgorithm
from techne.data import TrainingSample, Trajectory
from techne.transform import FullHistoryTransform, TrajectoryTransform


class TechneTrainer:
    """Simple unified trainer wrapper."""

    def __init__(self, config: TechneConfig, transform: TrajectoryTransform | None = None):
        self.config = config
        self.transform = transform or FullHistoryTransform()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.dtype,
            attn_implementation=config.model.attn_implementation,
            device_map="auto",
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA if enabled
        if config.model.lora.enabled:
            lora_config = LoraConfig(
                r=config.model.lora.r,
                lora_alpha=config.model.lora.alpha,
                lora_dropout=config.model.lora.dropout,
                target_modules=config.model.lora.target_modules,
                bias=config.model.lora.bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

    async def train(
        self,
        data: list[Trajectory] | list[TrainingSample] | Any | None = None,
        dataset: Any | None = None,
        reward_fn: Any | None = None,
        **kwargs,
    ):
        """Unified training entry point.

        Routes to appropriate training method based on algorithm in config.
        """
        algo = self.config.training.algorithm

        # 1. On-Policy RL Training (GRPO/PPO/GSPO/DISTILL) - async with Ray
        if dataset is not None and algo in [
            TrainingAlgorithm.GRPO,
            TrainingAlgorithm.PPO,
            TrainingAlgorithm.GSPO,
            TrainingAlgorithm.DISTILL,
        ]:
            from techne.training.rl import train_rl

            return await train_rl(
                self.config, self.model, self.tokenizer, dataset, reward_fn, **kwargs
            )

        # 2. Offline Distillation
        if algo == TrainingAlgorithm.DISTILL_OFFLINE:
            from techne.training.distill import train_distill_offline

            return await train_distill_offline(
                self.config, self.model, self.tokenizer, data, **kwargs
            )

        # 3. Offline Training (SFT/DFT)
        if not data:
            print("No data provided for training.")
            return

        # Process trajectories if needed
        samples = data
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], Trajectory):
            samples = self.transform.process(data, self.tokenizer)

        from techne.training.sft import get_sft_trainer

        trainer = get_sft_trainer(self.config, self.model, self.tokenizer, samples, **kwargs)
        return trainer.train()

    def get_state_dict(self):
        """Get model state dict for checkpointing."""
        return self.model.state_dict()

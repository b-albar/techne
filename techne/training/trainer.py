"""Simple unified trainer wrapper for Techne.

This is the main entry point for all training types:
- SFT/DFT: Supervised/Direct fine-tuning
- GRPO/PPO/GSPO: On-policy RL
- DISTILL: On-policy distillation
- DISTILL_OFFLINE: Offline distillation
- OFFLINE_RL: Off-policy RL
"""

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from techne.agent import Agent
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
            torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
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
        agent: Agent | None = None,
        dataset: Any | None = None,
        reward_fn: Any | None = None,
        **kwargs,
    ):
        """Unified training entry point.

        Routes to appropriate training method based on algorithm in config.
        """
        algo = self.config.training.algorithm

        # 1. On-Policy RL Training (GRPO/PPO/GSPO/DISTILL)
        if dataset is not None and algo in [
            TrainingAlgorithm.GRPO,
            TrainingAlgorithm.PPO,
            TrainingAlgorithm.GSPO,
            TrainingAlgorithm.DISTILL,
            TrainingAlgorithm.OFFLINE_RL,
        ]:
            from techne.training.rl import train_rl

            return train_rl(self.config, self.model, self.tokenizer, dataset, reward_fn, **kwargs)

        # 2. Offline Distillation
        if algo == TrainingAlgorithm.DISTILL_OFFLINE:
            from techne.training.distill import train_distill_offline

            return train_distill_offline(self.config, self.model, self.tokenizer, data, **kwargs)

        # 3. Legacy on-policy loop (for custom agents with tool use)
        if agent and dataset:
            return await self._train_loop(agent, dataset, reward_fn, **kwargs)

        # 4. Offline Training (SFT/DFT)
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

    async def _train_loop(
        self,
        agent: Agent,
        dataset: Any,
        reward_fn: Any = None,
        get_prompts: Any = lambda batch: batch["prompt"],
        get_ground_truths: Any = lambda batch: batch["ground_truth"]
        if "ground_truth" in batch.column_names
        else None,
    ):
        """Legacy manual training loop for custom agents with tool use."""
        print(f"Starting Manual Training Loop ({self.config.training.algorithm.value.upper()})...")

        for i in range(self.config.training.max_steps):
            # 1. Batching
            batch = dataset.shuffle().select(range(self.config.training.batch_size))

            prompts = get_prompts(batch)
            ground_truths = get_ground_truths(batch)

            # 2. Rollout
            trajectories = await agent.collect_trajectories(prompts)

            # 3. Reward assignment
            if reward_fn and ground_truths:
                for traj, gt in zip(trajectories, ground_truths):
                    traj.reward = reward_fn(traj, gt)

            # 4. Training update
            samples = self.transform.process(trajectories, self.tokenizer)
            from techne.training.sft import get_sft_trainer

            trainer = get_sft_trainer(self.config, self.model, self.tokenizer, samples, max_steps=1)
            trainer.train()

            # 5. Logging
            if trajectories:
                avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
                print(f"Step {i} completed. Avg Reward: {avg_reward:.4f}")

    def get_state_dict(self):
        """Get model state dict for checkpointing."""
        return self.model.state_dict()

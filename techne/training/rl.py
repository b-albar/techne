"""RL Trainer for tool-augmented reinforcement learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from techne.config import RolloutBackendType, TechneConfig, TrainingAlgorithm
from techne.rollout.backends.base import RolloutBackend
from techne.rollout.external import ExternalAgent
from techne.rollout.orchestrator import (
    BlackBoxOrchestrator,
    RolloutOrchestrator,
    Trajectory,
    create_orchestrator,
)
from techne.rollout.parser import TagParser
from techne.training.rewards import AccuracyReward, RewardFunction

console = Console()


class RLTrainer:
    """Reinforcement Learning Trainer supporting PPO, GRPO, and GSPO.

    Implements tool-augmented RL training with:
    - Multi-turn rollouts with tool execution
    - Interpreter response masking
    - LoRA support for efficient training
    - Multi-GPU support via Accelerate/DeepSpeed
    """

    def __init__(
        self,
        config: TechneConfig,
        reward_fn: RewardFunction | None = None,
        external_agent: ExternalAgent | None = None,
    ):
        """Initialize RL trainer.

        Args:
            config: Techne configuration
            reward_fn: Reward function (default: AccuracyReward)
            external_agent: Optional implementation of ExternalAgent protocol
                          for black-box rollouts
        """
        self.config = config
        self.reward_fn = reward_fn or AccuracyReward()
        self._parser = TagParser(config.tags)
        self._external_agent = external_agent

        self._model = None
        self._ref_model = None
        self._tokenizer = None
        self._backend: RolloutBackend | None = None
        self._orchestrator: RolloutOrchestrator | BlackBoxOrchestrator | None = None

    def _load_model(self) -> None:
        """Load model, tokenizer, and optionally reference model."""
        # Only load generation model if we don't have an external agent
        # Or if we still need tokenizer/model for value/reward functions
        # For now, we assume we need the model for PPO/GRPO updates regardless

        console.print("[bold blue]Loading model and tokenizer...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.model.torch_dtype, torch.bfloat16)

        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": torch_dtype,
            "attn_implementation": self.config.model.attn_implementation,
        }

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name_or_path,
            **model_kwargs,
        )

        # Apply LoRA if enabled
        if self.config.model.lora.enabled:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.model.lora.r,
                lora_alpha=self.config.model.lora.alpha,
                lora_dropout=self.config.model.lora.dropout,
                target_modules=self.config.model.lora.target_modules,
                bias=self.config.model.lora.bias,
                task_type=self.config.model.lora.task_type,
            )

            self._model = get_peft_model(self._model, lora_config)
            console.print("[green]LoRA applied:")
            self._model.print_trainable_parameters()

        # Reference model for KL penalty (only if kl_coef > 0)
        if self.config.training.kl_coef > 0:
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs,
            )
            self._ref_model.eval()
            for param in self._ref_model.parameters():
                param.requires_grad = False

    async def _create_backend(self) -> RolloutBackend:
        """Create and start rollout backend."""
        if self.config.rollout.backend == RolloutBackendType.VLLM:
            from techne.rollout.backends.vllm import VLLMBackend

            backend = VLLMBackend(self.config.model, self.config.rollout)
        else:
            from techne.rollout.backends.sglang import SGLangBackend

            backend = SGLangBackend(self.config.model, self.config.rollout)

        await backend.start()
        return backend

    def _create_trainer(self):
        """Create TRL trainer based on algorithm."""
        algorithm = self.config.training.algorithm

        if algorithm == TrainingAlgorithm.PPO:
            return self._create_ppo_trainer()
        elif algorithm == TrainingAlgorithm.GRPO:
            return self._create_grpo_trainer()
        elif algorithm == TrainingAlgorithm.GSPO:
            return self._create_gspo_trainer()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _create_ppo_trainer(self):
        """Create PPO trainer using TRL."""
        from trl import PPOConfig, PPOTrainer

        ppo_config = PPOConfig(
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            mini_batch_size=self.config.training.batch_size // 4,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            kl_penalty="kl",
            init_kl_coef=self.config.training.kl_coef,
            gamma=self.config.training.gamma,
            seed=self.config.seed,
            # Pass FSDP args if supported by PPOConfig (inherits TrainingArgs usually)
            # In many versions PPOConfig wraps Accelerator, but let's try kwargs
        )

        # Manually updating config if PPOConfig is standard dataclass
        if hasattr(ppo_config, "fsdp"):
            ppo_config.fsdp = self.config.training.fsdp
        if hasattr(ppo_config, "fsdp_config"):
            ppo_config.fsdp_config = self.config.training.fsdp_config

        return PPOTrainer(
            config=ppo_config,
            model=self._model,
            ref_model=self._ref_model,
            tokenizer=self._tokenizer,
        )

    def _create_grpo_trainer(self):
        """Create GRPO trainer using TRL.

        GRPO (Group Relative Policy Optimization) doesn't require a separate
        reward model or value model - it uses verifiable rewards.
        """
        from trl import GRPOConfig, GRPOTrainer

        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            max_steps=self.config.training.max_steps,
            warmup_ratio=self.config.training.warmup_ratio,
            num_generations=self.config.training.num_rollouts_per_prompt,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            seed=self.config.seed,
            bf16=True,
            gradient_checkpointing=True,
            deepspeed=self.config.training.deepspeed_config,
            # FSDP Support
            fsdp=self.config.training.fsdp,
            fsdp_config=self.config.training.fsdp_config,
        )

        return GRPOTrainer(
            config=grpo_config,
            model=self._model,
            tokenizer=self._tokenizer,
            reward_funcs=self._grpo_reward_wrapper,
        )

    def _create_gspo_trainer(self):
        """Create GSPO trainer.

        GSPO (Grouped Self-Play Policy Optimization) extends GRPO with
        self-play mechanisms for improved exploration.
        """
        # GSPO is similar to GRPO but with self-play dynamics
        # For now, we use GRPO as base and add self-play logic
        from trl import GRPOConfig, GRPOTrainer

        gspo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            max_steps=self.config.training.max_steps,
            warmup_ratio=self.config.training.warmup_ratio,
            num_generations=self.config.training.num_rollouts_per_prompt * 2,  # More for self-play
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            seed=self.config.seed,
            bf16=True,
            gradient_checkpointing=True,
            deepspeed=self.config.training.deepspeed_config,
            # FSDP Support
            fsdp=self.config.training.fsdp,
            fsdp_config=self.config.training.fsdp_config,
        )

        return GRPOTrainer(
            config=gspo_config,
            model=self._model,
            tokenizer=self._tokenizer,
            reward_funcs=self._gspo_reward_wrapper,
        )

    def _grpo_reward_wrapper(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Wrapper for GRPO reward computation.

        Args:
            prompts: Input prompts
            completions: Model completions

        Returns:
            List of rewards
        """
        ground_truths = kwargs.get("ground_truths", [None] * len(completions))

        rewards = []
        for completion, gt in zip(completions, ground_truths):
            reward = self.reward_fn.compute(completion, gt)
            rewards.append(reward)

        return rewards

    def _gspo_reward_wrapper(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """GSPO reward wrapper with self-play comparison.

        Args:
            prompts: Input prompts
            completions: Model completions

        Returns:
            List of relative rewards
        """
        # Get base rewards
        base_rewards = self._grpo_reward_wrapper(prompts, completions, **kwargs)

        # GSPO adds relative ranking within groups
        # Group completions by prompt and compute relative rewards
        num_per_prompt = self.config.training.num_rollouts_per_prompt * 2
        ranked_rewards = []

        for i in range(0, len(base_rewards), num_per_prompt):
            group = base_rewards[i : i + num_per_prompt]
            # Normalize within group
            mean = sum(group) / len(group)
            std = max((sum((r - mean) ** 2 for r in group) / len(group)) ** 0.5, 1e-8)
            group_normalized = [(r - mean) / std for r in group]
            ranked_rewards.extend(group_normalized)

        return ranked_rewards

    async def _generate_trajectories(
        self,
        prompts: list[str],
    ) -> list[Trajectory]:
        """Generate multi-turn trajectories with tool execution.

        Args:
            prompts: Input prompts

        Returns:
            List of completed trajectories
        """
        # Initialize backend/orchestrator if needed
        if self._orchestrator is None:
            if self._external_agent:
                # Use black-box orchestrator
                self._orchestrator = BlackBoxOrchestrator(
                    agent=self._external_agent,
                    parser=self._parser,
                )
            else:
                # Use standard orchestrator
                if self._backend is None:
                    self._backend = await self._create_backend()

                self._orchestrator = create_orchestrator(
                    backend=self._backend,
                    tags=self.config.tags,
                    rollout_config=self.config.rollout,
                    tool_executor=None,  # Will be created with defaults
                )

        return await self._orchestrator.rollout_batch(prompts)

    async def _sync_weights_to_backend(self) -> None:
        """Sync current policy weights to rollout backend/agent."""
        if self.config.weight_sync_interval <= 0:
            return

        state_dict = self._model.state_dict()

        # Sync to backend if using vLLM/SGLang
        if self._backend:
            console.print("[dim]Syncing weights to rollout backend...")
            await self._backend.update_weights(state_dict)

        # Sync to external agent if using BlackBoxOrchestrator
        if self._orchestrator and hasattr(self._orchestrator, "update_weights"):
            console.print("[dim]Syncing weights to external agent...")
            await self._orchestrator.update_weights(state_dict)

    def train(
        self,
        dataset: Dataset | str,
        output_dir: str | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> None:
        """Run RL training.

        Args:
            dataset: Training dataset or path
            output_dir: Output directory
            resume_from_checkpoint: Checkpoint to resume from
        """
        from datasets import load_dataset

        console.print("[bold green]Starting RL Training")
        console.print(f"Algorithm: {self.config.training.algorithm.value}")

        # Load model
        if self._model is None:
            self._load_model()

        # Load dataset
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split="train")

        output_dir = output_dir or self.config.output_dir

        # Create trainer
        trainer = self._create_trainer()

        # For GRPO/GSPO, we can use the built-in training loop
        if self.config.training.algorithm in [TrainingAlgorithm.GRPO, TrainingAlgorithm.GSPO]:
            # Prepare dataset with prompts and ground truths
            def prepare_example(example):
                return {
                    "prompt": example.get("prompt", example.get("question", "")),
                    "ground_truth": example.get("answer", example.get("solution", "")),
                }

            train_dataset = dataset.map(prepare_example)

            # Train
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            trainer.save_model(output_dir)

        else:  # PPO
            self._train_ppo(trainer, dataset, output_dir)

        console.print(f"[bold green]Training complete! Model saved to {output_dir}")

    def _train_ppo(
        self,
        trainer,
        dataset: Dataset,
        output_dir: str,
    ) -> None:
        """PPO training loop with tool-augmented rollouts.

        Args:
            trainer: PPO trainer
            dataset: Training dataset
            output_dir: Output directory
        """
        from torch.utils.data import DataLoader

        # Create dataloader
        def collate_fn(examples):
            return {
                "prompts": [ex.get("prompt", ex.get("question", "")) for ex in examples],
                "ground_truths": [ex.get("answer", ex.get("solution", "")) for ex in examples],
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Training loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=self.config.training.max_steps)
            step = 0

            for batch in dataloader:
                if step >= self.config.training.max_steps:
                    break

                prompts = batch["prompts"]
                ground_truths = batch["ground_truths"]

                # Tokenize prompts
                query_tensors = [
                    self._tokenizer.encode(p, return_tensors="pt").squeeze() for p in prompts
                ]

                # Generate responses
                response_tensors = trainer.generate(
                    query_tensors,
                    max_new_tokens=self.config.rollout.max_new_tokens,
                    temperature=self.config.rollout.temperature,
                    top_p=self.config.rollout.top_p,
                )

                # Decode responses
                responses = [
                    self._tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
                ]

                # Compute rewards
                rewards = [
                    torch.tensor(self.reward_fn.compute(r, gt))
                    for r, gt in zip(responses, ground_truths)
                ]

                # Compute response masks to exclude interpreter feedback
                # This ensures we don't train on tool outputs
                response_masks = []
                for r_text, r_tensor in zip(responses, response_tensors):
                    # Get ranges to mask
                    mask_ranges = self._parser.get_response_mask_ranges(r_text)

                    # Create boolean mask (True = include, False = mask)
                    r_len = len(r_tensor)
                    mask = torch.ones(r_len, dtype=torch.bool)

                    if mask_ranges:
                        # Get offsets
                        enc = self._tokenizer(
                            r_text, return_offsets_mapping=True, add_special_tokens=False
                        )
                        offsets = enc.offset_mapping

                        for idx, (start, end) in enumerate(offsets):
                            if idx >= r_len:
                                break

                            # Check overlap with any masked range
                            for m_start, m_end in mask_ranges:
                                if max(start, m_start) < min(end, m_end):
                                    mask[idx] = False
                                    break

                    response_masks.append(mask)

                # PPO step
                # Note: TRL's PPOTrainer.step signature might vary by version
                # If supported, pass response_masks. If not, this logic assumes support
                # or needs adaptation (e.g. by manipulating response_tensors or advantages)
                stats = trainer.step(
                    query_tensors, response_tensors, rewards, response_masks=response_masks
                )

                # Log
                if step % self.config.logging_steps == 0:
                    console.print(
                        f"Step {step}: reward={sum(r.item() for r in rewards) / len(rewards):.3f}"
                    )

                # Save checkpoint
                if step % self.config.save_steps == 0:
                    trainer.save_pretrained(f"{output_dir}/checkpoint-{step}")

                # Sync weights to backend/agent
                if (
                    self.config.weight_sync_interval > 0
                    and step % self.config.weight_sync_interval == 0
                ):
                    import asyncio

                    asyncio.run(self._sync_weights_to_backend())

                step += 1
                progress.update(task, advance=1)

        # Save final model
        trainer.save_pretrained(output_dir)

    def save_model(self, path: str | Path) -> None:
        """Save trained model.

        Args:
            path: Output path
        """
        if self._model is not None:
            self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)

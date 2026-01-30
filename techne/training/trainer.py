"""Simple unified trainer wrapper for Techne.

Training types:
- SFT/DFT: Supervised/Direct fine-tuning
- GRPO/PPO/GSPO/DISTILL: Async on-policy RL with Ray
- DISTILL_OFFLINE: Offline distillation
"""

import logging
import os
from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from techne.config import DistributedBackend, TechneConfig, TrainingAlgorithm
from techne.data import TrainingSample, Trajectory

logger = logging.getLogger(__name__)


def _is_distributed_env() -> bool:
    """Check if we're already running inside a distributed launcher (torchrun/accelerate)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


class TechneTrainer:
    """Simple unified trainer wrapper."""

    def __init__(self, config: TechneConfig):
        self.config = config

        # device_map="auto" enables model/tensor parallelism (sharding layers
        # across GPUs in a single process). This is the right choice for:
        #   - Single-process training (NONE backend)
        #   - Tensor parallelism (TP backend)
        # FSDP and DDP do their own sharding/replication and require
        # device_map=None so HF Trainer can manage device placement.
        backend = config.training.distributed_backend
        self._wants_data_parallel = (
            backend in (DistributedBackend.FSDP, DistributedBackend.DDP)
            and config.training.num_training_workers > 1
        )
        device_map = None if self._wants_data_parallel else "auto"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.dtype,
            attn_implementation=config.model.attn_implementation,
            device_map=device_map,
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
        reward_fn_class: type | None = None,
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
                self.config, self.model, self.tokenizer, dataset, reward_fn_class, **kwargs
            )

        # 2. Offline Distillation (synchronous — no await)
        if algo == TrainingAlgorithm.DISTILL_OFFLINE:
            from techne.training.distill import train_distill_offline

            return train_distill_offline(
                self.config, self.model, self.tokenizer, data, **kwargs
            )

        # 3. Offline Training (SFT/DFT)
        if not data:
            logger.warning("No data provided for training.")
            return

        samples = data

        from techne.training.sft import get_sft_trainer

        # FSDP/DDP require a multi-process launcher. TP does not (it runs
        # in a single process with device_map="auto").
        if self._wants_data_parallel and not _is_distributed_env():
            n = self.config.training.num_training_workers
            backend = self.config.training.distributed_backend
            if backend == DistributedBackend.FSDP:
                hint = f"accelerate launch --use_fsdp --num_processes {n} your_script.py"
            else:
                hint = f"torchrun --nproc_per_node={n} your_script.py"
            raise RuntimeError(
                f"Distributed SFT training requires a multi-process launcher. "
                f"Config requests {backend.value.upper()} with {n} workers, but the "
                f"current process was not launched in a distributed context "
                f"(RANK/WORLD_SIZE not set). Launch with:\n  {hint}"
            )

        trainer = get_sft_trainer(self.config, self.model, self.tokenizer, samples, **kwargs)
        return trainer.train()

    def get_state_dict(self):
        """Get model state dict for checkpointing."""
        return self.model.state_dict()

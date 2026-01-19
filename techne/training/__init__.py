"""Training module for Techne.

Provides modular training utilities:
- trainer: Main TechneTrainer class (unified entry point)
- sft: SFT/DFT training utilities
- rl: Async RL training with Ray (GRPO/PPO/GSPO/DISTILL)
- distill: Distillation training (offline)
- data_gen: Distributed data generation with Ray
"""

from techne.training.data_gen import (
    GeneratedSample,
    generate_distill_data,
    generate_distill_data_sync,
    generate_sft_data,
    generate_sft_data_sync,
    samples_to_dataset,
)
from techne.training.trainer import TechneTrainer

__all__ = [
    "TechneTrainer",
    "GeneratedSample",
    "generate_sft_data",
    "generate_sft_data_sync",
    "generate_distill_data",
    "generate_distill_data_sync",
    "samples_to_dataset",
]

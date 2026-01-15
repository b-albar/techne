"""Training module for Techne.

Provides modular training utilities:
- trainer: Main TechneTrainer class (unified entry point)
- sft: SFT/DFT training utilities
- rl: RL training with GRPO/PPO
- distill: Distillation training (on-policy and offline)
"""

from techne.training.trainer import TechneTrainer

__all__ = ["TechneTrainer"]

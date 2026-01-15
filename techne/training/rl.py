"""RL training utilities using TRL's GRPO/PPO trainers."""

from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from techne.config import TechneConfig, TrainingAlgorithm


def train_rl(
    config: TechneConfig,
    model,
    tokenizer,
    dataset: Any,
    reward_fn: Callable | None = None,
    **kwargs,
):
    """Train using online RL with GRPO/PPO.

    Supports:
    - GRPO/PPO/GSPO: User-provided reward function
    - DISTILL: Built-in KL reward from teacher model
    - OFFLINE_RL: Off-policy training on pre-collected data

    Args:
        config: Techne configuration
        model: Model to train
        tokenizer: Tokenizer
        dataset: HF Dataset with prompts
        reward_fn: Optional reward function (trajectory, ground_truth) -> float
        **kwargs: Additional arguments

    Returns:
        Training result
    """
    algo = config.training.algorithm
    reward_funcs = []

    if algo == TrainingAlgorithm.DISTILL:
        # On-policy distillation: reward = -reverse_KL from teacher
        reward_funcs.append(_create_kl_reward_fn(config, tokenizer))
        print("Using KL reward for on-policy distillation")

    elif reward_fn:
        # User-provided reward function
        reward_funcs.append(_create_reward_wrapper(reward_fn))
    else:
        # Default: zero reward
        reward_funcs.append(lambda prompts, completions, **kw: [0.0] * len(completions))

    # GRPO Config
    args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps,
        logging_steps=config.logging_steps,
        report_to=getattr(config.training, "report_to", "none"),
        bf16=config.model.dtype == "bfloat16",
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    algo_name = algo.value.upper()
    print(f"Starting {algo_name} training on {len(dataset)} samples...")
    return trainer.train()


def _create_kl_reward_fn(config: TechneConfig, student_tokenizer):
    """Create KL-based reward function for on-policy distillation."""
    teacher_model_path = config.training.teacher_model
    if not teacher_model_path:
        raise ValueError("On-policy distillation requires teacher_model in config")

    print(f"Loading teacher model: {teacher_model_path}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    from techne.training.distill import compute_distillation_reward

    def kl_reward_fn(prompts, completions, completion_ids=None, **kwargs):
        """Compute -reverse_KL as reward for distillation."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                reward = compute_distillation_reward(
                    completion_text=completion,
                    prompt_text=prompt,
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tokenizer,
                    student_tokenizer=student_tokenizer,
                    use_kl=False,  # RL trainer computes its own KL penalty usually, so we use Teacher Likelihood
                )
                rewards.append(reward)
            except Exception as e:
                print(f"Reward computation failed: {e}")
                rewards.append(0.0)
        return rewards

    return kl_reward_fn


def _create_reward_wrapper(reward_fn: Callable):
    """Wrap user reward function to match TRL's signature."""

    def trl_reward_wrapper(prompts, completions, **kwargs):
        """TRL reward signature: (prompts, completions) -> list[float]"""
        rewards = []
        ground_truths = kwargs.get("ground_truth", [None] * len(completions))
        for i, completion in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else None

            # Create a simple trajectory-like object
            class SimpleTrajectory:
                def __init__(self, text):
                    self.steps = [type("Step", (), {"role": "assistant", "content": str(text)})()]

            traj = SimpleTrajectory(completion)
            rewards.append(reward_fn(traj, gt) if gt else 0.0)
        return rewards

    return trl_reward_wrapper

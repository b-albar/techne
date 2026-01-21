"""Async RL training using Ray for on-policy methods (GRPO/PPO/GSPO/DISTILL)."""

from collections.abc import Callable
from typing import Any

from techne.config import TechneConfig, TrainingAlgorithm
from techne.training.async_rl import train_async_rl


async def train_rl(
    config: TechneConfig,
    model,
    tokenizer,
    dataset: Any,
    reward_fn: Callable | None = None,
    **kwargs,
):
    """Train using async on-policy RL with Ray workers.

    Supports: GRPO, PPO, GSPO, DISTILL

    Args:
        config: Techne configuration
        model: Model to train
        tokenizer: Tokenizer
        dataset: HF Dataset with "prompt" column
        reward_fn: Reward function (prompts, completions) -> list[float]
        **kwargs: Additional arguments

    Returns:
        Training result dict
    """
    algo = config.training.algorithm

    # Validate dataset
    assert len(dataset) > 0, "RL dataset is empty!"
    sample = dataset[0]
    assert "prompt" in sample or "text" in sample, (
        f"Dataset must have 'prompt' or 'text' column. Found: {list(sample.keys())}"
    )

    # Convert TRL-style reward_fn to simple (prompt, completion) -> float
    simple_reward_fn = None
    if reward_fn:

        def simple_reward_fn(sample: dict, completion: str) -> float:
            prompt = sample.get("prompt") or sample.get("text") or str(sample)
            if isinstance(prompt, list):
                # Cannot use TRL reward fn with list prompts easily
                return 0.0
            return reward_fn([prompt], [completion])[0]

    # For DISTILL, create KL reward from teacher
    if algo == TrainingAlgorithm.DISTILL:
        simple_reward_fn = _create_kl_reward_fn(config, tokenizer)

    print(f"Starting async {algo.value.upper()} training on {len(dataset)} samples...")

    return await train_async_rl(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=simple_reward_fn,
        algorithm=algo.value,
        **kwargs,
    )


def _create_kl_reward_fn(config: TechneConfig, student_tokenizer):
    """Create KL-based reward function for on-policy distillation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def kl_reward_fn(sample: dict, completion: str) -> float:
        """Compute -reverse_KL as reward for distillation."""
        # Extract prompt string for teacher
        prompt = sample.get("prompt") or sample.get("text") or str(sample)
        if isinstance(prompt, list):
            # For teacher, we need text. Use apply_chat_template if available?
            # Or assume teacher handles it. compute_distillation_reward takes prompt_text.
            # We should probably convert list to string via tokenizer template
            prompt = student_tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

        try:
            return compute_distillation_reward(
                completion_text=completion,
                prompt_text=prompt,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                use_kl=False,
            )
        except Exception as e:
            print(f"Reward computation failed: {e}")
            return 0.0

    return kl_reward_fn

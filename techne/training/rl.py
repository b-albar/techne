"""Async RL training using Ray for on-policy methods (GRPO/PPO/GSPO/DISTILL)."""

import logging
from typing import Any

from techne.config import TechneConfig
from techne.training.async_rl import train_async_rl

logger = logging.getLogger(__name__)


async def train_rl(
    config: TechneConfig,
    model,
    tokenizer,
    dataset: Any,
    reward_fn_class: type | None = None,
    **kwargs,
):
    """Train using async on-policy RL with Ray workers.

    Supports: GRPO, PPO, GSPO, DISTILL

    Args:
        config: Techne configuration
        model: Model to train
        tokenizer: Tokenizer
        dataset: HF Dataset with "prompt" column
        reward_fn_class: Reward function class (instantiated in workers)
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

    logger.info("Starting async %s training on %d samples", algo.value.upper(), len(dataset))

    return await train_async_rl(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn_class=reward_fn_class,
        algorithm=algo.value,
        **kwargs,
    )

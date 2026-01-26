import argparse
import asyncio
import os

from techne.config import TechneConfig, TrainingAlgorithm
from techne.training.trainer import TechneTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/maths/configs/distill.yaml")
    parser.add_argument("--dataset", type=str, default="examples/maths/data/rl")
    args = parser.parse_args()

    # 1. Load Config
    config = TechneConfig.from_yaml(args.config)

    # Ensure algorithm is DISTILL
    if config.training.algorithm != TrainingAlgorithm.DISTILL:
        print(f"Warning: Config algorithm is {config.training.algorithm}, overriding to DISTILL")
        config.training.algorithm = TrainingAlgorithm.DISTILL

    # 2. Load Data
    print(f"Loading dataset from {args.dataset}...")
    from datasets import load_from_disk

    try:
        ds = load_from_disk(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # 3. Initialize Trainer
    trainer = TechneTrainer(config)

    # 4. Start Distillation
    # For on-policy distillation, we pass the dataset.
    # The trainer will use _create_kl_reward_fn internally since algo is DISTILL.
    # We don't implement a custom reward function here.

    train_ds = ds["train"]

    await trainer.train(
        dataset=train_ds,
    )


if __name__ == "__main__":
    asyncio.run(main())

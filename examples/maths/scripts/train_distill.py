"""Train a Math Tool Agent using distillation.

This script demonstrates offline distillation where a smaller student
model learns from a larger teacher model's outputs.

Usage:
    python train_distill.py --config ../configs/distill.yaml --dataset ../data/sft
"""

import argparse
import asyncio
import os

from datasets import load_from_disk

from techne.config import TechneConfig
from techne.training.trainer import TechneTrainer


async def main():
    parser = argparse.ArgumentParser(description="Train using distillation")
    parser.add_argument("--config", type=str, default="examples/maths/configs/distill.yaml")
    parser.add_argument("--dataset", type=str, default="examples/maths/data/sft")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        if os.path.exists(f"../{config_path}"):
            config_path = f"../{config_path}"
        elif os.path.exists("../configs/distill.yaml"):
            config_path = "../configs/distill.yaml"
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    config = TechneConfig.from_yaml(config_path)
    print(f"Loaded config: {config_path}")
    print(f"  Student model: {config.model.name_or_path}")
    print(f"  Teacher model: {config.training.teacher_model}")
    print(f"  Algorithm: {config.training.algorithm}")

    # 2. Load Dataset
    print(f"\nLoading dataset from {args.dataset}...")
    try:
        data_path = args.dataset
        if not os.path.exists(data_path):
            if os.path.exists(f"../{data_path}"):
                data_path = f"../{data_path}"
            elif os.path.exists("../data/sft"):
                data_path = "../data/sft"

        ds = load_from_disk(data_path)
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    print(f"  Dataset size: {len(ds)} samples")

    # 3. Initialize Trainer
    trainer = TechneTrainer(config)

    # 4. Preprocess dataset (same as SFT)
    from math_agent import MathToolAgent

    agent = MathToolAgent(config, model=None, tokenizer=trainer.tokenizer)

    def preprocess_incremental(sample):
        """Use agent's tokenize_messages for consistent incremental tokenization."""
        messages = sample["prompt"]
        return agent.tokenize_messages(messages)

    # Select subset and tokenize
    train_dataset = ds.select(range(min(100, len(ds))))
    print(f"\nTokenizing {len(train_dataset)} samples...")
    train_dataset = train_dataset.map(preprocess_incremental, remove_columns=["prompt"])

    # 5. Train
    print(f"\nStarting distillation training...")
    await trainer.train(train_dataset)


if __name__ == "__main__":
    asyncio.run(main())

"""RL training script for math tool-use (white-box approach with vLLM)."""

import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from techne.config import TechneConfig
from techne.rollout.backends.vllm import VLLMBackend
from techne.rollout.orchestrator import create_orchestrator
from techne.training.rewards import AccuracyReward
from techne.training.rl import RLTrainer


def load_data(data_path: Path):
    """Load and prepare RL training data."""
    dataset = load_from_disk(str(data_path))

    # Extract prompts and ground truth answers
    prompts = [ex["problem"] for ex in dataset["train"]]
    answers = [ex["answer"] for ex in dataset["train"]]

    return prompts, answers


def main():
    parser = argparse.ArgumentParser(description="RL training (white-box)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/rl"),
        help="Path to RL dataset",
    )

    args = parser.parse_args()

    # Load configuration
    config = TechneConfig.from_yaml(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Model: {config.model.name_or_path}")
    print(f"Algorithm: {config.training.algorithm}")
    print(f"Backend: {config.rollout.backend} (white-box)")

    # Load data
    print("\nLoading dataset...")
    prompts, answers = load_data(args.data_path)
    print(f"Loaded {len(prompts)} training examples")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    # Create vLLM backend for rollouts
    print("\nInitializing vLLM backend...")
    backend = VLLMBackend(
        model_name_or_path=config.model.name_or_path,
        tensor_parallel_size=config.rollout.tensor_parallel_size,
        gpu_memory_utilization=config.rollout.gpu_memory_utilization,
    )

    # Create orchestrator (white-box)
    orchestrator = create_orchestrator(
        backend=backend,
        tags=config.tags,
        rollout_config=config.rollout,
    )

    # Create reward function
    reward_fn = AccuracyReward(ground_truth=answers)

    # Create RL trainer
    print("\nInitializing RL trainer...")
    trainer = RLTrainer(
        config=config,
        tokenizer=tokenizer,
        orchestrator=orchestrator,
        reward_fn=reward_fn,
        train_prompts=prompts,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting RL training (WHITE-BOX with vLLM)")
    print("=" * 60)
    trainer.train()

    print("\n✓ Training complete!")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

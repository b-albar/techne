"""RL training script for math tool-use (black-box approach with external agent)."""

import argparse
from pathlib import Path

from black_box_agent import MathToolAgent
from datasets import load_from_disk
from transformers import AutoTokenizer

from techne.config import TechneConfig
from techne.rollout.orchestrator import BlackBoxOrchestrator
from techne.rollout.parser import TagParser
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
    parser = argparse.ArgumentParser(description="RL training (black-box)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("examples/maths/data/rl"),
        help="Path to RL dataset",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        required=True,
        help="Model path (should be same as policy model for consistency check)",
    )
    parser.add_argument(
        "--inference-server",
        type=str,
        required=True,
        help="Inference server URL (e.g., http://localhost:8000/v1 for local vLLM server)",
    )

    args = parser.parse_args()

    # Load configuration
    config = TechneConfig.from_yaml(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Policy model: {config.model.name_or_path}")
    print(f"Rollout agent: {args.agent_model} (black-box)")
    print(f"Algorithm: {config.training.algorithm}")

    # Load data
    print("\nLoading dataset...")
    prompts, answers = load_data(args.data_path)
    print(f"Loaded {len(prompts)} training examples")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    # Create black-box agent
    print("\nInitializing black-box agent...")
    agent = MathToolAgent(
        model_name=args.agent_model,
        tags=config.tags,
        api_base=args.inference_server,
    )

    # Create black-box orchestrator
    parser = TagParser(config.tags)
    orchestrator = BlackBoxOrchestrator(agent=agent, parser=parser)

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
    print(f"Starting RL training (BLACK-BOX)")
    print("=" * 60)
    print(f"Policy model: {config.model.name_or_path}")
    print(f"Rollout agent: {args.agent_model} via {args.inference_server}")
    print("Note: Using same model for consistency check")
    print("      Black-box tests re-tokenization path")
    print("=" * 60)
    trainer.train()

    print("\n✓ Training complete!")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

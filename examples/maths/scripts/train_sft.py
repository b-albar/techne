"""SFT training script for math tool-use."""

import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from techne.config import TechneConfig
from techne.training.sft import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="SFT training for math tool-use")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/sft"),
        help="Path to SFT dataset",
    )

    args = parser.parse_args()

    # Load configuration
    config = TechneConfig.from_yaml(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Model: {config.model.name_or_path}")

    # Load data
    print("\nLoading dataset...")
    dataset = load_from_disk(str(args.data_path))
    print(f"Loaded {len(dataset['train'])} training examples")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create SFT trainer
    print("\nInitializing SFT trainer...")
    print(f"Loss type: {config.training.loss_type}")
    if config.training.loss_type == "dft":
        print("→ Using Dynamic Fine-Tuning (DFT) for better generalization")

    trainer = SFTTrainer(
        config=config,
        loss_type=config.training.loss_type,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting SFT training")
    print("=" * 60)
    trainer.train(dataset=dataset["train"])

    print("\n✓ Training complete!")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

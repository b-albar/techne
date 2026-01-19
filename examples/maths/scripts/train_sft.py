import argparse
import asyncio
import os

from datasets import load_dataset, load_from_disk

from techne.config import TechneConfig
from techne.training.trainer import TechneTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/maths/configs/sft.yaml")
    parser.add_argument("--dataset", type=str, default="examples/maths/data/sft")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback relative check
        if os.path.exists(f"../{config_path}"):
            config_path = f"../{config_path}"
        elif os.path.exists("configs/sft.yaml"):
            config_path = "configs/sft.yaml"

    config = TechneConfig.from_yaml(config_path)

    # 2. Load Data (ReTool SFT)
    print(f"Loading SFT dataset from {args.dataset}...")
    try:
        data_path = args.dataset
        if not os.path.exists(data_path):
            # Fallback relative checks
            if os.path.exists(f"../{data_path}"):
                data_path = f"../{data_path}"
            elif os.path.exists("../data/sft"):
                data_path = "../data/sft"

        ds = load_from_disk(data_path)
        if hasattr(ds, "keys") and "train" in ds:
            train_dataset = ds["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # 3. Initialize Trainer
    trainer = TechneTrainer(config)

    # Print model parameters
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in trainer.model.parameters())
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_params:,d} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}"
        )

    # 4. Preprocess with incremental tokenization
    # Import MathToolAgent for its tokenize_messages method
    from math_agent import MathToolAgent

    # Create agent with shared tokenizer (no model needed for tokenization)
    agent = MathToolAgent(config, model=None, tokenizer=trainer.tokenizer)

    def preprocess_incremental(sample):
        """Use agent's tokenize_messages for consistent incremental tokenization."""
        messages = sample["prompt"]
        return agent.tokenize_messages(messages)

    # Tokenize
    print(f"Tokenizing {len(train_dataset)} samples incrementally...")
    train_dataset = train_dataset.map(preprocess_incremental, remove_columns=["prompt"])

    # 5. Train
    print(f"Starting SFT on {len(train_dataset)} samples...")
    await trainer.train(train_dataset)


if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import os


from math_agent import MathToolAgent
from rewards import MathReward
from techne.config import TechneConfig
from techne.training.trainer import TechneTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/maths/configs/agent_training.yaml")
    parser.add_argument("--dataset", type=str, default="examples/maths/data/rl")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        # Only simple relative check
        if os.path.exists(f"../{config_path}"):
            config_path = f"../{config_path}"
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    config = TechneConfig.from_yaml(config_path)

    # 2. Load Real Data (DAPO-Math-17k for ReTool)
    print(f"Loading RL dataset from {args.dataset}...")
    from datasets import load_from_disk

    try:
        data_path = args.dataset
        if not os.path.exists(data_path):
            if os.path.exists(f"../{data_path}"):
                data_path = f"../{data_path}"
            elif os.path.exists("../data/rl"):
                data_path = "../data/rl"

        ds = load_from_disk(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # 3. Initialize Trainer
    trainer = TechneTrainer(config)

    # 4. Initialize Agent (On-policy)
    agent = MathToolAgent(config, model=trainer.model, tokenizer=trainer.tokenizer)
    reward_fn = MathReward()

    # 5. Start Training Loop
    train_ds = ds["train"]

    # Preprocess dataset to ensure it has 'ground_truth' column at top level
    if "reward_model" in train_ds.column_names and "ground_truth" not in train_ds.column_names:
        print("Preprocessing dataset to extract ground truths...")
        train_ds = train_ds.map(lambda x: {"ground_truth": x["reward_model"]["ground_truth"]})

    # 5. Start Training Loop
    await trainer.train(
        agent=agent,
        dataset=train_ds,
        reward_fn=reward_fn,
    )


if __name__ == "__main__":
    asyncio.run(main())

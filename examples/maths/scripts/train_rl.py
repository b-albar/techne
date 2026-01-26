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
    config = TechneConfig.from_yaml(args.config)

    # 2. Load Real Data (DAPO-Math-17k for ReTool)
    print(f"Loading RL dataset from {args.dataset}...")
    from datasets import load_from_disk

    try:
        data_path = args.dataset
        ds = load_from_disk(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # 3. Initialize Trainer
    trainer = TechneTrainer(config)

    train_ds = ds["train"]

    # Preprocess dataset if it has nested reward_model
    if "reward_model" in train_ds.column_names and "ground_truth" not in train_ds.column_names:
        print("Preprocessing dataset to extract ground truths...")
        train_ds = train_ds.map(lambda x: {"ground_truth": x["reward_model"]["ground_truth"]})

    # 4. Prepare Reward Function
    # The new async RL passes (prompt, completion) to the reward function.
    # We use a lookup table to find the ground truth for each prompt.
    def reward_fn_wrapper(sample: dict, completion: str) -> float:
        gt = sample.get("ground_truth")
        if gt is None:
            # Fallback if nested in reward_model (should be handled by preprocessing but just in case)
            if "reward_model" in sample and isinstance(sample["reward_model"], dict):
                gt = sample["reward_model"].get("ground_truth")

        if gt is None:
            return 0.0

        predicted = MathReward.extract_answer(completion)
        return 1.0 if MathReward.is_correct(predicted, gt) else 0.0

    # 5. Start Async Training
    # We pass the class MathToolAgent because workers will instantiate it themselves.
    await trainer.train(
        dataset=train_ds,
        reward_fn=reward_fn_wrapper,
        agent_class=MathToolAgent,
    )


if __name__ == "__main__":
    asyncio.run(main())

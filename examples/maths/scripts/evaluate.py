import argparse
import asyncio
from pathlib import Path

from datasets import load_dataset, load_from_disk
from math_agent import MathToolAgent
from rewards import MathReward
from techne.config import TechneConfig


def extract_qa(example):
    """Extract question and answer from common math dataset formats."""
    # 1. Answer extraction first (easier)
    a = example.get("answer") or example.get("solution") or example.get("ground_truth")
    if not a and "reward_model" in example:
        rm = example["reward_model"]
        a = rm.get("ground_truth") if isinstance(rm, dict) else None

    # 2. Question/Prompt extraction
    # If 'prompt' is present and is a list, it's often the best representation
    q = example.get("prompt")
    if q and isinstance(q, list):
        return q, a

    # Fallback to string keys
    q = example.get("question") or example.get("problem") or example.get("problem_content")
    if not q and "prompt" in example:
        q = example["prompt"]

    return q, a


async def run_evaluation(args):
    # 1. Load Config & Agent
    config_path = Path(args.config)
    config = (
        TechneConfig.from_yaml(config_path)
        if config_path.exists()
        else TechneConfig(model={"name_or_path": args.model})
    )
    config.model.name_or_path = args.model

    agent = MathToolAgent(config)
    reward_fn = MathReward()

    # 2. Load Data
    data_path = Path(args.dataset)
    if not data_path.exists():
        # Try relative to project root or examples dir
        possibilities = [
            Path("examples/maths") / args.dataset,
            Path(__file__).parent.parent / args.dataset,
            Path(__file__).parent.parent / "data" / args.dataset
            if "data" not in args.dataset
            else None,
        ]
        for p in possibilities:
            if p and p.exists():
                data_path = p
                break

    print(f"Loading dataset from: {data_path}")
    if data_path.exists():
        ds = load_from_disk(str(data_path))
        eval_data = ds["test"] if "test" in ds else (ds["train"] if "train" in ds else ds)
    else:
        eval_data = load_dataset(args.dataset, split="train")

    if args.max_samples > 0:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    # 3. Process Evaluation
    correct = 0
    total = len(eval_data)
    print(f"Starting evaluation on {total} samples...\n")

    for i, example in enumerate(eval_data):
        q, a = extract_qa(example)
        if not q:
            continue

        traj = (await agent.collect_trajectories([q]))[0]
        score = reward_fn(traj, a)

        # Determine what was predicted for logging
        last_assist = next((s.content for s in reversed(traj.steps) if s.role == "assistant"), "")
        pred = MathReward.extract_answer(last_assist)

        is_correct = score > 0
        correct += int(is_correct)

        icon = "✓" if is_correct else "✗"
        q_str = q[-1]["content"] if isinstance(q, list) else str(q)
        print(f"[{i + 1}/{total}] {icon} | Q: {q_str[:50]}... | Target: {a} | Pred: {pred}")

    # 4. Final Summary
    acc = 100 * correct / total if total > 0 else 0
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/eval", help="Path to evaluation data")
    parser.add_argument("--config", type=str, default="examples/maths/configs/sft.yaml")
    parser.add_argument(
        "--max-samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)"
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

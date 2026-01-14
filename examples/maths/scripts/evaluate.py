"""Evaluation script for math problem solving with tool use."""

import argparse
import asyncio
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from techne.config import TechneConfig
from techne.rollout.backends.vllm import VLLMBackend
from techne.rollout.orchestrator import create_orchestrator


def extract_answer(text: str) -> str:
    """Extract the final answer from generated text."""
    # Look for common patterns like "The answer is X" or "= X"
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "answer is" in line.lower():
            # Extract number after "answer is"
            parts = line.lower().split("answer is")
            if len(parts) > 1:
                return parts[1].strip().strip(".")
        if line.strip().startswith("="):
            return line.strip()[1:].strip()
    # Default: return last line
    return lines[-1].strip() if lines else ""


async def evaluate_problem(orchestrator, prompt: str, ground_truth: str) -> bool:
    """Evaluate a single math problem.

    Returns:
        True if the answer is correct
    """
    trajectory = await orchestrator.rollout_single(prompt)

    # Extract predicted answer
    predicted = extract_answer(trajectory.final_response)

    # Compare (normalize both)
    predicted_normalized = predicted.strip().lower().replace(",", "")
    ground_truth_normalized = str(ground_truth).strip().lower().replace(",", "")

    return predicted_normalized == ground_truth_normalized


async def main():
    parser = argparse.ArgumentParser(description="Evaluate math tool-use model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("examples/maths/data/eval"),
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/maths/configs/rl_whitebox.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )

    args = parser.parse_args()

    # Load configuration
    config = TechneConfig.from_yaml(args.config)
    config.model.name_or_path = str(args.checkpoint)

    print(f"Evaluating: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")

    # Load evaluation data
    dataset = load_from_disk(str(args.dataset))
    eval_data = dataset["test"] if "test" in dataset else dataset["train"]

    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    print(f"Evaluating on {len(eval_data)} problems\n")

    # Create backend and orchestrator
    backend = VLLMBackend(
        model_name_or_path=str(args.checkpoint),
        tensor_parallel_size=config.rollout.tensor_parallel_size,
        gpu_memory_utilization=0.9,
    )

    orchestrator = create_orchestrator(
        backend=backend,
        tags=config.tags,
        rollout_config=config.rollout,
    )

    # Evaluate
    correct = 0
    total = len(eval_data)

    async with backend:
        for idx, example in enumerate(eval_data):
            problem = example["problem"]
            answer = example["answer"]

            is_correct = await evaluate_problem(orchestrator, problem, answer)

            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            print(
                f"[{idx + 1}/{total}] {status} Accuracy: {correct}/{idx + 1} ({100 * correct / (idx + 1):.1f}%)"
            )

    print("\n" + "=" * 60)
    print(f"Final Accuracy: {correct}/{total} ({100 * correct / total:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

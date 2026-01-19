import argparse
import asyncio
import os
from pathlib import Path

import ray
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_agent import MathToolAgent
from rewards import MathReward
from techne.config import TechneConfig


@ray.remote
class EvaluationWorker:
    """Distributed worker for evaluating MathToolAgent."""

    def __init__(self, config_path: str, model_name: str, torch_compile: bool = False):
        self.config = (
            TechneConfig.from_yaml(config_path)
            if os.path.exists(config_path)
            else TechneConfig(model={"name_or_path": model_name})
        )
        self.config.model.name_or_path = model_name
        self.config.model.compile = torch_compile

        # Force eager attention for evaluation if not specified or problems arise
        # (similar to what we did for distillation)
        if not hasattr(self.config.model, "attn_implementation"):
            self.config.model.attn_implementation = "eager"

        # Initialize Agent (which loads model/tokenizer)
        self.agent = MathToolAgent(self.config)
        self.reward_fn = MathReward()

    async def evaluate_batch(self, batch: list[dict]) -> list[dict]:
        """Evaluate a batch of examples."""
        results = []
        for example in batch:
            try:
                q, a = self._extract_qa(example)
                if not q:
                    continue

                traj = (await self.agent.collect_trajectories([q]))[0]
                score = self.reward_fn(traj, a)

                # Determine what was predicted for logging
                last_assist = next(
                    (s.content for s in reversed(traj.steps) if s.role == "assistant"), ""
                )
                pred = MathReward.extract_answer(last_assist)

                is_correct = score > 0

                results.append(
                    {
                        "question": str(q)[:100],
                        "target": str(a),
                        "prediction": pred,
                        "correct": is_correct,
                        "error": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "question": str(example)[:50],
                        "target": None,
                        "prediction": None,
                        "correct": False,
                        "error": str(e),
                    }
                )
        return results

    def _extract_qa(self, example):
        """Extract question and answer from common math dataset formats."""
        # 1. Answer extraction first (easier)
        a = example.get("answer") or example.get("solution") or example.get("ground_truth")
        if not a and "reward_model" in example:
            rm = example["reward_model"]
            a = rm.get("ground_truth") if isinstance(rm, dict) else None

        # 2. Question/Prompt extraction
        q = example.get("prompt")

        # Fallback to string keys
        if not q:
            q = example.get("question") or example.get("problem") or example.get("problem_content")

        return q, a


async def run_distributed_evaluation(args):
    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Calculate resources
    num_workers = args.parallel
    num_gpus = torch.cuda.device_count()
    gpu_per_worker = num_gpus / num_workers if num_gpus > 0 else 0

    # 2. Create Workers
    print(f"Initializing {num_workers} workers with {gpu_per_worker:.2f} GPU each...")
    workers = [
        EvaluationWorker.options(num_gpus=gpu_per_worker).remote(
            args.config, args.model, args.torch_compile
        )
        for _ in range(num_workers)
    ]

    # 3. Load Data
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

    # 4. Distribute Work
    total = len(eval_data)
    batch_size = (total + num_workers - 1) // num_workers
    batches = [eval_data[i : i + batch_size] for i in range(0, total, batch_size)]

    # Convert HF dataset slices to list of dicts for Ray serialization
    batches_list = []
    for i in range(0, total, batch_size):
        # HuggingFace Dataset slicing returns a dict of lists {col: [vals]},
        # we want list of dicts [{col: val}, ...]
        slice_dict = eval_data[i : i + batch_size]
        # Transpose
        keys = list(slice_dict.keys())
        num_items = len(slice_dict[keys[0]])
        items = [{k: slice_dict[k][j] for k in keys} for j in range(num_items)]
        batches_list.append(items)

    print(f"Starting distributed evaluation on {total} samples across {len(workers)} workers...")

    # Assign batches to workers round-robin
    futures = []
    for i, batch in enumerate(batches_list):
        worker = workers[i % len(workers)]
        futures.append(worker.evaluate_batch.remote(batch))

    # 5. Collect Results
    results_flat = []

    # Wrap Ray ObjectRefs into asyncio futures
    completed = await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in futures])

    for batch_results in completed:
        results_flat.extend(batch_results)

    # 6. Report
    correct = sum(1 for r in results_flat if r["correct"])
    acc = 100 * correct / total if total > 0 else 0

    # Print sample results
    for i, r in enumerate(results_flat):
        icon = "✓" if r["correct"] else "✗"
        if r["error"]:
            print(f"[{i + 1}/{total}] {icon} Error: {r['error']}")
        else:
            print(
                f"[{i + 1}/{total}] {icon} | Q: {r['question']}... | Target: {r['target']} | Pred: {r['prediction']}"
            )

    print("\n" + "=" * 40)
    print("Distributed Evaluation Results")
    print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")
    print("=" * 40)

    ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/eval", help="Path to evaluation data")
    parser.add_argument("--config", type=str, default="examples/maths/configs/sft.yaml")
    parser.add_argument(
        "--max-samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)"
    )
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    asyncio.run(run_distributed_evaluation(args))


if __name__ == "__main__":
    main()

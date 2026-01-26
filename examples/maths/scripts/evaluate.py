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

    def __init__(self, config: TechneConfig):
        self.config = config

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
        """Extract question and answer."""
        q = example.get("question") or example.get("prompt")
        a = example.get("answer") or example.get("solution")
        return q, a


async def run_distributed_evaluation(args):
    # 0. Prepare Config
    config = (
        TechneConfig.from_yaml(args.config)
        if os.path.exists(args.config)
        else TechneConfig(model={"name_or_path": args.model})
    )
    config.model.name_or_path = args.model
    # Force eager attention for evaluation if not specified
    if not hasattr(config.model, "attn_implementation"):
        config.model.attn_implementation = "eager"

    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            _metrics_export_port=None,
            configure_logging=False,
            log_to_driver=False,
            _system_config={"metrics_report_interval_ms": 0},
        )

    # Calculate resources
    num_workers = args.parallel
    num_gpus = torch.cuda.device_count()
    gpu_per_worker = num_gpus / num_workers if num_gpus > 0 else 0

    # 2. Create Workers
    print(f"Initializing {num_workers} workers with {gpu_per_worker:.2f} GPU each...")
    workers = [
        EvaluationWorker.options(num_gpus=gpu_per_worker).remote(config) for _ in range(num_workers)
    ]

    # 3. Load Data
    data_path = Path(args.dataset)
    print(f"Loading dataset from: {data_path}")

    if data_path.exists():
        ds = load_from_disk(str(data_path))
        eval_data = ds["test"] if "test" in ds else (ds["train"] if "train" in ds else ds)
    else:
        eval_data = load_dataset(args.dataset, split="train")

    if args.max_samples > 0:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    # 4. Distribute Work
    data_list = eval_data.to_list()
    total = len(data_list)
    batch_size = (total + num_workers - 1) // num_workers
    batches_list = [data_list[i : i + batch_size] for i in range(0, total, batch_size)]

    print(f"Starting distributed evaluation on {total} samples across {len(workers)} workers...")

    # Assign batches to workers round-robin
    futures = []
    for i, batch in enumerate(batches_list):
        worker = workers[i % len(workers)]
        futures.append(worker.evaluate_batch.remote(batch))

    import tqdm

    # Wrap Ray ObjectRefs into asyncio futures
    async_futures = [asyncio.wrap_future(f.future()) for f in futures]

    # 5. Collect Results with Progress Bar
    results_flat = []
    print("\nWaiting for results...")
    for f in tqdm.tqdm(
        asyncio.as_completed(async_futures), total=len(async_futures), desc="Evaluating batches"
    ):
        batch_results = await f
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
    args = parser.parse_args()

    asyncio.run(run_distributed_evaluation(args))


if __name__ == "__main__":
    main()

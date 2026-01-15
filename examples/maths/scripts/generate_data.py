"""
Script to generate datasets for SFT and Distillation by running an agent.

Usage:
    python generate_data.py --config ../configs/distill.yaml --input_dataset "gsm8k" --output_dir ../data/generated --num_samples 100
"""

import argparse
import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.dirname(__file__))  # For math_agent import

from datasets import load_dataset, load_from_disk, Dataset

from techne.config import TechneConfig
from techne.data import Trajectory
from math_agent import MathToolAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data using MathToolAgent")
    parser.add_argument("--config", type=str, required=True, help="Path to Techne config")
    parser.add_argument("--input_dataset", type=str, required=True, help="HF dataset name or path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the dataset")
    parser.add_argument(
        "--model", type=str, default=None, help="Override model path (e.g. teacher model)"
    )
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to generate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Generation batch size (sequential in this script)",
    )
    return parser.parse_args()


def trajectory_to_dict(traj: Trajectory) -> dict:
    """Convert trajectory to serializable dict."""
    steps = []
    for s in traj.steps:
        step_dict = {
            "role": s.role,
            "content": s.content,
            "token_ids": s.token_ids,
            "log_probs": s.log_probs,
        }
        steps.append(step_dict)

    return {"steps": steps, "full_text": traj.to_text(), "total_reward": traj.total_reward}


async def main():
    args = parse_args()

    # Load config
    config = TechneConfig.from_yaml(args.config)

    # Override model if specified
    if args.model:
        config.model.name_or_path = args.model
        print(f"Overriding model with: {args.model}")

    print(f"Initializing agent with model: {config.model.name_or_path}")

    # Load Model & Tokenizer explicitly to share or ensure correct loading
    # (Agent can load it itself, but strictly speaking we might want control)
    # We let the Agent load it for simplicity as implemented in MathToolAgent.__init__

    agent = MathToolAgent(config)

    # Load Input Dataset
    print(f"Loading input dataset: {args.input_dataset} [{args.split}]")
    try:
        if os.path.exists(args.input_dataset):
            ds = load_from_disk(args.input_dataset)
        else:
            ds = load_dataset(args.input_dataset, split=args.split)

        if hasattr(ds, "keys") and args.split in ds:
            ds = ds[args.split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select samples
    if args.num_samples > 0:
        ds = ds.select(range(min(len(ds), args.num_samples)))

    print(f"Generating data for {len(ds)} samples...")

    generated_data = []

    for i, sample in enumerate(ds):
        # Extract prompt: assumes 'question' or 'prompt' field
        if "question" in sample:
            prompt_text = sample["question"]
        elif "prompt" in sample:
            prompt_text = sample["prompt"]
        else:
            print(f"Skipping sample {i}: No 'question' or 'prompt' field found.")
            continue

        print(f"Processing sample {i + 1}/{len(ds)}...")

        try:
            # Run Agent
            trajectory = await agent._run_rollout(prompt_text)

            # Serialize
            record = trajectory_to_dict(trajectory)
            record["original_prompt"] = prompt_text
            record["problem_id"] = i

            generated_data.append(record)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback

            traceback.print_exc()

    # Save to disk
    if not generated_data:
        print("No data generated.")
        return

    print(f"Saving {len(generated_data)} samples to {args.output_dir}...")

    # Create HF Dataset
    # We flatten the steps for easier usage, or keep recursive structure?
    # Usually for SFT we want 'prompt' (messages format)

    # Convert to messages format for SFT
    final_records = []
    for item in generated_data:
        messages = []
        for s in item["steps"]:
            messages.append({"role": s["role"], "content": s["content"]})

        # Construct pre-tokenized fields to ensure exact consistency
        input_ids = []
        labels = []
        log_probs_list = []

        for s in item["steps"]:
            # Flatten token_ids
            s_ids = s["token_ids"] if s["token_ids"] else []
            input_ids.extend(s_ids)

            # Handle Log Probs
            # If present, extend. If not (e.g. user prompt), pad with placeholder (e.g. 0.0)
            if s["log_probs"]:
                log_probs_list.extend(s["log_probs"])
            else:
                log_probs_list.extend([0.0] * len(s_ids))

            # Create labels (mask non-trainable steps)
            if s["role"] == "assistant":
                labels.extend(s_ids)
            else:
                labels.extend([-100] * len(s_ids))

        final_records.append(
            {
                "prompt": messages,  # Keep for reference/debug
                "input_ids": input_ids,  # Pre-tokenized
                "labels": labels,  # Pre-computed labels
                "log_probs": log_probs_list,
                "original_question": item["original_prompt"],
                "steps_data": item["steps"],
            }
        )

    output_ds = Dataset.from_list(final_records)
    output_ds.save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

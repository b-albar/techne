"""Download and preprocess datasets for math tool-use training."""

import argparse
import json
import re
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from techne.data import Step, Trajectory, save_trajectories


def download_sft_dataset(output_dir: Path) -> None:
    """Download and save SFT dataset."""
    print("Downloading SFT dataset: swordfaith/ReTool-SFT-multi-turn")
    dataset = load_dataset("swordfaith/ReTool-SFT-multi-turn")

    output_path = output_dir / "sft"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))
    print(f"✓ Saved SFT dataset to {output_path}")


def download_rl_dataset(output_dir: Path) -> None:
    """Download and save RL training dataset."""
    print("Downloading RL dataset: BytedTsinghua-SIA/DAPO-Math-17k")
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")

    output_path = output_dir / "rl"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))
    print(f"✓ Saved RL dataset to {output_path}")


def download_eval_dataset(output_dir: Path) -> None:
    """Download and save evaluation dataset."""
    print("Downloading evaluation dataset: BytedTsinghua-SIA/AIME-2024")
    dataset = load_dataset("BytedTsinghua-SIA/AIME-2024")

    output_path = output_dir / "eval"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))
    print(f"✓ Saved evaluation dataset to {output_path}")


SYSTEM_PROMPT = (
    "You are a helpful assistant capable of solving math problems.\n"
    "You can use a Python interpreter to calculate results.\n"
    "To execute Python code, wrap it in a markdown block:\n"
    "```python\n"
    "print(12 * 12)\n"
    "```\n"
    "The output will be provided to you in a ```output block.\n"
    "The last line of your response should be of the form:\n"
    "<answer>\n"
    "\\\\boxed{Answer}\n"
    "</answer>\n"
    "where Answer is the answer to the problem.\n"
    "Solve the following problem:\n"
)

NEW_INSTR = (
    "Remember to place the final answer in the last part using the format: "
    "\n<answer>\n\\\\boxed{The final answer goes here.}\n</answer>"
)


def preprocess_sft_data(data_dir: Path) -> None:
    """Preprocess SFT data into serialized trajectories with trainability marking.

    Creates Trajectory objects where:
    - Assistant messages: trainable=True (model learns to generate these)
    - Tool results/outputs: trainable=False (masked, not trained on)
    - System/user messages: trainable=False (context only)
    """
    print("Preprocessing SFT dataset into trajectories...")

    sft_path = data_dir / "sft"
    dataset = load_from_disk(str(sft_path))

    trajectories: list[Trajectory] = []

    for idx, example in enumerate(dataset["train"]):
        messages = example["messages"]
        steps: list[Step] = []

        # Add system prompt as first step
        steps.append(
            Step(
                role="system",
                content=SYSTEM_PROMPT,
                trainable=False,
            )
        )

        for msg in messages:
            role = msg["role"]
            content = msg.get("content") or ""

            # Skip existing system prompts to enforce uniformity
            if role == "system":
                continue

            # Clean user instructions
            if role == "user":
                content = re.sub(
                    r"Remember to place the final answer in the last part using the format:.*?</answer>",
                    NEW_INSTR,
                    content,
                    flags=re.DOTALL,
                )

            # Standardize tool calls (convert JSON tool_calls to markdown)
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if tc.get("type") == "function":
                            func = tc.get("function", {})
                            args = func.get("arguments", "")
                            code = ""
                            if isinstance(args, str):
                                try:
                                    args_json = json.loads(args)
                                    code = args_json.get("code", "")
                                except (json.JSONDecodeError, TypeError, KeyError):
                                    code = args
                            elif isinstance(args, dict):
                                code = args.get("code", "")

                            if code:
                                content = content.rstrip()
                                content += f"\n```python\n{code}\n```"

                # Heuristic fallback for inline code
                elif ("import " in content or "print(" in content) and "```" not in content:
                    content = f"```python\n{content}\n```"

                content = content.replace("<code>", "```python\n").replace("</code>", "\n```")

                # Standardize final answer format
                content = content.replace("<answer>", "").replace("</answer>", "")
                boxed_match = re.search(r"\\boxed\{([^}]+)\}", content)
                hash_match = re.search(r"####\s*(.*)", content)
                if boxed_match:
                    ans = boxed_match.group(1).strip()
                    content = content.replace(
                        boxed_match.group(0), f"\n<answer>\n\\boxed{{{ans}}}\n</answer>"
                    )
                elif hash_match:
                    ans = hash_match.group(1).strip()
                    content = content.replace(
                        hash_match.group(0), f"\n<answer>\n\\boxed{{{ans}}}\n</answer>"
                    )

            # Standardize tool outputs
            is_tool_output = role == "tool" or (role == "user" and "output" in content.lower())
            if is_tool_output:
                if "```" not in content:
                    content = f"```output\n{content}\n```"
                content = content.replace("<interpreter>", "```output\n").replace(
                    "</interpreter>", "\n```"
                )

            # Determine trainability:
            # - Assistant messages: trainable (model learns to generate)
            # - Tool outputs: NOT trainable (environment feedback)
            # - User/system: NOT trainable (context only)
            if role == "assistant":
                trainable = True
                step_role = "assistant"
            elif is_tool_output:
                trainable = False
                step_role = "tool"
            else:
                trainable = False
                step_role = role

            steps.append(
                Step(
                    role=step_role,
                    content=content,
                    trainable=trainable,
                )
            )

        trajectory = Trajectory(
            steps=steps,
            metadata={"source": "ReTool-SFT-multi-turn", "example_idx": idx},
        )
        trajectories.append(trajectory)

    # Save trajectories as JSONL
    output_path = sft_path / "trajectories.jsonl"
    save_trajectories(trajectories, output_path)

    # Also save as HuggingFace dataset with prompt column for backwards compatibility
    def trajectory_to_prompt(traj: Trajectory) -> list[dict]:
        return [{"role": s.role, "content": s.content} for s in traj.steps]

    prompt_data = [{"prompt": trajectory_to_prompt(t)} for t in trajectories]
    hf_dataset = Dataset.from_list(prompt_data)

    # Save HF dataset (overwrite original)
    temp_path = sft_path.with_name("sft_temp")
    hf_dataset.save_to_disk(str(temp_path))

    # Move files
    for f in sft_path.iterdir():
        if f.name != "trajectories.jsonl":
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    for f in temp_path.iterdir():
        shutil.move(str(f), str(sft_path / f.name))
    temp_path.rmdir()

    print(f"✓ Created {len(trajectories)} trajectories")
    print(f"✓ Saved to {output_path}")

    # Print sample statistics
    sample = trajectories[0]
    trainable_steps = sum(1 for s in sample.steps if s.trainable)
    total_steps = len(sample.steps)
    print(f"✓ Sample: {trainable_steps}/{total_steps} steps trainable")


def preprocess_rl_data(data_dir: Path) -> None:
    """Preprocess RL data with complex answer format instructions."""
    print("Preprocessing RL dataset...")
    rl_path = data_dir / "rl"
    dataset = load_from_disk(str(rl_path))

    def update_prompt(example):
        prompt = example["prompt"]
        new_prompt = []
        for msg in prompt:
            content = msg["content"]
            if msg["role"] == "user":
                content = content.replace(
                    "The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.",
                    "",
                )
                content = content.replace(
                    'Remember to put your answer on its own line after "Answer:".', NEW_INSTR
                )
            new_prompt.append({"role": msg["role"], "content": content})

        if not any(msg.get("role") == "system" for msg in new_prompt):
            new_prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + new_prompt

        return {"prompt": new_prompt}

    dataset = dataset.map(update_prompt)

    # Save to temp and move because datasets can't overwrite itself
    temp_path = rl_path.with_name("rl_temp")
    dataset.save_to_disk(str(temp_path))

    if rl_path.exists():
        shutil.rmtree(rl_path)
    shutil.move(str(temp_path), str(rl_path))

    print(f"✓ Prepended system prompt to {len(dataset['train'])} RL examples")


def preprocess_eval_data(data_dir: Path) -> None:
    """Preprocess EVAL data to include tool-use system prompt."""
    print("Preprocessing evaluation dataset...")
    eval_path = data_dir / "eval"
    if not eval_path.exists():
        return
    dataset = load_from_disk(str(eval_path))

    def add_system_prompt(example):
        prompt = example.get("prompt")
        if not prompt:
            return example
        # Only add if not already present
        # Prepare new prompt list
        new_prompt = []

        # Add system prompt if missing
        if not any(msg.get("role") == "system" for msg in prompt):
            new_prompt.append({"role": "system", "content": SYSTEM_PROMPT})

        # Clean user messages
        for msg in prompt:
            new_msg = msg.copy()
            if new_msg["role"] == "user":
                content = new_msg.get("content", "")
                # Remove conflicting format instructions
                content = content.replace(
                    "The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.",
                    "",
                )
                content = content.replace(
                    'Remember to put your answer on its own line after "Answer:".', ""
                )
                new_msg["content"] = content.strip()

            new_prompt.append(new_msg)

        return {"prompt": new_prompt}

    dataset = dataset.map(add_system_prompt)

    # Save to temp and move
    temp_path = eval_path.with_name("eval_temp")
    dataset.save_to_disk(str(temp_path))

    if eval_path.exists():
        shutil.rmtree(eval_path)
    shutil.move(str(temp_path), str(eval_path))

    print("✓ Prepended system prompt to evaluation examples")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare math datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data"),
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sft", "rl", "eval"],
        help="Datasets to process (sft, rl, eval)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets

    print(f"Downloading datasets to: {args.output_dir}")
    print("=" * 60)

    if "sft" in datasets_to_download:
        download_sft_dataset(args.output_dir)
        preprocess_sft_data(args.output_dir)
        print()

    if "rl" in datasets_to_download:
        download_rl_dataset(args.output_dir)
        preprocess_rl_data(args.output_dir)
        print()

    if "eval" in datasets_to_download:
        download_eval_dataset(args.output_dir)
        preprocess_eval_data(args.output_dir)
        print()

    print("=" * 60)
    print("✓ All datasets downloaded successfully!")
    print(f"\nDatasets location: {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()

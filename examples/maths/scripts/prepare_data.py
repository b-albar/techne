"""Download and preprocess datasets for math tool-use training."""

import argparse
from pathlib import Path

from datasets import load_dataset, load_from_disk


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
    """Preprocess SFT data to ensure tool_call format matches Agent expectations."""
    print("Preprocessing SFT dataset...")
    import json
    import re

    sft_path = data_dir / "sft"
    dataset = load_from_disk(str(sft_path))

    def reformat_and_add_system_prompt(example):
        messages = example["messages"]
        new_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content") or ""

            # Skip existing system prompts to enforce uniformity
            if role == "system":
                continue

            # Clean User Instructions
            if role == "user":
                # Replace broken/legacy instructions - permissive regex
                content = re.sub(
                    r"Remember to place the final answer in the last part using the format:.*?</answer>",
                    NEW_INSTR,
                    content,
                    flags=re.DOTALL,
                )

            # 1. Standardize Tool Calls (Convert JSON tool_calls to Markdown)
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
                                # Debug print for extracted tool code
                                # print(f"Debug: Extracted tool code for message {idx}: {code[:50]}...")
                            else:
                                # Debug print for missing tool code
                                print(
                                    f"Warning: Tool call found but no code extracted for message {idx}. Args: {args}"
                                )

                # Heuristic fallback
                elif ("import " in content or "print(" in content) and "```" not in content:
                    content = f"```python\n{content}\n```"

                content = content.replace("<code>", "```python\n").replace("</code>", "\n```")

            # 2. Standardize Tool Outputs
            if role == "tool" or (role == "user" and "output" in content.lower()):
                if "```" not in content:
                    content = f"```output\n{content}\n```"
                content = content.replace("<interpreter>", "```output\n").replace(
                    "</interpreter>", "\n```"
                )

            # 3. Standardize Final Answer
            if role == "assistant":
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

            new_messages.append({"role": role, "content": content})

        return {"prompt": new_messages}

    # Apply transformation
    dataset = dataset.map(
        reformat_and_add_system_prompt, remove_columns=["messages"], load_from_cache_file=False
    )

    # Save back (using temp path to avoid self-overwrite)
    temp_path = sft_path.with_name("sft_temp")
    dataset.save_to_disk(str(temp_path))

    import shutil

    if sft_path.exists():
        shutil.rmtree(sft_path)
    shutil.move(str(temp_path), str(sft_path))

    sample = dataset["train"][0]
    print(f"✓ Processed sample messages: {len(sample.get('prompt', []))}")
    print(f"✓ Sample content end: {str(sample['prompt'][-1]['content'])[-50:]}")


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

    import shutil

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

    import shutil

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

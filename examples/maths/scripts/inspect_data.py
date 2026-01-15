import argparse
import sys
from pathlib import Path

from datasets import load_from_disk


def inspect_dataset(dataset_path: str, index: int, split: str = "train"):
    """View a specific sample from a dataset."""
    path = Path(dataset_path)
    if not path.exists():
        # Try relative to script
        if (Path.cwd() / dataset_path).exists():
            path = Path.cwd() / dataset_path
        elif (Path(__file__).parent / dataset_path).exists():
            path = Path(__file__).parent / dataset_path
        elif (Path(__file__).parent / "../data" / dataset_path).exists():
            path = Path(__file__).parent / "../data" / dataset_path
        else:
            print(f"Error: Dataset not found at {dataset_path}")
            sys.exit(1)

    try:
        ds = load_from_disk(str(path))
        if hasattr(ds, "keys") and split in ds:
            ds = ds[split]

        if index < 0 or index >= len(ds):
            print(f"Error: Index {index} out of bounds (0 to {len(ds) - 1})")
            sys.exit(1)

        sample = ds[index]
        print(f"\n{'=' * 20} SAMPLE {index} {'=' * 20}")

        if "messages" in sample:
            print(f"Format: Conversation (Messages: {len(sample['messages'])})")
            for i, msg in enumerate(sample["messages"]):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                print(f"\n[{i}] {role}:")
                print("-" * 40)
                print(content)
                print("-" * 40)

        elif "prompt" in sample:
            print("Format: Prompt/Response")
            print(f"\nPROMPT:\n{sample['prompt']}")

            ans = (
                sample.get("answer")
                or sample.get("solution")
                or sample.get("completion")
                or sample.get("ground_truth")
            )
            if not ans and "reward_model" in sample and isinstance(sample["reward_model"], dict):
                ans = sample["reward_model"].get("ground_truth")

            print(f"\nRESPONSE/ANSWER:\n{ans}")

        else:
            print("Format: Unknown, printing raw dict")
            import pprint

            pprint.pprint(sample)

        print(f"\n{'=' * 50}")

    except Exception as e:
        print(f"Error loading/reading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dataset samples")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset (e.g. ../data/sft)"
    )
    parser.add_argument("--index", type=int, default=0, help="Index of sample to view")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")

    args = parser.parse_args()
    inspect_dataset(args.dataset, args.index, args.split)

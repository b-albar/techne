"""Download and preprocess datasets for math tool-use training."""

import argparse
from pathlib import Path

from datasets import load_dataset


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


def preprocess_sft_data(data_dir: Path) -> None:
    """Preprocess SFT data to ensure tool_call format."""
    print("Preprocessing SFT dataset...")
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_dir / "sft"))

    # The ReTool-SFT-multi-turn dataset already has tool_call attribute
    # Just verify the format
    sample = dataset["train"][0]
    print(f"✓ Sample conversation turns: {len(sample.get('messages', []))}")
    print(f"✓ Has tool calls: {'tool_calls' in sample or 'code' in str(sample)}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare math datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["sft", "rl", "eval", "all"],
        default=["all"],
        help="Which datasets to download",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["sft", "rl", "eval"]

    print(f"Downloading datasets to: {args.output_dir}")
    print("=" * 60)

    if "sft" in datasets_to_download:
        download_sft_dataset(args.output_dir)
        preprocess_sft_data(args.output_dir)
        print()

    if "rl" in datasets_to_download:
        download_rl_dataset(args.output_dir)
        print()

    if "eval" in datasets_to_download:
        download_eval_dataset(args.output_dir)
        print()

    print("=" * 60)
    print("✓ All datasets downloaded successfully!")
    print(f"\nDatasets location: {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()

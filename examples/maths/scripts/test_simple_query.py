"""Simple test script for MathToolAgent."""

import argparse
import asyncio
import os

from math_agent import MathToolAgent
from techne.config import TechneConfig


async def run_test(args):
    # 1. Initialize Config
    config = TechneConfig(model={"name_or_path": args.model}, max_turns=args.max_turns)

    # 2. Force backend if requested
    if args.backend == "huggingface" and "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    # 3. Initialize Agent
    print(f"Initializing agent: {args.model}")
    agent = MathToolAgent(config)
    print(f"Backend: {agent.backend}")
    print("=" * 60)

    # 4. Generate Trajectory
    print(f"Prompt: {args.prompt}\n")
    try:
        trajectories = await agent.collect_trajectories([args.prompt])
        traj = trajectories[0]

        print("--- Full Trajectory ---")
        print(traj.to_text())
        print("=" * 60)

        if traj.reward is not None:
            print(f"Reward: {traj.reward}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test Math Agent")
    parser.add_argument("--prompt", type=str, default="Calculate 2+2 and then its square.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--backend", type=str, choices=["openai", "huggingface", "auto"], default="auto"
    )
    parser.add_argument("--max-turns", type=int, default=5)

    args = parser.parse_args()
    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()

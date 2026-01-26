"""Benchmark KV cache vs no cache for math agent."""

import asyncio
import time

from math_agent import MathToolAgent
from techne.config import TechneConfig

# A harder problem that requires more computation steps
PROMPT = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$."


def main():
    config = TechneConfig(
        **{
            "model": {"name_or_path": "Qwen/Qwen3-0.6B", "dtype": "bfloat16", "compile": False},
            "rollout": {
                "max_turns": 8,
                "max_new_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 50,
            },
        }
    )

    agent = MathToolAgent(config)

    # Warmup
    print("Warming up...")
    asyncio.run(agent._run_rollout(PROMPT, use_kv_cache=False))

    # Benchmark without cache
    print("\n" + "=" * 50)
    print("Running WITHOUT KV cache...")
    start = time.perf_counter()
    traj_no_cache = asyncio.run(agent._run_rollout(PROMPT, use_kv_cache=False))
    time_no_cache = time.perf_counter() - start
    turns_no_cache = len([s for s in traj_no_cache.steps if s.role == "assistant"])
    tokens_no_cache = sum(len(s.token_ids) for s in traj_no_cache.steps if s.role == "assistant")

    # Benchmark with cache
    print("\nRunning WITH KV cache...")
    start = time.perf_counter()
    traj_cache = asyncio.run(agent._run_rollout(PROMPT, use_kv_cache=True))
    time_cache = time.perf_counter() - start
    turns_cache = len([s for s in traj_cache.steps if s.role == "assistant"])
    tokens_cache = sum(len(s.token_ids) for s in traj_cache.steps if s.role == "assistant")

    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    tps_no_cache = tokens_no_cache / time_no_cache if time_no_cache > 0 else 0
    tps_cache = tokens_cache / time_cache if time_cache > 0 else 0

    print(
        f"Without cache: {time_no_cache:.2f}s | {turns_no_cache} turns | {tokens_no_cache} tokens | {tps_no_cache:.1f} tok/s"
    )
    print(
        f"With cache:    {time_cache:.2f}s | {turns_cache} turns | {tokens_cache} tokens | {tps_cache:.1f} tok/s"
    )
    if tps_no_cache > 0:
        speedup = tps_cache / tps_no_cache
        print(f"Speedup:       {speedup:.2f}x (tokens/sec)")


if __name__ == "__main__":
    main()

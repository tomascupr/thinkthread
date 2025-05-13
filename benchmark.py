"""Benchmark script for ThinkThread SDK optimizations.

This script tests the performance of the ThinkThread SDK with different
optimization configurations using the OpenAI API.
"""

import os
import time
import asyncio
from typing import Dict, Any

from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm.openai_client import OpenAIClient
from thinkthread_sdk.monitoring import GLOBAL_MONITOR


async def run_benchmark(config: ThinkThreadConfig, question: str) -> Dict[str, Any]:
    """Run a benchmark with the given configuration.

    Args:
        config: The configuration to use
        question: The question to ask

    Returns:
        A dictionary with benchmark results
    """
    client = OpenAIClient()

    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(True)

    session = ThinkThreadSession(
        llm_client=client,
        alternatives=config.alternatives,
        max_rounds=config.max_rounds,
        config=config,
    )

    start_time = time.time()
    result = await session.run_async(question)
    elapsed = time.time() - start_time

    stats = GLOBAL_MONITOR.get_stats()

    return {
        "elapsed": elapsed,
        "result": result,
        "stats": stats,
    }


async def main():
    """Run benchmarks with different optimization configurations."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print(f"Using OpenAI API key: {api_key[:10]}...")

    question = (
        "What are the key differences between supervised and unsupervised learning?"
    )

    configs = [
        {
            "name": "Baseline (no optimizations)",
            "config": ThinkThreadConfig(
                alternatives=2,
                max_rounds=2,
                parallel_alternatives=False,
                parallel_evaluation=False,
                use_caching=False,
                early_termination=False,
                enable_monitoring=True,
            ),
        },
        {
            "name": "Parallel alternatives only",
            "config": ThinkThreadConfig(
                alternatives=2,
                max_rounds=2,
                parallel_alternatives=True,
                parallel_evaluation=False,
                use_caching=False,
                early_termination=False,
                enable_monitoring=True,
            ),
        },
        {
            "name": "All optimizations",
            "config": ThinkThreadConfig(
                alternatives=2,
                max_rounds=2,
                parallel_alternatives=True,
                parallel_evaluation=True,
                use_caching=True,
                early_termination=True,
                early_termination_threshold=0.95,
                enable_monitoring=True,
            ),
        },
    ]

    results = []

    for config_info in configs:
        print(f"\nRunning benchmark: {config_info['name']}")
        try:
            result = await run_benchmark(config_info["config"], question)
            results.append(
                {
                    "name": config_info["name"],
                    "elapsed": result["elapsed"],
                    "stats": result["stats"],
                }
            )
            print(f"  Elapsed time: {result['elapsed']:.2f}s")
            for operation, op_stats in result["stats"].items():
                print(
                    f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}"
                )
        except Exception as e:
            print(f"  Error: {e}")

    if len(results) > 1:
        print("\nSummary:")
        baseline = results[0]["elapsed"]
        for result in results:
            speedup = baseline / result["elapsed"] if result["elapsed"] > 0 else 0
            print(
                f"{result['name']}: {result['elapsed']:.2f}s ({speedup:.2f}x speedup)"
            )


if __name__ == "__main__":
    asyncio.run(main())

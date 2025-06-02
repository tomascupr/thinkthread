"""Simple benchmark script for ThinkThread SDK optimizations.

This script tests the performance of the ThinkThread SDK with different
optimization configurations using the OpenAI API, focusing on parallel processing.
"""

import os
import time
import asyncio
from typing import Dict, Any

from thinkthread.config import ThinkThreadConfig
from thinkthread.session import ThinkThreadSession
from thinkthread.llm.openai_client import OpenAIClient


async def run_benchmark(
    config_name: str, use_parallel: bool, question: str
) -> Dict[str, Any]:
    """Run a benchmark with the given configuration.

    Args:
        config_name: Name of the configuration for reporting
        use_parallel: Whether to use parallel processing
        question: The question to ask

    Returns:
        A dictionary with benchmark results
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    client = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")

    config = ThinkThreadConfig(
        alternatives=2,
        max_rounds=2,
        parallel_alternatives=use_parallel,
        parallel_evaluation=use_parallel,
        use_caching=False,
        early_termination=False,
    )

    session = ThinkThreadSession(
        llm_client=client,
        alternatives=config.alternatives,
        max_rounds=config.max_rounds,
        config=config,
    )

    print(f"\nRunning benchmark: {config_name}")
    start_time = time.time()

    try:
        result = await session.run_async(question)
        elapsed = time.time() - start_time

        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  Result: {result[:100]}...")

        return {
            "name": config_name,
            "elapsed": elapsed,
            "result": result,
        }
    except Exception as e:
        print(f"  Error: {str(e)}")
        return {
            "name": config_name,
            "error": str(e),
        }


async def main():
    """Run benchmarks with different optimization configurations."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print(f"Using OpenAI API key: {api_key[:10]}...")

    question = "What are three key benefits of exercise?"

    results = []

    baseline_result = await run_benchmark(
        "Baseline (no parallel processing)",
        use_parallel=False,
        question=question,
    )
    if "elapsed" in baseline_result:
        results.append(baseline_result)

    parallel_result = await run_benchmark(
        "With parallel processing",
        use_parallel=True,
        question=question,
    )
    if "elapsed" in parallel_result:
        results.append(parallel_result)

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

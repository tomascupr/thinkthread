"""Patched benchmark script for ThinkThread SDK optimizations.

This script tests the performance of the ThinkThread SDK with different
optimization configurations using the OpenAI API.
"""

import os
import time
import asyncio
from typing import Dict, Any, AsyncIterator

from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm.openai_client import OpenAIClient
from thinkthread_sdk.monitoring import GLOBAL_MONITOR


class PatchedOpenAIClient(OpenAIClient):
    """Patched OpenAI client that implements the _generate_uncached method."""

    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text without using the cache.

        This method implements the abstract method from LLMClient.
        It simply calls the original generate method implementation.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters

        Returns:
            The generated text response from the model
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )

            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""

        except Exception as e:
            error_message = f"Error when calling OpenAI API: {str(e)}"
            return error_message

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from the language model.

        This is a minimal implementation to satisfy the abstract method.

        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters

        Yields:
            Chunks of the generated text response
        """
        response = await self.acomplete(prompt, **kwargs)
        yield response


async def run_benchmark(
    config_name: str, config: ThinkThreadConfig, question: str
) -> Dict[str, Any]:
    """Run a benchmark with the given configuration.

    Args:
        config_name: Name of the configuration for reporting
        config: The configuration to use
        question: The question to ask

    Returns:
        A dictionary with benchmark results
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    client = PatchedOpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")
    client.enable_cache(config.use_caching)

    if config.concurrency_limit > 0:
        client.set_concurrency_limit(config.concurrency_limit)

    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(True)

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

        stats = GLOBAL_MONITOR.get_stats()

        print(f"  Elapsed time: {elapsed:.2f}s")
        for operation, op_stats in stats.items():
            print(
                f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}"
            )

        return {
            "name": config_name,
            "elapsed": elapsed,
            "result": result,
            "stats": stats,
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
                concurrency_limit=5,
                enable_monitoring=True,
            ),
        },
    ]

    results = []

    for config_info in configs:
        try:
            result = await run_benchmark(
                config_info["name"],
                config_info["config"],
                question,
            )
            if "elapsed" in result:
                results.append(result)
        except Exception as e:
            print(f"Error running benchmark {config_info['name']}: {str(e)}")

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

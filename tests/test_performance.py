"""Tests for performance profiling of the ThinkThread SDK."""

import time
from typing import Dict, Any, List

from thinkthread_sdk.llm.dummy import DummyLLMClient
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.config import ThinkThreadConfig


class TimingStats:
    """Simple class to collect timing statistics."""

    def __init__(self):
        self.timing_data: Dict[str, List[float]] = {}

    def record(self, key: str, time_taken: float):
        if key not in self.timing_data:
            self.timing_data[key] = []
        self.timing_data[key].append(time_taken)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for key, values in self.timing_data.items():
            if not values:
                continue
            stats[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values),
                "count": len(values),
            }
        return stats


class ProfilingLLMClient(DummyLLMClient):
    """LLM client wrapper that profiles execution time."""

    def __init__(self, *args, **kwargs):
        self.stats = TimingStats()
        super().__init__(*args, **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        start_time = time.time()
        result = super().generate(prompt, **kwargs)
        elapsed = time.time() - start_time
        self.stats.record("generate", elapsed)
        return result

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        start_time = time.time()
        result = await super().acomplete(prompt, **kwargs)
        elapsed = time.time() - start_time
        self.stats.record("acomplete", elapsed)
        return result


def test_cort_session_performance():
    """Test and profile the performance of ThinkThread session with multiple rounds."""
    client = ProfilingLLMClient(
        responses=["Initial"] + ["Alt" for _ in range(9)] + ["Best" for _ in range(3)]
    )
    config = ThinkThreadConfig(use_pairwise_evaluation=False)

    overall_stats = TimingStats()

    rounds_to_test = [1, 2, 3]

    for rounds in rounds_to_test:
        client.reset()

        start_time = time.time()

        session = ThinkThreadSession(
            llm_client=client, max_rounds=rounds, alternatives=3, config=config
        )

        session.run("Test question for performance profiling")

        elapsed = time.time() - start_time
        overall_stats.record(f"total_rounds_{rounds}", elapsed)

    print("\nPerformance Statistics:")
    print("LLM Client Operations:")
    for key, stats in client.stats.get_stats().items():
        print(
            f"  {key}: avg={stats['avg']:.4f}s, total={stats['total']:.4f}s, count={stats['count']}"
        )

    print("\nOverall Session Performance:")
    for key, stats in overall_stats.get_stats().items():
        print(f"  {key}: {stats['avg']:.4f}s")

    stats = overall_stats.get_stats()

    assert "total_rounds_1" in stats
    assert "total_rounds_2" in stats
    assert "total_rounds_3" in stats

    time_per_round = (
        stats["total_rounds_3"]["avg"] - stats["total_rounds_1"]["avg"]
    ) / 2

    print(f"\nEstimated overhead per round: {time_per_round:.4f}s")

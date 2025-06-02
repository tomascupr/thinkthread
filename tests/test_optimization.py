"""Tests for performance optimizations in the ThinkThread SDK."""

import pytest
import time
import asyncio
from typing import Dict, Any

from thinkthread.llm.dummy import DummyLLMClient
from thinkthread.session import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig
from thinkthread.monitoring import GLOBAL_MONITOR


class DelayedDummyLLMClient(DummyLLMClient):
    """Dummy LLM client with configurable delay for performance testing."""

    def __init__(self, delay: float = 0.5, **kwargs: Any) -> None:
        """Initialize the delayed dummy client.

        Args:
            delay: Delay in seconds for each call
            **kwargs: Additional arguments for DummyLLMClient
        """
        super().__init__(**kwargs)
        self.delay = delay
        self._use_cache = False
        self._semaphore = None
        self._cache = {}

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching.

        Args:
            enabled: Whether to enable caching
        """
        self._use_cache = enabled

    def set_concurrency_limit(self, limit: int) -> None:
        """Set the concurrency limit.

        Args:
            limit: Maximum number of concurrent calls
        """
        self._semaphore = asyncio.Semaphore(limit) if limit > 0 else None

    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response with a delay.

        Args:
            prompt: The input text
            **kwargs: Additional parameters

        Returns:
            The generated response
        """
        time.sleep(self.delay)
        return super().generate(prompt, **kwargs)

    async def _acomplete_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response with a delay.

        Args:
            prompt: The input text
            **kwargs: Additional parameters

        Returns:
            The generated response
        """
        await asyncio.sleep(self.delay)
        return await super().acomplete(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response with optional caching.

        Args:
            prompt: The input text
            **kwargs: Additional parameters

        Returns:
            The generated response
        """
        if self._use_cache:
            cache_key = f"{prompt}_{kwargs.get('temperature', 0.7)}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            result = self._generate_uncached(prompt, **kwargs)
            self._cache[cache_key] = result
            return result

        return self._generate_uncached(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response with optional caching.

        Args:
            prompt: The input text
            **kwargs: Additional parameters

        Returns:
            The generated response
        """
        if self._use_cache:
            cache_key = f"{prompt}_{kwargs.get('temperature', 0.7)}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            if self._semaphore is not None:
                async with self._semaphore:
                    result = await self._acomplete_uncached(prompt, **kwargs)
            else:
                result = await self._acomplete_uncached(prompt, **kwargs)

            self._cache[cache_key] = result
            return result

        if self._semaphore is not None:
            async with self._semaphore:
                return await self._acomplete_uncached(prompt, **kwargs)

        return await self._acomplete_uncached(prompt, **kwargs)

    def reset(self) -> None:
        """Reset the client state."""
        self._cache.clear()
        if hasattr(self, "_responses_index"):
            self._responses_index = 0


@pytest.mark.parametrize(
    "optimization_config,expected_speedup",
    [
        (
            {
                "parallel_alternatives": False,
                "parallel_evaluation": False,
                "use_caching": False,
            },
            1.0,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": False,
                "use_caching": False,
            },
            1.8,
        ),
        (
            {
                "parallel_alternatives": False,
                "parallel_evaluation": True,
                "use_caching": False,
            },
            1.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": False,
            },
            2.5,
        ),
        (
            {
                "parallel_alternatives": False,
                "parallel_evaluation": False,
                "use_caching": True,
            },
            1.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
            },
            3.0,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.9,
            },
            3.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "use_batched_requests": True,
            },
            3.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "use_fast_similarity": True,
            },
            3.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "use_adaptive_temperature": True,
                "initial_temperature": 0.7,
                "generation_temperature": 0.9,
                "min_generation_temperature": 0.5,
                "temperature_decay_rate": 0.8,
            },
            3.5,
        ),
        (
            {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.9,
                "use_batched_requests": True,
                "use_fast_similarity": True,
                "use_adaptive_temperature": True,
                "initial_temperature": 0.7,
                "generation_temperature": 0.9,
                "min_generation_temperature": 0.5,
                "temperature_decay_rate": 0.8,
            },
            5.0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_optimization_performance(
    optimization_config: Dict[str, Any], expected_speedup: float
) -> None:
    """Test the performance impact of various optimizations."""
    alternatives = 3
    max_rounds = 2
    delay = 0.2  # 200ms delay per LLM call

    client = DelayedDummyLLMClient(
        delay=delay,
        responses=["Initial"]
        + ["Alt" for _ in range(alternatives * max_rounds)]
        + ["Best" for _ in range(max_rounds)],
    )

    if optimization_config.get("use_caching", False):
        client.enable_cache(True)

    if "concurrency_limit" in optimization_config:
        client.set_concurrency_limit(optimization_config["concurrency_limit"])
    else:
        client.set_concurrency_limit(5)  # Default concurrency limit

    config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        use_pairwise_evaluation=False,
        enable_monitoring=True,
        **optimization_config,
    )

    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(True)

    session = ThinkThreadSession(
        llm_client=client,
        alternatives=alternatives,
        max_rounds=max_rounds,
        config=config,
    )

    start_time = time.time()
    await session.run_async("Test question for performance optimization")
    elapsed = time.time() - start_time

    stats = GLOBAL_MONITOR.get_stats()

    print(f"\nPerformance with {optimization_config}:")
    print(f"Total time: {elapsed:.2f}s")
    for operation, op_stats in stats.items():
        print(
            f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}"
        )

    theoretical_min = delay  # Initial call

    if optimization_config.get("parallel_alternatives", False):
        theoretical_min += delay * max_rounds  # One delay per round for alternatives
    else:
        theoretical_min += delay * alternatives * max_rounds

    if optimization_config.get("parallel_evaluation", False):
        theoretical_min += delay * max_rounds  # One delay per round for evaluation
    else:
        theoretical_min += delay * max_rounds  # One evaluation per round

    if optimization_config.get("use_batched_requests", False):
        theoretical_min = theoretical_min * 0.8

    # Adjust theoretical minimum for fast similarity
    if optimization_config.get(
        "use_fast_similarity", False
    ) and optimization_config.get("early_termination", False):
        theoretical_min = theoretical_min * 0.9

    assert (
        elapsed <= theoretical_min * 1.5
    ), f"Expected time close to {theoretical_min}s, got {elapsed}s"

    if any(optimization_config.values()):
        client.reset()

        baseline_config = ThinkThreadConfig(
            alternatives=alternatives,
            max_rounds=max_rounds,
            use_pairwise_evaluation=False,
            parallel_alternatives=False,
            parallel_evaluation=False,
            use_caching=False,
            enable_monitoring=True,
        )

        GLOBAL_MONITOR.reset()

        baseline_session = ThinkThreadSession(
            llm_client=client,
            alternatives=alternatives,
            max_rounds=max_rounds,
            config=baseline_config,
        )

        start_time = time.time()
        await baseline_session.run_async("Test question for performance optimization")
        baseline_elapsed = time.time() - start_time

        print(f"Baseline (no optimizations): {baseline_elapsed:.2f}s")
        print(f"Speedup: {baseline_elapsed / elapsed:.2f}x")

        assert (
            elapsed < baseline_elapsed
        ), "Expected optimized version to be faster than baseline"

        min_expected_speedup = expected_speedup * 0.8  # Allow 20% tolerance
        actual_speedup = baseline_elapsed / elapsed

        assert (
            actual_speedup >= min_expected_speedup
        ), f"Expected at least {min_expected_speedup:.2f}x speedup, got {actual_speedup:.2f}x"

        if all(
            v
            for k, v in optimization_config.items()
            if k
            in [
                "parallel_alternatives",
                "parallel_evaluation",
                "use_caching",
                "early_termination",
                "use_batched_requests",
                "use_fast_similarity",
                "use_adaptive_temperature",
            ]
        ):
            assert (
                actual_speedup >= 3.0
            ), f"Expected at least 3x speedup with all optimizations, got {actual_speedup:.2f}x"


def test_early_termination() -> None:
    """Test the early termination optimization."""
    alternatives = 3
    max_rounds = 3
    delay = 0.2

    client = DelayedDummyLLMClient(
        delay=delay,
        responses=[
            "Initial answer",  # Initial response
            "Alternative 1",  # First round alternatives
            "Alternative 2",
            "Alternative 3",
            "Best answer",  # Best answer after first round
            "Alternative 4",  # Second round alternatives (should be skipped)
            "Alternative 5",
            "Alternative 6",
        ],
    )

    config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        early_termination=True,
        early_termination_threshold=0.9,  # High threshold to ensure termination
        enable_monitoring=True,
    )

    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(True)

    session = ThinkThreadSession(
        llm_client=client,
        alternatives=alternatives,
        max_rounds=max_rounds,
        config=config,
    )

    start_time = time.time()

    original_calculate_similarity = session._calculate_similarity

    def mock_calculate_similarity(str1: str, str2: str) -> float:
        if str1 == "Best answer" and str2 == "Initial answer":
            return 0.95  # Above the threshold
        return original_calculate_similarity(str1, str2)

    session._calculate_similarity = mock_calculate_similarity

    result = session.run("Test question for early termination")
    elapsed = time.time() - start_time

    assert result == "Best answer"

    stats = GLOBAL_MONITOR.get_stats()

    print("\nPerformance with early termination:")
    print(f"Total time: {elapsed:.2f}s")
    for operation, op_stats in stats.items():
        print(
            f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}"
        )

    client.reset()

    baseline_config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        early_termination=False,
        enable_monitoring=True,
    )

    GLOBAL_MONITOR.reset()

    baseline_session = ThinkThreadSession(
        llm_client=client,
        alternatives=alternatives,
        max_rounds=max_rounds,
        config=baseline_config,
    )

    start_time = time.time()
    _ = baseline_session.run("Test question for early termination")
    baseline_elapsed = time.time() - start_time

    print(f"Baseline (no early termination): {baseline_elapsed:.2f}s")
    print(f"Speedup: {baseline_elapsed / elapsed:.2f}x")

    assert elapsed < baseline_elapsed, "Expected early termination to be faster"

    expected_speedup = max_rounds / (
        max_rounds - 1
    )  # If we skip 1 out of 3 rounds, expect ~1.5x speedup
    assert (
        baseline_elapsed / elapsed >= expected_speedup * 0.8
    ), f"Expected at least {expected_speedup}x speedup, got {baseline_elapsed / elapsed:.2f}x"


def test_fast_similarity() -> None:
    """Test the fast similarity optimization."""
    alternatives = 3
    max_rounds = 2
    delay = 0.2

    client = DelayedDummyLLMClient(
        delay=delay,
        responses=[
            "Initial answer that is quite long with many details about the topic at hand",
            "Alternative 1 that is also quite verbose and contains many words to make similarity calculation slower",
            "Alternative 2 with even more text to process during similarity calculation",
            "Alternative 3 with a substantial amount of content to analyze",
            "Best answer with comprehensive information that needs to be compared",
        ],
    )

    config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        early_termination=True,
        early_termination_threshold=0.8,
        use_fast_similarity=True,
        enable_monitoring=True,
    )

    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(True)

    session = ThinkThreadSession(
        llm_client=client,
        alternatives=alternatives,
        max_rounds=max_rounds,
        config=config,
    )

    original_calculate_similarity = session._calculate_similarity
    original_calculate_fast_similarity = getattr(
        session, "_calculate_fast_similarity", None
    )

    similarity_times = []

    def timed_calculate_similarity(str1: str, str2: str) -> float:
        start_time = time.time()
        result = original_calculate_similarity(str1, str2)
        elapsed = time.time() - start_time
        similarity_times.append(elapsed)
        return result

    def timed_calculate_fast_similarity(str1: str, str2: str) -> float:
        start_time = time.time()
        if original_calculate_fast_similarity is not None:
            result = original_calculate_fast_similarity(str1, str2)
        else:
            result = 0.0  # Default value if the function doesn't exist
        elapsed = time.time() - start_time
        similarity_times.append(elapsed)
        return result

    session._calculate_similarity = timed_calculate_similarity
    if original_calculate_fast_similarity:
        session._calculate_fast_similarity = timed_calculate_fast_similarity

    start_time = time.time()
    session.run("Test question for fast similarity")
    fast_elapsed = time.time() - start_time

    fast_similarity_times = similarity_times.copy()

    # Test with standard similarity
    client.reset()
    similarity_times.clear()

    baseline_config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        early_termination=True,
        early_termination_threshold=0.8,
        use_fast_similarity=False,
        enable_monitoring=True,
    )

    GLOBAL_MONITOR.reset()

    baseline_session = ThinkThreadSession(
        llm_client=client,
        alternatives=alternatives,
        max_rounds=max_rounds,
        config=baseline_config,
    )

    baseline_session._calculate_similarity = timed_calculate_similarity

    start_time = time.time()
    baseline_session.run("Test question for fast similarity")
    standard_elapsed = time.time() - start_time

    standard_similarity_times = similarity_times.copy()

    print("\nPerformance with fast similarity:")
    print(f"Total time with fast similarity: {fast_elapsed:.4f}s")
    print(f"Total time with standard similarity: {standard_elapsed:.4f}s")
    print(
        f"Average similarity calculation time (fast): {sum(fast_similarity_times) / len(fast_similarity_times):.4f}s"
    )
    print(
        f"Average similarity calculation time (standard): {sum(standard_similarity_times) / len(standard_similarity_times):.4f}s"
    )
    print(f"Speedup: {standard_elapsed / fast_elapsed:.2f}x")

    assert sum(fast_similarity_times) <= sum(
        standard_similarity_times
    ), "Expected fast similarity to be faster than standard similarity"


@pytest.mark.asyncio
async def test_semantic_caching() -> None:
    """Test the semantic caching optimization."""
    delay = 0.2

    class SemanticCachingDummyClient(DelayedDummyLLMClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._embedding_cache = {}
            self._semantic_cache = {}
            self._use_semantic_cache = False
            self._semantic_similarity_threshold = 0.9

        def enable_semantic_cache(self, enabled=True, similarity_threshold=0.9):
            self._use_semantic_cache = enabled
            self._semantic_similarity_threshold = similarity_threshold

        async def aembed(self, text):
            await asyncio.sleep(0.05)  # Small delay for embedding generation
            return [len(text), text.count(" "), text.count("."), text.count("a")]

        def _cosine_similarity(self, v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_v1 = sum(a * a for a in v1) ** 0.5
            norm_v2 = sum(b * b for b in v2) ** 0.5
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            return dot_product / (norm_v1 * norm_v2)

    # Create test prompts with semantic similarity
    prompts = [
        "What are the benefits of exercise?",
        "What are the advantages of physical activity?",  # Semantically similar to first prompt
        "How does regular exercise improve health?",  # Semantically similar to first prompt
        "What is the capital of France?",  # Different topic
        "How do I make chocolate chip cookies?",  # Different topic
    ]

    client = SemanticCachingDummyClient(
        delay=delay,
        responses=["Response " + str(i) for i in range(10)],
    )

    client.enable_cache(True)
    client.enable_semantic_cache(True, 0.8)

    start_time = time.time()
    _ = await client.acomplete(prompts[0])
    first_request_time = time.time() - start_time

    start_time = time.time()
    _ = await client.acomplete(prompts[1])
    semantic_hit_time = time.time() - start_time

    start_time = time.time()
    _ = await client.acomplete(prompts[3])
    different_topic_time = time.time() - start_time

    start_time = time.time()
    _ = await client.acomplete(prompts[0])
    exact_hit_time = time.time() - start_time

    print("\nSemantic caching performance:")
    print(f"First request (cache miss): {first_request_time:.4f}s")
    print(f"Semantically similar request: {semantic_hit_time:.4f}s")
    print(f"Different topic request: {different_topic_time:.4f}s")
    print(f"Exact same request: {exact_hit_time:.4f}s")

    assert (
        semantic_hit_time < first_request_time
    ), "Expected semantic cache hit to be faster than cache miss"
    assert (
        semantic_hit_time > exact_hit_time
    ), "Expected semantic cache hit to be slower than exact cache hit"
    assert (
        different_topic_time > semantic_hit_time
    ), "Expected different topic to be a cache miss"

    client.reset()
    client.enable_cache(True)
    client.enable_semantic_cache(False)

    await client.acomplete(prompts[0])

    start_time = time.time()
    await client.acomplete(prompts[1])
    no_semantic_time = time.time() - start_time

    print(
        f"Semantically similar request (without semantic caching): {no_semantic_time:.4f}s"
    )
    assert (
        no_semantic_time > semantic_hit_time
    ), "Expected request without semantic caching to be slower"

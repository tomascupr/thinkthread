"""Tests for performance optimizations in the ThinkThread SDK."""

import pytest
import time
import asyncio
from typing import Dict, Any, List, Optional

from thinkthread_sdk.llm.dummy import DummyLLMClient
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.monitoring import GLOBAL_MONITOR


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
    "optimization_config",
    [
        {"parallel_alternatives": False, "parallel_evaluation": False, "use_caching": False},
        {"parallel_alternatives": True, "parallel_evaluation": False, "use_caching": False},
        {"parallel_alternatives": False, "parallel_evaluation": True, "use_caching": False},
        {"parallel_alternatives": True, "parallel_evaluation": True, "use_caching": False},
        {"parallel_alternatives": False, "parallel_evaluation": False, "use_caching": True},
        {"parallel_alternatives": True, "parallel_evaluation": True, "use_caching": True},
    ],
)
@pytest.mark.asyncio
async def test_optimization_performance(optimization_config: Dict[str, bool]) -> None:
    """Test the performance impact of various optimizations."""
    alternatives = 3
    max_rounds = 2
    delay = 0.2  # 200ms delay per LLM call
    
    client = DelayedDummyLLMClient(
        delay=delay,
        responses=["Initial"] + ["Alt" for _ in range(alternatives * max_rounds)] + ["Best" for _ in range(max_rounds)],
    )
    
    config = ThinkThreadConfig(
        alternatives=alternatives,
        max_rounds=max_rounds,
        use_pairwise_evaluation=False,
        enable_monitoring=True,
        concurrency_limit=5,
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
        print(f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}")
    
    
    theoretical_min = delay  # Initial call
    
    if optimization_config["parallel_alternatives"]:
        theoretical_min += delay * max_rounds  # One delay per round for alternatives
    else:
        theoretical_min += delay * alternatives * max_rounds
    
    if optimization_config["parallel_evaluation"]:
        theoretical_min += delay * max_rounds  # One delay per round for evaluation
    else:
        theoretical_min += delay * max_rounds  # One evaluation per round
    
    
    assert elapsed <= theoretical_min * 1.5, f"Expected time close to {theoretical_min}s, got {elapsed}s"
    
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
        
        assert elapsed < baseline_elapsed, f"Expected optimized version to be faster than baseline"
        
        if all(v for k, v in optimization_config.items() if k != "early_termination"):
            assert baseline_elapsed / elapsed >= 3.0, f"Expected at least 3x speedup, got {baseline_elapsed / elapsed:.2f}x"


def test_early_termination() -> None:
    """Test the early termination optimization."""
    alternatives = 3
    max_rounds = 3
    delay = 0.2
    
    client = DelayedDummyLLMClient(
        delay=delay,
        responses=[
            "Initial answer",  # Initial response
            "Alternative 1",   # First round alternatives
            "Alternative 2",
            "Alternative 3",
            "Best answer",     # Best answer after first round
            "Alternative 4",   # Second round alternatives (should be skipped)
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
    
    print(f"\nPerformance with early termination:")
    print(f"Total time: {elapsed:.2f}s")
    for operation, op_stats in stats.items():
        print(f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}")
    
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
    baseline_result = baseline_session.run("Test question for early termination")
    baseline_elapsed = time.time() - start_time
    
    print(f"Baseline (no early termination): {baseline_elapsed:.2f}s")
    print(f"Speedup: {baseline_elapsed / elapsed:.2f}x")
    
    assert elapsed < baseline_elapsed, f"Expected early termination to be faster"
    
    expected_speedup = max_rounds / (max_rounds - 1)  # If we skip 1 out of 3 rounds, expect ~1.5x speedup
    assert baseline_elapsed / elapsed >= expected_speedup * 0.8, f"Expected at least {expected_speedup}x speedup, got {baseline_elapsed / elapsed:.2f}x"

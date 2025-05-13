"""Performance monitoring for ThinkThread SDK.

This module provides utilities for monitoring the performance of the ThinkThread SDK.
"""

import time
from typing import Dict, List, Any, Callable, TypeVar
from functools import wraps

T = TypeVar("T")


class PerformanceMonitor:
    """Performance monitor for tracking execution time of operations.

    This class provides utilities for tracking the execution time of
    operations in the ThinkThread SDK, such as LLM calls, alternative
    generation, and evaluation.
    """

    def __init__(self) -> None:
        """Initialize an empty performance monitor."""
        self.timing_data: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        self.enabled = False

    def enable(self, enabled: bool = True) -> None:
        """Enable or disable performance monitoring.

        Args:
            enabled: Whether to enable monitoring
        """
        self.enabled = enabled

    def start(self, operation: str) -> None:
        """Start timing an operation.

        Args:
            operation: Name of the operation
        """
        if not self.enabled:
            return

        self.start_times[operation] = time.time()

    def end(self, operation: str) -> None:
        """End timing an operation and record the elapsed time.

        Args:
            operation: Name of the operation
        """
        if not self.enabled or operation not in self.start_times:
            return

        elapsed = time.time() - self.start_times[operation]
        if operation not in self.timing_data:
            self.timing_data[operation] = []

        self.timing_data[operation].append(elapsed)
        del self.start_times[operation]

    def record(self, operation: str, elapsed: float) -> None:
        """Record the elapsed time for an operation.

        Args:
            operation: Name of the operation
            elapsed: Elapsed time in seconds
        """
        if not self.enabled:
            return

        if operation not in self.timing_data:
            self.timing_data[operation] = []

        self.timing_data[operation].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations.

        Returns:
            A dictionary mapping operation names to statistics
        """
        stats = {}
        for operation, times in self.timing_data.items():
            if not times:
                continue

            stats[operation] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "total": sum(times),
                "count": len(times),
            }

        return stats

    def reset(self) -> None:
        """Reset all timing data."""
        self.timing_data.clear()
        self.start_times.clear()


GLOBAL_MONITOR = PerformanceMonitor()


def timed(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for timing function execution.

    This decorator can be applied to methods to track their execution time.

    Args:
        operation: Name of the operation

    Returns:
        The decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not GLOBAL_MONITOR.enabled:
                return func(*args, **kwargs)

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            GLOBAL_MONITOR.record(operation, elapsed)
            return result

        return wrapper

    return decorator


async def timed_async(
    operation: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Time an async function execution.

    Args:
        operation: Name of the operation
        func: The async function to time
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function
    """
    if not GLOBAL_MONITOR.enabled:
        return await func(*args, **kwargs)

    start_time = time.time()
    result = await func(*args, **kwargs)
    elapsed = time.time() - start_time

    GLOBAL_MONITOR.record(operation, elapsed)
    return result

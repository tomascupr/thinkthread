"""
ThinkThread: Advanced reasoning for AI applications

Basic usage:
    from thinkthread import reason
    answer = reason("What is consciousness?")

Advanced usage:
    answer = reason.explore("Design a mars colony", visualize=True)
    answer = reason.debate("Is AI sentient?")
    answer = reason.solve("Climate change solutions")
"""

from .reason import reason, explore, refine, debate, solve
from .modes import available, search
from .transparency import ReasoningDebugger, LiveReasoningView, ReasoningProfiler

__version__ = "0.8.0"

__all__ = [
    'reason',
    'explore',
    'refine',
    'debate',
    'solve',
    'available',
    'search',
    'ReasoningDebugger',
    'LiveReasoningView',
    'ReasoningProfiler',
]


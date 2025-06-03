"""ThinkThread: Make your AI think before it speaks."""

__version__ = "0.8.1"

# Simple API
from .api import reason, explore, solve, debate, refine

# Advanced SDK access
from .session import ThinkThreadSession
from .tree_thinker import TreeThinker
from .config import create_config, ThinkThreadConfig

__all__ = [
    # Simple API
    "reason",
    "explore",
    "solve",
    "debate",
    "refine",
    # Advanced SDK
    "ThinkThreadSession",
    "TreeThinker",
    "create_config",
    "ThinkThreadConfig",
]

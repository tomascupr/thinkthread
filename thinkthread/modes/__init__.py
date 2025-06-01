"""
Reasoning modes for ThinkThread

This module provides different reasoning strategies as first-class citizens.
"""

from .base import ReasoningMode, ReasoningResult, ChainedMode, ParallelMode
from .explore import ExploreMode
from .refine import RefineMode
from .debate import DebateMode
from .solve import SolveMode

# Make modes available as functions
explore = ExploreMode
refine = RefineMode
debate = DebateMode
solve = SolveMode

# List of available modes
AVAILABLE_MODES = {
    'explore': ExploreMode,
    'refine': RefineMode,
    'debate': DebateMode,
    'solve': SolveMode,
}


def available() -> list[str]:
    """Get list of available reasoning modes"""
    return list(AVAILABLE_MODES.keys())


def search(keyword: str) -> list[str]:
    """Search for modes by keyword"""
    keyword_lower = keyword.lower()
    matching_modes = []
    
    mode_keywords = {
        'explore': ['creative', 'brainstorm', 'design', 'possibilities'],
        'refine': ['improve', 'enhance', 'polish', 'edit'],
        'debate': ['argue', 'perspective', 'compare', 'versus'],
        'solve': ['solution', 'fix', 'problem', 'resolve'],
    }
    
    for mode, keywords in mode_keywords.items():
        if keyword_lower in mode or any(keyword_lower in kw for kw in keywords):
            matching_modes.append(mode)
    
    return matching_modes


__all__ = [
    'ReasoningMode',
    'ReasoningResult',
    'ChainedMode',
    'ParallelMode',
    'ExploreMode',
    'RefineMode',
    'DebateMode',
    'SolveMode',
    'explore',
    'refine',
    'debate',
    'solve',
    'available',
    'search',
]
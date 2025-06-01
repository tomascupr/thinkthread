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

# Auto-configuration on import
def _auto_configure():
    """Automatically configure ThinkThread on import"""
    import os
    
    # Check for API keys in environment
    api_keys_found = []
    if os.environ.get('OPENAI_API_KEY'):
        api_keys_found.append('OpenAI')
    if os.environ.get('ANTHROPIC_API_KEY'):
        api_keys_found.append('Anthropic')
    if os.environ.get('HF_API_TOKEN'):
        api_keys_found.append('HuggingFace')
        
    if api_keys_found:
        print(f"ThinkThread: Auto-detected API keys for: {', '.join(api_keys_found)}")
    else:
        print("ThinkThread: No API keys detected. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or HF_API_TOKEN")

# Run auto-configuration
_auto_configure()
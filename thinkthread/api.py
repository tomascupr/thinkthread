"""Simple API wrapper for the proven ThinkThread SDK.

This module provides a beautiful, minimal API that wraps the battle-tested
SDK functionality without adding complexity.
"""

from typing import Optional, Dict, Any
import os
from .session import ThinkThreadSession
from .tree_thinker import TreeThinker
from .llm import DummyLLMClient, OpenAIClient, AnthropicClient


def _get_client(test_mode: bool = False):
    """Get the appropriate LLM client based on environment and test mode."""
    if test_mode:
        return DummyLLMClient()
    
    # Check for API keys in environment
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
        )
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name=os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        )
    else:
        # Fallback to dummy for testing
        return DummyLLMClient()


def reason(prompt: str, test_mode: bool = False, **kwargs) -> str:
    """
    Make AI think before responding using Chain-of-Recursive-Thoughts.
    
    Args:
        prompt: The question or prompt to reason about
        test_mode: If True, use dummy LLM client (no API calls)
        **kwargs: Additional arguments passed to ThinkThreadSession
        
    Returns:
        The refined answer after recursive reasoning
        
    Example:
        >>> answer = reason("What are the implications of quantum computing?")
        >>> print(answer)
    """
    client = _get_client(test_mode)
    session = ThinkThreadSession(
        llm_client=client,
        alternatives=kwargs.get('alternatives', 3),
        rounds=kwargs.get('rounds', 2),
        **{k: v for k, v in kwargs.items() if k not in ['alternatives', 'rounds']}
    )
    return session.run(prompt)


def explore(prompt: str, test_mode: bool = False, **kwargs) -> str:
    """
    Explore multiple paths of thinking using Tree-of-Thoughts.
    
    Args:
        prompt: The question or prompt to explore
        test_mode: If True, use dummy LLM client (no API calls)
        **kwargs: Additional arguments passed to TreeThinker
        
    Returns:
        The best solution after tree-based exploration
        
    Example:
        >>> ideas = explore("Creative solutions for climate change")
        >>> print(ideas)
    """
    client = _get_client(test_mode)
    thinker = TreeThinker(
        llm_client=client,
        max_tree_depth=kwargs.get('max_tree_depth', 3),
        branching_factor=kwargs.get('branching_factor', 3),
        **{k: v for k, v in kwargs.items() if k not in ['max_tree_depth', 'branching_factor']}
    )
    return thinker.solve(prompt, beam_width=kwargs.get('beam_width', 3))


# Convenience functions with better prompts
def solve(problem: str, test_mode: bool = False, **kwargs) -> str:
    """
    Get step-by-step solutions to specific problems.
    
    Args:
        problem: The problem to solve
        test_mode: If True, use dummy LLM client (no API calls)
        **kwargs: Additional arguments passed to reason()
        
    Returns:
        A detailed solution with actionable steps
        
    Example:
        >>> solution = solve("Our deployment takes 45 minutes")
        >>> print(solution)
    """
    prompt = f"""Problem to solve step-by-step:
{problem}

Provide a detailed solution with:
1. Root cause analysis
2. Step-by-step implementation plan
3. Expected outcomes
4. Potential risks and mitigations"""
    return reason(prompt, test_mode, **kwargs)


def debate(question: str, test_mode: bool = False, **kwargs) -> str:
    """
    Analyze a question from multiple perspectives.
    
    Args:
        question: The question to analyze
        test_mode: If True, use dummy LLM client (no API calls)
        **kwargs: Additional arguments passed to reason()
        
    Returns:
        A balanced analysis with multiple viewpoints
        
    Example:
        >>> analysis = debate("Should we use microservices?")
        >>> print(analysis)
    """
    prompt = f"""Analyze this question from multiple perspectives:
{question}

Consider:
- Arguments in favor
- Arguments against
- Key tradeoffs
- Contextual factors
- Balanced recommendation"""
    return reason(prompt, test_mode, **kwargs)


def refine(text: str, instructions: str = "", test_mode: bool = False, **kwargs) -> str:
    """
    Refine and improve existing text or ideas.
    
    Args:
        text: The text to refine
        instructions: Optional specific refinement instructions
        test_mode: If True, use dummy LLM client (no API calls)
        **kwargs: Additional arguments passed to reason()
        
    Returns:
        The refined and improved version
        
    Example:
        >>> better = refine("We need to fix the bug", "Make it more professional")
        >>> print(better)
    """
    if instructions:
        prompt = f"""Refine the following text according to these instructions:
Instructions: {instructions}

Text to refine:
{text}"""
    else:
        prompt = f"""Improve and refine the following:
{text}

Make it clearer, more concise, and more effective."""
    return reason(prompt, test_mode, **kwargs)
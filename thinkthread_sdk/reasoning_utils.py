"""Reasoning utilities for ThinkThread SDK.

This module contains shared utilities for different reasoning approaches,
including functions for generating alternatives, evaluating answers, and
calculating similarity between answers.
"""

import difflib
from typing import List, Dict, Any, Optional, Union, Callable

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig


def generate_alternatives(
    question: str,
    current_answer: str,
    llm_client: LLMClient,
    template_manager: TemplateManager,
    count: int = 3,
    temperature: float = 0.9,
) -> List[str]:
    """Generate alternative answers to a question.

    Args:
        question: The original question
        current_answer: The current best answer
        llm_client: LLM client to use for generation
        template_manager: Template manager for prompt templates
        count: Number of alternatives to generate
        temperature: Temperature for generation

    Returns:
        List of alternative answers
    """
    alternatives = []

    for i in range(count):
        prompt = template_manager.render_template(
            "alternative_prompt.j2",
            {"question": question, "current_answer": current_answer},
        )

        alternative = llm_client.generate(prompt, temperature=temperature)
        alternatives.append(alternative)

    return alternatives


async def generate_alternatives_async(
    question: str,
    current_answer: str,
    llm_client: LLMClient,
    template_manager: TemplateManager,
    count: int = 3,
    temperature: float = 0.9,
    parallel: bool = False,
) -> List[str]:
    """Generate alternative answers to a question asynchronously.

    Args:
        question: The original question
        current_answer: The current best answer
        llm_client: LLM client to use for generation
        template_manager: Template manager for prompt templates
        count: Number of alternatives to generate
        temperature: Temperature for generation
        parallel: Whether to generate alternatives in parallel

    Returns:
        List of alternative answers
    """
    if not parallel:
        alternatives = []
        for i in range(count):
            prompt = template_manager.render_template(
                "alternative_prompt.j2",
                {"question": question, "current_answer": current_answer},
            )

            alternative = await llm_client.acomplete(prompt, temperature=temperature)
            alternatives.append(alternative)

        return alternatives

    import asyncio

    async def generate_alternative(i):
        prompt = template_manager.render_template(
            "alternative_prompt.j2",
            {"question": question, "current_answer": current_answer},
        )
        return await llm_client.acomplete(prompt, temperature=temperature)

    tasks = [generate_alternative(i) for i in range(count)]
    alternatives = await asyncio.gather(*tasks)
    return alternatives


def calculate_similarity(str1: str, str2: str, fast: bool = False) -> float:
    """Calculate the similarity between two strings.

    Args:
        str1: First string
        str2: Second string
        fast: Whether to use a faster algorithm for large texts

    Returns:
        A similarity score between 0.0 and 1.0
    """
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    if fast:
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard = intersection / union

        len1, len2 = len(str1), len(str2)
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0

        similarity = (0.8 * jaccard) + (0.2 * length_ratio)

        return similarity

    return difflib.SequenceMatcher(None, str1, str2).ratio()

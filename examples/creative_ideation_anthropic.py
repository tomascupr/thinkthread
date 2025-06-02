"""Example of Tree-of-Thoughts creative ideation using Anthropic with tiered fallbacks.

This example demonstrates how to use the TreeThinker with Anthropic's Claude models
for creative problem-solving, with tiered fallbacks to ensure completion within
a specified time limit.
"""

import os
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkthread.tree_thinker import TreeThinker
from thinkthread.llm import AnthropicClient, DummyLLMClient
from thinkthread.config import create_config
from thinkthread.evaluation import ModelEvaluator


async def solve_creative_problem(problem, timeout=150):
    """Solve a creative problem using Tree-of-Thoughts with tiered fallbacks.

    This function demonstrates a pattern for creative ideation with fallbacks:
    1. Start with Claude-3-Haiku for high-quality results
    2. Fall back to Claude-3-Sonnet if approaching the time limit
    3. Fall back to DummyLLMClient as a last resort

    Args:
        problem: The creative problem to solve
        timeout: Maximum execution time in seconds

    Returns:
        The best solution found
    """
    start_time = time.time()

    config = create_config()
    config.parallel_alternatives = True
    config.use_caching = True

    evaluator = ModelEvaluator()

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        client = AnthropicClient(api_key=api_key, model="claude-3-haiku-20240307")

        tree_thinker = TreeThinker(
            llm_client=client,
            max_tree_depth=2,  # Limited for performance
            branching_factor=2,
            config=config,
            evaluator=evaluator,
        )

        if time.time() - start_time > timeout / 3:
            print("Approaching time limit, switching to Claude-3-Sonnet")
            client = AnthropicClient(
                api_key=api_key, model="claude-3-sonnet-20240229"  # Faster model
            )
            tree_thinker = TreeThinker(
                llm_client=client,
                max_tree_depth=1,
                branching_factor=2,
                config=config,
                evaluator=evaluator,
            )

        result = await tree_thinker.solve_async(
            problem=problem, beam_width=2, max_iterations=1
        )

        if time.time() - start_time > timeout * 0.8:
            print(f"Approaching time limit of {timeout}s, falling back to dummy model")
            dummy_client = DummyLLMClient()
            dummy_thinker = TreeThinker(
                llm_client=dummy_client,
                max_tree_depth=1,
                branching_factor=1,
                config=config,
                evaluator=evaluator,
            )
            dummy_result = await dummy_thinker.solve_async(
                problem=problem, beam_width=1, max_iterations=1
            )

            best_node_id = None
            best_score = -1.0

            for node_id, node in dummy_thinker.threads.items():
                if node.score > best_score:
                    best_score = node.score
                    best_node_id = node_id

            if best_node_id:
                best_node = dummy_thinker.threads[best_node_id]
                best_answer = best_node.state.get("current_answer", "Fallback answer")
                return best_answer

            return "No answer found with fallback model"

        best_node_id = None
        best_score = -1.0

        for node_id, node in tree_thinker.threads.items():
            if node.score > best_score:
                best_score = node.score
                best_node_id = node_id

        if best_node_id:
            best_node = tree_thinker.threads[best_node_id]
            best_answer = best_node.state.get("current_answer", "No answer found")
            return best_answer

        return "No answer found"

    except Exception as e:
        print(f"Error: {e}, falling back to dummy solution")
        dummy_client = DummyLLMClient()
        dummy_thinker = TreeThinker(
            llm_client=dummy_client,
            max_tree_depth=1,
            branching_factor=1,
            config=config,
            evaluator=evaluator,
        )

        try:
            dummy_result = await dummy_thinker.solve_async(
                problem=problem, beam_width=1, max_iterations=1
            )

            best_node_id = None
            best_score = -1.0

            for node_id, node in dummy_thinker.threads.items():
                if node.score > best_score:
                    best_score = node.score
                    best_node_id = node_id

            if best_node_id:
                best_node = dummy_thinker.threads[best_node_id]
                best_answer = best_node.state.get("current_answer", "Fallback answer")
                return best_answer
        except:
            pass

        return "Fallback answer due to errors"


if __name__ == "__main__":
    problem = "Generate 5 innovative product ideas combining AI and renewable energy"
    print(f"Problem: {problem}")
    solution = asyncio.run(solve_creative_problem(problem))
    print(f"Solution: {solution}")

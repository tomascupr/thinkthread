"""Test script for the unified CLI interface.

This script tests both CoRT and ToT reasoning approaches using the unified CLI interface.
"""

import os
import asyncio
from thinkthread_sdk.llm import OpenAIClient
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.config import create_config

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Please set the OPENAI_API_KEY environment variable")
    exit(1)

test_question = "What are the benefits of unified code architecture?"
config = create_config()

client = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")


async def test_cort():
    """Test Chain-of-Recursive-Thoughts reasoning."""
    print("\n=== Testing Chain-of-Recursive-Thoughts ===")
    print(f"Question: {test_question}")

    session = ThinkThreadSession(
        llm_client=client, alternatives=2, rounds=1, config=config
    )

    print("Thinking...")
    answer = await session.run_async(test_question)

    print("\nAnswer:")
    print(answer)
    print("\nCoRT test completed successfully!")


async def test_tot():
    """Test Tree-of-Thoughts reasoning."""
    print("\n=== Testing Tree-of-Thoughts ===")
    print(f"Question: {test_question}")

    tree_thinker = TreeThinker(
        llm_client=client, max_tree_depth=2, branching_factor=2, config=config
    )

    print("Thinking...")
    result = await tree_thinker.solve_async(
        test_question, beam_width=2, max_iterations=1
    )

    best_node_id = None
    best_score = -1.0

    for node_id, node in tree_thinker.threads.items():
        if node.score > best_score:
            best_score = node.score
            best_node_id = node_id

    if best_node_id:
        best_node = tree_thinker.threads[best_node_id]
        best_answer = best_node.state.get("current_answer", "No answer found")

        print("\nBest solution found:")
        print(f"Node: {best_node_id}")
        print(f"Score: {best_score:.2f}")
        print("\nAnswer:")
        print(best_answer)
    else:
        print("\nNo solution found")

    print("\nToT test completed successfully!")


async def main():
    """Run all tests."""
    await test_cort()
    await test_tot()
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

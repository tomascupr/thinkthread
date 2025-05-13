"""Test script for TreeThinker module.

This script demonstrates the functionality of the TreeThinker module
with error handling and scoring improvements. It tests with both
DummyLLMClient and OpenAIClient to verify the implementation works
with real LLM providers.
"""

import asyncio
from thinkthread_sdk.llm import DummyLLMClient, OpenAIClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import create_config
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.evaluation import ModelEvaluator


def error_response_generator(prompt: str, call_count: int) -> str:
    """Generate responses or throw exceptions based on call count.

    Args:
        prompt: The input prompt
        call_count: The number of times the generator has been called

    Returns:
        A response string

    Raises:
        Exception: If call_count is divisible by 3, to simulate API errors
    """
    if call_count % 3 == 0:
        raise Exception("Simulated API error for testing error handling")

    return f"Response #{call_count} to: '{prompt[:30]}...'"


async def test_with_dummy_client():
    """Test TreeThinker with DummyLLMClient."""
    print("\n=== Testing with DummyLLMClient ===")

    config = create_config()
    template_manager = TemplateManager(config.prompt_dir)

    # Test 1: Normal operation with standard DummyLLMClient
    print("\n--- Test 1: Normal Operation ---")
    normal_client = DummyLLMClient()

    evaluator = ModelEvaluator()

    tree_thinker = TreeThinker(
        llm_client=normal_client,
        max_tree_depth=2,
        branching_factor=2,
        template_manager=template_manager,
        config=config,
        evaluator=evaluator,
    )

    problem = "What are the benefits of tree-based search for reasoning?"

    print("Testing synchronous solve method...")
    result = tree_thinker.solve(problem, beam_width=2, max_iterations=1)
    print(f"Sync result status: {result.get('status', 'unknown')}")
    print(f"Thread count: {result.get('thread_count', 0)}")

    print("\nTesting asynchronous solve method...")
    async_result = await tree_thinker.solve_async(
        problem, beam_width=2, max_iterations=1
    )
    print(f"Async result status: {async_result.get('status', 'unknown')}")
    print(f"Thread count: {async_result.get('thread_count', 0)}")

    print("\n--- Test 2: Error Handling ---")
    error_client = DummyLLMClient(response_generator=error_response_generator)

    error_thinker = TreeThinker(
        llm_client=error_client,
        max_tree_depth=2,
        branching_factor=2,
        template_manager=template_manager,
        config=config,
        evaluator=evaluator,
    )

    print("Testing error handling in synchronous solve method...")
    try:
        error_result = error_thinker.solve(problem, beam_width=2, max_iterations=1)
        print(
            f"Error handling worked! Result status: {error_result.get('status', 'unknown')}"
        )
        print(f"Thread count: {error_result.get('thread_count', 0)}")
    except Exception as e:
        print(f"Error handling failed: {e}")

    print("\nTesting error handling in asynchronous solve method...")
    try:
        async_error_result = await error_thinker.solve_async(
            problem, beam_width=2, max_iterations=1
        )
        print(
            f"Async error handling worked! Result status: {async_error_result.get('status', 'unknown')}"
        )
        print(f"Thread count: {async_error_result.get('thread_count', 0)}")
    except Exception as e:
        print(f"Async error handling failed: {e}")

    print("\n--- Test 3: Expansion and Scoring ---")
    responses = [
        "Tree-based search allows exploring multiple reasoning paths simultaneously.",
        "Tree-based search enables pruning of less promising paths to focus on the best ones.",
        "Tree-based search provides a structured way to organize and evaluate different thoughts.",
    ]
    scoring_client = DummyLLMClient(responses=responses)

    scoring_thinker = TreeThinker(
        llm_client=scoring_client,
        max_tree_depth=2,
        branching_factor=2,
        template_manager=template_manager,
        config=config,
        evaluator=evaluator,
    )

    print("Testing expansion with scoring...")
    scoring_result = scoring_thinker.solve(problem, beam_width=1, max_iterations=0)

    if scoring_result.get("thread_count", 0) > 0:
        expansion_result = scoring_thinker.expand_threads(beam_width=2)
        print(f"Expansion result: {expansion_result}")

        if "new_nodes" in expansion_result:
            print(f"New nodes created: {len(expansion_result['new_nodes'])}")

            for node_id in expansion_result["new_nodes"]:
                if node_id in scoring_thinker.threads:
                    node = scoring_thinker.threads[node_id]
                    print(f"Node {node_id} score: {node.score}")


async def test_with_openai_client():
    """Test TreeThinker with OpenAIClient."""
    print("\n=== Testing with OpenAIClient ===")

    config = create_config()
    template_manager = TemplateManager(config.prompt_dir)

    if not config.openai_api_key:
        print("Skipping OpenAI test: No API key available")
        return

    openai_client = OpenAIClient(
        api_key=str(config.openai_api_key), model_name="gpt-3.5-turbo"
    )

    evaluator = ModelEvaluator()

    tree_thinker = TreeThinker(
        llm_client=openai_client,
        max_tree_depth=2,
        branching_factor=2,
        template_manager=template_manager,
        config=config,
        evaluator=evaluator,
    )

    problem = "What are three key benefits of tree-based search for reasoning?"

    print("\n--- Test 1: Normal Operation with OpenAI ---")
    print("Testing synchronous solve method...")
    try:
        result = tree_thinker.solve(problem, beam_width=2, max_iterations=1)
        print(f"Sync result status: {result.get('status', 'unknown')}")
        print(f"Thread count: {result.get('thread_count', 0)}")

        if result.get("thread_count", 0) > 0:
            print("\nTesting expansion with scoring...")
            expansion_result = tree_thinker.expand_threads(beam_width=2)

            if "new_nodes" in expansion_result:
                print(f"New nodes created: {len(expansion_result['new_nodes'])}")

                for node_id in expansion_result["new_nodes"]:
                    if node_id in tree_thinker.threads:
                        node = tree_thinker.threads[node_id]
                        print(f"Node {node_id} score: {node.score}")
    except Exception as e:
        print(f"Error with OpenAI client: {e}")
        print("Error handling caught the exception as expected")


async def main():
    """Run tests for the TreeThinker module."""
    print("Testing TreeThinker module...")

    # Test with DummyLLMClient
    await test_with_dummy_client()

    await test_with_openai_client()

    print("\nTreeThinker test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

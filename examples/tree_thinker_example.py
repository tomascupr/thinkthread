"""Example usage of TreeThinker module.

This example demonstrates how to use the TreeThinker module to solve problems
using tree-based search for reasoning. It shows both synchronous and asynchronous
usage with different LLM providers.
"""

import os
import asyncio
from typing import Dict, Any

from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import OpenAIClient, AnthropicClient, DummyLLMClient
from thinkthread_sdk.config import create_config
from thinkthread_sdk.evaluation import ModelEvaluator


def example_sync_usage():
    """Demonstrate synchronous usage of TreeThinker."""
    print("\n=== Synchronous TreeThinker Example ===\n")
    
    config = create_config()
    
    llm_client = OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model_name="gpt-3.5-turbo"
    )
    
    evaluator = ModelEvaluator()
    
    tree_thinker = TreeThinker(
        llm_client=llm_client,
        max_tree_depth=3,         # Maximum depth of the thinking tree
        branching_factor=3,       # Number of branches per node
        config=config,
        evaluator=evaluator,
    )
    
    problem = "What are three key benefits of tree-based search for reasoning?"
    
    print(f"Problem: {problem}")
    print("Thinking...", flush=True)
    
    result = tree_thinker.solve(
        problem=problem,
        beam_width=2,             # Number of parallel thought threads
        max_iterations=2          # Number of expansion iterations
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
    
    print(f"\nTotal nodes explored: {len(tree_thinker.threads)}")
    print(f"Final layer nodes: {len(tree_thinker.current_layer)}")


async def example_async_usage():
    """Demonstrate asynchronous usage of TreeThinker."""
    print("\n=== Asynchronous TreeThinker Example ===\n")
    
    config = create_config()
    
    llm_client = AnthropicClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model="claude-2"
    )
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No Anthropic API key found, using DummyLLMClient instead")
        llm_client = DummyLLMClient()
    
    evaluator = ModelEvaluator()
    
    tree_thinker = TreeThinker(
        llm_client=llm_client,
        max_tree_depth=3,         # Maximum depth of the thinking tree
        branching_factor=3,       # Number of branches per node
        config=config,
        evaluator=evaluator,
    )
    
    problem = "How can we address climate change through technology innovation?"
    
    print(f"Problem: {problem}")
    print("Thinking asynchronously...", flush=True)
    
    result = await tree_thinker.solve_async(
        problem=problem,
        beam_width=3,             # Number of parallel thought threads
        max_iterations=2          # Number of expansion iterations
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
    
    print(f"\nTotal nodes explored: {len(tree_thinker.threads)}")
    print(f"Final layer nodes: {len(tree_thinker.current_layer)}")
    
    print("\nReasoning process details:")
    print(f"Root threads: {result.get('root_threads', [])}")
    print(f"Expanded threads: {len(result.get('expanded_threads', []))}")
    print(f"Pruned count: {result.get('pruned_count', 0)}")


def example_custom_provider():
    """Demonstrate using TreeThinker with a custom LLM provider."""
    print("\n=== Custom Provider TreeThinker Example ===\n")
    
    config = create_config()
    
    llm_client = DummyLLMClient(
        responses=[
            "Tree-based search allows exploring multiple reasoning paths simultaneously.",
            "Tree-based search enables pruning of less promising paths to focus on the best ones.",
            "Tree-based search provides a structured way to organize and evaluate different thoughts."
        ]
    )
    
    tree_thinker = TreeThinker(
        llm_client=llm_client,
        max_tree_depth=2,
        branching_factor=2,
        config=config,
    )
    
    problem = "What are the benefits of tree-based search for reasoning?"
    
    print(f"Problem: {problem}")
    print("Thinking with custom provider...", flush=True)
    
    result = tree_thinker.solve(
        problem=problem,
        beam_width=1,
        max_iterations=1
    )
    
    best_node_id = max(
        tree_thinker.threads.keys(),
        key=lambda node_id: tree_thinker.threads[node_id].score
    )
    best_node = tree_thinker.threads[best_node_id]
    
    print("\nBest solution found:")
    print(f"Node: {best_node_id}")
    print(f"Score: {best_node.score:.2f}")
    print("\nAnswer:")
    print(best_node.state.get("current_answer", "No answer found"))


async def main():
    """Run all examples."""
    example_sync_usage()
    
    example_custom_provider()
    
    await example_async_usage()


if __name__ == "__main__":
    asyncio.run(main())

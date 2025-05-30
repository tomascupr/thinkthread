"""Command-line interface for the TreeThinker module.

This module provides a CLI for running Tree-of-Thoughts reasoning using different LLM providers
and visualizing the tree of thoughts.
"""

import typer
import asyncio
import logging
from typing import Optional, Dict
from rich.console import Console
from rich.tree import Tree as RichTree
from rich.panel import Panel

from thinkthread_sdk.config import create_config
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import (
    OpenAIClient,
    DummyLLMClient,
    AnthropicClient,
    HuggingFaceClient,
    LLMClient,
)
from thinkthread_sdk.evaluation import ModelEvaluator

console = Console()


def tot(
    problem: str = typer.Argument(..., help="The problem to solve"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    beam_width: int = typer.Option(3, help="Number of parallel thought threads"),
    max_depth: int = typer.Option(3, help="Maximum depth of the thinking tree"),
    branching_factor: int = typer.Option(3, help="Number of branches per node"),
    iterations: int = typer.Option(2, help="Number of expansion iterations"),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
    visualize: bool = typer.Option(True, help="Visualize the tree of thoughts"),
) -> None:
    """Solve a problem using Tree-of-Thoughts reasoning.

    This command provides a CLI interface to the Tree-of-Thoughts reasoning process,
    which explores multiple reasoning paths in parallel and selects the most promising
    ones using beam search. It supports multiple LLM providers and offers visualization
    of the thinking tree.

    The process involves:
    1. Initializing multiple root thought threads
    2. For each iteration:
       a. Expanding each active thought thread
       b. Scoring the new branches
       c. Selecting the top branches based on beam width
    3. Returning the best solution found

    When visualization is enabled, the command will display the tree of thoughts
    showing the reasoning paths explored and their scores.

    Examples:
        $ thinkthread tot "How can we solve the climate crisis?"

        $ thinkthread tot "Design a system for autonomous vehicles" --provider anthropic

        $ thinkthread tot "What are the ethical implications of AI?" --beam-width 5

        $ thinkthread tot "Solve the traveling salesman problem" --max-depth 4 --iterations 3

        $ thinkthread tot "Explain quantum computing" --verbose
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.debug("Verbose logging enabled")
        logging.debug(f"Problem: {problem}")
        logging.debug(f"Provider: {provider}")
        logging.debug(f"Beam width: {beam_width}")
        logging.debug(f"Max depth: {max_depth}")
        logging.debug(f"Iterations: {iterations}")

    config = create_config()
    client: LLMClient

    if provider == "openai":
        client = OpenAIClient(
            api_key=str(config.openai_api_key or ""),
            model_name=model or config.openai_model,
        )
    elif provider == "anthropic":
        client = AnthropicClient(
            api_key=str(config.anthropic_api_key or ""),
            model=model or config.anthropic_model,
        )
    elif provider == "hf":
        client = HuggingFaceClient(
            api_token=str(config.hf_api_token or ""),
            model=model or config.hf_model,
        )
    elif provider == "dummy":
        client = DummyLLMClient(model_name=model or "dummy-model")
    else:
        print(f"Unknown provider: {provider}")
        return

    evaluator = ModelEvaluator()

    tree_thinker = TreeThinker(
        llm_client=client,
        max_tree_depth=max_depth,
        branching_factor=branching_factor,
        config=config,
        evaluator=evaluator,
    )

    if verbose:
        logging.debug("Starting TreeThinker session")

    asyncio.run(
        run_tot(tree_thinker, problem, beam_width, iterations, visualize, verbose)
    )


async def run_tot(
    tree_thinker: TreeThinker,
    problem: str,
    beam_width: int,
    iterations: int,
    visualize: bool,
    verbose: bool = False,
) -> None:
    """Run the TreeThinker session asynchronously with visualization.

    This function handles the execution of the Tree-of-Thoughts reasoning process
    and visualizes the thinking tree when enabled.

    Args:
        tree_thinker: The TreeThinker instance to use for reasoning
        problem: The problem to solve
        beam_width: Number of parallel thought threads
        iterations: Number of expansion iterations
        visualize: Whether to visualize the tree of thoughts
        verbose: Whether to enable verbose logging
    """
    if verbose:
        logging.debug("Starting run_tot")

    print(f"Problem: {problem}")
    print(f"Beam width: {beam_width}")
    print(f"Iterations: {iterations}")
    print("Thinking...", flush=True)

    result = await tree_thinker.solve_async(
        problem, beam_width=beam_width, max_iterations=iterations
    )

    if verbose:
        logging.debug("TreeThinker reasoning completed")
        logging.debug(f"Result: {result}")

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

    if visualize:
        visualize_tree(tree_thinker)


def visualize_tree(tree_thinker: TreeThinker) -> None:
    """Visualize the tree of thoughts.

    This function creates a visual representation of the thinking tree
    showing the reasoning paths explored and their scores.

    Args:
        tree_thinker: The TreeThinker instance containing the tree
    """
    print("\nTree of Thoughts:")

    tree_map: Dict[str, list] = {}
    root_nodes = []

    for node_id, node in tree_thinker.threads.items():
        if node.parent_id:
            if node.parent_id not in tree_map:
                tree_map[node.parent_id] = []
            tree_map[node.parent_id].append(node_id)
        else:
            root_nodes.append(node_id)

    rich_tree = RichTree("Tree of Thoughts")

    def build_tree(parent_rich_node, node_id):
        node = tree_thinker.threads[node_id]
        answer = node.state.get("current_answer", "")
        display_answer = (answer[:100] + "...") if len(answer) > 100 else answer

        node_label = f"[bold]{node_id}[/bold] (Score: {node.score:.2f})"

        node_rich = parent_rich_node.add(
            Panel(
                f"{node_label}\n\n{display_answer}",
                expand=False,
                padding=(1, 2),
            )
        )

        if node_id in tree_map:
            for child_id in tree_map[node_id]:
                build_tree(node_rich, child_id)

    for root_id in root_nodes:
        root_node = tree_thinker.threads[root_id]
        root_label = f"[bold]{root_id}[/bold] (Score: {root_node.score:.2f})"
        root_answer = root_node.state.get("current_answer", "")
        display_answer = (
            (root_answer[:100] + "...") if len(root_answer) > 100 else root_answer
        )

        root_rich = rich_tree.add(
            Panel(
                f"{root_label}\n\n{display_answer}",
                expand=False,
                padding=(1, 2),
            )
        )

        if root_id in tree_map:
            for child_id in tree_map[root_id]:
                build_tree(root_rich, child_id)

    console.print(rich_tree)

    total_nodes = len(tree_thinker.threads)
    leaf_nodes = sum(1 for node_id in tree_thinker.threads if node_id not in tree_map)

    print("\nStatistics:")
    print(f"Total nodes: {total_nodes}")
    print(f"Leaf nodes: {leaf_nodes}")
    print(f"Internal nodes: {total_nodes - leaf_nodes}")


if __name__ == "__main__":
    app()

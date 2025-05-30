"""Unified command-line interface for the ThinkThread SDK.

This module provides a CLI for running both Chain-of-Recursive-Thoughts and
Tree-of-Thoughts reasoning using different LLM providers and viewing the results.
"""

import typer
import asyncio
import logging
from typing import Optional, Dict, Literal, Any
from rich.console import Console
from rich.tree import Tree as RichTree
from rich.panel import Panel

from thinkthread_sdk import __version__
from thinkthread_sdk.config import create_config
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.base_reasoner import BaseReasoner
from thinkthread_sdk.utils import ThinkThreadUtils
from thinkthread_sdk.llm import (
    OpenAIClient,
    DummyLLMClient,
    AnthropicClient,
    HuggingFaceClient,
    LLMClient,
)
from thinkthread_sdk.evaluation import ModelEvaluator

console = Console()


def setup_logging(verbose: bool, **log_data: Any) -> None:
    """Set up logging with consistent format.

    Args:
        verbose: Whether to enable verbose logging
        **log_data: Additional data to log as key-value pairs
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.debug("Verbose logging enabled")
        for key, value in log_data.items():
            logging.debug(f"{key}: {value}")


# Removed app decorators - these functions are imported by cli.py


def create_llm_client(
    provider: str, model: Optional[str] = None, config=None
) -> LLMClient:
    """Create an LLM client based on the provider.

    Args:
        provider: LLM provider to use (openai, anthropic, hf, dummy)
        model: Model name to use (provider-specific)
        config: Configuration object

    Returns:
        An LLM client instance
    """
    if config is None:
        config = create_config()

    if provider == "openai":
        return OpenAIClient(
            api_key=str(config.openai_api_key or ""),
            model_name=model or config.openai_model,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=str(config.anthropic_api_key or ""),
            model=model or config.anthropic_model,
        )
    elif provider == "hf":
        return HuggingFaceClient(
            api_token=str(config.hf_api_token or ""),
            model=model or config.hf_model,
        )
    elif provider == "dummy":
        return DummyLLMClient(model_name=model or "dummy-model")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def think(
    question: str = typer.Argument(..., help="The question to answer"),
    approach: str = typer.Option("cort", help="Reasoning approach to use (cort, tot)"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    alternatives: int = typer.Option(3, help="Number of alternative answers per round"),
    rounds: int = typer.Option(2, help="Number of refinement rounds"),
    beam_width: int = typer.Option(
        3, help="Number of parallel thought threads (ToT only)"
    ),
    max_depth: int = typer.Option(
        3, help="Maximum depth of the thinking tree (ToT only)"
    ),
    branching_factor: int = typer.Option(
        3, help="Number of branches per node (ToT only)"
    ),
    stream: bool = typer.Option(True, help="Stream the final answer as it's generated"),
    visualize: bool = typer.Option(
        True, help="Visualize the tree of thoughts (ToT only)"
    ),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
    self_evaluation: bool = typer.Option(False, help="Toggle self-evaluation on/off"),
) -> None:
    """Think about a question using recursive or tree-based reasoning.

    This command provides a unified CLI interface to both Chain-of-Recursive-Thoughts
    and Tree-of-Thoughts reasoning processes. It supports multiple LLM providers and
    offers both synchronous and streaming output modes.

    The CoRT process involves:
    1. Generating an initial answer to the question
    2. For each round:
       a. Generating alternative answers
       b. Evaluating all answers to select the best one
       c. Using the best answer as input for the next round
    3. Returning the final best answer

    The ToT process involves:
    1. Initializing multiple root thought threads
    2. For each iteration:
       a. Expanding each active thought thread
       b. Scoring the new branches
       c. Selecting the top branches based on beam width
    3. Returning the best solution found

    Examples:
        $ thinkthread think "What is the meaning of life?"

        $ thinkthread think "How to solve climate change?" --approach tot

        $ thinkthread think "Explain quantum computing" --provider anthropic

        $ thinkthread think "Pros and cons of renewable energy" --rounds 3 --alternatives 5

        $ thinkthread think "Design a system for autonomous vehicles" --approach tot --beam-width 5
    """
    setup_logging(verbose, question=question, approach=approach, provider=provider)

    config = create_config()

    if self_evaluation:
        config.use_self_evaluation = True
        if verbose:
            logging.debug("Self-evaluation enabled")

    try:
        client = create_llm_client(provider, model, config)
    except ValueError as e:
        print(str(e))
        return

    if approach.lower() == "cort":
        if verbose:
            logging.debug(
                f"Using CoRT with rounds={rounds}, alternatives={alternatives}"
            )

        reasoner = ThinkThreadSession(
            llm_client=client, alternatives=alternatives, rounds=rounds, config=config
        )

        asyncio.run(run_cort(reasoner, question, stream, verbose))

    elif approach.lower() == "tot":
        if verbose:
            logging.debug(
                f"Using ToT with beam_width={beam_width}, max_depth={max_depth}"
            )

        evaluator = ModelEvaluator()

        reasoner = TreeThinker(
            llm_client=client,
            max_tree_depth=max_depth,
            branching_factor=branching_factor,
            config=config,
            evaluator=evaluator,
        )

        asyncio.run(run_tot(reasoner, question, beam_width, rounds, visualize, verbose))

    else:
        print(f"Unknown reasoning approach: {approach}")
        print("Available approaches: cort, tot")


async def run_cort(
    reasoner: ThinkThreadSession, question: str, stream: bool, verbose: bool = False
) -> None:
    """Run the CoRT reasoning process asynchronously with optional streaming.

    Args:
        reasoner: The ThinkThreadSession instance to use for reasoning
        question: The question to answer
        stream: Whether to stream the final answer as it's generated
        verbose: Whether to enable verbose logging
    """
    if verbose:
        logging.debug("Starting run_cort")

    if stream:
        print(f"Question: {question}")
        print("Thinking...", end="", flush=True)

        answer = await reasoner.run_async(question)

        print("\r" + " " * 20 + "\r", end="", flush=True)
        print("Answer:")

        prompt = reasoner.template_manager.render_template(
            "final_answer.j2", {"question": question, "answer": answer}
        )

        async for token in reasoner.llm_client.astream(prompt):
            print(token, end="", flush=True)
        print()
    else:
        print(f"Question: {question}")
        print("Thinking...", flush=True)

        answer = await reasoner.run_async(question)

        print("Answer:")
        print(answer)


async def run_tot(
    reasoner: TreeThinker,
    problem: str,
    beam_width: int,
    iterations: int,
    visualize: bool,
    verbose: bool = False,
) -> None:
    """Run the ToT reasoning process asynchronously with visualization.

    Args:
        reasoner: The TreeThinker instance to use for reasoning
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

    result = await reasoner.solve_async(
        problem, beam_width=beam_width, max_iterations=iterations
    )

    if verbose:
        logging.debug("TreeThinker reasoning completed")
        logging.debug(f"Result: {result}")

    best_node_id = None
    best_score = -1.0

    for node_id, node in reasoner.threads.items():
        if node.score > best_score:
            best_score = node.score
            best_node_id = node_id

    if best_node_id:
        best_node = reasoner.threads[best_node_id]
        best_answer = best_node.state.get("current_answer", "No answer found")

        print("\nBest solution found:")
        print(f"Node: {best_node_id}")
        print(f"Score: {best_score:.2f}")
        print("\nAnswer:")
        print(best_answer)
    else:
        print("\nNo solution found")

    if visualize:
        visualize_tree(reasoner)


def visualize_tree(tree_thinker: TreeThinker) -> None:
    """Visualize the tree of thoughts.

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


def refine(
    question: str = typer.Argument(..., help="The question to answer"),
    initial_answer: str = typer.Argument(..., help="The initial answer to refine"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    rounds: int = typer.Option(2, help="Number of refinement rounds"),
    alternatives: int = typer.Option(3, help="Number of alternative answers per round"),
    stream: bool = typer.Option(True, help="Stream the final answer as it's generated"),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    """Refine an answer through multiple rounds of critique and revision.

    This command implements a generate → critique → revise loop that iteratively
    improves an initial answer through self-refinement. It leverages the
    Chain-of-Recursive-Thoughts (CoRT) technique.

    Examples:
        $ thinkthread refine "What is quantum computing?" "Quantum computing uses qubits."

        $ thinkthread refine "Explain climate change" "Climate change refers to significant changes in global temperature, precipitation, wind patterns, and other measures of climate that occur over several decades or longer." --rounds 3

        $ thinkthread refine "Compare democracy and autocracy" "Democracy is rule by the people while autocracy is rule by one person." --provider anthropic
    """
    setup_logging(
        verbose, question=question, initial_answer=initial_answer, provider=provider
    )

    config = create_config()

    try:
        client = create_llm_client(provider, model, config)
    except ValueError as e:
        print(str(e))
        return

    utils = ThinkThreadUtils(llm_client=client, config=config)

    asyncio.run(
        run_refine(
            utils, question, initial_answer, rounds, alternatives, stream, verbose
        )
    )


def brainstorm(
    question: str = typer.Argument(..., help="The question to answer"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    candidates: int = typer.Option(5, help="Number of candidate answers to generate"),
    max_depth: int = typer.Option(1, help="Maximum depth of the thinking tree"),
    stream: bool = typer.Option(True, help="Stream the final answer as it's generated"),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    """Generate multiple answers and select the best one.

    This command implements a shallow Tree-of-Thoughts approach that generates
    multiple answers in parallel and evaluates them to select the best one.

    Examples:
        $ thinkthread brainstorm "Ideas for reducing plastic waste"

        $ thinkthread brainstorm "Creative uses for old smartphones" --candidates 10

        $ thinkthread brainstorm "Solutions to urban traffic congestion" --provider anthropic
    """
    setup_logging(verbose, question=question, provider=provider, candidates=candidates)

    config = create_config()

    try:
        client = create_llm_client(provider, model, config)
    except ValueError as e:
        print(str(e))
        return

    utils = ThinkThreadUtils(llm_client=client, config=config)

    asyncio.run(run_brainstorm(utils, question, candidates, max_depth, stream, verbose))


async def run_refine(
    utils: ThinkThreadUtils,
    question: str,
    initial_answer: str,
    rounds: int,
    alternatives: int,
    stream: bool,
    verbose: bool = False,
) -> None:
    """Run the refinement process asynchronously with optional streaming.

    Args:
        utils: ThinkThreadUtils instance
        question: The question to answer
        initial_answer: The initial answer to refine
        rounds: Number of refinement rounds
        alternatives: Number of alternatives per round
        stream: Whether to stream the final answer
        verbose: Enable verbose logging
    """
    if verbose:
        logging.debug("Starting run_refine")

    if stream:
        print(f"Question: {question}")
        print(f"Initial answer: {initial_answer}")
        print("Refining...", end="", flush=True)

        refined_answer = await utils.self_refine_async(
            question, initial_answer, rounds=rounds, alternatives=alternatives
        )

        print("\r" + " " * 20 + "\r", end="", flush=True)
        print("Refined answer:")

        prompt = utils.template_manager.render_template(
            "final_answer.j2", {"question": question, "answer": refined_answer}
        )

        async for token in utils.llm_client.astream(prompt):
            print(token, end="", flush=True)
        print()
    else:
        print(f"Question: {question}")
        print(f"Initial answer: {initial_answer}")
        print("Refining...", flush=True)

        refined_answer = await utils.self_refine_async(
            question, initial_answer, rounds=rounds, alternatives=alternatives
        )

        print("Refined answer:")
        print(refined_answer)


async def run_brainstorm(
    utils: ThinkThreadUtils,
    question: str,
    candidates: int,
    max_depth: int,
    stream: bool,
    verbose: bool = False,
) -> None:
    """Run the brainstorming process asynchronously with optional streaming.

    Args:
        utils: ThinkThreadUtils instance
        question: The question to answer
        candidates: Number of candidate answers to generate
        max_depth: Maximum depth of the thinking tree
        stream: Whether to stream the final answer
        verbose: Enable verbose logging
    """
    if verbose:
        logging.debug("Starting run_brainstorm")

    print(f"Question: {question}")
    print(f"Generating {candidates} answers...", flush=True)

    best_answer = await utils.n_best_brainstorm_async(
        question, n=candidates, max_depth=max_depth, return_metadata=True
    )

    if isinstance(best_answer, dict):
        print(f"\nBest answer (score: {best_answer['best_score']:.2f}):")

        if stream:
            prompt = utils.template_manager.render_template(
                "final_answer.j2",
                {"question": question, "answer": best_answer["best_answer"]},
            )

            async for token in utils.llm_client.astream(prompt):
                print(token, end="", flush=True)
            print()
        else:
            print(best_answer["best_answer"])
    else:
        print("\nBest answer:")
        print(best_answer)


if __name__ == "__main__":
    app()

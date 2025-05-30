"""Command-line interface for the ThinkThread SDK.

This module provides a CLI for running ThinkThread reasoning using different LLM providers
and viewing the results.
"""

import typer
import asyncio
import logging
from typing import Optional
from thinkthread_sdk import __version__
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.config import create_config
from thinkthread_sdk.llm import (
    OpenAIClient,
    DummyLLMClient,
    AnthropicClient,
    HuggingFaceClient,
    LLMClient,
)
from thinkthread_sdk.cli_tot import tot
from thinkthread_sdk.unified_cli import think, refine, brainstorm

app = typer.Typer()

# Register imported commands
app.command(name="tot", help="Run Tree-of-Thoughts reasoning")(tot)
app.command(name="think", help="Think about a question using recursive or tree-based reasoning")(think)
app.command(name="refine", help="Refine an existing answer using self-refinement")(refine)
app.command(name="brainstorm", help="Generate multiple answers and select the best one")(brainstorm)


@app.callback()
def callback() -> None:
    """ThinkThread SDK - Command Line Interface."""


@app.command()
def version() -> None:
    """Show the current version of ThinkThread SDK."""
    print(f"ThinkThread SDK version: {__version__}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    alternatives: int = typer.Option(3, help="Number of alternative answers per round"),
    rounds: int = typer.Option(2, help="Number of refinement rounds"),
    stream: bool = typer.Option(True, help="Stream the final answer as it's generated"),
) -> None:
    """Ask a question and get an answer using ThinkThread reasoning.

    This command provides a CLI interface to the Chain-of-Recursive-Thoughts
    reasoning process. It supports multiple LLM providers and offers both
    synchronous and streaming output modes.

    The command uses the async implementation of ThinkThread internally, even when
    called from the synchronous CLI context. This is achieved by using
    asyncio.run() to run the async code in the event loop.

    When streaming is enabled (the default), the final answer will be displayed
    token by token as it's generated, providing a more responsive user experience.
    When streaming is disabled, the command will wait for the complete answer
    before displaying it.

    Examples:
        $ python -m thinkthread_sdk ask "What is the meaning of life?"

        $ python -m thinkthread_sdk ask "What is the meaning of life?" --provider anthropic

        $ python -m thinkthread_sdk ask "What is the meaning of life?" --no-stream

        $ python -m thinkthread_sdk ask "What is the meaning of life?" --alternatives 5 --rounds 3

    """
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

    session = ThinkThreadSession(
        llm_client=client, alternatives=alternatives, rounds=rounds, config=config
    )

    asyncio.run(run_session(session, question, stream))


async def run_session(
    session: ThinkThreadSession, question: str, stream: bool, verbose: bool = False
) -> None:
    """Run the ThinkThread session asynchronously with optional streaming.

    This function handles the execution of the ThinkThread reasoning process in an
    asynchronous manner, with support for streaming the final answer as it's
    generated. It provides two modes of operation:

    1. Streaming mode: Shows the answer being generated token by token
       in real-time, providing immediate feedback to the user and reducing
       perceived latency.

    2. Non-streaming mode: Waits for the complete answer before displaying it,
       which is useful for scripting or when the output needs to be captured
       as a single block.

    The implementation uses the async ThinkThread session and, when streaming is enabled,
    leverages the LLM client's astream method to progressively display tokens
    as they're generated.

    Args:
        session: The ThinkThread session instance to use for reasoning
        question: The question to answer
        stream: Whether to stream the final answer as it's generated
        verbose: Whether to enable verbose logging

    """
    if verbose:
        logging.debug("Starting run_session")

    if stream:
        print(f"Question: {question}")
        print("Thinking...", end="", flush=True)

        if verbose:
            logging.debug("Running with streaming enabled")

        answer = await session.run_async(question)

        if verbose:
            logging.debug("Received answer from ThinkThread session")

        print("\r" + " " * 20 + "\r", end="", flush=True)

        print("Answer:")

        prompt = session.template_manager.render_template(
            "final_answer.j2", {"question": question, "answer": answer}
        )

        if verbose:
            logging.debug("Streaming final answer")

        async for token in session.llm_client.astream(prompt):
            print(token, end="", flush=True)
        print()
    else:
        print(f"Question: {question}")
        print("Thinking...", flush=True)

        if verbose:
            logging.debug("Running without streaming")

        answer = await session.run_async(question)

        if verbose:
            logging.debug("Received answer from ThinkThread session")

        print("Answer:")
        print(answer)


@app.command()
def run(
    question: str = typer.Argument(..., help="The question to answer"),
    provider: str = typer.Option(
        "openai", help="LLM provider to use (openai, anthropic, hf, dummy)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name to use (provider-specific)"
    ),
    alternatives: int = typer.Option(3, help="Number of alternative answers per round"),
    rounds: int = typer.Option(2, help="Number of refinement rounds"),
    stream: bool = typer.Option(
        False, help="Stream the final answer as it's generated"
    ),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
    self_evaluation: bool = typer.Option(False, help="Toggle self-evaluation on/off"),
) -> None:
    """Run recursive reasoning on a question and get a refined answer.

    This command provides a CLI interface to the Chain-of-Recursive-Thoughts
    reasoning process, which performs multiple rounds of self-refinement to
    improve the answer quality. It supports multiple LLM providers and offers
    both synchronous and streaming output modes.

    The process involves:
    1. Generating an initial answer to the question
    2. For each round:
       a. Generating alternative answers
       b. Evaluating all answers to select the best one
       c. Using the best answer as input for the next round
    3. Returning the final best answer

    When streaming is enabled, the final answer will be displayed token by token
    as it's generated. When verbose is enabled, the command will print additional
    debug information about the reasoning process.

    Examples:
        $ thinkthread run "What is the most effective way to combat climate change?"

        $ thinkthread run "Explain quantum computing" --provider anthropic

        $ thinkthread run "Summarize the Iliad" --stream

        $ thinkthread run "Pros and cons of renewable energy" --rounds 3 --alternatives 5

        $ thinkthread run "How do neural networks work?" --verbose

        $ thinkthread run "Explain blockchain technology" --self-evaluation

    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.debug("Verbose logging enabled")
        logging.debug(f"Question: {question}")
        logging.debug(f"Provider: {provider}")
        logging.debug(f"Rounds: {rounds}")

    config = create_config()
    client: LLMClient

    if self_evaluation:
        config.use_self_evaluation = True
        if verbose:
            logging.debug("Self-evaluation enabled")

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

    session = ThinkThreadSession(
        llm_client=client, alternatives=alternatives, rounds=rounds, config=config
    )

    if verbose:
        logging.debug("Starting ThinkThread session")

    asyncio.run(run_session(session, question, stream, verbose))


if __name__ == "__main__":
    app()

import typer
import asyncio
import sys
from typing import Optional
from cort_sdk import __version__
from cort_sdk.cort_session import CoRTSession
from cort_sdk.config import create_config
from cort_sdk.llm import OpenAIClient, DummyLLMClient, AnthropicClient, HuggingFaceClient
from cort_sdk.prompting import TemplateManager

app = typer.Typer()

@app.callback()
def callback():
    """
    CORT SDK - Command Line Interface
    """

@app.command()
def version():
    """
    Show the current version of CORT SDK
    """
    print(f"CORT SDK version: {__version__}")

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer"),
    provider: str = typer.Option("openai", help="LLM provider to use (openai, anthropic, hf, dummy)"),
    model: Optional[str] = typer.Option(None, help="Model name to use (provider-specific)"),
    alternatives: int = typer.Option(3, help="Number of alternative answers per round"),
    rounds: int = typer.Option(2, help="Number of refinement rounds"),
    stream: bool = typer.Option(True, help="Stream the final answer as it's generated"),
):
    """
    Ask a question and get an answer using CoRT reasoning.
    
    This command provides a CLI interface to the Chain-of-Recursive-Thoughts
    reasoning process. It supports multiple LLM providers and offers both
    synchronous and streaming output modes.
    
    The command uses the async implementation of CoRT internally, even when
    called from the synchronous CLI context. This is achieved by using
    asyncio.run() to run the async code in the event loop.
    
    When streaming is enabled (the default), the final answer will be displayed
    token by token as it's generated, providing a more responsive user experience.
    When streaming is disabled, the command will wait for the complete answer
    before displaying it.
    
    Examples:
        $ python -m cort_sdk ask "What is the meaning of life?"
        
        $ python -m cort_sdk ask "What is the meaning of life?" --provider anthropic
        
        $ python -m cort_sdk ask "What is the meaning of life?" --no-stream
        
        $ python -m cort_sdk ask "What is the meaning of life?" --alternatives 5 --rounds 3
    """
    config = create_config()
    
    if provider == "openai":
        client = OpenAIClient(api_key=str(config.openai_api_key or ""), model_name=model or config.openai_model)
    elif provider == "anthropic":
        client = AnthropicClient(api_key=str(config.anthropic_api_key or ""), model_name=model or config.anthropic_model)
    elif provider == "hf":
        client = HuggingFaceClient(api_token=str(config.hf_api_token or ""), model_name=model or config.hf_model)
    elif provider == "dummy":
        client = DummyLLMClient(model_name=model or "dummy-model")
    else:
        print(f"Unknown provider: {provider}")
        return
    
    session = CoRTSession(
        llm_client=client,
        alternatives=alternatives,
        rounds=rounds,
        config=config
    )
    
    asyncio.run(run_session(session, question, stream))

async def run_session(session: CoRTSession, question: str, stream: bool):
    """
    Run the CoRT session asynchronously with optional streaming.
    
    This function handles the execution of the CoRT reasoning process in an
    asynchronous manner, with support for streaming the final answer as it's
    generated. It provides two modes of operation:
    
    1. Streaming mode (default): Shows the answer being generated token by token
       in real-time, providing immediate feedback to the user and reducing
       perceived latency.
       
    2. Non-streaming mode: Waits for the complete answer before displaying it,
       which is useful for scripting or when the output needs to be captured
       as a single block.
    
    The implementation uses the async CoRT session and, when streaming is enabled,
    leverages the LLM client's astream method to progressively display tokens
    as they're generated.
    
    Args:
        session: The CoRTSession instance to use for reasoning
        question: The question to answer
        stream: Whether to stream the final answer as it's generated
    """
    if stream:
        print(f"Question: {question}")
        print("Thinking...", end="", flush=True)
        
        answer = await session.run_async(question)
        
        print("\r" + " " * 20 + "\r", end="", flush=True)
        
        print("Answer:")
        
        prompt = session.template_manager.render_template(
            "final_answer.j2", 
            {"question": question, "answer": answer}
        )
        
        async for token in await session.llm_client.astream(prompt):
            print(token, end="", flush=True)
        print()
    else:
        print(f"Question: {question}")
        print("Thinking...", flush=True)
        
        answer = await session.run_async(question)
        
        print("Answer:")
        print(answer)

if __name__ == "__main__":
    app()

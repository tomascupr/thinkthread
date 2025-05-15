"""Example of recursive refinement using OpenAI with tiered fallbacks.

This example demonstrates how to use the ThinkThreadSession with OpenAI models
for recursive refinement, with tiered fallbacks to ensure completion within
a specified time limit.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient, DummyLLMClient
from thinkthread_sdk.config import create_config


def run_with_fallback(question, timeout=150):
    """Run recursive refinement with tiered fallbacks.

    This function demonstrates a pattern for recursive refinement with fallbacks:
    1. Start with GPT-4 for high-quality results
    2. Fall back to GPT-3.5-Turbo if approaching the time limit
    3. Fall back to DummyLLMClient as a last resort

    Args:
        question: The question to answer
        timeout: Maximum execution time in seconds

    Returns:
        The final refined answer
    """
    start_time = time.time()

    config = create_config()
    config.parallel_alternatives = True
    config.use_caching = True
    config.early_termination = True

    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")

        client_gpt4 = OpenAIClient(api_key=api_key, model_name="gpt-4")

        session = ThinkThreadSession(
            llm_client=client_gpt4, alternatives=3, rounds=2, config=config
        )

        if time.time() - start_time > timeout / 2:
            print("Approaching time limit, switching to GPT-3.5-Turbo")
            client_gpt35 = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")

            session = ThinkThreadSession(
                llm_client=client_gpt35, alternatives=2, rounds=1, config=config
            )

        answer = session.run(question)

        if time.time() - start_time > timeout:
            print(f"Exceeded time limit of {timeout}s, falling back to dummy model")
            dummy_client = DummyLLMClient()
            dummy_session = ThinkThreadSession(
                llm_client=dummy_client, alternatives=1, rounds=1, config=config
            )
            return dummy_session.run(question)

        return answer

    except Exception as e:
        print(f"Error: {e}, falling back to dummy model")
        dummy_client = DummyLLMClient()
        dummy_session = ThinkThreadSession(
            llm_client=dummy_client, alternatives=1, rounds=1, config=config
        )
        return dummy_session.run(question)


if __name__ == "__main__":
    question = "Explain the concept of recursive self-improvement in AI"
    print(f"Question: {question}")
    print(f"Answer: {run_with_fallback(question)}")

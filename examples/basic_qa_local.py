"""Example of basic Q&A using a local model with fallback mechanisms.

This example demonstrates how to use a small local HuggingFace model for
answering questions, with fallback to a DummyLLMClient if the primary model
exceeds the time limit or encounters an error.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkthread_sdk.llm import HuggingFaceClient, DummyLLMClient


def ask_question_with_fallback(question, timeout=60):
    """Ask a question using a small local model with fallback mechanisms.

    This function demonstrates a pattern for making LLM calls resilient:
    1. Try with a small local HuggingFace model first
    2. Fall back to DummyLLMClient if:
       - The primary model exceeds the timeout
       - An exception occurs with the primary model

    Args:
        question: The question to ask
        timeout: Maximum execution time in seconds

    Returns:
        The answer from the model
    """
    start_time = time.time()

    try:
        api_token = os.environ.get("HF_API_TOKEN", "")

        hf_client = HuggingFaceClient(
            api_token=api_token,  # Optional for some local models
            model="distilgpt2",  # Small model that can run on CPU
        )

        answer = hf_client.generate(f"Question: {question}\nAnswer:", max_tokens=100)

        if time.time() - start_time > timeout:
            print(
                f"Primary model exceeded time limit of {timeout}s, falling back to dummy model"
            )
            dummy_client = DummyLLMClient()
            return dummy_client.generate(f"Question: {question}\nAnswer:")

        return answer

    except Exception as e:
        print(f"Error with primary model: {e}, falling back to dummy model")
        dummy_client = DummyLLMClient()
        return dummy_client.generate(f"Question: {question}\nAnswer:")


if __name__ == "__main__":
    question = "What is machine learning?"
    print(f"Question: {question}")
    print(f"Answer: {ask_question_with_fallback(question)}")

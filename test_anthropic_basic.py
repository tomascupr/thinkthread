"""Basic test script for ThinkThread SDK with Anthropic.

This script tests the basic functionality of the ThinkThread SDK with Anthropic.
"""

import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import AnthropicClient

def main():
    """Run a basic test of the ThinkThread SDK with Anthropic."""
    api_key = "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"
    os.environ["ANTHROPIC_API_KEY"] = api_key

    client = AnthropicClient(api_key=api_key)

    session = ThinkThreadSession(llm_client=client)

    question = "What is the capital of France?"
    print(f"Question: {question}")
    print("Running CoRT reasoning with default settings...")
    answer = session.run(question)
    print(f"Answer: {answer}")
    print("\nBasic Anthropic test completed successfully!")

if __name__ == "__main__":
    main()

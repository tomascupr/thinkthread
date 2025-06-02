"""Basic test script for ThinkThread SDK with Anthropic.

This script tests the basic functionality of the ThinkThread SDK with Anthropic.
"""

import os
from thinkthread.cort_session import ThinkThreadSession
from thinkthread.llm import AnthropicClient


def main():
    """Run a basic test of the ThinkThread SDK with Anthropic."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

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

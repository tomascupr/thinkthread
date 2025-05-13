"""Advanced test script for ThinkThread SDK with Anthropic.

This script tests the ThinkThread SDK with Anthropic using multiple rounds and alternatives.
"""

import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import AnthropicClient

def main():
    """Run an advanced test of the ThinkThread SDK with Anthropic."""
    api_key = "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"
    os.environ["ANTHROPIC_API_KEY"] = api_key

    client = AnthropicClient(api_key=api_key)

    session = ThinkThreadSession(llm_client=client, alternatives=3, rounds=2)

    question = "Explain the concept of recursion in programming"
    print(f"Question: {question}")
    print("Running CoRT reasoning with 3 alternatives and 2 rounds...")
    answer = session.run(question)
    print(f"Answer: {answer}")
    print("\nAdvanced Anthropic test completed successfully!")

if __name__ == "__main__":
    main()

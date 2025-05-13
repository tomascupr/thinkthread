"""Basic test script for ThinkThread SDK with OpenAI.

This script tests the basic functionality of the ThinkThread SDK with OpenAI.
"""

import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient


def main():
    """Run a basic test of the ThinkThread SDK with OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")

    session = ThinkThreadSession(llm_client=client)

    question = "What is the capital of France?"
    print(f"Question: {question}")
    print("Running CoRT reasoning with default settings...")
    answer = session.run(question)
    print(f"Answer: {answer}")
    print("\nBasic OpenAI test completed successfully!")


if __name__ == "__main__":
    main()

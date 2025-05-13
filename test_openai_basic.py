"""Basic test script for ThinkThread SDK with OpenAI.

This script tests the basic functionality of the ThinkThread SDK with OpenAI.
"""

import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

def main():
    """Run a basic test of the ThinkThread SDK with OpenAI."""
    api_key = "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"
    os.environ["OPENAI_API_KEY"] = api_key

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

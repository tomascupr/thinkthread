"""Advanced test script for ThinkThread SDK with OpenAI.

This script tests the ThinkThread SDK with OpenAI using multiple rounds and alternatives.
"""

import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

def main():
    """Run an advanced test of the ThinkThread SDK with OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")

    session = ThinkThreadSession(llm_client=client, alternatives=3, rounds=2)

    question = "What are the pros and cons of renewable energy?"
    print(f"Question: {question}")
    print("Running CoRT reasoning with 3 alternatives and 2 rounds...")
    answer = session.run(question)
    print(f"Answer: {answer}")
    print("\nAdvanced OpenAI test completed successfully!")

if __name__ == "__main__":
    main()

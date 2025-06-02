import os
from thinkthread.cort_session import ThinkThreadSession
from thinkthread.llm import OpenAIClient

# Set API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set")
    exit(1)

# Initialize client
client = OpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")

# Create session
session = ThinkThreadSession(llm_client=client, alternatives=2, rounds=1)

# Run test
question = "What is the capital of France?"
print(f"Question: {question}")
print("Running CoRT reasoning...")
answer = session.run(question)
print(f"Answer: {answer}")

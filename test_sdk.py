import os
from thinkthread_sdk.cort_session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

# Set API key
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

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

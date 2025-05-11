import os
from cort_sdk.llm.anthropic_client import AnthropicClient

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not api_key:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    exit(1)

client = AnthropicClient(api_key=api_key)

print("Sending request to Claude API...")
response = client.generate("Hello, how are you?")
print("\nResponse from Claude:")
print(response)

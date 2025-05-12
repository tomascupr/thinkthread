# ThinkThread SDK Usage Guide

This guide provides detailed instructions on how to use the ThinkThread SDK for various scenarios.

## Basic Usage with CLI

The ThinkThread SDK provides a command-line interface for quick access to its functionality.

### Running a Simple Query

```bash
thinkthread run "What is the significance of the Fibonacci sequence in nature?"
```

This will:
1. Generate an initial answer
2. Create alternative answers
3. Evaluate and select the best answer
4. Repeat for the configured number of rounds
5. Return the final answer

### Customising Parameters

You can customise the behaviour with various options:

```bash
# Use a different LLM provider
thinkthread run "Explain quantum entanglement" --provider anthropic

# Increase the number of alternatives and rounds
thinkthread run "What are the ethical implications of AI?" --alternatives 5 --rounds 3

# Enable streaming output
thinkthread run "Describe the formation of black holes" --stream

# Enable verbose logging
thinkthread run "Explain blockchain technology" --verbose
```

### Getting Help

To see all available commands and options:

```bash
thinkthread --help
thinkthread run --help
```

## Using the Python API

For more control or to integrate with your applications, use the Python API.

### Basic Example

```python
from thinkthread_sdk.cort_session import CoRTSession
from thinkthread_sdk.llm import OpenAIClient

# Initialize an LLM client
client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")

# Create a CoRT session
session = CoRTSession(llm_client=client, alternatives=3, rounds=2)

# Run recursive reasoning on a question
question = "What are the implications of quantum computing on cryptography?"
answer = session.run(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Using Asynchronous API

For non-blocking operation in async applications:

```python
import asyncio
from thinkthread_sdk.cort_session import CoRTSession
from thinkthread_sdk.llm import OpenAIClient

async def main():
    # Initialize client and session
    client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
    session = CoRTSession(llm_client=client, alternatives=3, rounds=2)
    
    # Run recursively with async API
    question = "Explain the impact of climate change on biodiversity"
    answer = await session.run_async(question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Streaming the final answer
    print("Streaming the final answer:")
    prompt = session.template_manager.render_template(
        "final_answer.j2", {"question": question, "answer": answer}
    )
    
    async for token in await session.llm_client.astream(prompt):
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Different LLM Providers

The SDK supports multiple LLM providers:

```python
# OpenAI
from thinkthread_sdk.llm import OpenAIClient
openai_client = OpenAIClient(
    api_key="your-openai-api-key",
    model_name="gpt-4"
)

# Anthropic
from thinkthread_sdk.llm import AnthropicClient
anthropic_client = AnthropicClient(
    api_key="your-anthropic-api-key",
    model_name="claude-2"
)

# HuggingFace
from thinkthread_sdk.llm import HuggingFaceClient
hf_client = HuggingFaceClient(
    api_token="your-hf-token",
    model_name="gpt2"
)
```

## Example: Solving a Problem with and without CoRT

### Without CoRT (Direct LLM Response)

```python
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
question = "What is the best strategy for addressing climate change?"
direct_answer = client.generate(
    f"Question: {question}\n\nAnswer:", temperature=0.7
)

print(f"Direct Answer: {direct_answer}")
```

### With CoRT (Recursive Refinement)

```python
from thinkthread_sdk.cort_session import CoRTSession
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
session = CoRTSession(llm_client=client, alternatives=3, rounds=2)

question = "What is the best strategy for addressing climate change?"
cort_answer = session.run(question)

print(f"CoRT Answer: {cort_answer}")
```

The CoRT answer will typically:
- Be more comprehensive
- Consider multiple perspectives
- Address counterarguments
- Provide more nuanced analysis
- Have fewer factual errors

This improvement comes from the recursive process of generating alternatives, evaluating them, and selecting the best answer over multiple iterations.

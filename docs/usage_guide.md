# ThinkThread SDK Usage Guide

This guide provides detailed instructions on how to use the ThinkThread SDK for various scenarios.

## Basic Usage with CLI

The ThinkThread SDK provides a command-line interface for quick access to its functionality, supporting both Chain-of-Recursive-Thoughts (CoRT) and Tree-of-Thoughts (ToT) reasoning approaches.

### Running a Simple Query with CoRT

```bash
thinkthread run "What is the significance of the Fibonacci sequence in nature?"
```

This will:
1. Generate an initial answer
2. Create alternative answers
3. Evaluate and select the best answer
4. Repeat for the configured number of rounds
5. Return the final answer

### Running a Simple Query with ToT

```bash
thinkthread tot "How would you solve the Tower of Hanoi puzzle with 3 disks?"
```

This will:
1. Generate multiple initial thoughts
2. Expand each thought into multiple branches
3. Evaluate all branches and prune less promising paths
4. Continue expanding the most promising branches
5. Select the highest-scoring solution path

### Using the Unified Interface

The SDK provides a unified `think` command that supports both reasoning approaches:

```bash
# Using CoRT approach (default)
thinkthread think "What is the significance of the Fibonacci sequence in nature?"

# Using ToT approach
thinkthread think "How would you solve the Tower of Hanoi puzzle with 3 disks?" --approach tot
```

### Customising Parameters

You can customise the behaviour with various options:

```bash
# Use a different LLM provider
thinkthread think "Explain quantum entanglement" --provider anthropic

# CoRT-specific options
thinkthread run "What are the ethical implications of AI?" --alternatives 5 --rounds 3

# ToT-specific options
thinkthread tot "Design an algorithm for the traveling salesman problem" --beam-width 4 --max-tree-depth 3

# Enable streaming output
thinkthread think "Describe the formation of black holes" --stream

# Enable verbose logging
thinkthread think "Explain blockchain technology" --verbose
```

### Getting Help

To see all available commands and options:

```bash
thinkthread --help
thinkthread run --help
thinkthread tot --help
thinkthread think --help
```

## Using the Python API

For more control or to integrate with your applications, use the Python API.

### Chain-of-Recursive-Thoughts (CoRT) Example

```python
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

# Initialize an LLM client
client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")

# Create a ThinkThread session (CoRT approach)
session = ThinkThreadSession(llm_client=client, alternatives=3, rounds=2)

# Run recursive reasoning on a question
question = "What are the implications of quantum computing on cryptography?"
answer = session.run(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Tree-of-Thoughts (ToT) Example

```python
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import OpenAIClient

# Initialize an LLM client
client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")

# Create a TreeThinker session
tree_thinker = TreeThinker(
    llm_client=client,
    max_tree_depth=3,
    branching_factor=3
)

# Run tree-based reasoning on a problem
problem = "How would you solve the Tower of Hanoi puzzle with 3 disks?"
solution = tree_thinker.solve(problem, beam_width=3)

print(f"Problem: {problem}")
print(f"Solution: {solution}")
```

### Using Asynchronous API

For non-blocking operation in async applications:

#### Asynchronous CoRT Example

```python
import asyncio
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

async def cort_async_example():
    # Initialize client and session
    client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
    session = ThinkThreadSession(llm_client=client, alternatives=3, rounds=2)
    
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
    asyncio.run(cort_async_example())
```

#### Asynchronous ToT Example

```python
import asyncio
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import OpenAIClient

async def tot_async_example():
    # Initialize client and tree thinker
    client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
    tree_thinker = TreeThinker(
        llm_client=client,
        max_tree_depth=3,
        branching_factor=3
    )
    
    # Run tree-based reasoning with async API
    problem = "Design an algorithm for the traveling salesman problem"
    solution = await tree_thinker.solve_async(problem, beam_width=3)
    
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")

if __name__ == "__main__":
    asyncio.run(tot_async_example())
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

## Comparing Reasoning Approaches

### Direct LLM Response (No Reasoning)

```python
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
question = "What is the best strategy for addressing climate change?"
direct_answer = client.generate(
    f"Question: {question}\n\nAnswer:", temperature=0.7
)

print(f"Direct Answer: {direct_answer}")
```

### Chain-of-Recursive-Thoughts (CoRT)

```python
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
session = ThinkThreadSession(llm_client=client, alternatives=3, rounds=2)

question = "What is the best strategy for addressing climate change?"
cort_answer = session.run(question)

print(f"CoRT Answer: {cort_answer}")
```

### Tree-of-Thoughts (ToT)

```python
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
tree_thinker = TreeThinker(llm_client=client, max_tree_depth=3, branching_factor=3)

question = "What is the best strategy for addressing climate change?"
tot_answer = tree_thinker.solve(question, beam_width=3)

print(f"ToT Answer: {tot_answer}")
```

## Benefits of Advanced Reasoning

### Benefits of CoRT

The CoRT approach typically produces answers that:
- Are more comprehensive
- Consider multiple perspectives
- Address counterarguments
- Provide more nuanced analysis
- Have fewer factual errors

This improvement comes from the recursive process of generating alternatives, evaluating them, and selecting the best answer over multiple iterations.

### Benefits of ToT

The ToT approach is particularly effective for:
- Complex problem-solving tasks
- Strategic planning
- Creative ideation
- Multi-step reasoning
- Research questions

This effectiveness comes from exploring multiple reasoning paths simultaneously and selecting the most promising solution path.

## Choosing the Right Approach

For detailed guidance on when to use each approach, see the [Comparison Guide](comparison.md).

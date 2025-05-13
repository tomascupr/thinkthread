# Tree-of-Thoughts Reasoning

The TreeThinker module implements tree-based search for reasoning, allowing exploration of multiple reasoning paths in parallel. This approach can lead to more thorough and higher-quality solutions compared to linear reasoning.

## Overview

Tree-of-Thoughts (ToT) is an advanced reasoning technique that extends Chain-of-Thoughts by exploring multiple reasoning paths simultaneously. The key advantages include:

- **Breadth of exploration**: Considers multiple alternative reasoning paths
- **Pruning ineffective paths**: Focuses computational resources on promising directions
- **Parallel processing**: Explores multiple threads of thought concurrently
- **Structured evaluation**: Compares partial solutions systematically

## Basic Usage

```python
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import OpenAIClient
from thinkthread_sdk.config import create_config

# Setup
client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")
config = create_config()

# Initialize TreeThinker
tree_thinker = TreeThinker(
    llm_client=client,
    max_tree_depth=3,         # Maximum depth of the thinking tree
    branching_factor=3,       # Number of branches per node
    config=config,
)

# Solve a problem
problem = "What are three key benefits of tree-based search for reasoning?"
result = tree_thinker.solve(
    problem=problem,
    beam_width=2,             # Number of parallel thought threads
    max_iterations=2          # Number of expansion iterations
)

# Find the best solution
best_node_id = max(
    tree_thinker.threads.keys(),
    key=lambda node_id: tree_thinker.threads[node_id].score
)
best_node = tree_thinker.threads[best_node_id]

print(f"Best solution (score: {best_node.score:.2f}):")
print(best_node.state.get("current_answer", "No answer found"))
```

## Asynchronous Usage

For non-blocking operation, use the asynchronous API:

```python
import asyncio
from thinkthread_sdk.tree_thinker import TreeThinker
from thinkthread_sdk.llm import AnthropicClient

async def solve_with_tree_thinking():
    client = AnthropicClient(api_key="your-api-key", model="claude-2")
    tree_thinker = TreeThinker(llm_client=client)
    
    result = await tree_thinker.solve_async(
        problem="How can we address climate change through technology?",
        beam_width=3,
        max_iterations=2
    )
    
    # Process results
    best_node_id = max(
        tree_thinker.threads.keys(),
        key=lambda node_id: tree_thinker.threads[node_id].score
    )
    best_node = tree_thinker.threads[best_node_id]
    
    return best_node.state.get("current_answer", "No answer found")

# Run the async function
answer = asyncio.run(solve_with_tree_thinking())
print(answer)
```

## Command Line Interface

The TreeThinker module can be used from the command line:

```bash
# Basic usage
thinkthread tot "What are the benefits of tree-based search for reasoning?"

# With specific provider
thinkthread tot "Design a system for autonomous vehicles" --provider anthropic

# Advanced configuration
thinkthread tot "What are the ethical implications of AI?" --beam-width 5 --max-depth 4 --iterations 3
```

## Configuration Options

The TreeThinker module supports various configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_tree_depth` | Maximum depth of the thinking tree | 3 |
| `branching_factor` | Number of branches per node | 3 |
| `beam_width` | Number of parallel thought threads | 3 |
| `max_iterations` | Number of expansion iterations | 2 |

## Working with Different LLM Providers

TreeThinker works with any LLM provider that implements the `LLMClient` interface:

```python
# OpenAI
from thinkthread_sdk.llm import OpenAIClient
client = OpenAIClient(api_key="your-api-key", model_name="gpt-4")

# Anthropic
from thinkthread_sdk.llm import AnthropicClient
client = AnthropicClient(api_key="your-api-key", model="claude-2")

# HuggingFace
from thinkthread_sdk.llm import HuggingFaceClient
client = HuggingFaceClient(api_token="your-token", model="mistralai/Mistral-7B-Instruct-v0.2")

# Custom or mock client for testing
from thinkthread_sdk.llm import DummyLLMClient
client = DummyLLMClient()
```

## Advanced Features

### Custom Evaluation

You can provide a custom evaluator for scoring thought branches:

```python
from thinkthread_sdk.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
tree_thinker = TreeThinker(
    llm_client=client,
    evaluator=evaluator,
    config=config,
)
```

### Accessing the Thinking Tree

After solving a problem, you can access the full thinking tree:

```python
# Get all nodes
all_nodes = tree_thinker.threads

# Get the final layer (leaf nodes after pruning)
final_layer = tree_thinker.current_layer

# Traverse the tree
root_nodes = [node for node_id, node in all_nodes.items() if node.parent_id is None]
for root in root_nodes:
    children = [node for node_id, node in all_nodes.items() if node.parent_id == root.node_id]
    # Process children...
```

## Example Applications

- **Complex problem solving**: Breaking down multi-step problems
- **Creative ideation**: Exploring multiple creative directions
- **Decision making**: Evaluating different options systematically
- **Planning**: Developing and comparing alternative plans

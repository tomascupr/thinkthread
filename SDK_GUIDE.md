# ThinkThread SDK Guide

A comprehensive guide to using the ThinkThread SDK for advanced reasoning capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Basic SDK Usage](#basic-sdk-usage)
5. [Advanced Features](#advanced-features)
6. [Configuration Deep Dive](#configuration-deep-dive)
7. [Custom Components](#custom-components)
8. [Performance Optimization](#performance-optimization)
9. [Production Best Practices](#production-best-practices)

## Introduction

While ThinkThread's simple API (`reason()`, `explore()`, etc.) covers most use cases, the SDK provides fine-grained control over the reasoning process. Use the SDK when you need:

- Custom evaluation strategies
- Specific LLM configurations
- Performance optimizations
- Integration with existing systems
- Advanced monitoring and debugging

## Core Concepts

### 1. Chain-of-Recursive-Thoughts (CoRT)

The `ThinkThreadSession` implements CoRT, which iteratively improves answers through:

```
Initial Answer → Generate Alternatives → Evaluate → Select Best → Repeat
```

Each round explores different reasoning paths, challenges assumptions, and synthesizes insights.

### 2. Tree-of-Thoughts (ToT)

The `TreeThinker` explores multiple reasoning branches simultaneously:

```
                Problem
               /   |   \
         Branch A  B   C
            / \    |   / \
          A1  A2  B1  C1 C2
```

### 3. Evaluation Strategies

ThinkThread uses three evaluation approaches:

- **Self-Evaluation**: Each alternative compared against current best
- **Pairwise Evaluation**: Head-to-head comparisons between alternatives
- **Batch Evaluation**: All alternatives evaluated together

## Architecture Overview

```
┌─────────────────┐
│   Application   │
└────────┬────────┘
         │
┌────────▼────────┐
│   Simple API    │ ◄── reason(), explore(), solve()
└────────┬────────┘
         │
┌────────▼────────────────────────────┐
│          ThinkThread SDK            │
│  ┌────────────┐  ┌──────────────┐  │
│  │  Session   │  │  TreeThinker │  │
│  └─────┬──────┘  └──────┬───────┘  │
│        │                 │          │
│  ┌─────▼─────────────────▼──────┐  │
│  │         LLM Client           │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │         Evaluators           │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Basic SDK Usage

### Creating a Session

```python
from thinkthread import ThinkThreadSession
from thinkthread.llm import OpenAIClient

# Initialize LLM client
client = OpenAIClient(
    api_key="your-api-key",
    model_name="gpt-4-turbo"
)

# Create session with custom parameters
session = ThinkThreadSession(
    llm_client=client,
    alternatives=5,  # Generate 5 alternatives per round
    rounds=3,        # Run 3 refinement rounds
)

# Run reasoning
answer = session.run("How can we reduce technical debt?")
```

### Using Tree-of-Thoughts

```python
from thinkthread import TreeThinker

thinker = TreeThinker(
    llm_client=client,
    max_tree_depth=3,    # Explore 3 levels deep
    branching_factor=4,  # 4 branches per node
)

# Solve with beam search
solution = thinker.solve(
    "Design a scalable microservices architecture",
    beam_width=2,        # Keep top 2 paths
    max_iterations=5     # Maximum iterations
)
```

## Advanced Features

### 1. Async Operations

```python
import asyncio
from thinkthread import ThinkThreadSession

async def parallel_reasoning():
    session = ThinkThreadSession(llm_client=client)
    
    # Run multiple questions in parallel
    tasks = [
        session.run_async("Question 1"),
        session.run_async("Question 2"),
        session.run_async("Question 3"),
    ]
    
    answers = await asyncio.gather(*tasks)
    return answers

# Execute
answers = asyncio.run(parallel_reasoning())
```

### 2. Parallel Processing

```python
from thinkthread.config import ThinkThreadConfig

config = ThinkThreadConfig(
    parallel_alternatives=True,  # Generate alternatives in parallel
    parallel_evaluation=True,    # Evaluate in parallel
    concurrency_limit=10,       # Max concurrent requests
)

session = ThinkThreadSession(
    llm_client=client,
    config=config
)
```

### 3. Early Termination

Stop reasoning when confidence is high:

```python
config = ThinkThreadConfig(
    early_termination=True,
    early_termination_threshold=0.95,  # Stop at 95% confidence
)

session = ThinkThreadSession(
    llm_client=client,
    config=config,
    max_rounds=10  # Will terminate early if threshold met
)
```

### 4. Adaptive Temperature

Dynamically adjust creativity during reasoning:

```python
config = ThinkThreadConfig(
    use_adaptive_temperature=True,
    initial_temperature=0.7,
    generation_temperature=0.9,
    min_generation_temperature=0.5,
    temperature_decay_rate=0.8  # Reduce by 20% each round
)
```

### 5. Caching

Enable caching for repeated queries:

```python
config = ThinkThreadConfig(use_caching=True)

session = ThinkThreadSession(
    llm_client=client,
    config=config
)

# First call hits the API
answer1 = session.run("Complex question")

# Subsequent identical calls use cache
answer2 = session.run("Complex question")  # Much faster!
```

## Configuration Deep Dive

### Key Configuration Parameters

```python
from thinkthread.config import ThinkThreadConfig

config = ThinkThreadConfig(
    # API Keys
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    
    # Provider Selection
    provider="openai",  # or "anthropic", "hf"
    
    # Model Selection
    openai_model="gpt-4-turbo",
    anthropic_model="claude-3-opus-20240229",
    
    # Core Parameters
    alternatives=3,      # Alternatives per round
    rounds=2,           # Default rounds
    max_rounds=5,       # Maximum rounds (with early termination)
    
    # Evaluation Strategy
    use_pairwise_evaluation=True,   # Compare pairs
    use_self_evaluation=False,      # Self-evaluate
    
    # Performance
    parallel_alternatives=True,     # Parallel generation
    parallel_evaluation=True,       # Parallel evaluation
    concurrency_limit=5,           # Max concurrent requests
    use_batched_requests=False,    # Batch API calls
    
    # Optimization
    use_caching=True,              # Cache responses
    early_termination=True,        # Stop when confident
    early_termination_threshold=0.95,
    
    # Temperature Control
    use_adaptive_temperature=True,
    initial_temperature=0.7,
    generation_temperature=0.9,
    min_generation_temperature=0.5,
    temperature_decay_rate=0.8,
    
    # Advanced
    enable_monitoring=True,        # Enable metrics
    use_fast_similarity=True,      # Fast similarity checks
    prompt_dir="/custom/prompts",  # Custom prompt directory
)
```

### Loading from Environment

```python
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo
ALTERNATIVES=5
ROUNDS=3
USE_CACHING=true
PARALLEL_ALTERNATIVES=true

# Python
from thinkthread.config import create_config

config = create_config(".env")  # Loads from .env file
```

## Custom Components

### 1. Custom Evaluator

```python
from thinkthread.evaluation import BaseEvaluator

class DomainSpecificEvaluator(BaseEvaluator):
    def evaluate(self, alternatives, context=None):
        """Evaluate alternatives based on domain criteria."""
        scores = []
        
        for alt in alternatives:
            score = 0.0
            
            # Check for specific keywords
            if "scalable" in alt.lower():
                score += 0.2
            
            # Penalize overly complex solutions
            if len(alt) > 1000:
                score -= 0.1
                
            # Custom domain logic
            score += self._domain_specific_scoring(alt)
            
            scores.append(score)
            
        # Return index of best alternative
        return scores.index(max(scores))
    
    def _domain_specific_scoring(self, text):
        # Your domain logic here
        return 0.5

# Use custom evaluator
session = ThinkThreadSession(
    llm_client=client,
    evaluation_strategy=DomainSpecificEvaluator()
)
```

### 2. Custom Scoring Function

```python
def security_focused_scorer(answer: str, context: dict) -> float:
    """Score answers based on security considerations."""
    score = 0.0
    
    security_keywords = [
        "authentication", "authorization", "encryption",
        "security", "vulnerability", "protection"
    ]
    
    # Boost score for security mentions
    for keyword in security_keywords:
        if keyword in answer.lower():
            score += 0.15
    
    # Check for security anti-patterns
    if "plaintext password" in answer.lower():
        score -= 0.5
        
    return min(max(score, 0.0), 1.0)

# Use with TreeThinker
thinker = TreeThinker(
    llm_client=client,
    scoring_function=security_focused_scorer
)
```

### 3. Custom Prompt Templates

Create custom Jinja2 templates:

```python
# custom_prompts/initial_prompt.j2
You are an expert in {{ domain }}.

Question: {{ question }}

Provide a detailed answer considering:
- Industry best practices
- Common pitfalls
- Real-world constraints

Answer:

# Python
from thinkthread.prompting import TemplateManager

template_manager = TemplateManager(
    template_dir="custom_prompts"
)

session = ThinkThreadSession(
    llm_client=client,
    template_manager=template_manager
)
```

## Performance Optimization

### 1. Reduce API Calls

```python
# Minimize rounds and alternatives
session = ThinkThreadSession(
    llm_client=client,
    alternatives=2,  # Fewer alternatives
    rounds=1,        # Fewer rounds
)
```

### 2. Enable Parallel Processing

```python
config = ThinkThreadConfig(
    parallel_alternatives=True,
    parallel_evaluation=True,
    concurrency_limit=10  # Adjust based on API limits
)
```

### 3. Use Caching

```python
config = ThinkThreadConfig(
    use_caching=True,
    use_fast_similarity=True  # Faster cache lookups
)
```

### 4. Early Termination

```python
config = ThinkThreadConfig(
    early_termination=True,
    early_termination_threshold=0.9  # Lower threshold = faster
)
```

### 5. Optimize Tree Search

```python
# Smaller trees for faster results
thinker = TreeThinker(
    llm_client=client,
    max_tree_depth=2,    # Shallower tree
    branching_factor=2   # Fewer branches
)

solution = thinker.solve(
    problem,
    beam_width=1,        # Single path
    max_iterations=3     # Fewer iterations
)
```

## Production Best Practices

### 1. Error Handling

```python
from thinkthread.llm import LLMException

try:
    answer = session.run(question)
except LLMException as e:
    # Handle API errors
    logger.error(f"LLM error: {e}")
    # Fallback logic
except Exception as e:
    # Handle other errors
    logger.error(f"Unexpected error: {e}")
```

### 2. Monitoring

```python
from thinkthread.monitoring import GLOBAL_MONITOR

config = ThinkThreadConfig(enable_monitoring=True)

session = ThinkThreadSession(
    llm_client=client,
    config=config
)

# After running
metrics = GLOBAL_MONITOR.get_metrics()
print(f"Total API calls: {metrics['api_calls']}")
print(f"Average latency: {metrics['avg_latency']}ms")
```

### 3. Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_minute):
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            min_interval = 60.0 / calls_per_minute
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
            
        return wrapper
    return decorator

# Apply rate limiting
@rate_limit(calls_per_minute=20)
def process_question(question):
    return session.run(question)
```

### 4. Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('thinkthread')

# Log session details
logger.info(f"Starting session with {config.alternatives} alternatives")
answer = session.run(question)
logger.info(f"Completed reasoning in {session.rounds} rounds")
```

### 5. Configuration Management

```python
# config/development.env
OPENAI_API_KEY=${OPENAI_API_KEY}
ALTERNATIVES=5
ROUNDS=3
USE_CACHING=false
ENABLE_MONITORING=true

# config/production.env
OPENAI_API_KEY=${OPENAI_API_KEY}
ALTERNATIVES=3
ROUNDS=2
USE_CACHING=true
ENABLE_MONITORING=true
PARALLEL_ALTERNATIVES=true

# Load based on environment
import os
env = os.getenv('ENVIRONMENT', 'development')
config = create_config(f"config/{env}.env")
```

## Next Steps

- See [API_REFERENCE.md](API_REFERENCE.md) for detailed parameter documentation
- Check [EXAMPLES.md](EXAMPLES.md) for practical use cases
- Review the [source code](https://github.com/tomcupr/thinkthread) for implementation details

Remember: The SDK gives you power and flexibility, but the simple API is often sufficient. Start simple and add complexity only when needed.
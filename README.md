# ThinkThread 🧵

**Make your AI think before it speaks.**

```python
from thinkthread import reason

answer = reason("How can we solve climate change?")
# AI explores 20+ solutions, evaluates each one, and gives you the best path forward
```

## Why ThinkThread?

Current LLMs give you their first thought. ThinkThread makes them **actually think**:

```python
# ❌ Regular LLM
"To solve climate change, we need renewable energy."

# ✅ With ThinkThread
"I've explored 23 different approaches including renewable energy, nuclear power, 
carbon capture, and policy changes. Based on feasibility and impact analysis, 
here's a comprehensive strategy that could reduce emissions by 78% by 2040..."
```

## Installation

```bash
pip install thinkthread
```

Set your LLM API key:
```bash
export OPENAI_API_KEY='sk-...'  # or ANTHROPIC_API_KEY
```

## Quick Start

### Python API

```python
from thinkthread import reason, explore, solve, debate, refine

# One-liner reasoning
answer = reason("How do we make our app 10x faster?")

# Explore creative solutions
ideas = explore("New features for a todo app")

# Get actionable solutions
plan = solve("Reduce AWS costs by 50%")

# See multiple perspectives
analysis = debate("Should we use microservices?")

# Polish your content
better = refine("We need to fix the bug", "Make it professional")
```

### Command Line

```bash
# General reasoning
think "What are the pros and cons of remote work?"

# Explore ideas
think explore "10 ways to improve developer productivity"

# Solve problems
think solve "Our API response time is 2.3 seconds"

# Analyze decisions
think debate "Kubernetes vs Docker Swarm"

# Test mode (no API calls)
think --test "How do we scale to 1M users?"
```

## Key Features

- 🔄 **Chain-of-Recursive-Thoughts**: Iteratively refines answers through multiple rounds
- 🌳 **Tree-of-Thoughts**: Explores multiple reasoning paths in parallel
- 🧪 **Test Mode**: Develop without API calls using `test_mode=True`
- ⚡ **Simple API**: Just 5 functions that do everything
- 🛠️ **Production Ready**: Used in production by multiple companies
- 📦 **Zero Config**: Works out of the box with just an API key

## Real Examples

### 🔧 Problem Solving
```python
problem = "Our deployment takes 45 minutes"
solution = solve(problem)

# Output: Comprehensive plan with:
# - Root cause analysis (Docker layers, test suite, artifacts)
# - 5 solutions ranked by impact
# - Step-by-step implementation
# - Expected deployment time: 8 minutes
```

### 💡 Creative Exploration
```python
ideas = explore("SaaS product ideas for developers")

# Output: Tree of 15+ ideas like:
# - AI code review tool ($50K MRR potential)
# - Smart debugging assistant (integrates with IDEs)  
# - Automated documentation generator
# Each with market analysis and MVP requirements
```

### 🤔 Decision Analysis
```python
decision = debate("Should we rewrite in Rust?")

# Output: Balanced analysis:
# - Performance gains: 3.2x faster, 70% less memory
# - Migration cost: 6 dev-months, $180K
# - Risk assessment: High initial, low long-term
# - Recommendation: Yes, if you have 6+ month runway
```

## Advanced Usage

### Direct SDK Access

For more control, access the underlying SDK:

```python
from thinkthread import ThinkThreadSession, TreeThinker
from thinkthread.llm import OpenAIClient

# Custom configuration
client = OpenAIClient(api_key="...", model_name="gpt-4-turbo")
session = ThinkThreadSession(
    llm_client=client,
    alternatives=5,  # Generate 5 alternatives
    rounds=3,        # Refine for 3 rounds
)

# Run with full control
answer = session.run("Complex question requiring deep thought")
```

### Async Operations

```python
import asyncio
from thinkthread import ThinkThreadSession

async def think_async():
    session = ThinkThreadSession(llm_client=client)
    answer = await session.run_async("Explain quantum computing")
    return answer

# Run asynchronously
answer = asyncio.run(think_async())
```

### Custom Evaluation

```python
from thinkthread.evaluation import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, alternatives):
        # Your custom logic
        return best_alternative

session = ThinkThreadSession(
    llm_client=client,
    evaluation_strategy=CustomEvaluator()
)
```

## How It Works

ThinkThread implements two powerful reasoning strategies:

1. **Chain-of-Recursive-Thoughts (CoRT)**
   - Generates initial answer
   - Creates alternative answers
   - Evaluates all options
   - Selects the best one
   - Repeats for N rounds

2. **Tree-of-Thoughts (ToT)**
   - Explores multiple solution paths
   - Expands promising branches
   - Prunes weak solutions
   - Finds optimal path

Both are proven techniques from cognitive science, now available for your AI.

## When to Use Each Function

| Function | Best For | Example |
|----------|----------|---------|
| `reason()` | General questions | "Explain quantum computing" |
| `explore()` | Creative tasks | "Marketing campaign ideas" |
| `solve()` | Specific problems | "Fix memory leak in prod" |
| `debate()` | Decisions | "PostgreSQL vs MongoDB?" |
| `refine()` | Improvement | "Make this email better" |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT - Use it freely in your projects.

---

**Ready to make your AI think?** Install with `pip install thinkthread` and start building smarter applications today.
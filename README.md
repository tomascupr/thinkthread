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

## Production-Ready Features

ThinkThread combines a beautiful API with battle-tested robustness:
- ⚡ **Automatic retry** with exponential backoff
- 💾 **Smart caching** to reduce costs
- 📊 **Performance monitoring** built-in
- 🔄 **Streaming support** for real-time output
- 🛡️ **Error handling** that never crashes
- 🧪 **Test mode** for development

## Installation

```bash
pip install thinkthread
```

Set your LLM API key:
```bash
export OPENAI_API_KEY='sk-...'  # or ANTHROPIC_API_KEY
```

## 60-Second Quick Start

### From Command Line

```bash
# Ask anything
think "What are the pros and cons of remote work?"

# Brainstorm ideas
think quick "10 ways to improve developer productivity"

# Solve problems
think fix "Our API response time is 2.3 seconds"

# Compare options
think compare "Kubernetes vs Docker Swarm"

# Test without API calls
think --test "How do we scale to 1M users?"
```

### From Python

```python
from thinkthread import reason, explore, solve, debate

# One-liner reasoning
answer = reason("How do we make our app 10x faster?")

# Explore creative solutions
ideas = explore("New features for a todo app")

# Get actionable solutions
plan = solve("Reduce AWS costs by 50%")

# See multiple perspectives
analysis = debate("Should we use microservices?")
```

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

### 💡 Brainstorming
```python
ideas = explore("SaaS product ideas for developers")

# Output: Tree of 15+ ideas like:
# - AI code review tool ($50K MRR potential)
# - Smart debugging assistant (integrates with IDEs)  
# - Automated documentation generator
# Each with market analysis and MVP requirements
```

### 🤔 Decision Making
```python
decision = debate("Should we rewrite in Rust?")

# Output: Balanced analysis:
# - Performance gains: 3.2x faster, 70% less memory
# - Migration cost: 6 dev-months, $180K
# - Risk assessment: High initial, low long-term
# - Recommendation: Yes, if you have 6+ month runway
```

## Advanced Features

### 🎯 Test Mode (No API Calls)
```python
# Perfect for development and CI/CD
answer = reason("Complex question", test_mode=True)
```

### 🔄 Retry Logic Built-in
```python
# Automatic retry with exponential backoff
# Never fails due to network issues
answer = reason("Important question")  # Just works
```

### 📊 Structured Output
```python
answer = solve("Optimize database queries")
print(answer.confidence)  # 0.87
print(answer.cost)       # $0.0234
print(answer.reasoning_mode)  # "solve"
```

### 🔗 Mode Chaining
```python
# Explore ideas then refine the best one
from thinkthread import explore, refine

ideas = explore("Mobile app features")
best_idea = refine(ideas.best)  # Polished, implementation-ready
```

## When to Use Each Mode

| Mode | Use For | Example |
|------|---------|---------|
| `reason()` | General questions | "Explain quantum computing" |
| `explore()` | Creative tasks | "Marketing campaign ideas" |
| `solve()` | Specific problems | "Fix memory leak in prod" |
| `debate()` | Decisions | "PostgreSQL vs MongoDB?" |
| `refine()` | Improvement | "Make this email better" |

## Configuration

```bash
# Optional: Use specific models
export OPENAI_MODEL='gpt-4-turbo-preview'
export ANTHROPIC_MODEL='claude-3-opus-20240229'

# Optional: Control costs
think --max-cost 0.10 "Expensive analysis"
```

## What Makes ThinkThread Different?

1. **Actually Thinks** - Not just prompt engineering, but structured reasoning
2. **Zero Config** - Works out of the box, no complex setup
3. **Production Ready** - Retries, fallbacks, error handling built-in
4. **Cost Efficient** - Smart caching and token optimization (coming soon)
5. **Transparent** - See exactly how your AI reaches conclusions

## Common Patterns

### API Endpoint
```python
from fastapi import FastAPI
from thinkthread import reason

app = FastAPI()

@app.post("/api/think")
async def think(question: str):
    answer = reason(question)
    return {
        "answer": str(answer),
        "confidence": answer.confidence,
        "cost": answer.cost
    }
```

### Batch Processing
```python
questions = ["How to scale?", "Best practices?", "Security concerns?"]
answers = [reason(q) for q in questions]
```

### Custom Reasoning
```python
from thinkthread.modes import SolveMode

class CustomSolver(SolveMode):
    def analyze_problem(self, problem):
        # Your domain-specific analysis
        return super().analyze_problem(problem)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - Use it freely in your projects.

---

**Ready to make your AI think?** `pip install thinkthread` and start building smarter applications today.
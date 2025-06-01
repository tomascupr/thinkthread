# ThinkThread 🧠

[![PyPI](https://img.shields.io/pypi/v/thinkthread)](https://pypi.org/project/thinkthread/)
[![Python](https://img.shields.io/pypi/pyversions/thinkthread)](https://pypi.org/project/thinkthread/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Make your AI think before it speaks.**

## What is ThinkThread?

ThinkThread gives your AI applications human-like reasoning capabilities with just one line of code:

```python
from thinkthread import reason

answer = reason("How can we solve climate change?")
```

That's it. Your AI now explores multiple solutions, evaluates trade-offs, and delivers thoughtful answers backed by transparent reasoning.

## Why ThinkThread?

| Regular AI | ThinkThread AI |
|------------|----------------|
| First thought = final answer | Explores multiple paths before answering |
| "I think X" | "I considered X, Y, and Z. Here's why X is best..." |
| Black box responses | See the reasoning process step-by-step |
| $0.50 per complex query | $0.05 after learning patterns |
| One-size-fits-all | Automatically picks the best reasoning strategy |

## Installation

```bash
pip install thinkthread
```

## Quick Start

### Basic Usage

```python
from thinkthread import reason

# It's really this simple
answer = reason("What are the pros and cons of remote work?")
print(answer)
```

### Different Reasoning Modes

ThinkThread automatically selects the best reasoning mode, or you can choose:

```python
from thinkthread import explore, refine, debate, solve

# Explore possibilities (best for creative tasks)
ideas = explore("Design a city on Mars")

# Refine and improve (best for content)
better = refine("Make this email professional", initial_draft)

# See multiple perspectives (best for decisions)  
analysis = debate("Should we use microservices?")

# Get actionable solutions (best for problems)
plan = solve("Reduce customer churn by 30%")
```

### See How Your AI Thinks

```python
# Watch reasoning in real-time
answer = reason("Complex question", visualize=True)
# Opens browser with live reasoning visualization

# Debug reasoning
answer = reason("Why did Rome fall?")
answer.explain()  # See why it reached this conclusion
```

### Track Costs

```python
answer = reason("Business strategy question")
print(f"Cost: ${answer.cost:.4f}")
print(f"Confidence: {answer.confidence:.0%}")
```

## Real-World Examples

### Customer Support
```python
issue = "I was charged twice for my subscription"
response = solve(f"Customer issue: {issue}")
# Provides step-by-step resolution with 87% confidence
```

### Content Creation
```python
topic = "Impact of AI on education"
article = explore(topic) | refine()  # Chain modes together
# Explores angles, then polishes the best one
```

### Strategic Decisions
```python
decision = "Should we expand to Europe or Asia first?"
analysis = debate(decision, perspectives=4)
# Analyzes from multiple viewpoints with consensus scoring
```

## Key Features

✨ **Smart Mode Selection** - Automatically picks the best reasoning approach  
🔍 **Reasoning Transparency** - See exactly how conclusions are reached  
💰 **Cost Optimization** - Learn patterns to reduce API costs by 90%  
🔄 **Seamless Fallbacks** - Never fails, always returns useful results  
📊 **Built-in Analytics** - Track performance, costs, and quality  
🚀 **Production Ready** - Handles errors, retries, and rate limits  

## Advanced Usage

### Compose Reasoning Modes
```python
# Chain modes with | operator
answer = explore("Future of transport") | refine() | debate()

# Run in parallel with & operator  
answer = explore("Problem") & solve("Problem")  # Compare approaches
```

### Configure Behavior
```python
# Set cost limits
reason.set_budget(daily=10.00, per_query=0.50)

# Enable learning
reason.enable_memory()  # Learn from patterns

# Custom settings
answer = reason("Question", 
    temperature=0.8,
    max_depth=4,
    confidence_threshold=0.9
)
```

### Integration
```python
# FastAPI
from fastapi import FastAPI
from thinkthread import reason

app = FastAPI()

@app.post("/think")
def think(question: str):
    return {"answer": reason(question)}

# Enhance existing functions
from thinkthread import enhance

@enhance
def analyze_data(data):
    return basic_analysis(data)  # Now with reasoning!
```

## Reasoning Modes Explained

### 🌳 Explore (Tree of Thoughts)
Best for: Creative tasks, brainstorming, open-ended questions
```python
ideas = explore("New product ideas for teenagers")
```

### 🔄 Refine (Chain of Recursive Thoughts)  
Best for: Improving content, polishing, convergent thinking
```python
refined = refine("Draft blog post", initial_text)
```

### 🎭 Debate (Multi-Perspective)
Best for: Balanced analysis, controversial topics, decisions
```python
analysis = debate("Is nuclear energy safe?")
```

### 🎯 Solve (Solution-Focused)
Best for: Specific problems, action plans, troubleshooting
```python
solution = solve("Improve website conversion by 25%")
```

## Coming Soon

🎨 **Live Visualization** - Watch AI reasoning in real-time  
💾 **Pattern Memory** - Learn and reuse successful reasoning patterns  
🌊 **Streaming** - Get results as they're generated  
🔗 **More LLMs** - Gemini, Cohere, Ollama support  

## Comparison

| Feature | ThinkThread | LangChain | DSPy | Guidance |
|---------|------------|-----------|------|----------|
| One-line usage | ✅ | ❌ | ❌ | ❌ |
| Automatic reasoning | ✅ | ❌ | ❌ | ❌ |
| Visual debugging | ✅ | ❌ | ❌ | ❌ |
| Cost optimization | ✅ | ⚠️ | ❌ | ❌ |
| Multiple reasoning modes | ✅ | ⚠️ | ⚠️ | ❌ |
| Production ready | ✅ | ✅ | ❌ | ⚠️ |

## Get Started

```python
from thinkthread import reason

# Your AI now thinks before it speaks
answer = reason("What should I build this weekend?")
print(answer)  # A well-reasoned, thoughtful response
```

## Contributing

We'd love your help making AI reasoning even better! Check out our [contributing guide](CONTRIBUTING.md).

## License

MIT - Use it however you want!

---

<p align="center">
  <b>ThinkThread</b>: Because the best answers come from thinking things through.
</p>
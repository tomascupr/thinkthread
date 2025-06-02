# ThinkThread Usage Guide

This guide shows you how to use ThinkThread to make your AI think before it speaks.

## Installation

```bash
pip install thinkthread
```

Set your API key:
```bash
export OPENAI_API_KEY='sk-...'  # or ANTHROPIC_API_KEY
```

## Quick Start

### Python API

```python
from thinkthread import reason

# Simple one-liner - AI thinks through the problem
answer = reason("How can we reduce technical debt?")
print(answer)
```

### Command Line

```bash
# Ask anything
think "What are the pros and cons of remote work?"

# Quick brainstorming
think quick "10 ways to improve code quality"

# Fix specific problems
think fix "Our deployment takes 45 minutes"

# Compare options
think compare "PostgreSQL vs MongoDB"
```

## The New Simple API

ThinkThread provides five main functions that cover different thinking styles:

### 1. `reason()` - General Reasoning
Auto-selects the best approach based on your question.

```python
from thinkthread import reason

# It automatically detects what kind of thinking is needed
answer = reason("Explain the CAP theorem")
answer = reason("How to scale to 1M users?")  # Switches to problem-solving mode
answer = reason("Should we use microservices?")  # Switches to debate mode
```

### 2. `explore()` - Creative Exploration
Uses Tree-of-Thoughts to explore multiple possibilities.

```python
from thinkthread import explore

# Brainstorm and explore ideas
ideas = explore("New features for a todo app")
print(ideas)  # Returns explored paths and best ideas

# Great for:
# - Product ideation
# - Creative solutions
# - Research directions
# - Architecture options
```

### 3. `solve()` - Problem Solving
Combines exploration with focused refinement for actionable solutions.

```python
from thinkthread import solve

# Get specific, implementable solutions
solution = solve("Our API response time is 2.3 seconds")
print(solution)  # Detailed plan with steps, risks, and outcomes

# Perfect for:
# - Performance issues
# - Bug fixing strategies
# - System design problems
# - Process improvements
```

### 4. `debate()` - Multiple Perspectives
Generates different viewpoints and synthesizes them.

```python
from thinkthread import debate

# See all sides of a decision
analysis = debate("Should we migrate to Kubernetes?")
print(analysis)  # Pros, cons, and balanced recommendation

# Specify number of perspectives
analysis = debate("Remote vs office work", perspectives=4)

# Ideal for:
# - Technical decisions
# - Architecture choices
# - Process changes
# - Tool selection
```

### 5. `refine()` - Iterative Improvement
Polishes and improves existing content or ideas.

```python
from thinkthread import refine

# Improve existing text
email = "Hey team, we need to fix the bug in prod. It's breaking stuff."
better_email = refine("Make this email more professional", initial_text=email)

# Refine ideas without initial text
plan = refine("Create a 90-day onboarding plan for engineers")

# Great for:
# - Documentation
# - Communication
# - Proposals
# - Code reviews
```

## Advanced Features

### Test Mode (No API Calls)
Perfect for development and testing:

```python
# Python
answer = reason("Complex question", test_mode=True)

# CLI
think --test "How do we scale our database?"
```

### Structured Output
All functions return structured results:

```python
answer = solve("Optimize database queries")

# Access metadata
print(answer.answer)        # The actual response
print(answer.confidence)    # 0.87 (87% confidence)
print(answer.cost)         # $0.0234
print(answer.mode)         # "solve"
print(answer.time_elapsed) # 3.4 seconds
```

### Streaming Responses
Get real-time output (coming soon):

```python
# Stream tokens as they're generated
for chunk in reason.stream("Explain machine learning"):
    print(chunk, end="", flush=True)
```

### Cost Control
Set spending limits:

```python
from thinkthread import reason

# Set per-query limit
answer = reason("Expensive analysis", max_cost=0.10)

# Set global budget (coming soon)
reason.set_budget(daily=10.00, per_query=0.50)
```

### Custom Configuration

```python
from thinkthread import reason

# Use specific providers
answer = reason("Question", provider="anthropic", model="claude-3-opus")

# Control reasoning depth
answer = explore("Ideas", max_tree_depth=4, branching_factor=5)

# Adjust confidence thresholds
answer = refine("Text", rounds=5, confidence_threshold=0.95)
```

## CLI Commands

### Basic Commands

```bash
# General thinking (auto-selects mode)
think "Your question here"

# Quick brainstorming (uses explore mode)
think quick "Product ideas"

# Problem fixing (uses solve mode)  
think fix "Performance issue"

# Comparison (uses debate mode)
think compare "Option A vs Option B"

# Polish content (uses refine mode)
think polish "Improve this text" --text "Your text here"
```

### Options

```bash
# Use test mode (no API calls)
think --test "Question"

# Specify provider
think --provider anthropic "Question"

# Enable debug output
think --debug "Question"

# Limit cost
think --max-cost 0.10 "Question"

# Show reasoning visualization (coming soon)
think --visualize "Question"
```

## Real-World Examples

### 1. Debugging Production Issues

```python
from thinkthread import solve

issue = """
Production API throwing 502 errors intermittently.
Happens 10-15 times per hour. No pattern detected.
Logs show timeout errors but DB queries are fast.
"""

solution = solve(issue)
# Returns: Root cause analysis + step-by-step fix
```

### 2. Architecture Decisions

```python
from thinkthread import debate

decision = debate("Should we switch from REST to GraphQL?")
# Returns: Balanced analysis with team-specific recommendations
```

### 3. Code Review Feedback

```python
from thinkthread import refine

feedback = """
This code doesn't follow our patterns.
The error handling is missing.
No tests were added.
"""

better_feedback = refine(
    "Make this code review feedback more constructive",
    initial_text=feedback
)
```

### 4. Sprint Planning

```python
from thinkthread import explore

ideas = explore("Ways to improve our sprint velocity")
# Returns: Tree of ideas with practical suggestions
```

## Integration Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI
from thinkthread import reason
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    text: str
    mode: str = "auto"

@app.post("/think")
async def think(question: Question):
    answer = reason(question.text, mode=question.mode)
    return {
        "answer": str(answer),
        "confidence": answer.confidence,
        "cost": answer.cost,
        "mode": answer.mode
    }
```

### Slack Bot

```python
from slack_bolt import App
from thinkthread import reason

app = App(token=SLACK_BOT_TOKEN)

@app.message("think:")
def handle_think(message, say):
    question = message['text'].replace("think:", "").strip()
    answer = reason(question)
    say(f"🤔 {answer}")

app.start(port=3000)
```

### Jupyter Notebook

```python
from thinkthread import reason, explore
import pandas as pd

# Analyze data
df = pd.read_csv("sales.csv")
question = f"Given this sales data summary: {df.describe()}, what patterns do you see?"

analysis = reason(question)
display(Markdown(analysis.answer))

# Explore improvements
ideas = explore("Ways to increase sales based on this data")
```

## Best Practices

### 1. Choose the Right Mode

- **General questions**: Use `reason()` - it auto-selects
- **Creative tasks**: Use `explore()` for brainstorming  
- **Specific problems**: Use `solve()` for actionable plans
- **Decisions**: Use `debate()` for balanced analysis
- **Improvement**: Use `refine()` for polishing

### 2. Provide Context

```python
# ❌ Too vague
answer = reason("Fix the bug")

# ✅ Specific context
answer = solve("""
Bug: Login fails for users with special characters in email
Error: "Invalid credentials" even with correct password  
Affects: 12% of users
Started: After last Tuesday's deployment
""")
```

### 3. Use Test Mode During Development

```python
# Develop without burning API credits
if DEBUG:
    answer = reason(question, test_mode=True)
else:
    answer = reason(question)
```

### 4. Handle Costs Wisely

```python
# For expensive operations, set limits
answer = explore(
    "Complex analysis of 500-page document",
    max_cost=1.00
)

# Check cost before proceeding
if answer.cost > 0.50:
    logger.warning(f"High cost operation: ${answer.cost}")
```

## Troubleshooting

### API Key Issues

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set the key
export OPENAI_API_KEY='sk-...'

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### Rate Limiting

ThinkThread automatically handles rate limits with exponential backoff. No action needed.

### Timeout Issues

For long-running operations:

```python
# Increase timeout (coming soon)
answer = reason(question, timeout=60)
```

## Next Steps

- Check out [examples/](../examples/) for more code samples
- Read the [Developer Guide](developer_guide.md) for customization
- See [Configuration Reference](configuration_reference.md) for all options
- Join our [Discord](https://discord.gg/thinkthread) for support
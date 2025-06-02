# ThinkThread Examples

Practical examples demonstrating ThinkThread SDK usage patterns.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Session Examples](#advanced-session-examples)
3. [Tree-of-Thoughts Examples](#tree-of-thoughts-examples)
4. [Custom Components](#custom-components)
5. [Production Patterns](#production-patterns)
6. [Domain-Specific Applications](#domain-specific-applications)

## Basic Examples

### Simple Question Answering

```python
from thinkthread import reason

# Basic reasoning
answer = reason("What causes climate change?")
print(answer)

# With custom parameters
answer = reason(
    "How can we reduce software bugs?",
    alternatives=5,  # More alternatives for better coverage
    rounds=3         # More rounds for deeper analysis
)
```

### Problem Solving

```python
from thinkthread import solve

# Technical problem
solution = solve("""
Our e-commerce site crashes during Black Friday sales.
Peak traffic: 100k concurrent users
Current setup: 3 servers, no caching
""")

print(solution)
# Output includes:
# - Root cause analysis
# - Step-by-step scaling solution
# - Caching strategy
# - Load balancing approach
```

### Decision Making

```python
from thinkthread import debate

# Architecture decision
analysis = debate("Should we migrate from monolith to microservices?")

# Technology choice
comparison = debate("PostgreSQL vs MongoDB for our social media app?")
```

## Advanced Session Examples

### Custom LLM Configuration

```python
from thinkthread import ThinkThreadSession
from thinkthread.llm import OpenAIClient, AnthropicClient

# Using GPT-4 Turbo
openai_client = OpenAIClient(
    api_key="sk-...",
    model_name="gpt-4-turbo-preview"
)

session = ThinkThreadSession(
    llm_client=openai_client,
    alternatives=4,
    rounds=2
)

# Using Claude 3
claude_client = AnthropicClient(
    api_key="sk-ant-...",
    model_name="claude-3-opus-20240229"
)

session_claude = ThinkThreadSession(
    llm_client=claude_client,
    alternatives=3,
    rounds=3
)
```

### Parallel Processing

```python
from thinkthread import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig
from thinkthread.llm import OpenAIClient

# Configure for parallel execution
config = ThinkThreadConfig(
    parallel_alternatives=True,
    parallel_evaluation=True,
    concurrency_limit=10
)

client = OpenAIClient(api_key="sk-...")
session = ThinkThreadSession(
    llm_client=client,
    config=config,
    alternatives=10  # Can handle more with parallel processing
)

# Process multiple questions concurrently
import asyncio

async def process_questions(questions):
    tasks = [session.run_async(q) for q in questions]
    return await asyncio.gather(*tasks)

questions = [
    "How to optimize database queries?",
    "Best practices for API design?",
    "How to implement caching?"
]

answers = asyncio.run(process_questions(questions))
```

### Early Termination

```python
from thinkthread import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig

# Stop when confidence is high
config = ThinkThreadConfig(
    early_termination=True,
    early_termination_threshold=0.9,  # 90% confidence
    enable_monitoring=True
)

session = ThinkThreadSession(
    llm_client=client,
    config=config,
    max_rounds=10  # Will likely terminate before 10 rounds
)

answer = session.run("What is 2+2?")  # Should terminate early
```

### Adaptive Temperature

```python
from thinkthread.config import ThinkThreadConfig

# Start creative, become more focused
config = ThinkThreadConfig(
    use_adaptive_temperature=True,
    initial_temperature=0.9,      # Creative start
    generation_temperature=0.8,   # Still creative alternatives
    min_generation_temperature=0.3,  # Very focused final rounds
    temperature_decay_rate=0.7    # 30% reduction per round
)

session = ThinkThreadSession(
    llm_client=client,
    config=config,
    rounds=5
)

# Good for creative tasks that need refinement
story = session.run("Write a sci-fi story opening")
```

## Tree-of-Thoughts Examples

### Basic Tree Search

```python
from thinkthread import TreeThinker
from thinkthread.llm import OpenAIClient

client = OpenAIClient(api_key="sk-...")

# Create tree thinker
thinker = TreeThinker(
    llm_client=client,
    max_tree_depth=3,
    branching_factor=3
)

# Solve complex problem
solution = thinker.solve(
    "Design a fault-tolerant distributed system",
    beam_width=2,  # Keep top 2 paths
    max_iterations=5
)
```

### Custom Scoring Function

```python
def performance_scorer(answer: str, context: dict) -> float:
    """Score based on performance considerations."""
    score = 0.0
    
    # Keywords that indicate performance focus
    perf_keywords = ["optimize", "cache", "parallel", "async", "efficient"]
    for keyword in perf_keywords:
        if keyword in answer.lower():
            score += 0.1
    
    # Penalize overly complex solutions
    if answer.count("\n") > 50:  # Too many steps
        score -= 0.2
        
    # Bonus for mentioning metrics
    if any(word in answer.lower() for word in ["benchmark", "measure", "profile"]):
        score += 0.15
        
    return max(0, min(1, score))

thinker = TreeThinker(
    llm_client=client,
    scoring_function=performance_scorer
)

solution = thinker.solve("How to make our API 10x faster?")
```

### Exploring Design Alternatives

```python
from thinkthread import explore

# Generate multiple design approaches
designs = explore("""
Design a real-time chat system that supports:
- 1M concurrent users
- Message history
- File uploads
- End-to-end encryption
""")

# The tree exploration will consider multiple architectures:
# - WebSocket vs Server-Sent Events
# - SQL vs NoSQL for message storage
# - CDN strategies for file uploads
# - Encryption implementation approaches
```

## Custom Components

### Domain-Specific Evaluator

```python
from thinkthread.evaluation import BaseEvaluator
from thinkthread import ThinkThreadSession

class CodeQualityEvaluator(BaseEvaluator):
    """Evaluate code-based answers for quality."""
    
    def evaluate(self, alternatives, context=None):
        scores = []
        
        for alt in alternatives:
            score = 0.0
            
            # Check for code blocks
            if "```" in alt:
                score += 0.2
            
            # Error handling
            if any(term in alt.lower() for term in ["try", "except", "error handling"]):
                score += 0.15
            
            # Documentation
            if any(term in alt for term in ["#", "//", "/*", "docstring"]):
                score += 0.1
            
            # Security considerations
            if any(term in alt.lower() for term in ["sanitize", "validate", "security"]):
                score += 0.2
                
            # Testing
            if any(term in alt.lower() for term in ["test", "unittest", "pytest"]):
                score += 0.15
                
            scores.append(score)
        
        return scores.index(max(scores))

# Use custom evaluator
session = ThinkThreadSession(
    llm_client=client,
    evaluation_strategy=CodeQualityEvaluator()
)

code_solution = session.run("Write a Python function to process user input")
```

### Custom Prompt Templates

```python
# Create directory: custom_prompts/
# File: custom_prompts/initial_prompt.j2
"""
You are a senior {{ role }} with expertise in {{ domain }}.

Context: {{ context }}

Question: {{ question }}

Provide a detailed answer that includes:
1. Best practices
2. Common pitfalls to avoid
3. Real-world considerations
4. Example implementation

Answer:
"""

# Python code
from thinkthread.prompting import TemplateManager
from thinkthread import ThinkThreadSession

template_manager = TemplateManager(template_dir="custom_prompts")

session = ThinkThreadSession(
    llm_client=client,
    template_manager=template_manager
)

# Use with custom context
answer = session.run(
    "How to implement authentication?",
    context={
        "role": "security engineer",
        "domain": "web application security",
        "context": "Building a financial services app"
    }
)
```

### Chaining Multiple Sessions

```python
from thinkthread import ThinkThreadSession, TreeThinker

# First session: Generate ideas
idea_session = ThinkThreadSession(
    llm_client=client,
    alternatives=5,
    rounds=2
)

ideas = idea_session.run("Innovative features for a fitness app")

# Second session: Evaluate feasibility
eval_session = ThinkThreadSession(
    llm_client=client,
    alternatives=3,
    rounds=1
)

feasibility = eval_session.run(f"""
Evaluate the technical feasibility of these features:
{ideas}

Consider: complexity, time to implement, required resources
""")

# Tree search: Implementation plan
thinker = TreeThinker(llm_client=client)
implementation = thinker.solve(f"""
Create implementation plan for the most feasible features:
{feasibility}
""")
```

## Production Patterns

### Error Handling and Retry Logic

```python
from thinkthread import ThinkThreadSession
from thinkthread.llm import LLMException
import time
import logging

logger = logging.getLogger(__name__)

class RobustSession:
    def __init__(self, session, max_retries=3):
        self.session = session
        self.max_retries = max_retries
    
    def run_with_retry(self, question):
        for attempt in range(self.max_retries):
            try:
                return self.session.run(question)
            except LLMException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
# Usage
session = ThinkThreadSession(llm_client=client)
robust_session = RobustSession(session)

try:
    answer = robust_session.run_with_retry("Complex question")
except LLMException:
    logger.error("All retry attempts failed")
    # Fallback logic
```

### Caching Layer

```python
from thinkthread import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig
import hashlib
import json
import redis

class CachedSession:
    def __init__(self, session, redis_client, ttl=3600):
        self.session = session
        self.cache = redis_client
        self.ttl = ttl
    
    def _cache_key(self, question, params):
        # Create deterministic cache key
        content = f"{question}:{json.dumps(params, sort_keys=True)}"
        return f"thinkthread:{hashlib.md5(content.encode()).hexdigest()}"
    
    def run(self, question, **kwargs):
        cache_key = self._cache_key(question, kwargs)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached.decode('utf-8')
        
        # Generate answer
        answer = self.session.run(question)
        
        # Cache result
        self.cache.setex(cache_key, self.ttl, answer)
        
        return answer

# Usage
redis_client = redis.Redis(host='localhost', port=6379)
session = ThinkThreadSession(llm_client=client)
cached_session = CachedSession(session, redis_client)

# First call: generates answer
answer1 = cached_session.run("Explain microservices")

# Second call: from cache
answer2 = cached_session.run("Explain microservices")
```

### Monitoring and Observability

```python
from thinkthread import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig
from thinkthread.monitoring import GLOBAL_MONITOR
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SessionMetrics:
    total_time: float
    rounds_completed: int
    alternatives_generated: int
    api_calls: int
    cache_hits: int = 0
    
class MonitoredSession:
    def __init__(self, session):
        self.session = session
        self.metrics: Dict[str, Any] = {}
    
    def run(self, question: str) -> tuple[str, SessionMetrics]:
        start_time = time.time()
        
        # Enable monitoring
        GLOBAL_MONITOR.enable(True)
        GLOBAL_MONITOR.reset()
        
        # Run session
        answer = self.session.run(question)
        
        # Collect metrics
        monitor_metrics = GLOBAL_MONITOR.get_metrics()
        
        metrics = SessionMetrics(
            total_time=time.time() - start_time,
            rounds_completed=self.session.rounds,
            alternatives_generated=self.session.alternatives * self.session.rounds,
            api_calls=monitor_metrics.get('api_calls', 0),
            cache_hits=monitor_metrics.get('cache_hits', 0)
        )
        
        return answer, metrics

# Usage
config = ThinkThreadConfig(enable_monitoring=True)
session = ThinkThreadSession(llm_client=client, config=config)
monitored = MonitoredSession(session)

answer, metrics = monitored.run("How to scale a database?")
print(f"Generated in {metrics.total_time:.2f}s with {metrics.api_calls} API calls")
```

### Rate Limiting

```python
from thinkthread import ThinkThreadSession
import threading
import time
from collections import deque

class RateLimitedSession:
    def __init__(self, session, calls_per_minute=20):
        self.session = session
        self.calls_per_minute = calls_per_minute
        self.call_times = deque()
        self.lock = threading.Lock()
    
    def _wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            while self.call_times and self.call_times[0] < now - 60:
                self.call_times.popleft()
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.call_times.append(now)
    
    def run(self, question):
        self._wait_if_needed()
        return self.session.run(question)

# Usage
session = ThinkThreadSession(llm_client=client)
rate_limited = RateLimitedSession(session, calls_per_minute=20)

# Process many questions without hitting rate limits
questions = ["Q1", "Q2", "Q3", ...]
for q in questions:
    answer = rate_limited.run(q)
```

## Domain-Specific Applications

### Code Review Assistant

```python
from thinkthread import ThinkThreadSession, reason
from thinkthread.config import ThinkThreadConfig

def review_code(code: str, language: str = "python") -> str:
    """AI-powered code review."""
    
    review_prompt = f"""
    Review this {language} code for:
    1. Bugs and potential issues
    2. Performance problems
    3. Security vulnerabilities
    4. Code style and best practices
    5. Suggestions for improvement
    
    Code:
    ```{language}
    {code}
    ```
    """
    
    # Use more alternatives for thorough review
    return reason(review_prompt, alternatives=5, rounds=3)

# Example usage
code_to_review = """
def process_user_input(user_data):
    query = "SELECT * FROM users WHERE name = '" + user_data + "'"
    result = db.execute(query)
    return result
"""

review = review_code(code_to_review)
print(review)
# Will identify SQL injection vulnerability and suggest parameterized queries
```

### Content Generation Pipeline

```python
from thinkthread import reason, refine, debate

class ContentPipeline:
    def __init__(self):
        self.test_mode = False
    
    def generate_article(self, topic: str) -> dict:
        """Generate a complete article with multiple perspectives."""
        
        # 1. Generate initial outline
        outline = reason(
            f"Create a detailed outline for an article about: {topic}",
            alternatives=4,
            rounds=2
        )
        
        # 2. Generate content
        content = reason(
            f"Write an article based on this outline:\n{outline}",
            alternatives=3,
            rounds=2
        )
        
        # 3. Add balanced perspective
        balanced = debate(
            f"What are different viewpoints on: {topic}"
        )
        
        # 4. Refine and polish
        final = refine(
            content,
            "Make it more engaging and add the balanced perspectives"
        )
        
        return {
            "outline": outline,
            "content": final,
            "perspectives": balanced
        }

# Usage
pipeline = ContentPipeline()
article = pipeline.generate_article("The future of remote work")
```

### Technical Documentation Generator

```python
from thinkthread import ThinkThreadSession
from thinkthread.config import ThinkThreadConfig

class DocGenerator:
    def __init__(self, llm_client):
        config = ThinkThreadConfig(
            alternatives=4,
            rounds=2,
            use_adaptive_temperature=True,
            initial_temperature=0.3,  # More factual
            generation_temperature=0.5
        )
        self.session = ThinkThreadSession(
            llm_client=llm_client,
            config=config
        )
    
    def generate_api_docs(self, code: str) -> str:
        """Generate API documentation from code."""
        
        prompt = f"""
        Generate comprehensive API documentation for this code:
        
        {code}
        
        Include:
        1. Function/class descriptions
        2. Parameter types and descriptions
        3. Return values
        4. Example usage
        5. Error cases
        """
        
        return self.session.run(prompt)
    
    def generate_readme(self, project_info: dict) -> str:
        """Generate README.md for a project."""
        
        prompt = f"""
        Create a professional README.md for:
        
        Project: {project_info['name']}
        Description: {project_info['description']}
        Language: {project_info['language']}
        Dependencies: {project_info.get('dependencies', [])}
        
        Include all standard sections.
        """
        
        return self.session.run(prompt)
```

### Decision Support System

```python
from thinkthread import TreeThinker, debate
from typing import List, Dict

class DecisionSupport:
    def __init__(self, llm_client):
        self.thinker = TreeThinker(
            llm_client=llm_client,
            max_tree_depth=4,
            branching_factor=3
        )
    
    def analyze_decision(self, 
                        decision: str, 
                        criteria: List[str],
                        constraints: List[str]) -> Dict[str, Any]:
        """Comprehensive decision analysis."""
        
        # 1. Explore all options
        options = self.thinker.solve(
            f"""
            Decision: {decision}
            
            Generate and evaluate all possible options considering:
            Criteria: {', '.join(criteria)}
            Constraints: {', '.join(constraints)}
            """,
            beam_width=3,
            max_iterations=5
        )
        
        # 2. Get balanced analysis
        analysis = debate(decision)
        
        # 3. Risk assessment
        risks = self.thinker.solve(
            f"What are the risks for each option in: {options}",
            beam_width=2,
            max_iterations=3
        )
        
        return {
            "options": options,
            "analysis": analysis,
            "risks": risks,
            "recommendation": self._synthesize_recommendation(options, analysis, risks)
        }
    
    def _synthesize_recommendation(self, options, analysis, risks):
        return self.thinker.solve(
            f"""
            Based on:
            Options: {options}
            Analysis: {analysis}
            Risks: {risks}
            
            Provide a final recommendation with rationale.
            """,
            beam_width=1,
            max_iterations=2
        )

# Usage
support = DecisionSupport(llm_client)
result = support.analyze_decision(
    decision="Should we rewrite our monolith as microservices?",
    criteria=["performance", "maintainability", "cost", "team expertise"],
    constraints=["6 month timeline", "$500k budget", "maintain uptime"]
)
```

These examples demonstrate the flexibility and power of the ThinkThread SDK across various use cases. Start with simple examples and gradually incorporate more advanced features as needed.
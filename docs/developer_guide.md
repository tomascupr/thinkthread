# ThinkThread Developer Guide

This guide explains the architecture of ThinkThread and how to extend it with new providers, reasoning approaches, or evaluation strategies.

## Architecture Overview

ThinkThread uses a hybrid architecture that combines a beautiful, simple API with the battle-tested robustness of the Chain-of-Recursive-Thoughts (CoRT) SDK.

### Key Components

1. **New API Layer** (`thinkthread/`)
   - **reason.py**: Main entry point with functions like `reason()`, `explore()`, `solve()`
   - **modes/**: Reasoning mode interfaces (currently placeholders)
   - **cli.py**: New simplified CLI with commands like `think`, `quick`, `fix`

2. **Adapter Layer** (`thinkthread/core/adapter.py`)
   - Bridges the new API to the old SDK
   - Translates simple function calls to complex SDK operations
   - Maintains all production features (retry, caching, monitoring)

3. **Core SDK** (`thinkthread/core/`)
   - **session.py**: Chain-of-Recursive-Thoughts implementation
   - **tree_thinker.py**: Tree-of-Thoughts implementation
   - **evaluation.py**: Answer evaluation strategies
   - **monitoring.py**: Performance tracking
   - **config.py**: Configuration management

4. **LLM Clients** (`thinkthread/llm/`)
   - **base.py**: Abstract LLM interface with caching
   - **openai_client.py**: OpenAI integration
   - **anthropic_client.py**: Anthropic integration
   - **dummy.py**: Test mode implementation

### How It Works

```
User Code
    ↓
reason("How to scale?")
    ↓
ReasoningEngine (new API)
    ↓
SDKAdapter (translator)
    ↓
ThinkThreadSession/TreeThinker (proven SDK)
    ↓
LLMClient (with retry/cache)
    ↓
OpenAI/Anthropic API
```

## The Integration Architecture

### Why Hybrid?

During development, we discovered that:
- The new API had better developer experience
- The old SDK had better production features (retry logic, caching, monitoring)
- Best solution: Keep both! New API powered by old engine

### Key Integration Points

1. **Test Mode Switching**
   ```python
   # In reason.py - reinitializes LLM client when test_mode changes
   if kwargs.get('test_mode', False) != self.adapter.config.test_mode:
       self.adapter.config.test_mode = kwargs.get('test_mode', False)
       self.adapter._llm_client = self.adapter._initialize_llm_client()
   ```

2. **Mode Detection**
   ```python
   # In adapter.py - auto-detects reasoning mode from prompt
   def _detect_mode(self, prompt: str) -> str:
       if "explore" in prompt or "brainstorm" in prompt:
           return "explore"
       elif "debate" in prompt or "pros and cons" in prompt:
           return "debate"
       # ... etc
   ```

3. **Result Conversion**
   ```python
   # Converts old SDK results to new structured format
   def _convert_result(self, old_result, mode, metadata=None):
       return {
           "answer": str(old_result),
           "mode": mode,
           "cost": extract_cost(old_result),
           "confidence": extract_confidence(old_result),
           "metadata": metadata
       }
   ```

## Extending ThinkThread

### Adding a New Reasoning Mode

1. Create mode class in `thinkthread/modes/`:
```python
# thinkthread/modes/analyze.py
from .base import ReasoningMode

class AnalyzeMode(ReasoningMode):
    def process(self, question: str, **kwargs):
        # Your custom reasoning logic
        pass
```

2. Add to adapter in `thinkthread/core/adapter.py`:
```python
def analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
    # Custom logic using SDK components
    session = self.create_session(alternatives=4, rounds=3)
    result = session.run(f"Analyze in detail: {prompt}")
    return self._convert_result(result, mode="analyze")
```

3. Expose in API in `thinkthread/reason.py`:
```python
def analyze(question: str, **kwargs) -> ReasoningResult:
    """Deep analysis of a topic"""
    return _engine.analyze(question, **kwargs)

# Attach to reason function
reason.analyze = analyze
```

### Adding a New LLM Provider

1. Create client in `thinkthread/llm/`:
```python
# thinkthread/llm/gemini_client.py
from .base import LLMClient

class GeminiClient(LLMClient):
    def _generate_uncached(self, prompt: str, **kwargs) -> str:
        # Call Gemini API
        response = gemini.generate(prompt, **kwargs)
        return response.text
        
    async def astream(self, prompt: str, **kwargs):
        # Streaming implementation
        async for chunk in gemini.stream(prompt, **kwargs):
            yield chunk
```

2. Update adapter initialization:
```python
# In adapter.py _initialize_llm_client()
elif self.config.provider == "gemini":
    return GeminiClient(
        api_key=os.environ.get("GEMINI_API_KEY"),
        model_name=self.config.model or "gemini-pro"
    )
```

### Adding Retry Logic

The SDK already includes retry logic in `thinkthread/llm_integration.py`:

```python
class RetryableLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        for attempt in range(self.max_retries):
            try:
                return self.client.generate(prompt, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise
```

To customize retry behavior, modify the `RetryableLLMClient` class.

### Custom Evaluation Strategies

1. Create new strategy:
```python
# thinkthread/core/custom_evaluation.py
from .evaluation import EvaluationStrategy

class CustomEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, question, answers, llm_client, template_manager):
        # Your evaluation logic
        # Return index of best answer
        return best_index
```

2. Use in session creation:
```python
# In adapter.py
session = ThinkThreadSession(
    llm_client=self._llm_client,
    evaluation_strategy=CustomEvaluationStrategy(),
    config=config
)
```

## Testing and Development

### Test Mode

Test mode uses `DummyLLMClient` to avoid API calls:

```python
# Python
from thinkthread import reason
result = reason("test question", test_mode=True)

# CLI
think --test "test question"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dummy_llm.py

# Run with coverage
pytest --cov=thinkthread
```

### Debugging Integration Issues

1. Check LLM client initialization:
```python
# Add debug prints in adapter._initialize_llm_client()
print(f"Test mode: {self.config.test_mode}")
print(f"Provider: {self.config.provider}")
```

2. Monitor mode detection:
```python
# In adapter.reason()
print(f"Detected mode: {mode}")
```

3. Verify result conversion:
```python
# In adapter._convert_result()
print(f"Converting result: {old_result}")
```

## Performance Optimization

### Caching

The LLM base client includes built-in caching:

```python
# Enable caching
client.enable_cache(True)

# Enable semantic caching (finds similar prompts)
client.enable_semantic_cache(True, similarity_threshold=0.95)
```

### Concurrency

Set concurrency limits to avoid rate limiting:

```python
# Limit concurrent API calls
client.set_concurrency_limit(5)
```

### Monitoring

Access performance metrics:

```python
from thinkthread.core.monitoring import GLOBAL_MONITOR

# Get metrics
metrics = GLOBAL_MONITOR.get_summary()
print(f"Total time: {metrics['total_time']}")
print(f"LLM calls: {metrics['llm_calls']}")
```

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='...'

# Model Selection
export OPENAI_MODEL='gpt-4-turbo-preview'
export ANTHROPIC_MODEL='claude-3-opus-20240229'

# Custom Settings
export THINKTHREAD_MAX_RETRIES=5
export THINKTHREAD_CACHE_TTL=3600
```

### Programmatic Configuration

```python
from thinkthread import reason

# Configure globally
reason.set_budget(daily=10.00, per_query=0.50)
reason.enable_memory()

# Per-call configuration
answer = reason(
    "Question",
    provider="anthropic",
    model="claude-3-opus",
    max_cost=0.10,
    alternatives=5,
    rounds=3
)
```

## Contributing

### Code Style

- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Type hints required for public APIs
- Docstrings for all public functions

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Update documentation
5. Submit PR with clear description

### Adding Examples

Add examples to `examples/` directory:

```python
# examples/custom_mode_example.py
"""Example of using custom reasoning mode"""

from thinkthread import reason

# Your example code
result = reason.analyze("Complex topic")
print(result)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure package is installed: `pip install -e .`
   - Check Python path includes project root

2. **API Key Errors**
   - Verify environment variables are set
   - Check key permissions and quotas

3. **Test Mode Not Working**
   - Ensure test_mode=True is passed
   - Check adapter is reinitializing LLM client

4. **Performance Issues**
   - Enable caching to reduce API calls
   - Use test mode during development
   - Monitor costs with result.cost

### Getting Help

- Check [examples/](../examples/) for working code
- Read test files for usage patterns
- Open an issue on GitHub
- Join our Discord community

## Future Enhancements

Planned improvements:
- Streaming support in new API
- Websocket live view for reasoning process
- Plugin system for custom modes
- Better cost prediction
- Reasoning explanation UI
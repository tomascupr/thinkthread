# ThinkThread Documentation 🧵

**Make your AI think before it speaks.**

Welcome to ThinkThread - a production-ready reasoning engine that combines a beautiful, simple API with battle-tested robustness from the Chain-of-Recursive-Thoughts (CoRT) SDK.

## What is ThinkThread?

ThinkThread makes your AI actually think through problems instead of giving you its first response. It uses the proven Chain-of-Recursive-Thoughts (CoRT) technique under the hood, wrapped in a developer-friendly API.

### How It Works

1. **Initial Reasoning**: Generates a thoughtful initial answer
2. **Exploration**: Creates multiple alternative approaches
3. **Evaluation**: Compares all answers using sophisticated strategies
4. **Refinement**: Iteratively improves the best answer
5. **Delivery**: Returns a well-reasoned, high-confidence response

### Architecture

ThinkThread uses a hybrid architecture that combines:
- **Beautiful API**: Simple functions like `reason()`, `explore()`, `debate()`
- **Robust Engine**: Battle-tested CoRT SDK with retry logic, caching, and monitoring
- **Smart Adapter**: Seamlessly bridges the new API to the proven SDK implementation

## Key Features

### Simple API
- **Zero Config**: Works out of the box with just `reason("your question")`
- **Multiple Modes**: `explore()`, `solve()`, `debate()`, `refine()` for different thinking styles
- **Test Mode**: Development without API calls using `test_mode=True`

### Production Ready
- **Automatic Retry**: Exponential backoff for transient failures
- **Smart Caching**: Reduces costs by reusing previous results
- **Performance Monitoring**: Built-in metrics and profiling
- **Error Handling**: Never crashes, always returns usable results

### Advanced Capabilities
- **Multiple LLM Providers**: OpenAI, Anthropic, HuggingFace, and more
- **Async Support**: Non-blocking operations for web applications
- **Streaming**: Real-time token-by-token output
- **Custom Templates**: Jinja2-based prompt customization
- **Evaluation Strategies**: Self-evaluation, pairwise comparison, and more

## Quick Start

```python
from thinkthread import reason

# That's it! AI now thinks before answering
answer = reason("How can we reduce technical debt?")
print(answer)
```

## Documentation

- [Usage Guide](usage_guide.md): Examples and best practices
- [CLI Reference](cli_reference.md): Command-line interface
- [Developer Guide](developer_guide.md): Architecture and customization
- [Configuration Reference](configuration_reference.md): All settings explained
- [Performance Guide](performance_optimization.md): Speed and cost optimization

## Common Use Cases

1. **Problem Solving**: `solve("Our API is slow")`
2. **Brainstorming**: `explore("New product features")`
3. **Decision Making**: `debate("Microservices vs monolith?")`
4. **Content Improvement**: `refine("Make this email better", initial_text)`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.

# ThinkThread SDK Documentation

Welcome to the documentation for the Chain-of-Recursive-Thoughts (CoRT) SDK.

## What is CoRT?

Chain-of-Recursive-Thoughts is a technique that improves the quality of answers from large language models through a recursive self-refinement process:

1. Generate an initial answer to a question
2. For each refinement round:
   - Generate alternative answers
   - Evaluate all answers (current and alternatives)
   - Select the best answer for the next round
3. Return the final selected answer

This process enables the model to critically examine its own responses, consider alternative perspectives, and ultimately produce higher-quality answers.

## Key Features

- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and HuggingFace models
- **Prompt Templating**: Customisable Jinja2 templates for all prompting needs
- **Recursive Reasoning**: Multi-round refinement process for improved answers
- **Self-Evaluation**: Ability to evaluate answer quality without external criteria
- **Pairwise Evaluation**: Compare answers head-to-head for better selection
- **Asynchronous Support**: Non-blocking API for integration with async applications
- **Streaming Responses**: Real-time token-by-token output for better user experience
- **Extensible Architecture**: Easily add new providers or evaluation strategies

## Getting Started

- [Installation and Quickstart](../README.md)
- [Usage Guide](usage_guide.md): Detailed instructions for using the SDK
- [Developer Guide](developer_guide.md): Architecture and extension information
- [Configuration Reference](configuration_reference.md): All configuration options
- [CLI Reference](cli_reference.md): Command-line interface documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.

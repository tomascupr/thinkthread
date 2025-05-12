# ThinkThread SDK CLI Reference

This document provides detailed information about the Command Line Interface (CLI) provided by the ThinkThread SDK.

## Overview

The ThinkThread SDK CLI allows you to interact with the Chain-of-Recursive-Thoughts reasoning functionality from the command line. It's implemented using the Typer library.

## Basic Usage

```bash
# Run using the Python module
python -m thinkthread_sdk [COMMAND] [OPTIONS]

# Or using the installed entry point
thinkthread [COMMAND] [OPTIONS]
```

## Commands

### `version`

Display the current version of the ThinkThread SDK.

```bash
thinkthread version
```

### `ask`

Ask a question and get an answer using ThinkThread reasoning.

```bash
thinkthread ask "What is the meaning of life?"
```

Options:
- `--provider TEXT`: LLM provider to use (openai, anthropic, hf, dummy) [default: openai]
- `--model TEXT`: Model name to use (provider-specific)
- `--alternatives INTEGER`: Number of alternative answers per round [default: 3]
- `--rounds INTEGER`: Number of refinement rounds [default: 2]
- `--stream / --no-stream`: Stream the final answer as it's generated [default: stream]

### `run`

Run recursive reasoning on a question and get a refined answer. This is similar to `ask` but with more options.

```bash
thinkthread run "What is the most effective way to combat climate change?"
```

Options:
- `--provider TEXT`: LLM provider to use (openai, anthropic, hf, dummy) [default: openai]
- `--model TEXT`: Model name to use (provider-specific)
- `--alternatives INTEGER`: Number of alternative answers per round [default: 3]
- `--rounds INTEGER`: Number of refinement rounds [default: 2]
- `--stream / --no-stream`: Stream the final answer as it's generated [default: no-stream]
- `--verbose / --no-verbose`: Enable debug logging [default: no-verbose]
- `--self-evaluation / --no-self-evaluation`: Toggle self-evaluation on/off [default: no-self-evaluation]

## Examples

### Basic Query

```bash
thinkthread run "Explain the concept of entropy in thermodynamics"
```

### Using a Different Provider

```bash
thinkthread run "What are the implications of quantum computing for cryptography?" --provider anthropic
```

### Increasing Reasoning Rounds and Alternatives

```bash
thinkthread run "Compare different approaches to artificial general intelligence" --rounds 3 --alternatives 5
```

### Enable Streaming

```bash
thinkthread run "Explain blockchain technology" --stream
```

### Enable Verbose Logging

```bash
thinkthread run "Describe the water cycle" --verbose
```

### Enable Self-Evaluation

```bash
thinkthread run "What are the ethical considerations of genetic engineering?" --self-evaluation
```

## Environment Variables

The CLI respects the same environment variables as the rest of the ThinkThread SDK:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `HF_API_TOKEN`: HuggingFace API token
- `PROVIDER`: Default LLM provider
- `OPENAI_MODEL`: Default OpenAI model
- `ANTHROPIC_MODEL`: Default Anthropic model
- `HF_MODEL`: Default HuggingFace model
- `ALTERNATIVES`: Default number of alternatives
- `ROUNDS`: Default number of rounds
- `MAX_ROUNDS`: Default maximum number of rounds
- `USE_PAIRWISE_EVALUATION`: Whether to use pairwise evaluation
- `USE_SELF_EVALUATION`: Whether to use self-evaluation
- `PROMPT_DIR`: Directory for custom prompt templates

For more details on configuration, see the [Configuration Reference](configuration_reference.md).

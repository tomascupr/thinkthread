# ThinkThread SDK CLI Reference

This document provides detailed information about the Command Line Interface (CLI) provided by the ThinkThread SDK.

## Overview

The ThinkThread SDK CLI allows you to interact with both Chain-of-Recursive-Thoughts (CoRT) and Tree-of-Thoughts (ToT) reasoning functionality from the command line. It's implemented using the Typer library.

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

Ask a question and get an answer using ThinkThread reasoning (CoRT approach).

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

Run Chain-of-Recursive-Thoughts reasoning on a question and get a refined answer. This is similar to `ask` but with more options.

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

### `tot`

Run Tree-of-Thoughts reasoning on a question and explore multiple reasoning paths in parallel.

```bash
thinkthread tot "What is the most effective way to combat climate change?"
```

Options:
- `--provider TEXT`: LLM provider to use (openai, anthropic, hf, dummy) [default: openai]
- `--model TEXT`: Model name to use (provider-specific)
- `--beam-width INTEGER`: Number of parallel thought threads to maintain [default: 3]
- `--max-tree-depth INTEGER`: Maximum depth of the thinking tree [default: 3]
- `--branching-factor INTEGER`: Number of branches to generate per node [default: 3]
- `--max-iterations INTEGER`: Maximum number of tree expansion iterations [default: 3]
- `--stream / --no-stream`: Stream the final answer as it's generated [default: no-stream]
- `--verbose / --no-verbose`: Enable debug logging [default: no-verbose]

### `think`

Unified command that supports both CoRT and ToT reasoning approaches.

```bash
thinkthread think "What is the most effective way to combat climate change?" --approach tot
```

Options:
- `--approach TEXT`: Reasoning approach to use (cort, tot) [default: cort]
- `--provider TEXT`: LLM provider to use (openai, anthropic, hf, dummy) [default: openai]
- `--model TEXT`: Model name to use (provider-specific)
- `--stream / --no-stream`: Stream the final answer as it's generated [default: no-stream]
- `--verbose / --no-verbose`: Enable debug logging [default: no-verbose]

CoRT-specific options (when using --approach cort):
- `--alternatives INTEGER`: Number of alternative answers per round [default: 3]
- `--rounds INTEGER`: Number of refinement rounds [default: 2]
- `--self-evaluation / --no-self-evaluation`: Toggle self-evaluation on/off [default: no-self-evaluation]

ToT-specific options (when using --approach tot):
- `--beam-width INTEGER`: Number of parallel thought threads to maintain [default: 3]
- `--max-tree-depth INTEGER`: Maximum depth of the thinking tree [default: 3]
- `--branching-factor INTEGER`: Number of branches to generate per node [default: 3]
- `--max-iterations INTEGER`: Maximum number of tree expansion iterations [default: 3]

## Examples

### Basic Chain-of-Recursive-Thoughts Query

```bash
thinkthread run "Explain the concept of entropy in thermodynamics"
```

### Basic Tree-of-Thoughts Query

```bash
thinkthread tot "Develop a strategy for solving the Tower of Hanoi puzzle"
```

### Using the Unified Interface

```bash
# Using CoRT approach
thinkthread think "Explain quantum computing" --approach cort --rounds 3

# Using ToT approach
thinkthread think "Solve this logic puzzle: A man has to get a fox, a chicken, and a sack of corn across a river" --approach tot --beam-width 5
```

### Using a Different Provider

```bash
thinkthread think "What are the implications of quantum computing for cryptography?" --provider anthropic
```

### Increasing Reasoning Depth

```bash
# For CoRT: more rounds and alternatives
thinkthread run "Compare different approaches to artificial general intelligence" --rounds 3 --alternatives 5

# For ToT: wider beam and deeper tree
thinkthread tot "Design an algorithm to solve the traveling salesman problem" --beam-width 4 --max-tree-depth 4
```

### Enable Streaming

```bash
thinkthread think "Explain blockchain technology" --stream
```

### Enable Verbose Logging

```bash
thinkthread think "Describe the water cycle" --verbose
```

## Environment Variables

The CLI respects the same environment variables as the rest of the ThinkThread SDK:

### Common Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `HF_API_TOKEN`: HuggingFace API token
- `PROVIDER`: Default LLM provider
- `OPENAI_MODEL`: Default OpenAI model
- `ANTHROPIC_MODEL`: Default Anthropic model
- `HF_MODEL`: Default HuggingFace model
- `PROMPT_DIR`: Directory for custom prompt templates
- `PARALLEL_ALTERNATIVES`: Whether to generate alternatives in parallel

### CoRT-Specific Environment Variables

- `ALTERNATIVES`: Default number of alternatives
- `ROUNDS`: Default number of rounds
- `MAX_ROUNDS`: Default maximum number of rounds
- `USE_PAIRWISE_EVALUATION`: Whether to use pairwise evaluation
- `USE_SELF_EVALUATION`: Whether to use self-evaluation

### ToT-Specific Environment Variables

- `BEAM_WIDTH`: Default beam width
- `MAX_TREE_DEPTH`: Default maximum tree depth
- `BRANCHING_FACTOR`: Default branching factor
- `MAX_ITERATIONS`: Default maximum iterations
- `SIMILARITY_THRESHOLD`: Default similarity threshold for pruning

For more details on configuration, see the [Configuration Reference](configuration_reference.md).

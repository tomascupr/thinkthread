# ThinkThread SDK Examples

This directory contains example scripts demonstrating key capabilities of the ThinkThread SDK.

## Overview

These examples showcase different reasoning strategies and LLM providers:

1. **Basic Q&A (Local Model)** - Simple question answering using a local HuggingFace model with fallback mechanism
2. **Recursive Refinement (OpenAI)** - Chain-of-Recursive-Thoughts (CoRT) reasoning with GPT models and tiered fallbacks
3. **Creative Ideation (Anthropic)** - Tree-of-Thoughts (ToT) reasoning for creative problems with Claude models and tiered fallbacks

All examples include robust fallback mechanisms to ensure they complete successfully even when primary models are unavailable or too slow.

## API Key Setup

The examples use placeholder API keys in the code. To run them successfully:

1. **Set environment variables** (recommended):
   - `HF_API_TOKEN` for HuggingFace examples
   - `OPENAI_API_KEY` for OpenAI examples
   - `ANTHROPIC_API_KEY` for Anthropic examples

2. **Or replace placeholders in the code**:
   - Replace `os.environ.get("HF_API_TOKEN", "")` with your actual HuggingFace API token
   - Replace `os.environ.get("OPENAI_API_KEY", "")` with your actual OpenAI API key
   - Replace `os.environ.get("ANTHROPIC_API_KEY", "")` with your actual Anthropic API key

**Note:** Never commit API keys to the repository.

## Requirements

These examples are designed to run on CPU-only setups with minimal requirements:

- Python 3.7+
- ThinkThread SDK dependencies
- For local models: Enough RAM to load small HuggingFace models (~500MB)

## Running the Examples

### Basic Q&A (Local Model)

```bash
python examples/basic_qa_local.py
```

This example:
- Uses a small HuggingFace model (distilgpt2) that runs efficiently on CPU
- Falls back to a DummyLLMClient if the primary model times out or encounters an error
- Answers a basic question about machine learning

### Recursive Refinement (OpenAI)

```bash
python examples/recursive_refinement_openai.py
```

This example:
- Uses ThinkThreadSession with GPT-4 for high-quality results
- Falls back to GPT-3.5-Turbo if approaching the time limit
- Falls back to DummyLLMClient as a last resort
- Demonstrates the Chain-of-Recursive-Thoughts approach
- Explains a concept with progressive refinement

### Creative Ideation (Anthropic)

```bash
python examples/creative_ideation_anthropic.py
```

This example:
- Uses TreeThinker with Claude-2 for creative problem-solving
- Falls back to Claude-instant-1 if approaching the time limit
- Falls back to DummyLLMClient as a last resort
- Demonstrates the Tree-of-Thoughts approach for divergent thinking
- Generates innovative product ideas

## Performance Optimization

All examples are configured to run efficiently on CPU within a 3-minute timeframe by:

- Using smaller models when necessary
- Implementing timeout-based fallbacks
- Configuring performance options like parallel processing and caching
- Limiting the model parameters (max tokens, tree depth, etc.)
- Using tiered fallbacks to ensure successful completion

## Customization

You can customize these examples by:

- Adjusting timeout parameters for your hardware
- Changing the questions/problems
- Modifying the model parameters
- Adding additional fallback tiers
- Integrating with your own LLM providers

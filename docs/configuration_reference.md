# ThinkThread SDK Configuration Reference

This document provides a comprehensive reference of all configuration options available in the ThinkThread SDK.

## Configuration Model

The ThinkThread SDK uses Pydantic for configuration management through the `CoRTConfig` class.

## Configuration Options

| Option | Type | Default | Environment Variable | Description |
|--------|------|---------|---------------------|-------------|
| `openai_api_key` | `Optional[str]` | `None` | `OPENAI_API_KEY` | OpenAI API key for using OpenAI models |
| `anthropic_api_key` | `Optional[str]` | `None` | `ANTHROPIC_API_KEY` | Anthropic API key for using Claude models |
| `hf_api_token` | `Optional[str]` | `None` | `HF_API_TOKEN` | HuggingFace API token for using HuggingFace models |
| `provider` | `str` | `"openai"` | `PROVIDER` | Default LLM provider to use (openai, anthropic, hf) |
| `openai_model` | `str` | `"gpt-4"` | `OPENAI_MODEL` | Default model name for OpenAI provider |
| `anthropic_model` | `str` | `"claude-2"` | `ANTHROPIC_MODEL` | Default model name for Anthropic provider |
| `hf_model` | `str` | `"gpt2"` | `HF_MODEL` | Default model name for HuggingFace provider |
| `alternatives` | `int` | `3` | `ALTERNATIVES` | Number of alternative answers to generate per round |
| `rounds` | `int` | `2` | `ROUNDS` | Number of refinement rounds |
| `max_rounds` | `int` | `3` | `MAX_ROUNDS` | Maximum number of refinement rounds |
| `use_pairwise_evaluation` | `bool` | `True` | `USE_PAIRWISE_EVALUATION` | Whether to use pairwise evaluation |
| `use_self_evaluation` | `bool` | `False` | `USE_SELF_EVALUATION` | Whether to use self-evaluation |
| `prompt_dir` | `Optional[str]` | `None` | `PROMPT_DIR` | Directory for custom prompt templates |

## Configuration Methods

### Environment Variables

The simplest way to configure the ThinkThread SDK is through environment variables:

```bash
# Set environment variables
export OPENAI_API_KEY=your-openai-api-key
export PROVIDER=openai
export OPENAI_MODEL=gpt-4
export ALTERNATIVES=5
export ROUNDS=3

# Run CoRT
thinkthread run "Your question here"
```

### .env File

You can also use a `.env` file in your project directory:

```
# .env file
OPENAI_API_KEY=your-openai-api-key
PROVIDER=openai
OPENAI_MODEL=gpt-4
ALTERNATIVES=5
ROUNDS=3
```

The ThinkThread SDK will automatically load this file when creating the configuration.

### Programmatic Configuration

For more control, create a configuration object programmatically:

```python
from thinkthread_sdk.config import CoRTConfig, create_config

# Use environment variables and .env file with overrides
config = create_config()

# Or create a custom configuration
custom_config = CoRTConfig(
    openai_api_key="your-api-key",
    provider="openai",
    openai_model="gpt-4",
    alternatives=5,
    rounds=3,
    use_pairwise_evaluation=True,
)

# Use with CoRTSession
from thinkthread_sdk.cort_session import CoRTSession
from thinkthread_sdk.llm import OpenAIClient

client = OpenAIClient(api_key=custom_config.openai_api_key, model_name=custom_config.openai_model)
session = CoRTSession(llm_client=client, config=custom_config)
```

## Configuration Validation

The `CoRTConfig` class includes validation for numeric fields:

```python
@field_validator("alternatives", "rounds", mode="before")
@classmethod
def validate_int_fields(cls, v):
    """Validate integer fields."""
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            raise ValueError(f"Value must be a valid integer, got {v}")
    return v
```

This ensures that environment variables (which are strings) are properly converted to integers for the appropriate fields.

## Best Practices

1. **API Keys**: Never hardcode API keys in your code. Use environment variables or a `.env` file.

2. **Environment-Specific Configurations**: Use different configurations for development, testing, and production:

   ```python
   # Development
   dev_config = create_config(".env.development")
   
   # Testing
   test_config = create_config(".env.testing")
   
   # Production
   prod_config = create_config(".env.production")
   ```

3. **Temperature Settings**: Use higher temperature (e.g., 0.9) for generating diverse alternatives and lower temperature (e.g., 0.2) for evaluations to get more deterministic results.

4. **Model Selection**: Use more capable models (e.g., GPT-4, Claude-2) for better results, especially for the evaluation steps.

5. **Rounds and Alternatives**: Start with default values (rounds=2, alternatives=3) and adjust based on your specific needs and performance requirements.

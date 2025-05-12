# CoRT SDK Developer Guide

This guide explains the architecture of the CoRT SDK and how to extend it with new providers or evaluation strategies.

## Architecture Overview

The CoRT SDK follows a modular architecture with these key components:

1. **CoRTSession**: The main orchestrator that manages the recursive reasoning process
2. **LLMClient**: Abstract interface for different LLM providers
3. **TemplateManager**: Manages Jinja2 templates for prompt generation
4. **Evaluation**: Strategies for evaluating and comparing answers
5. **Config**: Configuration management via Pydantic

### Chain-of-Recursive-Thoughts Loop

The core of the SDK is the Chain-of-Recursive-Thoughts loop, implemented in `CoRTSession.run()`:

1. Generate an initial answer using the configured LLM
2. For each round:
   - Generate multiple alternative answers
   - Evaluate all answers (current best and alternatives)
   - Select the best answer for the next round
3. Return the final answer after all rounds

This process enables the LLM to critically evaluate its own output and iteratively improve the quality of answers.

## Component Interfaces

### LLM Provider Interface

All LLM providers implement the `LLMClient` abstract base class:

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the language model."""
        pass
        
    @abstractmethod
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Asynchronously stream text generation from the language model."""
        pass
```

The interface includes both synchronous and asynchronous methods for text generation.

### Evaluation Interface

Evaluation strategies implement either the `EvaluationStrategy` interface (for comparing multiple answers) or the `Evaluator` interface (for pairwise comparison):

```python
class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(
        self,
        question: str,
        answers: List[str],
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> int:
        """Evaluate answers and return the index of the best one."""
        pass

class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        question: str,
        prev_answer: str,
        new_answer: str,
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> bool:
        """Evaluate whether the new answer is better than the previous."""
        pass
```

### Template Management

The `TemplateManager` class handles loading and rendering Jinja2 templates:

```python
class TemplateManager:
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        template = self.env.get_template(template_name)
        return template.render(**context)
```

## Adding a New LLM Provider

To add a new LLM provider, create a new class that implements the `LLMClient` interface:

1. Create a new file in the `cort_sdk/llm/` directory, e.g., `new_provider_client.py`
2. Implement the required methods:

```python
from typing import AsyncIterator
from cort_sdk.llm.base import LLMClient

class NewProviderClient(LLMClient):
    def __init__(self, api_key: str, model_name: str = "default-model"):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        # Initialize any provider-specific clients or configurations
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the new provider's API."""
        temperature = kwargs.get("temperature", 0.7)
        
        # Implement provider-specific API call
        # Example:
        # response = new_provider_api.generate(
        #     prompt=prompt,
        #     temperature=temperature,
        #     api_key=self.api_key,
        #     model=self.model_name,
        # )
        # return response.text
        
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Asynchronously stream text from the new provider's API."""
        temperature = kwargs.get("temperature", 0.7)
        
        # Implement provider-specific streaming API call
        # Example:
        # async for chunk in new_provider_api.stream(
        #     prompt=prompt,
        #     temperature=temperature,
        #     api_key=self.api_key,
        #     model=self.model_name,
        # ):
        #     yield chunk.text
        
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate text using the new provider's API."""
        # Implement native async completion or use the default implementation
        # which calls generate() in a thread
        return await super().acomplete(prompt, **kwargs)
        
    async def aclose(self) -> None:
        """Clean up any resources used by the client."""
        # Close any open connections, sessions, etc.
        pass
```

3. Add the new client to `cort_sdk/llm/__init__.py`:

```python
from .new_provider_client import NewProviderClient

__all__ = [
    # ... existing exports
    "NewProviderClient",
]
```

4. Update the CLI to support the new provider in `cort_sdk/cli.py`.

## Adding a New Evaluation Strategy

To add a new evaluation strategy:

1. Create a class that implements the `EvaluationStrategy` interface:

```python
from typing import List
from cort_sdk.evaluation import EvaluationStrategy
from cort_sdk.llm import LLMClient
from cort_sdk.prompting import TemplateManager

class NewEvaluationStrategy(EvaluationStrategy):
    def evaluate(
        self,
        question: str,
        answers: List[str],
        llm_client: LLMClient,
        template_manager: TemplateManager,
    ) -> int:
        """Evaluate answers using a new strategy."""
        # Implement your evaluation logic
        # Example:
        # 1. Create a prompt for evaluation
        # 2. Send it to the LLM
        # 3. Parse the response
        # 4. Return the index of the best answer
        
        # ...
        
        return best_index
```

2. Use your strategy when creating a CoRTSession:

```python
from cort_sdk.cort_session import CoRTSession
from your_module import NewEvaluationStrategy

session = CoRTSession(
    llm_client=client,
    evaluation_strategy=NewEvaluationStrategy(),
)
```

## Customising Prompt Templates

The SDK uses Jinja2 templates for all prompts. The default templates are in `cort_sdk/prompts/`, but you can provide your own:

1. Create a directory for your templates
2. Create template files with the `.j2` extension
3. Pass the directory path when creating a TemplateManager:

```python
from cort_sdk.prompting import TemplateManager
from cort_sdk.cort_session import CoRTSession

template_manager = TemplateManager(template_dir="/path/to/your/templates")
session = CoRTSession(llm_client=client, template_manager=template_manager)
```

Or set the path in configuration:

```
PROMPT_DIR=/path/to/your/templates
```

### Template Structure

Core templates that need to be implemented:

- `initial_prompt.j2`: Generates the initial answer
- `alternative_prompt.j2`: Generates alternative answers
- `evaluation_prompt.j2`: Evaluates multiple answers
- `pairwise_prompt.j2`: Compares two answers
- `final_answer.j2`: Formats the final answer

## Advanced Configuration

For advanced configuration, create a custom `CoRTConfig` instance:

```python
from cort_sdk.config import CoRTConfig
from cort_sdk.cort_session import CoRTSession

# Create custom configuration
config = CoRTConfig(
    openai_api_key="your-api-key",
    provider="openai",
    openai_model="gpt-4",
    alternatives=5,
    rounds=3,
    use_pairwise_evaluation=True,
    use_self_evaluation=False,
)

# Create session with custom config
session = CoRTSession(llm_client=client, config=config)
```

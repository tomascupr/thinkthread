# ThinkThread SDK Developer Guide

This guide explains the architecture of the ThinkThread SDK and how to extend it with new providers, reasoning approaches, or evaluation strategies.

## Architecture Overview

The ThinkThread SDK follows a modular architecture with these key components:

1. **BaseReasoner**: Abstract base class for all reasoning approaches
2. **ThinkThreadSession**: Implements Chain-of-Recursive-Thoughts (CoRT) reasoning
3. **TreeThinker**: Implements Tree-of-Thoughts (ToT) reasoning
4. **LLMClient**: Abstract interface for different LLM providers
5. **TemplateManager**: Manages Jinja2 templates for prompt generation
6. **Evaluation**: Strategies for evaluating and comparing answers
7. **Config**: Configuration management via Pydantic

### Reasoning Approaches

The SDK supports two complementary reasoning approaches:

#### Chain-of-Recursive-Thoughts Loop

The CoRT approach is implemented in `ThinkThreadSession.run()`:

1. Generate an initial answer using the configured LLM
2. For each round:
   - Generate multiple alternative answers
   - Evaluate all answers (current best and alternatives)
   - Select the best answer for the next round
3. Return the final answer after all rounds

This process enables the LLM to critically evaluate its own output and iteratively improve the quality of answers.

#### Tree-of-Thoughts Search

The ToT approach is implemented in `TreeThinker.solve()`:

1. Generate multiple initial thoughts (diverse starting points)
2. Expand each thought into multiple branches (parallel exploration)
3. Evaluate all branches and prune less promising paths (beam search)
4. Continue expanding the most promising branches
5. Select the highest-scoring solution path

This approach allows exploring multiple reasoning paths simultaneously, which is particularly effective for complex problems with multiple valid approaches.

## Component Interfaces

### BaseReasoner Interface

All reasoning approaches implement the `BaseReasoner` abstract base class:

```python
class BaseReasoner(ABC):
    def __init__(
        self,
        llm_client: LLMClient,
        template_manager: Optional[TemplateManager] = None,
        config: Optional[ThinkThreadConfig] = None,
    ):
        """Initialize the reasoner with an LLM client and optional components."""
        self.llm_client = llm_client
        self.config = config or create_config()
        self.template_manager = template_manager or TemplateManager(
            self.config.prompt_dir
        )

    @abstractmethod
    def run(self, question: str) -> str:
        """Execute the reasoning process on a question."""
        pass

    @abstractmethod
    async def run_async(self, question: str) -> str:
        """Execute the reasoning process asynchronously on a question."""
        pass
```

This interface ensures that all reasoning approaches provide both synchronous and asynchronous execution methods with a consistent API.

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

## Implementing a New Reasoning Approach

To implement a new reasoning approach, create a class that inherits from `BaseReasoner`:

```python
from thinkthread_sdk.base_reasoner import BaseReasoner
from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig

class NewReasoner(BaseReasoner):
    def __init__(
        self,
        llm_client: LLMClient,
        template_manager: Optional[TemplateManager] = None,
        config: Optional[ThinkThreadConfig] = None,
        **kwargs
    ):
        super().__init__(llm_client, template_manager, config)
        # Initialize any approach-specific attributes
        
    def run(self, question: str) -> str:
        """Execute the reasoning process on a question."""
        # Implement your reasoning approach
        # 1. Process the question
        # 2. Generate initial thoughts/answers
        # 3. Apply your reasoning algorithm
        # 4. Return the final answer
        
    async def run_async(self, question: str) -> str:
        """Execute the reasoning process asynchronously on a question."""
        # Implement the async version of your reasoning approach
        # You can use the shared utilities in reasoning_utils.py
```

## Adding a New LLM Provider

To add a new LLM provider, create a new class that implements the `LLMClient` interface:

1. Create a new file in the `thinkthread_sdk/llm/` directory, e.g., `new_provider_client.py`
2. Implement the required methods:

```python
from typing import AsyncIterator
from thinkthread_sdk.llm.base import LLMClient

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

3. Add the new client to `thinkthread_sdk/llm/__init__.py`:

```python
from .new_provider_client import NewProviderClient

__all__ = [
    # ... existing exports
    "NewProviderClient",
]
```

4. Update the CLI to support the new provider in `thinkthread_sdk/cli.py`.

## Adding a New Evaluation Strategy

To add a new evaluation strategy:

1. Create a class that implements the `EvaluationStrategy` interface:

```python
from typing import List
from thinkthread_sdk.evaluation import EvaluationStrategy
from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager

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

2. Use your strategy with any reasoner:

```python
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker
from your_module import NewEvaluationStrategy

# With Chain-of-Recursive-Thoughts
cort_session = ThinkThreadSession(
    llm_client=client,
    evaluation_strategy=NewEvaluationStrategy(),
)

# With Tree-of-Thoughts
tot_session = TreeThinker(
    llm_client=client,
    evaluator=NewEvaluationStrategy(),
)
```

## Shared Utilities

The SDK provides shared utilities in `reasoning_utils.py` for common operations used by different reasoning approaches:

```python
# Generate alternatives for a question and current answer
alternatives = generate_alternatives(
    question, 
    current_answer, 
    llm_client, 
    template_manager, 
    count=3, 
    temperature=0.9
)

# Generate alternatives asynchronously
alternatives = await generate_alternatives_async(
    question, 
    current_answer, 
    llm_client, 
    template_manager, 
    count=3, 
    temperature=0.9,
    parallel=True  # Use parallel processing
)

# Calculate similarity between two strings
similarity = calculate_similarity(str1, str2)
```

## Customising Prompt Templates

The SDK uses Jinja2 templates for all prompts. The default templates are in `thinkthread_sdk/prompts/`, but you can provide your own:

1. Create a directory for your templates
2. Create template files with the `.j2` extension
3. Pass the directory path when creating a TemplateManager:

```python
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker

template_manager = TemplateManager(template_dir="/path/to/your/templates")

# Use with Chain-of-Recursive-Thoughts
cort_session = ThinkThreadSession(
    llm_client=client, 
    template_manager=template_manager
)

# Use with Tree-of-Thoughts
tot_session = TreeThinker(
    llm_client=client, 
    template_manager=template_manager
)
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
- `tot_expansion_prompt.j2`: Generates continuations for tree nodes (ToT only)
- `tot_evaluation_prompt.j2`: Evaluates tree branches (ToT only)

## Advanced Configuration

For advanced configuration, create a custom `ThinkThreadConfig` instance:

```python
from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.tree_thinker import TreeThinker

# Create custom configuration
config = ThinkThreadConfig(
    openai_api_key="your-api-key",
    provider="openai",
    openai_model="gpt-4",
    alternatives=5,
    rounds=3,
    use_pairwise_evaluation=True,
    use_self_evaluation=False,
    parallel_alternatives=True,  # Enable parallel processing
)

# Use with Chain-of-Recursive-Thoughts
cort_session = ThinkThreadSession(
    llm_client=client, 
    config=config
)

# Use with Tree-of-Thoughts
tot_session = TreeThinker(
    llm_client=client,
    max_tree_depth=3,
    branching_factor=3,
    config=config
)
```

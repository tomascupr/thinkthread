# ThinkThread API Reference

Complete API documentation for all ThinkThread components.

## Table of Contents

1. [Simple API Functions](#simple-api-functions)
2. [ThinkThreadSession](#thinkthreadsession)
3. [TreeThinker](#treethinker)
4. [Configuration](#configuration)
5. [LLM Clients](#llm-clients)
6. [Evaluators](#evaluators)
7. [Exceptions](#exceptions)

## Simple API Functions

### `reason(prompt, test_mode=False, **kwargs)`

Make AI think before responding using Chain-of-Recursive-Thoughts.

**Parameters:**
- `prompt` (str): The question or prompt to reason about
- `test_mode` (bool): If True, use dummy LLM client (no API calls)
- `**kwargs`: Additional arguments passed to ThinkThreadSession
  - `alternatives` (int): Number of alternatives per round (default: 3)
  - `rounds` (int): Number of refinement rounds (default: 2)

**Returns:**
- `str`: The refined answer after recursive reasoning

**Example:**
```python
answer = reason("What are the implications of quantum computing?")
```

### `explore(prompt, test_mode=False, **kwargs)`

Explore multiple paths of thinking using Tree-of-Thoughts.

**Parameters:**
- `prompt` (str): The question or prompt to explore
- `test_mode` (bool): If True, use dummy LLM client
- `**kwargs`: Additional arguments passed to TreeThinker
  - `max_tree_depth` (int): Maximum tree depth (default: 2)
  - `branching_factor` (int): Branches per node (default: 2)
  - `beam_width` (int): Beam search width (default: 1)
  - `max_iterations` (int): Max iterations (default: 1)

**Returns:**
- `str`: The best solution after tree-based exploration

### `solve(problem, test_mode=False, **kwargs)`

Get step-by-step solutions to specific problems.

**Parameters:**
- `problem` (str): The problem to solve
- `test_mode` (bool): If True, use dummy LLM client
- `**kwargs`: Additional arguments passed to reason()

**Returns:**
- `str`: A detailed solution with actionable steps

### `debate(question, test_mode=False, **kwargs)`

Analyze a question from multiple perspectives.

**Parameters:**
- `question` (str): The question to analyze
- `test_mode` (bool): If True, use dummy LLM client
- `**kwargs`: Additional arguments passed to reason()

**Returns:**
- `str`: A balanced analysis with multiple viewpoints

### `refine(text, instructions="", test_mode=False, **kwargs)`

Refine and improve existing text or ideas.

**Parameters:**
- `text` (str): The text to refine
- `instructions` (str): Optional specific refinement instructions
- `test_mode` (bool): If True, use dummy LLM client
- `**kwargs`: Additional arguments passed to reason()

**Returns:**
- `str`: The refined and improved version

## ThinkThreadSession

### Class: `ThinkThreadSession`

Orchestrates multi-round questioning and refinement using LLMs.

### Constructor

```python
ThinkThreadSession(
    llm_client: LLMClient,
    alternatives: int = 3,
    rounds: int = 2,
    max_rounds: Optional[int] = None,
    template_manager: Optional[TemplateManager] = None,
    evaluation_strategy: Optional[EvaluationStrategy] = None,
    evaluator: Optional[Evaluator] = None,
    config: Optional[ThinkThreadConfig] = None
)
```

**Parameters:**
- `llm_client` (LLMClient): LLM client for generating/evaluating answers
- `alternatives` (int): Number of alternative answers per round
- `rounds` (int): Number of refinement rounds
- `max_rounds` (int, optional): Maximum rounds (overrides rounds if set)
- `template_manager` (TemplateManager, optional): Custom prompt templates
- `evaluation_strategy` (EvaluationStrategy, optional): Answer evaluation strategy
- `evaluator` (Evaluator, optional): Pairwise comparison evaluator
- `config` (ThinkThreadConfig, optional): Configuration object

### Methods

#### `run(question: str) -> str`

Execute the ThinkThread process on a question.

**Parameters:**
- `question` (str): The question to answer

**Returns:**
- `str`: The final best answer after all refinement rounds

**Process:**
1. Generate initial answer (temperature: 0.7)
2. For each round:
   - Generate alternatives (temperature: 0.9)
   - Evaluate using configured strategy
   - Select best as current answer
3. Return final best answer

#### `run_async(question: str) -> str`

Async version of run().

**Parameters:**
- `question` (str): The question to answer

**Returns:**
- `str`: The final best answer (async)

## TreeThinker

### Class: `TreeThinker`

Implements tree-based search for exploring multiple reasoning paths.

### Constructor

```python
TreeThinker(
    llm_client: LLMClient,
    max_tree_depth: int = 3,
    branching_factor: int = 3,
    template_manager: Optional[TemplateManager] = None,
    config: Optional[ThinkThreadConfig] = None,
    evaluator: Optional[Evaluator] = None,
    scoring_function: Optional[Callable[[str, Dict[str, Any]], float]] = None
)
```

**Parameters:**
- `llm_client` (LLMClient): LLM client for generating thoughts
- `max_tree_depth` (int): Maximum depth of thinking tree
- `branching_factor` (int): Number of branches per node
- `template_manager` (TemplateManager, optional): Custom templates
- `config` (ThinkThreadConfig, optional): Configuration
- `evaluator` (Evaluator, optional): Branch scoring evaluator
- `scoring_function` (Callable, optional): Custom scoring function

### Methods

#### `solve(problem: str, beam_width: int = 1, max_iterations: int = 10, **kwargs) -> Union[str, Dict[str, Any]]`

Solve a problem using tree-of-thoughts approach.

**Parameters:**
- `problem` (str): The problem to solve
- `beam_width` (int): Number of parallel thought threads
- `max_iterations` (int): Maximum iterations

**Returns:**
- `str` or `dict`: Best solution or solution with metadata

#### `solve_async(problem: str, beam_width: int = 1, max_iterations: int = 10, **kwargs) -> Union[str, Dict[str, Any]]`

Async version of solve().

## Configuration

### Class: `ThinkThreadConfig`

Configuration for the ThinkThread SDK.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **API Keys** ||||
| `openai_api_key` | str | None | OpenAI API key |
| `anthropic_api_key` | str | None | Anthropic API key |
| `hf_api_token` | str | None | HuggingFace API token |
| **Provider Settings** ||||
| `provider` | str | "openai" | LLM provider ("openai", "anthropic", "hf") |
| `openai_model` | str | "gpt-4" | OpenAI model name |
| `anthropic_model` | str | "claude-2" | Anthropic model name |
| `hf_model` | str | "gpt2" | HuggingFace model name |
| **Core Parameters** ||||
| `alternatives` | int | 3 | Alternatives per round |
| `rounds` | int | 2 | Default refinement rounds |
| `max_rounds` | int | 3 | Maximum rounds |
| **Evaluation** ||||
| `use_pairwise_evaluation` | bool | True | Use pairwise comparison |
| `use_self_evaluation` | bool | False | Use self-evaluation |
| **Performance** ||||
| `parallel_alternatives` | bool | False | Generate alternatives in parallel |
| `parallel_evaluation` | bool | False | Evaluate in parallel |
| `concurrency_limit` | int | 5 | Max concurrent requests |
| `use_batched_requests` | bool | False | Batch API calls |
| **Optimization** ||||
| `use_caching` | bool | False | Cache LLM responses |
| `early_termination` | bool | False | Stop when confident |
| `early_termination_threshold` | float | 0.95 | Confidence threshold |
| **Temperature** ||||
| `use_adaptive_temperature` | bool | True | Adjust temperature dynamically |
| `initial_temperature` | float | 0.7 | Initial generation temperature |
| `generation_temperature` | float | 0.9 | Alternative generation temperature |
| `min_generation_temperature` | float | 0.5 | Minimum temperature |
| `temperature_decay_rate` | float | 0.8 | Temperature decay per round |
| **Advanced** ||||
| `enable_monitoring` | bool | False | Enable performance monitoring |
| `use_fast_similarity` | bool | False | Fast similarity calculations |
| `prompt_dir` | str | None | Custom prompt directory |

### Function: `create_config(env_file: Optional[str] = ".env") -> ThinkThreadConfig`

Create configuration from environment variables.

**Parameters:**
- `env_file` (str, optional): Path to .env file

**Returns:**
- `ThinkThreadConfig`: Configuration instance

## LLM Clients

### Base Class: `LLMClient`

Abstract base class for LLM clients.

### Methods

#### `generate(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str`

Generate text from prompt.

#### `generate_async(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str`

Async text generation.

### OpenAI Client

```python
OpenAIClient(
    api_key: str,
    model_name: str = "gpt-4",
    max_retries: int = 3,
    timeout: int = 60
)
```

### Anthropic Client

```python
AnthropicClient(
    api_key: str,
    model_name: str = "claude-3-opus-20240229",
    max_retries: int = 3,
    timeout: int = 60
)
```

### HuggingFace Client

```python
HuggingFaceClient(
    api_token: str,
    model_name: str = "gpt2",
    max_retries: int = 3,
    timeout: int = 60
)
```

### Dummy Client

```python
DummyLLMClient()  # For testing without API calls
```

## Evaluators

### Base Class: `BaseEvaluator`

Abstract base class for evaluation strategies.

### Methods

#### `evaluate(alternatives: List[str], context: Optional[Dict[str, Any]] = None) -> int`

Evaluate alternatives and return best index.

### DefaultEvaluationStrategy

Default batch evaluation of all alternatives.

### ModelEvaluator

Uses LLM for pairwise comparison of alternatives.

### Custom Evaluator Example

```python
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, alternatives, context=None):
        # Custom evaluation logic
        scores = [self._score(alt) for alt in alternatives]
        return scores.index(max(scores))
    
    def _score(self, text):
        # Scoring logic
        return len(text) / 100  # Example: prefer longer answers
```

## Exceptions

### `LLMException`

Base exception for LLM-related errors.

```python
try:
    answer = session.run(question)
except LLMException as e:
    print(f"LLM error: {e}")
```

### `ConfigurationError`

Raised for invalid configuration.

```python
try:
    config = ThinkThreadConfig(provider="invalid")
except ConfigurationError as e:
    print(f"Config error: {e}")
```

### `EvaluationError`

Raised during evaluation failures.

```python
try:
    result = evaluator.evaluate(alternatives)
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
```

## Environment Variables

All configuration parameters can be set via environment variables:

| Config Parameter | Environment Variable |
|-----------------|---------------------|
| `openai_api_key` | `OPENAI_API_KEY` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| `hf_api_token` | `HF_API_TOKEN` |
| `provider` | `PROVIDER` |
| `openai_model` | `OPENAI_MODEL` |
| `anthropic_model` | `ANTHROPIC_MODEL` |
| `alternatives` | `ALTERNATIVES` |
| `rounds` | `ROUNDS` |
| `max_rounds` | `MAX_ROUNDS` |
| `use_pairwise_evaluation` | `USE_PAIRWISE_EVALUATION` |
| `parallel_alternatives` | `PARALLEL_ALTERNATIVES` |
| `use_caching` | `USE_CACHING` |
| `enable_monitoring` | `ENABLE_MONITORING` |

## Type Hints

All methods include full type hints for better IDE support:

```python
from typing import List, Optional, Dict, Any, Union, Callable

def solve(
    self,
    problem: str,
    beam_width: int = 1,
    max_iterations: int = 10,
    **kwargs: Any
) -> Union[str, Dict[str, Any]]:
    ...
```
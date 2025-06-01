"""
Base classes for reasoning modes

Provides the foundation for all reasoning modes with composability and introspection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import time
import json
import os


@dataclass
class ReasoningResult:
    """Result from a reasoning operation"""
    answer: str
    confidence: float
    reasoning_tree: Dict[str, Any]
    mode: str
    cost: float
    time_elapsed: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    
    def __str__(self):
        """Return just the answer for simple use"""
        return self.answer
    
    def explain(self, detail_level: str = 'summary') -> str:
        """Explain the reasoning process"""
        if detail_level == 'summary':
            return f"Used {self.mode} reasoning with {self.confidence:.2%} confidence. " \
                   f"Considered {len(self.alternatives)} alternatives."
        elif detail_level == 'full':
            return json.dumps(self.reasoning_tree, indent=2)
        else:
            return self.answer
    
    def visualize(self):
        """Open visualization of reasoning process"""
        # This would open the visualization in a browser
        print(f"Opening visualization for {self.mode} reasoning...")
        # Implementation would launch visualization server
    
    def improve(self):
        """Re-run reasoning with lessons learned"""
        # This would trigger a refinement with the current result
        print("Improving answer with refinement mode...")
        # Implementation would call refine mode
    
    @property
    def cost_breakdown(self) -> Dict[str, Any]:
        """Detailed cost analysis"""
        return {
            'total': self.cost,
            'per_token': self.cost / self.metadata.get('total_tokens', 1),
            'model_used': self.metadata.get('model', 'unknown'),
            'cache_savings': self.metadata.get('cache_savings', 0)
        }
    
    def save(self, filename: str):
        """Save result for later use"""
        with open(filename, 'w') as f:
            json.dump({
                'answer': self.answer,
                'confidence': self.confidence,
                'reasoning_tree': self.reasoning_tree,
                'mode': self.mode,
                'cost': self.cost,
                'metadata': self.metadata
            }, f, indent=2)
    
    def export_markdown(self) -> str:
        """Export as formatted markdown"""
        return f"""# Reasoning Result

## Question
{self.metadata.get('question', 'N/A')}

## Answer
{self.answer}

## Reasoning Details
- **Mode**: {self.mode}
- **Confidence**: {self.confidence:.2%}
- **Cost**: ${self.cost:.4f}
- **Time**: {self.time_elapsed:.2f}s
- **Alternatives Considered**: {len(self.alternatives)}

## Reasoning Tree
```json
{json.dumps(self.reasoning_tree, indent=2)}
```
"""


class ReasoningMode(ABC):
    """Base class that makes modes feel like built-in types"""
    
    def __init__(self, **config):
        self.config = config
        self._result = None
        self.debugger = config.get('debugger')
        self.visualizer = config.get('visualizer')
        self.profiler = config.get('profiler')
        
        # Initialize LLM integration
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM integration"""
        # Only initialize if not in test mode
        if not self.config.get('test_mode', False):
            from ..llm_integration import LLMIntegration
            self.llm = LLMIntegration(provider=self.config.get('provider', 'auto'))
        else:
            self.llm = None
    
    def __call__(self, question: str, **runtime_args) -> ReasoningResult:
        """Make modes callable like functions"""
        return self.execute(question, **runtime_args)
    
    def __or__(self, other: 'ReasoningMode') -> 'ChainedMode':
        """Chain modes with | operator"""
        return ChainedMode(self, other)
    
    def __and__(self, other: 'ReasoningMode') -> 'ParallelMode':
        """Run modes in parallel with & operator"""
        return ParallelMode(self, other)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"
    
    @abstractmethod
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute the reasoning mode"""
        pass
    
    def with_options(self, **options) -> 'ReasoningMode':
        """Fluent interface for configuration"""
        self.config.update(options)
        return self
    
    def then_if(self, confidence_below: float, fallback: 'ReasoningMode') -> 'ConditionalMode':
        """Conditional execution based on confidence"""
        return ConditionalMode(self, fallback, confidence_below)
    
    @property
    def description(self) -> str:
        """Human-readable description of the mode"""
        return self.__class__.__doc__ or "No description available"
    
    @property
    def characteristics(self) -> Dict[str, Any]:
        """Mode characteristics for introspection"""
        return {
            'thinking_style': 'unknown',
            'breadth_vs_depth': 'balanced',
            'token_usage': 'medium',
            'typical_duration': '5-15s',
            'best_for': []
        }
    
    @property
    def examples(self) -> List[str]:
        """Example usage of this mode"""
        return [f"{self.__class__.__name__.lower()}('Example question')"]
    
    def try_sample(self):
        """Run a quick demo of this mode"""
        sample_question = f"Sample question for {self.__class__.__name__}"
        print(f"Running {self.__class__.__name__} with: '{sample_question}'")
        # In real implementation, this would run actual reasoning
        return ReasoningResult(
            answer="Sample answer",
            confidence=0.85,
            reasoning_tree={'demo': True},
            mode=self.__class__.__name__,
            cost=0.01,
            time_elapsed=1.0
        )
    
    def estimate_cost(self, question: str) -> float:
        """Estimate cost for this question"""
        # Simple estimation based on question length
        return len(question) * 0.0001
    
    def estimate_time(self, question: str) -> float:
        """Estimate execution time"""
        # Simple estimation
        return 5.0 + len(question) * 0.01


class ChainedMode(ReasoningMode):
    """Sequential composition of reasoning modes"""
    
    def __init__(self, first: ReasoningMode, second: ReasoningMode):
        super().__init__()
        self.first = first
        self.second = second
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute modes in sequence"""
        # First mode
        result1 = self.first.execute(question, **kwargs)
        
        # Second mode uses first result
        kwargs['initial_answer'] = result1.answer
        kwargs['previous_reasoning'] = result1.reasoning_tree
        result2 = self.second.execute(question, **kwargs)
        
        # Combine results
        return ReasoningResult(
            answer=result2.answer,
            confidence=(result1.confidence + result2.confidence) / 2,
            reasoning_tree={
                'chain': [result1.reasoning_tree, result2.reasoning_tree]
            },
            mode=f"{result1.mode} | {result2.mode}",
            cost=result1.cost + result2.cost,
            time_elapsed=result1.time_elapsed + result2.time_elapsed,
            metadata={
                'chained': True,
                'modes': [result1.mode, result2.mode]
            }
        )


class ParallelMode(ReasoningMode):
    """Parallel execution of reasoning modes"""
    
    def __init__(self, first: ReasoningMode, second: ReasoningMode):
        super().__init__()
        self.first = first
        self.second = second
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute modes in parallel and merge results"""
        # In real implementation, these would run concurrently
        result1 = self.first.execute(question, **kwargs)
        result2 = self.second.execute(question, **kwargs)
        
        # Select best result based on confidence
        best_result = result1 if result1.confidence > result2.confidence else result2
        
        return ReasoningResult(
            answer=best_result.answer,
            confidence=max(result1.confidence, result2.confidence),
            reasoning_tree={
                'parallel': [result1.reasoning_tree, result2.reasoning_tree],
                'selected': best_result.mode
            },
            mode=f"{result1.mode} & {result2.mode}",
            cost=result1.cost + result2.cost,
            time_elapsed=max(result1.time_elapsed, result2.time_elapsed),
            alternatives=[result1.answer, result2.answer],
            metadata={
                'parallel': True,
                'modes': [result1.mode, result2.mode]
            }
        )


class ConditionalMode(ReasoningMode):
    """Conditional execution based on confidence threshold"""
    
    def __init__(self, primary: ReasoningMode, fallback: ReasoningMode, threshold: float):
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute primary, fall back if confidence is low"""
        result = self.primary.execute(question, **kwargs)
        
        if result.confidence < self.threshold:
            # Use fallback
            fallback_result = self.fallback.execute(question, **kwargs)
            fallback_result.metadata['used_fallback'] = True
            fallback_result.metadata['original_confidence'] = result.confidence
            return fallback_result
        
        return result
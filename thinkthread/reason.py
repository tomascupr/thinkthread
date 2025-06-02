"""
ThinkThread Reasoning Engine - Core implementation

This module provides the main reasoning interface that automatically selects
the best reasoning approach based on the question type.

Now powered by the robust old SDK implementation through the adapter layer.
"""

from typing import Optional, Dict, Any, Union, List
import time
import webbrowser
from dataclasses import dataclass
import asyncio

from .modes.base import ReasoningMode, ReasoningResult
from .transparency.debugger import ReasoningDebugger
from .transparency.live_view import LiveReasoningView
from .transparency.profiler import ReasoningProfiler

# Import the adapter that bridges to old SDK
from .core.adapter import SDKAdapter, AdapterConfig


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine"""
    enable_memory: bool = True
    enable_visualization: bool = False
    max_cost: Optional[float] = None
    preferred_models: List[str] = None
    cache_ttl: int = 3600
    auto_open_visualizer: bool = True


class ReasoningEngine:
    """
    Main reasoning engine that provides a unified interface to all reasoning modes.
    
    Now powered by the battle-tested SDK implementation while maintaining
    the beautiful new API.
    
    Usage:
        from thinkthread import reason
        
        # Automatic mode selection
        answer = reason("What is consciousness?")
        
        # Explicit mode selection
        answer = reason.explore("Design a mars colony", visualize=True)
        answer = reason.debate("Is AI sentient?")
    """
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.debugger = ReasoningDebugger()
        self.visualizer = LiveReasoningView() if self.config.enable_visualization else None
        self.profiler = ReasoningProfiler()
        
        # Initialize SDK adapter
        adapter_config = AdapterConfig(
            enable_visualization=self.config.enable_visualization,
            enable_debugging=False,
            enable_profiling=False,
            enable_monitoring=True,
            enable_caching=self.config.enable_memory,
        )
        self.adapter = SDKAdapter(adapter_config)
        
        # Last result for debugging
        self._last_result = None
    
    def __call__(self, question: str, mode: str = 'auto', **kwargs) -> ReasoningResult:
        """
        Main entry point for reasoning.
        
        Args:
            question: The question or prompt to reason about
            mode: Reasoning mode ('auto', 'explore', 'refine', 'debate', 'solve')
            **kwargs: Additional arguments passed to reasoning mode
            
        Returns:
            ReasoningResult object containing answer and metadata
        """
        start_time = time.time()
        
        # Update adapter config based on kwargs
        if kwargs.get('test_mode', False) != self.adapter.config.test_mode:
            # Need to reinitialize the LLM client if test mode changed
            self.adapter.config.test_mode = kwargs.get('test_mode', False)
            self.adapter._llm_client = self.adapter._initialize_llm_client()
        self.adapter.config.provider = kwargs.get('provider', 'auto')
        
        # Initialize visualizer if requested
        if kwargs.get('visualize', False):
            if not self.visualizer:
                self.visualizer = LiveReasoningView()
            self.visualizer.start_session(question)
            if self.config.auto_open_visualizer:
                # TODO: Make port configurable
                webbrowser.open("http://localhost:8080/reasoning-live")
        
        # Execute reasoning through adapter
        if mode == 'auto':
            adapter_result = self.adapter.reason(question, **kwargs)
        elif mode == 'explore':
            adapter_result = self.adapter.explore(question, **kwargs)
        elif mode == 'refine':
            initial_text = kwargs.pop('initial_text', '')
            adapter_result = self.adapter.refine(question, initial_text, **kwargs)
        elif mode == 'debate':
            perspectives = kwargs.pop('perspectives', 3)
            adapter_result = self.adapter.debate(question, perspectives, **kwargs)
        elif mode == 'solve':
            adapter_result = self.adapter.solve(question, **kwargs)
        else:
            raise ValueError(f"Unknown reasoning mode: {mode}")
        
        # Convert adapter result to ReasoningResult
        result = ReasoningResult(
            answer=adapter_result['answer'],
            confidence=adapter_result.get('confidence', 1.0),
            reasoning_tree=adapter_result.get('metadata', {}).get('reasoning_tree', {}),
            mode=adapter_result.get('mode', mode),
            cost=adapter_result.get('cost', 0.0),
            time_elapsed=time.time() - start_time,
            metadata={
                'mode_used': adapter_result.get('mode', mode),
                'auto_selected': mode == 'auto',
                **adapter_result.get('metadata', {})
            },
            alternatives=adapter_result.get('metadata', {}).get('alternatives', [])
        )
        
        # Add transparency data if enabled
        if kwargs.get('debug') and self.debugger:
            result.metadata['debug_trace'] = self.debugger.get_trace()
        
        if kwargs.get('profile') and self.profiler:
            result.metadata['profile'] = self.profiler.get_results()
        
        # Store last result
        self._last_result = result
        
        return result
    
    def explore(self, question: str, **kwargs) -> ReasoningResult:
        """Explore possibilities using Tree of Thoughts"""
        kwargs['mode'] = 'explore'
        return self.__call__(question, **kwargs)
    
    def refine(self, question: str, initial_text: str = "", **kwargs) -> ReasoningResult:
        """Refine and improve a response"""
        kwargs['mode'] = 'refine'
        kwargs['initial_text'] = initial_text
        return self.__call__(question, **kwargs)
    
    def debate(self, question: str, perspectives: int = 3, **kwargs) -> ReasoningResult:
        """Generate multiple perspectives on a topic"""
        kwargs['mode'] = 'debate'
        kwargs['perspectives'] = perspectives
        return self.__call__(question, **kwargs)
    
    def solve(self, question: str, **kwargs) -> ReasoningResult:
        """Focus on solving a specific problem"""
        kwargs['mode'] = 'solve'
        return self.__call__(question, **kwargs)
    
    def explain(self) -> Optional[str]:
        """Explain the last reasoning process"""
        if not self._last_result:
            return None
        
        explanation = f"""
Reasoning Mode: {self._last_result.metadata.get('mode_used', 'unknown')}
Confidence: {self._last_result.confidence:.2%}
Time: {self._last_result.metadata.get('total_time', 0):.2f}s
Cost: ${self._last_result.metadata.get('cost', 0):.4f}

Answer: {self._last_result.answer[:200]}...
"""
        return explanation
    
    def set_budget(self, daily: float = None, per_query: float = None):
        """Set cost limits (placeholder for future implementation)"""
        if daily:
            self.config.max_cost = daily
        # TODO: Implement budget tracking in adapter
    
    def enable_memory(self):
        """Enable pattern learning and caching"""
        self.config.enable_memory = True
        self.adapter.config.enable_caching = True


# Create singleton instance
_engine = ReasoningEngine()


# Public API functions
def reason(question: str, **kwargs) -> ReasoningResult:
    """Main reasoning function with automatic mode selection"""
    return _engine(question, **kwargs)


def explore(question: str, **kwargs) -> ReasoningResult:
    """Explore possibilities using Tree of Thoughts"""
    return _engine.explore(question, **kwargs)


def refine(question: str, initial_text: str = "", **kwargs) -> ReasoningResult:
    """Refine and improve a response"""
    return _engine.refine(question, initial_text, **kwargs)


def debate(question: str, perspectives: int = 3, **kwargs) -> ReasoningResult:
    """Generate multiple perspectives on a topic"""
    return _engine.debate(question, perspectives, **kwargs)


def solve(question: str, **kwargs) -> ReasoningResult:
    """Focus on solving a specific problem"""
    return _engine.solve(question, **kwargs)


# Attach methods to reason function for convenience
reason.explore = explore
reason.refine = refine
reason.debate = debate
reason.solve = solve
reason.explain = _engine.explain
reason.set_budget = _engine.set_budget
reason.enable_memory = _engine.enable_memory
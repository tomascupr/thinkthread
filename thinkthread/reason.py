"""
ThinkThread Reasoning Engine - Core implementation

This module provides the main reasoning interface that automatically selects
the best reasoning approach based on the question type.
"""

from typing import Optional, Dict, Any, Union, List
import time
import webbrowser
from dataclasses import dataclass
import asyncio

from .modes.base import ReasoningMode, ReasoningResult
from .modes.explore import ExploreMode
from .modes.refine import RefineMode
from .modes.debate import DebateMode
from .modes.solve import SolveMode
from .transparency.debugger import ReasoningDebugger
from .transparency.live_view import LiveReasoningView
from .transparency.profiler import ReasoningProfiler


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
        
        # Initialize reasoning modes
        self.modes = {
            'explore': ExploreMode,
            'refine': RefineMode,
            'debate': DebateMode,
            'solve': SolveMode,
        }
        
        # Mode detection keywords
        self.mode_patterns = {
            'explore': ['design', 'create', 'brainstorm', 'possibilities', 'what if'],
            'refine': ['improve', 'enhance', 'polish', 'revise', 'better'],
            'debate': ['argue', 'perspective', 'pros and cons', 'versus', 'compare'],
            'solve': ['solution', 'fix', 'resolve', 'problem', 'how to'],
        }
        
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
        
        # Auto-detect mode if needed
        if mode == 'auto':
            mode, confidence = self._detect_reasoning_mode(question)
            if kwargs.get('verbose'):
                print(f"Auto-selected mode: {mode} (confidence: {confidence:.2%})")
        
        # Initialize visualizer if requested
        if kwargs.get('visualize', False):
            if not self.visualizer:
                self.visualizer = LiveReasoningView()
            self.visualizer.start_session(question)
            if self.config.auto_open_visualizer:
                webbrowser.open("http://localhost:8080/reasoning-live")
        
        # Get reasoning mode class
        mode_class = self.modes.get(mode)
        if not mode_class:
            raise ValueError(f"Unknown reasoning mode: {mode}")
        
        # Create mode instance
        reasoner = mode_class(
            debugger=self.debugger if kwargs.get('debug') else None,
            visualizer=self.visualizer if kwargs.get('visualize') else None,
            profiler=self.profiler if kwargs.get('profile') else None
        )
        
        # Execute reasoning
        result = reasoner.execute(question, **kwargs)
        
        # Add metadata
        result.metadata['total_time'] = time.time() - start_time
        result.metadata['mode_used'] = mode
        result.metadata['auto_selected'] = mode == 'auto'
        
        # Store last result
        self._last_result = result
        
        return result
    
    def _detect_reasoning_mode(self, question: str) -> tuple[str, float]:
        """Detect the best reasoning mode based on question analysis"""
        question_lower = question.lower()
        
        # Check for mode-specific keywords
        mode_scores = {}
        for mode, keywords in self.mode_patterns.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            mode_scores[mode] = score
        
        # Get best mode
        if max(mode_scores.values()) == 0:
            # Default to explore for open-ended questions
            return 'explore', 0.6
        
        best_mode = max(mode_scores.items(), key=lambda x: x[1])
        confidence = min(best_mode[1] / 3.0, 1.0)  # Normalize confidence
        
        return best_mode[0], confidence
    
    # Convenience methods for specific modes
    def explore(self, question: str, **kwargs) -> ReasoningResult:
        """Use exploratory reasoning (Tree of Thoughts)"""
        return self(question, mode='explore', **kwargs)
    
    def refine(self, question: str, initial_answer: str = None, **kwargs) -> ReasoningResult:
        """Use refinement reasoning (Chain of Recursive Thoughts)"""
        if initial_answer:
            kwargs['initial_answer'] = initial_answer
        return self(question, mode='refine', **kwargs)
    
    def debate(self, question: str, **kwargs) -> ReasoningResult:
        """Use debate-style multi-perspective reasoning"""
        return self(question, mode='debate', **kwargs)
    
    def solve(self, question: str, **kwargs) -> ReasoningResult:
        """Use solution-focused reasoning"""
        return self(question, mode='solve', **kwargs)
    
    # Transparency methods
    @property
    def last(self) -> Optional[ReasoningResult]:
        """Get the last reasoning result"""
        return self._last_result
    
    def debug_last(self):
        """Open debugging interface for last reasoning operation"""
        if not self._last_result:
            print("No reasoning session to debug.")
            return
        
        self.debugger.inspect(self._last_result)
    
    def profile(self, reasoning_func):
        """Profile a reasoning operation"""
        return self.profiler.profile(reasoning_func)
    
    def compare(self, result1: ReasoningResult, result2: ReasoningResult):
        """Compare two reasoning results"""
        from .transparency.diff import ReasoningDiff
        diff = ReasoningDiff()
        return diff.compare(result1, result2)
    
    # Async support
    async def async_(self, question: str, mode: str = 'auto', **kwargs) -> ReasoningResult:
        """Async version of reasoning"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self, question, mode, **kwargs)


# Create global instance
reason = ReasoningEngine()

# Export convenience functions
explore = reason.explore
refine = reason.refine
debate = reason.debate
solve = reason.solve
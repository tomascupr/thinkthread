"""
Reasoning Debugger - Makes reasoning as debuggable as regular code
"""

from typing import Any, Dict, List, Callable, Optional
import json
import time


class ReasoningStep:
    """Represents a single step in reasoning"""
    
    def __init__(self, number: int, description: str, data: Dict[str, Any]):
        self.number = number
        self.description = description
        self.data = data
        self.timestamp = time.time()
        self.confidence = data.get('confidence', 0.0)
        self.alternatives = data.get('alternatives', [])


class ReasoningDebugger:
    """Makes reasoning as debuggable as regular code"""
    
    def __init__(self):
        self.breakpoints: List[Callable] = []
        self.watch_expressions: List[str] = []
        self.trace = []
        self.current_step = None
        self.paused = False
        
    def set_breakpoint(self, condition: Callable[[ReasoningStep], bool]):
        """Set a breakpoint condition"""
        self.breakpoints.append(condition)
        
    def watch(self, expression: str):
        """Watch how a value changes during reasoning"""
        self.watch_expressions.append(expression)
        
    def step(self, description: str, data: Dict[str, Any]):
        """Record a reasoning step"""
        step = ReasoningStep(
            number=len(self.trace) + 1,
            description=description,
            data=data
        )
        self.trace.append(step)
        self.current_step = step
        
        # Check breakpoints
        for breakpoint in self.breakpoints:
            if breakpoint(step):
                self.paused = True
                print(f"\n🔴 Breakpoint hit at step {step.number}: {step.description}")
                self._show_context(step)
                self._debug_prompt()
                
    def _show_context(self, step: ReasoningStep):
        """Show context at breakpoint"""
        print(f"\nStep {step.number}: {step.description}")
        print(f"Confidence: {step.confidence:.2%}")
        print(f"Data: {json.dumps(step.data, indent=2)}")
        
        if self.watch_expressions:
            print("\nWatched values:")
            for expr in self.watch_expressions:
                value = self._evaluate_watch(expr, step)
                print(f"  {expr}: {value}")
                
    def _evaluate_watch(self, expr: str, step: ReasoningStep) -> Any:
        """Evaluate a watch expression"""
        # Simple implementation - in real version would be more sophisticated
        if expr in step.data:
            return step.data[expr]
        elif expr == "confidence":
            return step.confidence
        elif expr == "alternatives_count":
            return len(step.alternatives)
        else:
            return "N/A"
            
    def _debug_prompt(self):
        """Interactive debug prompt"""
        while self.paused:
            command = input("\n(debug) ").strip().lower()
            
            if command == "continue" or command == "c":
                self.paused = False
            elif command == "step" or command == "s":
                self.paused = False
                # Will pause at next step
            elif command == "inspect" or command == "i":
                self.inspect(self.current_step)
            elif command == "trace" or command == "t":
                self.show_trace()
            elif command == "help" or command == "h":
                self._show_help()
            else:
                print(f"Unknown command: {command}")
                
    def _show_help(self):
        """Show debug commands"""
        print("""
Debug commands:
  continue (c) - Continue execution
  step (s)     - Step to next reasoning step
  inspect (i)  - Inspect current step in detail
  trace (t)    - Show full trace
  help (h)     - Show this help
        """)
        
    def inspect(self, target: Any):
        """Inspect a reasoning result or step"""
        if hasattr(target, 'reasoning_tree'):
            # Inspecting a full result
            print("\n=== Reasoning Result Inspection ===")
            print(f"Mode: {target.mode}")
            print(f"Confidence: {target.confidence:.2%}")
            print(f"Cost: ${target.cost:.4f}")
            print(f"Time: {target.time_elapsed:.2f}s")
            print("\nReasoning Tree:")
            print(json.dumps(target.reasoning_tree, indent=2))
            
            if target.alternatives:
                print(f"\nAlternatives ({len(target.alternatives)}):")
                for i, alt in enumerate(target.alternatives[:3]):
                    print(f"  {i+1}. {alt[:100]}...")
        else:
            # Inspecting a step
            print(f"\n=== Step {target.number} ===")
            print(f"Description: {target.description}")
            print(f"Timestamp: {target.timestamp}")
            print(f"Data: {json.dumps(target.data, indent=2)}")
            
    def show_trace(self):
        """Show full reasoning trace"""
        print("\n=== Reasoning Trace ===")
        for step in self.trace:
            print(f"{step.number:3d}. {step.description} "
                  f"[confidence: {step.confidence:.2%}]")
            
    def get_trace(self) -> List[ReasoningStep]:
        """Get the full trace for analysis"""
        return self.trace
        
    def clear(self):
        """Clear debugger state"""
        self.trace = []
        self.current_step = None
        self.paused = False
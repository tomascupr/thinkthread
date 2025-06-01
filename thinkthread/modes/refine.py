"""
Refine Mode - Iterative improvement of answers

Best for: improving existing content, polishing responses, convergent thinking
"""

from typing import Dict, Any, List, Optional
import time
from .base import ReasoningMode, ReasoningResult


class RefinementStep:
    """Represents one iteration of refinement"""
    
    def __init__(self, iteration: int, current_answer: str, alternatives: List[str], 
                 selected: str, confidence: float):
        self.iteration = iteration
        self.current_answer = current_answer
        self.alternatives = alternatives
        self.selected = selected
        self.confidence = confidence
        self.improvement = 0.0


class RefineMode(ReasoningMode):
    """
    Iterative refinement of answers through recursive improvement.
    
    This mode starts with an initial answer and iteratively generates
    alternatives, evaluates them, and selects the best improvement.
    """
    
    @property
    def characteristics(self) -> Dict[str, Any]:
        return {
            'thinking_style': 'convergent',
            'breadth_vs_depth': 'depth-focused',
            'token_usage': 'medium',
            'typical_duration': '5-15s',
            'best_for': ['editing', 'improvement', 'polishing', 'convergence']
        }
    
    @property
    def examples(self) -> List[str]:
        return [
            "refine('Improve this paragraph', initial_answer='...')",
            "refine('Make this explanation clearer')",
            "refine('Polish this email')",
            "refine('Enhance this product description')"
        ]
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute iterative refinement"""
        start_time = time.time()
        
        # Configuration
        max_rounds = kwargs.get('max_rounds', 3)
        alternatives_per_round = kwargs.get('alternatives', 3)
        initial_answer = kwargs.get('initial_answer', None)
        convergence_threshold = kwargs.get('convergence_threshold', 0.95)
        
        # Get initial answer if not provided
        if not initial_answer:
            initial_answer = self._generate_initial(question)
        
        # Track refinement history
        refinement_history = []
        current_answer = initial_answer
        current_confidence = 0.7  # Starting confidence
        
        # Refinement loop
        for round_num in range(max_rounds):
            # Generate alternatives
            alternatives = self._generate_alternatives(
                question, current_answer, alternatives_per_round
            )
            
            # Include current answer in evaluation
            candidates = [current_answer] + alternatives
            
            # Evaluate all candidates
            scores = self._evaluate_candidates(question, candidates)
            
            # Select best
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_answer = candidates[best_idx]
            best_score = scores[best_idx]
            
            # Record refinement step
            step = RefinementStep(
                iteration=round_num + 1,
                current_answer=current_answer,
                alternatives=alternatives,
                selected=best_answer,
                confidence=best_score
            )
            
            # Calculate improvement
            step.improvement = best_score - current_confidence
            refinement_history.append(step)
            
            # Notify visualizer if present
            if self.visualizer:
                self.visualizer.add_refinement_step({
                    'round': round_num + 1,
                    'candidates': candidates,
                    'scores': scores,
                    'selected': best_idx,
                    'improvement': step.improvement
                })
            
            # Update current answer and confidence
            current_answer = best_answer
            current_confidence = best_score
            
            # Check convergence
            if current_confidence >= convergence_threshold:
                break
            
            # No improvement, stop
            if step.improvement < 0.01:
                break
        
        # Build reasoning tree
        reasoning_tree = self._build_refinement_tree(refinement_history)
        
        # Calculate cost
        if self.llm:
            cost = self.llm.get_cost_estimate()
        else:
            # Simplified cost model for testing
            total_evaluations = sum(len(step.alternatives) + 1 for step in refinement_history)
            cost = total_evaluations * 0.001
        
        # Collect all alternatives considered
        all_alternatives = []
        for step in refinement_history:
            all_alternatives.extend(step.alternatives)
        
        return ReasoningResult(
            answer=current_answer,
            confidence=current_confidence,
            reasoning_tree=reasoning_tree,
            mode='refine',
            cost=cost,
            time_elapsed=time.time() - start_time,
            metadata={
                'question': question,
                'initial_answer': initial_answer,
                'rounds_completed': len(refinement_history),
                'total_alternatives': len(all_alternatives),
                'final_improvement': current_confidence - 0.7,
                'converged': current_confidence >= convergence_threshold
            },
            alternatives=all_alternatives[-3:]  # Last 3 alternatives
        )
    
    def _generate_initial(self, question: str) -> str:
        """Generate initial answer if not provided"""
        if self.llm:
            # Use LLM to generate initial answer
            prompt = f"Provide an initial answer to: {question}"
            return self.llm.client.generate(prompt, temperature=0.7)
        else:
            # Fallback for testing
            return f"Initial response to: {question}. This is a basic answer that will be refined."
    
    def _generate_alternatives(self, question: str, current: str, count: int) -> List[str]:
        """Generate alternative answers"""
        if self.llm:
            # Use real LLM to generate alternatives
            return self.llm.generate_alternatives(question, current, count)
        else:
            # Fallback for testing
            alternatives = []
            
            strategies = [
                "more detailed",
                "more concise",
                "different perspective",
                "with examples",
                "more formal",
                "more casual",
                "with analogies",
                "step-by-step"
            ]
            
            for i in range(count):
                strategy = strategies[i % len(strategies)]
                alt = f"{current} [Refined to be {strategy}]"
                alternatives.append(alt)
            
            return alternatives
    
    def _evaluate_candidates(self, question: str, candidates: List[str]) -> List[float]:
        """Evaluate all candidate answers"""
        if self.llm:
            # Use real LLM to evaluate candidates
            return self.llm.evaluate_answers(question, candidates)
        else:
            # Fallback for testing
            import random
            
            scores = []
            for i, candidate in enumerate(candidates):
                # Simulate evaluation with some improvement bias
                base_score = 0.7 + (i * 0.05)  # Later alternatives tend to be better
                noise = random.uniform(-0.1, 0.1)
                score = min(base_score + noise, 1.0)
                scores.append(score)
            
            return scores
    
    def _build_refinement_tree(self, history: List[RefinementStep]) -> Dict[str, Any]:
        """Build tree representation of refinement process"""
        tree = {
            'type': 'refinement',
            'steps': []
        }
        
        for step in history:
            tree['steps'].append({
                'iteration': step.iteration,
                'current': step.current_answer[:100] + '...',
                'alternatives_count': len(step.alternatives),
                'selected': step.selected[:100] + '...',
                'confidence': step.confidence,
                'improvement': step.improvement
            })
        
        return tree
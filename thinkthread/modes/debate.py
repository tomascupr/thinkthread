"""
Debate Mode - Multi-perspective reasoning through argument synthesis

Best for: controversial topics, balanced analysis, exploring different viewpoints
"""

from typing import Dict, Any, List, Optional
import time
from .base import ReasoningMode, ReasoningResult


class Perspective:
    """Represents one perspective in the debate"""
    
    def __init__(self, stance: str, argument: str, supporting_points: List[str]):
        self.stance = stance
        self.argument = argument
        self.supporting_points = supporting_points
        self.strength = 0.0
        self.weaknesses = []
        self.rebuttals = {}


class DebateMode(ReasoningMode):
    """
    Multi-perspective reasoning through structured debate.
    
    This mode generates multiple perspectives on a topic, has them
    debate each other, and synthesizes a balanced conclusion.
    """
    
    @property
    def characteristics(self) -> Dict[str, Any]:
        return {
            'thinking_style': 'dialectical',
            'breadth_vs_depth': 'balanced',
            'token_usage': 'high',
            'typical_duration': '10-20s',
            'best_for': ['controversial topics', 'balanced analysis', 'pros/cons', 'comparisons']
        }
    
    @property
    def examples(self) -> List[str]:
        return [
            "debate('Is AI consciousness possible?')",
            "debate('Remote work vs office work')",
            "debate('Nuclear energy: solution or problem?')",
            "debate('Privacy vs security in digital age')"
        ]
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute multi-perspective debate"""
        start_time = time.time()
        
        # Configuration
        perspectives_count = kwargs.get('perspectives', 3)
        debate_rounds = kwargs.get('rounds', 2)
        
        # Generate perspectives
        perspectives = self._generate_perspectives(question, perspectives_count)
        
        # Initial arguments
        for perspective in perspectives:
            if self.visualizer:
                self.visualizer.add_perspective({
                    'stance': perspective.stance,
                    'argument': perspective.argument,
                    'initial': True
                })
        
        # Debate rounds
        debate_history = []
        
        for round_num in range(debate_rounds):
            round_data = {
                'round': round_num + 1,
                'exchanges': []
            }
            
            # Each perspective responds to others
            for i, perspective in enumerate(perspectives):
                for j, other in enumerate(perspectives):
                    if i != j:
                        rebuttal = self._generate_rebuttal(
                            perspective, other, question
                        )
                        perspective.rebuttals[other.stance] = rebuttal
                        
                        round_data['exchanges'].append({
                            'from': perspective.stance,
                            'to': other.stance,
                            'rebuttal': rebuttal
                        })
                        
                        if self.visualizer:
                            self.visualizer.add_exchange({
                                'round': round_num + 1,
                                'from': perspective.stance,
                                'to': other.stance,
                                'content': rebuttal
                            })
            
            debate_history.append(round_data)
            
            # Strengthen or weaken arguments based on debate
            self._update_strengths(perspectives)
        
        # Synthesize conclusion
        synthesis = self._synthesize_conclusion(question, perspectives, debate_history)
        
        # Build reasoning tree
        reasoning_tree = self._build_debate_tree(perspectives, debate_history, synthesis)
        
        # Calculate confidence based on perspective agreement
        confidence = self._calculate_consensus(perspectives)
        
        # Cost calculation
        if self.llm:
            cost = self.llm.get_cost_estimate()
        else:
            # Simplified for testing
            total_exchanges = sum(len(r['exchanges']) for r in debate_history)
            cost = (len(perspectives) + total_exchanges) * 0.002
        
        return ReasoningResult(
            answer=synthesis,
            confidence=confidence,
            reasoning_tree=reasoning_tree,
            mode='debate',
            cost=cost,
            time_elapsed=time.time() - start_time,
            metadata={
                'question': question,
                'perspectives_count': len(perspectives),
                'debate_rounds': debate_rounds,
                'strongest_stance': max(perspectives, key=lambda p: p.strength).stance,
                'consensus_level': 'high' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'low'
            },
            alternatives=[p.argument for p in perspectives]
        )
    
    def _generate_perspectives(self, question: str, count: int) -> List[Perspective]:
        """Generate different perspectives on the question"""
        perspectives = []
        
        if self.llm:
            # Use real LLM to generate perspectives
            stances = ["supportive", "opposing", "neutral", "pragmatic"][:count]
            llm_perspectives = self.llm.generate_perspectives(question, stances)
            
            for persp_data in llm_perspectives:
                perspective = Perspective(
                    stance=persp_data["stance"],
                    argument=persp_data["argument"],
                    supporting_points=persp_data["supporting_points"]
                )
                perspective.strength = 0.5  # Neutral starting strength
                perspective.weaknesses = persp_data.get("weaknesses", [])
                perspectives.append(perspective)
        else:
            # Fallback for testing
            stance_templates = [
                ("supportive", "Yes, this is beneficial because"),
                ("opposing", "No, this is problematic because"),
                ("neutral", "It depends on the context because"),
                ("pragmatic", "From a practical standpoint")
            ]
            
            for i in range(min(count, len(stance_templates))):
                stance, prefix = stance_templates[i]
                
                perspective = Perspective(
                    stance=stance,
                    argument=f"{prefix} {question} - {stance} perspective",
                    supporting_points=[
                        f"Point 1 supporting {stance} view",
                        f"Point 2 supporting {stance} view",
                        f"Evidence for {stance} position"
                    ]
                )
                perspective.strength = 0.5  # Neutral starting strength
                perspectives.append(perspective)
        
        return perspectives
    
    def _generate_rebuttal(self, perspective: Perspective, other: Perspective, 
                          question: str) -> str:
        """Generate rebuttal from one perspective to another"""
        if self.llm:
            # Use real LLM to generate rebuttal
            persp_dict = {
                "stance": perspective.stance,
                "argument": perspective.argument,
                "supporting_points": perspective.supporting_points
            }
            other_dict = {
                "stance": other.stance,
                "argument": other.argument,
                "supporting_points": other.supporting_points
            }
            return self.llm.generate_rebuttal(persp_dict, other_dict, question)
        else:
            # Fallback for testing
            return f"{perspective.stance} response to {other.stance}: " \
                   f"While the {other.stance} view has merit, consider that..."
    
    def _update_strengths(self, perspectives: List[Perspective]):
        """Update argument strengths based on debate"""
        if self.llm:
            # Use LLM to evaluate strength after rebuttals
            for perspective in perspectives:
                # Evaluate based on rebuttals received and given
                context = f"After debate exchanges, evaluate the strength of {perspective.stance} perspective"
                strength = self.llm.evaluate_thought(perspective.argument, context)
                perspective.strength = strength
        else:
            # Fallback for testing
            import random
            for perspective in perspectives:
                adjustment = random.uniform(-0.1, 0.2)
                perspective.strength = max(0, min(1, perspective.strength + adjustment))
    
    def _synthesize_conclusion(self, question: str, perspectives: List[Perspective],
                              debate_history: List[Dict]) -> str:
        """Synthesize a balanced conclusion from all perspectives"""
        if self.llm:
            # Use real LLM to synthesize conclusion
            persp_list = [
                {
                    "stance": p.stance,
                    "argument": p.argument,
                    "strength": p.strength,
                    "supporting_points": p.supporting_points
                }
                for p in perspectives
            ]
            exchanges = [
                exchange for round_data in debate_history 
                for exchange in round_data.get('exchanges', [])
            ]
            return self.llm.synthesize_debate(question, persp_list, exchanges)
        else:
            # Fallback for testing
            strongest = max(perspectives, key=lambda p: p.strength)
            
            synthesis = f"After examining '{question}' from multiple perspectives:\n\n"
            
            # Summarize each perspective
            for p in perspectives:
                synthesis += f"- {p.stance.capitalize()} view: {p.argument}\n"
            
            synthesis += f"\nThe {strongest.stance} perspective appears strongest, "
            synthesis += "though each viewpoint contributes valuable insights. "
            synthesis += "A balanced approach would consider all these factors."
            
            return synthesis
    
    def _calculate_consensus(self, perspectives: List[Perspective]) -> float:
        """Calculate consensus level among perspectives"""
        if not perspectives:
            return 0.0
        
        # Calculate variance in strengths
        avg_strength = sum(p.strength for p in perspectives) / len(perspectives)
        variance = sum((p.strength - avg_strength) ** 2 for p in perspectives) / len(perspectives)
        
        # Lower variance means higher consensus
        consensus = 1.0 - min(variance * 2, 1.0)
        
        return consensus
    
    def _build_debate_tree(self, perspectives: List[Perspective], 
                          debate_history: List[Dict], synthesis: str) -> Dict[str, Any]:
        """Build tree representation of debate"""
        tree = {
            'type': 'debate',
            'perspectives': [
                {
                    'stance': p.stance,
                    'argument': p.argument,
                    'supporting_points': p.supporting_points,
                    'final_strength': p.strength,
                    'rebuttals': list(p.rebuttals.values())
                }
                for p in perspectives
            ],
            'debate_rounds': debate_history,
            'synthesis': synthesis
        }
        
        return tree
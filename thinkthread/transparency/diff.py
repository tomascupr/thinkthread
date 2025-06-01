"""
Reasoning Diff - Compare different reasoning attempts
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ReasoningComparison:
    """Result of comparing two reasoning attempts"""
    divergence_point: Optional[str]
    confidence_delta: float
    token_usage_delta: int
    cost_delta: float
    time_delta: float
    unique_insights_1: List[str]
    unique_insights_2: List[str]
    recommendation: str
    similarity_score: float


class ReasoningDiff:
    """Compare different reasoning attempts"""
    
    def compare(self, result1: 'ReasoningResult', result2: 'ReasoningResult') -> ReasoningComparison:
        """Compare two reasoning results"""
        
        # Find where reasoning paths diverged
        divergence = self._find_divergence_point(result1, result2)
        
        # Calculate deltas
        confidence_delta = result2.confidence - result1.confidence
        
        # Token usage (simplified - would need actual token counts)
        tokens1 = result1.metadata.get('total_tokens', 1000)
        tokens2 = result2.metadata.get('total_tokens', 1000)
        token_delta = tokens2 - tokens1
        
        # Cost and time deltas
        cost_delta = result2.cost - result1.cost
        time_delta = result2.time_elapsed - result1.time_elapsed
        
        # Extract unique insights
        insights1 = self._extract_unique_insights(result1, result2)
        insights2 = self._extract_unique_insights(result2, result1)
        
        # Calculate similarity
        similarity = self._calculate_similarity(result1.answer, result2.answer)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            result1, result2, confidence_delta, cost_delta
        )
        
        return ReasoningComparison(
            divergence_point=divergence,
            confidence_delta=confidence_delta,
            token_usage_delta=token_delta,
            cost_delta=cost_delta,
            time_delta=time_delta,
            unique_insights_1=insights1,
            unique_insights_2=insights2,
            recommendation=recommendation,
            similarity_score=similarity
        )
    
    def _find_divergence_point(self, result1: 'ReasoningResult', 
                               result2: 'ReasoningResult') -> Optional[str]:
        """Find where reasoning paths diverged"""
        # Compare reasoning trees
        tree1 = result1.reasoning_tree
        tree2 = result2.reasoning_tree
        
        # Different reasoning modes diverge immediately
        if result1.mode != result2.mode:
            return f"Different modes: {result1.mode} vs {result2.mode}"
        
        # For same mode, find structural differences
        if result1.mode == 'explore':
            return self._find_tree_divergence(tree1, tree2)
        elif result1.mode == 'refine':
            return self._find_refinement_divergence(tree1, tree2)
        
        return "Unknown divergence point"
    
    def _find_tree_divergence(self, tree1: Dict, tree2: Dict) -> str:
        """Find divergence in tree-based reasoning"""
        # Simplified - would traverse trees to find first difference
        if tree1.get('children', []) != tree2.get('children', []):
            return "Different exploration paths taken"
        return "Trees are structurally similar"
    
    def _find_refinement_divergence(self, tree1: Dict, tree2: Dict) -> str:
        """Find divergence in refinement reasoning"""
        steps1 = tree1.get('steps', [])
        steps2 = tree2.get('steps', [])
        
        if len(steps1) != len(steps2):
            return f"Different number of refinement rounds: {len(steps1)} vs {len(steps2)}"
        
        for i, (s1, s2) in enumerate(zip(steps1, steps2)):
            if s1.get('alternatives_count') != s2.get('alternatives_count'):
                return f"Different alternatives generated in round {i+1}"
                
        return "Similar refinement process"
    
    def _extract_unique_insights(self, result1: 'ReasoningResult', 
                                result2: 'ReasoningResult') -> List[str]:
        """Extract insights unique to result1"""
        # Simplified - in real implementation would use NLP
        insights = []
        
        # Check if result1 has unique alternatives
        if result1.alternatives:
            for alt in result1.alternatives:
                if alt not in result2.alternatives:
                    insights.append(f"Alternative approach: {alt[:50]}...")
                    
        # Check metadata for unique findings
        if result1.metadata.get('best_path'):
            insights.append(f"Unique path: {result1.metadata['best_path']}")
            
        return insights[:3]  # Limit to top 3
    
    def _calculate_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between answers"""
        # Simplified - in real implementation would use embeddings
        
        # Basic length-based similarity
        len_diff = abs(len(answer1) - len(answer2))
        max_len = max(len(answer1), len(answer2))
        
        if max_len == 0:
            return 1.0
            
        length_similarity = 1.0 - (len_diff / max_len)
        
        # Word overlap (simplified)
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return length_similarity
            
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        word_similarity = overlap / total if total > 0 else 0
        
        # Combined similarity
        return (length_similarity + word_similarity) / 2
    
    def _generate_recommendation(self, result1: 'ReasoningResult', 
                               result2: 'ReasoningResult',
                               confidence_delta: float,
                               cost_delta: float) -> str:
        """Generate recommendation based on comparison"""
        
        recommendations = []
        
        # Confidence comparison
        if abs(confidence_delta) < 0.05:
            recommendations.append("Both approaches yielded similar confidence")
        elif confidence_delta > 0:
            recommendations.append(f"{result2.mode} mode showed {confidence_delta:.0%} higher confidence")
        else:
            recommendations.append(f"{result1.mode} mode showed {-confidence_delta:.0%} higher confidence")
            
        # Cost comparison
        if cost_delta > 0:
            cost_increase = (cost_delta / result1.cost) * 100
            recommendations.append(f"{result2.mode} cost {cost_increase:.0f}% more")
        elif cost_delta < 0:
            cost_decrease = (-cost_delta / result1.cost) * 100
            recommendations.append(f"{result2.mode} cost {cost_decrease:.0f}% less")
            
        # Mode-specific recommendations
        if result1.mode != result2.mode:
            if result2.confidence > result1.confidence and cost_delta < result1.cost * 0.5:
                recommendations.append(f"Consider using {result2.mode} for this type of question")
                
        return ". ".join(recommendations)
    
    def visualize(self, comparison: ReasoningComparison):
        """Visualize the comparison (placeholder for actual implementation)"""
        print("\n=== Reasoning Comparison ===")
        print(f"Divergence: {comparison.divergence_point}")
        print(f"Confidence delta: {comparison.confidence_delta:+.2%}")
        print(f"Cost delta: ${comparison.cost_delta:+.4f}")
        print(f"Time delta: {comparison.time_delta:+.2f}s")
        print(f"Answer similarity: {comparison.similarity_score:.2%}")
        print(f"\nRecommendation: {comparison.recommendation}")
        
        if comparison.unique_insights_1:
            print("\nUnique to first approach:")
            for insight in comparison.unique_insights_1:
                print(f"  - {insight}")
                
        if comparison.unique_insights_2:
            print("\nUnique to second approach:")
            for insight in comparison.unique_insights_2:
                print(f"  - {insight}")
"""
Explore Mode - Tree-based exploration of possibilities

Best for: creative tasks, design problems, brainstorming, what-if scenarios
"""

from typing import Dict, Any, List, Optional
import time
import random
from .base import ReasoningMode, ReasoningResult


class ThoughtNode:
    """Represents a single thought in the exploration tree"""
    
    def __init__(self, content: str, parent: Optional['ThoughtNode'] = None):
        self.content = content
        self.parent = parent
        self.children: List['ThoughtNode'] = []
        self.score = 0.0
        self.pruned = False
        self.prune_reason = ""
        self.id = f"node_{id(self)}"
        self.depth = (parent.depth + 1) if parent else 0
    
    def add_child(self, content: str) -> 'ThoughtNode':
        """Add a child thought"""
        child = ThoughtNode(content, self)
        self.children.append(child)
        return child
    
    def prune(self, reason: str):
        """Prune this branch"""
        self.pruned = True
        self.prune_reason = reason
    
    def get_path(self) -> List[str]:
        """Get the path from root to this node"""
        path = []
        node = self
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))


class ExploreMode(ReasoningMode):
    """
    Tree-based exploration of possibilities.
    
    This mode generates multiple initial thoughts and explores each branch,
    pruning less promising paths and expanding successful ones.
    """
    
    @property
    def characteristics(self) -> Dict[str, Any]:
        return {
            'thinking_style': 'divergent',
            'breadth_vs_depth': 'breadth-focused',
            'token_usage': 'high',
            'typical_duration': '10-30s',
            'best_for': ['design', 'brainstorming', 'what-if scenarios', 'creative tasks']
        }
    
    @property
    def examples(self) -> List[str]:
        return [
            "explore('Design a sustainable city')",
            "explore('What if we eliminated money?')",
            "explore('New product ideas for teenagers')",
            "explore('How might we colonize Mars?')"
        ]
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute tree-based exploration"""
        start_time = time.time()
        
        # Configuration
        max_depth = kwargs.get('max_depth', 3)
        branching_factor = kwargs.get('branching_factor', 3)
        beam_width = kwargs.get('beam_width', 5)
        
        # Initialize root
        root = ThoughtNode(f"Exploring: {question}")
        
        # Track best nodes for beam search
        beam = [root]
        all_nodes = [root]
        
        # Notify visualizer if present
        if self.visualizer:
            self.visualizer.add_node({
                'id': root.id,
                'content': root.content,
                'score': 0.0,
                'parent': None
            })
        
        # Exploration loop
        for depth in range(max_depth):
            new_beam = []
            
            for node in beam:
                if node.pruned:
                    continue
                
                # Generate child thoughts
                children = self._generate_children(node, branching_factor)
                
                for child in children:
                    all_nodes.append(child)
                    
                    # Score the thought
                    child.score = self._score_thought(child)
                    
                    # Notify visualizer
                    if self.visualizer:
                        self.visualizer.add_node({
                            'id': child.id,
                            'content': child.content,
                            'score': child.score,
                            'parent': node.id,
                            'depth': child.depth
                        })
                    
                    # Prune if score is too low
                    if child.score < 0.3:
                        child.prune("Low score")
                        if self.visualizer:
                            self.visualizer.prune_branch(child.id, "Low score")
                    else:
                        new_beam.append(child)
            
            # Keep only top nodes (beam search)
            new_beam.sort(key=lambda n: n.score, reverse=True)
            beam = new_beam[:beam_width]
        
        # Find best path
        best_node = max(all_nodes, key=lambda n: n.score if not n.pruned else -1)
        best_path = best_node.get_path()
        
        # Generate final answer from best path
        answer = self._synthesize_answer(best_path, question)
        
        # Build reasoning tree for result
        reasoning_tree = self._build_tree_dict(root)
        
        # Calculate cost
        if self.llm:
            cost = self.llm.get_cost_estimate()
        else:
            # Simplified estimate for testing
            total_tokens = len(all_nodes) * 100
            cost = total_tokens * 0.00001
        
        return ReasoningResult(
            answer=answer,
            confidence=best_node.score,
            reasoning_tree=reasoning_tree,
            mode='explore',
            cost=cost,
            time_elapsed=time.time() - start_time,
            metadata={
                'question': question,
                'total_nodes': len(all_nodes),
                'best_path': best_path,
                'max_depth_reached': max(n.depth for n in all_nodes),
                'pruned_nodes': sum(1 for n in all_nodes if n.pruned)
            },
            alternatives=[n.content for n in beam[:3] if n != best_node]
        )
    
    def _generate_children(self, node: ThoughtNode, count: int) -> List[ThoughtNode]:
        """Generate child thoughts for a node"""
        children = []
        
        if self.llm:
            # Use real LLM to generate thoughts
            thoughts = self.llm.generate_thoughts(node.content, count)
            for thought in thoughts:
                child = node.add_child(thought)
                children.append(child)
        else:
            # Fallback for testing
            base_thoughts = [
                "Consider the environmental impact",
                "Think about scalability",
                "Explore technological solutions",
                "Examine social implications",
                "Analyze economic factors",
                "Look at historical precedents",
                "Consider ethical dimensions",
                "Evaluate practical constraints"
            ]
            
            for i in range(count):
                thought = f"{random.choice(base_thoughts)} of {node.content}"
                child = node.add_child(thought)
                children.append(child)
        
        return children
    
    def _score_thought(self, node: ThoughtNode) -> float:
        """Score a thought for quality/promise"""
        if self.llm:
            # Use real LLM to evaluate thought quality
            # Get the full context by traversing up the tree
            context = " -> ".join(node.get_path())
            score = self.llm.evaluate_thought(node.content, context)
            
            # Add depth bonus to encourage exploration
            depth_bonus = min(node.depth * 0.05, 0.15)
            return min(score + depth_bonus, 1.0)
        else:
            # Fallback for testing
            depth_bonus = min(node.depth * 0.1, 0.3)
            quality = random.uniform(0.4, 0.9)
            length_penalty = 0.1 if len(node.content) > 100 else 0
            return min(quality + depth_bonus - length_penalty, 1.0)
    
    def _synthesize_answer(self, path: List[str], question: str) -> str:
        """Synthesize final answer from best path"""
        if self.llm:
            # Use real LLM to synthesize comprehensive answer
            return self.llm.synthesize_answer(question, path[1:])  # Skip root
        else:
            # Fallback for testing
            answer = f"Based on exploring '{question}', here's a comprehensive approach:\n\n"
            
            for i, thought in enumerate(path[1:], 1):  # Skip root
                answer += f"{i}. {thought}\n"
            
            answer += "\nThis exploration considered multiple perspectives and paths, "
            answer += "ultimately focusing on the most promising approach based on "
            answer += "feasibility, impact, and alignment with the goal."
            
            return answer
    
    def _build_tree_dict(self, root: ThoughtNode) -> Dict[str, Any]:
        """Convert tree to dictionary for result"""
        def node_to_dict(node: ThoughtNode) -> Dict[str, Any]:
            return {
                'id': node.id,
                'content': node.content,
                'score': node.score,
                'pruned': node.pruned,
                'prune_reason': node.prune_reason,
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(root)
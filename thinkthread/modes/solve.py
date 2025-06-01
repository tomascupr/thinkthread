"""
Solve Mode - Solution-focused reasoning for specific problems

Best for: problem-solving, finding actionable solutions, step-by-step planning
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from .base import ReasoningMode, ReasoningResult


class Solution:
    """Represents a potential solution"""
    
    def __init__(self, approach: str, steps: List[str], pros: List[str], cons: List[str]):
        self.approach = approach
        self.steps = steps
        self.pros = pros
        self.cons = cons
        self.feasibility_score = 0.0
        self.impact_score = 0.0
        self.overall_score = 0.0


class SolveMode(ReasoningMode):
    """
    Solution-focused reasoning for specific problems.
    
    This mode identifies the problem, generates multiple solution approaches,
    evaluates them, and provides actionable steps.
    """
    
    @property
    def characteristics(self) -> Dict[str, Any]:
        return {
            'thinking_style': 'systematic',
            'breadth_vs_depth': 'depth-focused',
            'token_usage': 'medium',
            'typical_duration': '5-15s',
            'best_for': ['problem-solving', 'action plans', 'troubleshooting', 'optimization']
        }
    
    @property
    def examples(self) -> List[str]:
        return [
            "solve('How to reduce customer churn by 30%')",
            "solve('Fix memory leak in production')",
            "solve('Improve team productivity')",
            "solve('Reduce manufacturing costs')"
        ]
    
    def execute(self, question: str, **kwargs) -> ReasoningResult:
        """Execute solution-focused reasoning"""
        start_time = time.time()
        
        # Configuration
        max_solutions = kwargs.get('max_solutions', 3)
        prioritize = kwargs.get('prioritize', 'balanced')  # balanced, quick, impactful
        
        # Problem analysis
        problem_analysis = self._analyze_problem(question)
        
        # Generate solution approaches
        solutions = self._generate_solutions(question, problem_analysis, max_solutions)
        
        # Evaluate solutions
        for solution in solutions:
            self._evaluate_solution(solution, problem_analysis, prioritize)
            
            if self.visualizer:
                self.visualizer.add_solution({
                    'approach': solution.approach,
                    'feasibility': solution.feasibility_score,
                    'impact': solution.impact_score,
                    'overall': solution.overall_score
                })
        
        # Rank solutions
        solutions.sort(key=lambda s: s.overall_score, reverse=True)
        best_solution = solutions[0]
        
        # Create detailed action plan for best solution
        action_plan = self._create_action_plan(best_solution, problem_analysis)
        
        # Build final answer
        answer = self._format_solution(best_solution, action_plan, problem_analysis)
        
        # Build reasoning tree
        reasoning_tree = self._build_solution_tree(
            problem_analysis, solutions, best_solution, action_plan
        )
        
        # Calculate confidence based on solution quality
        confidence = best_solution.overall_score
        
        # Cost calculation
        if self.llm:
            cost = self.llm.get_cost_estimate()
        else:
            # Simplified for testing
            cost = (1 + len(solutions) * 2) * 0.001  # Analysis + solution generation
        
        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            reasoning_tree=reasoning_tree,
            mode='solve',
            cost=cost,
            time_elapsed=time.time() - start_time,
            metadata={
                'question': question,
                'problem_type': problem_analysis['type'],
                'solutions_generated': len(solutions),
                'best_approach': best_solution.approach,
                'feasibility': best_solution.feasibility_score,
                'expected_impact': best_solution.impact_score,
                'action_steps': len(action_plan['steps'])
            },
            alternatives=[s.approach for s in solutions[1:3]]  # Next best solutions
        )
    
    def _analyze_problem(self, question: str) -> Dict[str, Any]:
        """Analyze the problem to understand constraints and goals"""
        if self.llm:
            # Use real LLM to analyze problem
            return self.llm.analyze_problem(question)
        else:
            # Fallback for testing
            problem_type = 'optimization' if 'reduce' in question or 'improve' in question else 'general'
            
            return {
                'type': problem_type,
                'constraints': ['time', 'resources', 'complexity'],
                'success_criteria': 'measurable improvement',
                'stakeholders': ['users', 'team', 'business'],
                'urgency': 'medium'
            }
    
    def _generate_solutions(self, question: str, analysis: Dict, 
                           count: int) -> List[Solution]:
        """Generate potential solutions"""
        if self.llm:
            # Use real LLM to generate solutions
            solution_data = self.llm.generate_solutions(question, analysis, count)
            
            solutions = []
            for sol_dict in solution_data:
                solution = Solution(
                    approach=sol_dict.get('approach', f'Solution {len(solutions)+1}'),
                    steps=sol_dict.get('steps', []),
                    pros=sol_dict.get('pros', []),
                    cons=sol_dict.get('cons', [])
                )
                solutions.append(solution)
            
            return solutions
        else:
            # Fallback for testing
            solution_templates = [
                {
                    'approach': 'Quick wins approach',
                    'steps': [
                        'Identify low-hanging fruit',
                        'Implement quick fixes',
                        'Measure immediate impact',
                        'Iterate based on results'
                    ],
                    'pros': ['Fast implementation', 'Low risk', 'Quick feedback'],
                    'cons': ['Limited impact', 'May not address root cause']
                },
                {
                    'approach': 'Systematic overhaul',
                    'steps': [
                        'Comprehensive analysis',
                        'Design new system',
                        'Phased implementation',
                        'Monitor and optimize'
                    ],
                    'pros': ['Addresses root causes', 'Long-term solution', 'Scalable'],
                    'cons': ['Time-intensive', 'Higher risk', 'Resource heavy']
                },
                {
                    'approach': 'Incremental improvement',
                    'steps': [
                        'Baseline measurement',
                        'Small iterative changes',
                        'Continuous monitoring',
                        'Compound improvements'
                    ],
                    'pros': ['Low risk', 'Continuous progress', 'Team buy-in'],
                    'cons': ['Slower results', 'Requires discipline']
                },
                {
                    'approach': 'Technology-driven solution',
                    'steps': [
                        'Identify automation opportunities',
                        'Evaluate tools/platforms',
                        'Pilot implementation',
                        'Scale successful pilots'
                    ],
                    'pros': ['Efficiency gains', 'Modern approach', 'Scalable'],
                    'cons': ['Initial investment', 'Learning curve', 'Dependency']
                }
            ]
            
            solutions = []
            for i in range(min(count, len(solution_templates))):
                template = solution_templates[i]
                solution = Solution(
                    approach=template['approach'],
                    steps=template['steps'],
                    pros=template['pros'],
                    cons=template['cons']
                )
                solutions.append(solution)
            
            return solutions
    
    def _evaluate_solution(self, solution: Solution, analysis: Dict, 
                          priority: str) -> None:
        """Evaluate a solution's feasibility and impact"""
        if self.llm:
            # Use LLM to evaluate feasibility and impact
            context = f"Problem type: {analysis['type']}, Constraints: {analysis['constraints']}"
            
            # Evaluate feasibility
            feasibility_prompt = f"Evaluate feasibility of '{solution.approach}' considering: {context}"
            solution.feasibility_score = self.llm.evaluate_thought(solution.approach, feasibility_prompt)
            
            # Evaluate impact
            impact_prompt = f"Evaluate potential impact of '{solution.approach}' for solving the problem"
            solution.impact_score = self.llm.evaluate_thought(solution.approach, impact_prompt)
        else:
            # Fallback for testing
            import random
            solution.feasibility_score = random.uniform(0.6, 0.95)
            solution.impact_score = random.uniform(0.5, 0.9)
        
        # Adjust based on priority
        if priority == 'quick':
            weight_feasibility = 0.7
            weight_impact = 0.3
        elif priority == 'impactful':
            weight_feasibility = 0.3
            weight_impact = 0.7
        else:  # balanced
            weight_feasibility = 0.5
            weight_impact = 0.5
        
        solution.overall_score = (
            solution.feasibility_score * weight_feasibility +
            solution.impact_score * weight_impact
        )
    
    def _create_action_plan(self, solution: Solution, analysis: Dict) -> Dict[str, Any]:
        """Create detailed action plan for the chosen solution"""
        # In real implementation, this would expand on the solution steps
        
        action_plan = {
            'approach': solution.approach,
            'timeline': '4-6 weeks',
            'steps': []
        }
        
        for i, step in enumerate(solution.steps):
            action_plan['steps'].append({
                'week': i + 1,
                'action': step,
                'deliverables': [f"Deliverable for {step}"],
                'success_metrics': [f"Metric for {step}"],
                'responsible': 'Team'
            })
        
        return action_plan
    
    def _format_solution(self, solution: Solution, action_plan: Dict, 
                        analysis: Dict) -> str:
        """Format the solution as a clear answer"""
        answer = f"To address this problem, I recommend the '{solution.approach}' approach.\n\n"
        
        answer += "**Why this solution:**\n"
        for pro in solution.pros:
            answer += f"✓ {pro}\n"
        
        answer += "\n**Implementation steps:**\n"
        for i, step in enumerate(solution.steps, 1):
            answer += f"{i}. {step}\n"
        
        answer += "\n**Expected outcome:**\n"
        answer += f"- Feasibility: {solution.feasibility_score:.0%}\n"
        answer += f"- Impact: {solution.impact_score:.0%}\n"
        answer += f"- Timeline: {action_plan['timeline']}\n"
        
        if solution.cons:
            answer += "\n**Considerations:**\n"
            for con in solution.cons:
                answer += f"- {con}\n"
        
        return answer
    
    def _build_solution_tree(self, analysis: Dict, solutions: List[Solution],
                            best: Solution, action_plan: Dict) -> Dict[str, Any]:
        """Build tree representation of solution process"""
        tree = {
            'type': 'solution',
            'problem_analysis': analysis,
            'solutions_evaluated': [
                {
                    'approach': s.approach,
                    'feasibility': s.feasibility_score,
                    'impact': s.impact_score,
                    'overall': s.overall_score,
                    'pros': s.pros,
                    'cons': s.cons
                }
                for s in solutions
            ],
            'selected_solution': {
                'approach': best.approach,
                'reason': 'Highest overall score',
                'action_plan': action_plan
            }
        }
        
        return tree
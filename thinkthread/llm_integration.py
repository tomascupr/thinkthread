"""
LLM Integration Layer for ThinkThread

This module provides the connection between reasoning modes and actual LLMs.
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
from abc import ABC, abstractmethod

# Import existing LLM clients from thinkthread_sdk
from thinkthread_sdk.llm import OpenAIClient, AnthropicClient, HuggingFaceClient
from thinkthread_sdk.llm.base import LLMClient


class LLMIntegration:
    """Manages LLM connections for reasoning modes"""
    
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> LLMClient:
        """Initialize the appropriate LLM client"""
        if self.provider == "auto":
            # Auto-detect available API keys
            if os.environ.get("OPENAI_API_KEY"):
                return OpenAIClient(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model_name="gpt-4-turbo-preview"
                )
            elif os.environ.get("ANTHROPIC_API_KEY"):
                return AnthropicClient(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name="claude-3-opus-20240229"
                )
            else:
                # Fallback to dummy for testing
                from thinkthread_sdk.llm.dummy import DummyLLMClient
                return DummyLLMClient()
        
        # Specific provider requested
        if self.provider == "openai":
            return OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model_name="gpt-4-turbo-preview"
            )
        elif self.provider == "anthropic":
            return AnthropicClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model_name="claude-3-opus-20240229"
            )
        else:
            from thinkthread_sdk.llm.dummy import DummyLLMClient
            return DummyLLMClient()
    
    def generate_thoughts(self, context: str, count: int = 3) -> List[str]:
        """Generate multiple thoughts for exploration"""
        prompt = f"""Given the context: "{context}"
        
Generate {count} diverse and creative thoughts or approaches to explore this further.
Each thought should be a different perspective or direction.

Format your response as a JSON array of strings:
["thought 1", "thought 2", "thought 3"]"""
        
        response = self.client.generate(prompt, temperature=0.8)
        
        try:
            # Parse JSON response
            thoughts = json.loads(response)
            if isinstance(thoughts, list):
                return thoughts[:count]
        except:
            # Fallback: split by newlines
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:count]
        
        return [f"Thought {i+1} about {context}" for i in range(count)]
    
    def evaluate_thought(self, thought: str, context: str) -> float:
        """Evaluate the quality/promise of a thought"""
        prompt = f"""Evaluate the following thought in the context of "{context}":

Thought: "{thought}"

Rate this thought's quality, relevance, and potential on a scale of 0 to 1.
Consider:
- Relevance to the context
- Creativity and insight
- Practical feasibility
- Potential impact

Respond with just a number between 0 and 1."""
        
        response = self.client.generate(prompt, temperature=0.3)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default middle score
    
    def generate_alternatives(self, question: str, current_answer: str, count: int = 3) -> List[str]:
        """Generate alternative answers for refinement"""
        prompt = f"""Question: {question}

Current answer: {current_answer}

Generate {count} alternative answers that improve upon the current one.
Each alternative should take a different approach:
1. More detailed and comprehensive
2. More concise and clear
3. Different perspective or framework

Format as JSON array of strings."""
        
        response = self.client.generate(prompt, temperature=0.7)
        
        try:
            alternatives = json.loads(response)
            if isinstance(alternatives, list):
                return alternatives[:count]
        except:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:count]
        
        return [f"{current_answer} [Alternative {i+1}]" for i in range(count)]
    
    def evaluate_answers(self, question: str, answers: List[str]) -> List[float]:
        """Evaluate multiple answers and return scores"""
        prompt = f"""Question: {question}

Evaluate these answers and score each from 0 to 1 based on:
- Accuracy and correctness
- Completeness
- Clarity
- Usefulness

Answers to evaluate:
"""
        for i, answer in enumerate(answers):
            prompt += f"\n{i+1}. {answer}\n"
        
        prompt += "\nReturn scores as JSON array of numbers: [0.8, 0.6, 0.9]"
        
        response = self.client.generate(prompt, temperature=0.3)
        
        try:
            scores = json.loads(response)
            if isinstance(scores, list):
                return [float(s) for s in scores[:len(answers)]]
        except:
            # Fallback scoring
            return [0.7 + (i * 0.05) for i in range(len(answers))]
    
    def generate_perspectives(self, question: str, stances: List[str]) -> List[Dict[str, Any]]:
        """Generate different perspectives for debate"""
        perspectives = []
        
        for stance in stances:
            prompt = f"""Question: {question}

Provide a {stance} perspective on this question.
Include:
1. Main argument
2. 3 supporting points
3. Potential weaknesses

Format as JSON:
{{
    "argument": "main argument",
    "supporting_points": ["point 1", "point 2", "point 3"],
    "weaknesses": ["weakness 1", "weakness 2"]
}}"""
            
            response = self.client.generate(prompt, temperature=0.7)
            
            try:
                data = json.loads(response)
                perspectives.append({
                    "stance": stance,
                    "argument": data.get("argument", f"{stance} perspective"),
                    "supporting_points": data.get("supporting_points", []),
                    "weaknesses": data.get("weaknesses", [])
                })
            except:
                # Fallback
                perspectives.append({
                    "stance": stance,
                    "argument": f"{stance} perspective on {question}",
                    "supporting_points": [f"Point {i+1}" for i in range(3)],
                    "weaknesses": ["Potential weakness"]
                })
        
        return perspectives
    
    def generate_rebuttal(self, perspective: Dict, target: Dict, question: str) -> str:
        """Generate rebuttal from one perspective to another"""
        prompt = f"""In a debate about "{question}":

The {perspective['stance']} perspective argues: {perspective['argument']}
The {target['stance']} perspective argues: {target['argument']}

Write a thoughtful rebuttal from the {perspective['stance']} perspective to the {target['stance']} perspective.
Address their main points while reinforcing your position."""
        
        response = self.client.generate(prompt, temperature=0.6)
        return response.strip()
    
    def synthesize_debate(self, question: str, perspectives: List[Dict], exchanges: List[Dict]) -> str:
        """Synthesize a balanced conclusion from debate"""
        prompt = f"""Question: {question}

After considering these perspectives:
"""
        for p in perspectives:
            prompt += f"\n- {p['stance']}: {p['argument']}"
        
        prompt += "\n\nSynthesize a balanced, nuanced conclusion that:"
        prompt += "\n1. Acknowledges the validity of different viewpoints"
        prompt += "\n2. Identifies common ground"
        prompt += "\n3. Provides a thoughtful recommendation"
        prompt += "\n4. Addresses remaining uncertainties"
        
        response = self.client.generate(prompt, temperature=0.5)
        return response.strip()
    
    def analyze_problem(self, question: str) -> Dict[str, Any]:
        """Analyze a problem for solution mode"""
        prompt = f"""Analyze this problem: {question}

Provide a structured analysis including:
1. Problem type (optimization, troubleshooting, design, etc.)
2. Key constraints
3. Success criteria
4. Stakeholders affected
5. Urgency level

Format as JSON."""
        
        response = self.client.generate(prompt, temperature=0.4)
        
        try:
            return json.loads(response)
        except:
            return {
                "type": "general",
                "constraints": ["time", "resources"],
                "success_criteria": "measurable improvement",
                "stakeholders": ["users", "team"],
                "urgency": "medium"
            }
    
    def generate_solutions(self, question: str, analysis: Dict, count: int = 3) -> List[Dict[str, Any]]:
        """Generate solution approaches"""
        prompt = f"""Problem: {question}

Analysis: {json.dumps(analysis, indent=2)}

Generate {count} different solution approaches.
For each solution provide:
1. Approach name
2. Step-by-step implementation
3. Pros and cons
4. Resource requirements

Format as JSON array of solutions."""
        
        response = self.client.generate(prompt, temperature=0.7)
        
        try:
            solutions = json.loads(response)
            return solutions[:count]
        except:
            # Fallback solutions
            return [
                {
                    "approach": f"Solution {i+1}",
                    "steps": ["Step 1", "Step 2", "Step 3"],
                    "pros": ["Pro 1", "Pro 2"],
                    "cons": ["Con 1"],
                    "resources": ["Time", "Budget"]
                }
                for i in range(count)
            ]
    
    def synthesize_answer(self, question: str, reasoning_path: List[str]) -> str:
        """Synthesize final answer from reasoning path"""
        prompt = f"""Question: {question}

Based on this reasoning path:
"""
        for i, step in enumerate(reasoning_path):
            prompt += f"\n{i+1}. {step}"
        
        prompt += "\n\nSynthesize a comprehensive, well-structured answer that:"
        prompt += "\n- Directly addresses the question"
        prompt += "\n- Incorporates insights from the reasoning process"
        prompt += "\n- Provides actionable information"
        prompt += "\n- Is clear and well-organized"
        
        response = self.client.generate(prompt, temperature=0.5)
        return response.strip()
    
    def get_token_count(self) -> int:
        """Get approximate token count for cost tracking"""
        # This would integrate with the actual LLM client's token counting
        if hasattr(self.client, 'get_token_count'):
            return self.client.get_token_count()
        return 0
    
    def get_cost_estimate(self) -> float:
        """Get cost estimate for the operations"""
        # This would integrate with actual pricing
        tokens = self.get_token_count()
        # Rough estimate: $0.01 per 1k tokens for GPT-4
        return tokens * 0.00001
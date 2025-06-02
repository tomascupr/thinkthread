"""
LLM Integration Layer for ThinkThread

This module provides the connection between reasoning modes and actual LLMs.
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import time
from abc import ABC, abstractmethod

# Import LLM clients from new internal location
from .llm import OpenAIClient, AnthropicClient, HuggingFaceClient, DummyLLMClient
from .llm.base import LLMClient


class RetryableLLMClient:
    """Wrapper that adds retry logic to any LLM client"""
    
    def __init__(self, client: LLMClient, max_retries: int = 3, base_delay: float = 1.0):
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with exponential backoff retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.client.generate(prompt, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                continue
        
        # If all retries failed, return error message
        return f"Error after {self.max_retries} attempts: {str(last_error)}"
    
    def generate_stream(self, prompt: str, **kwargs):
        """Stream generation with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                yield from self.client.generate_stream(prompt, **kwargs)
                return
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                continue
        
        # If all retries failed, yield error message
        yield f"Error after {self.max_retries} attempts: {str(last_error)}"


class LLMIntegration:
    """Manages LLM connections for reasoning modes"""
    
    def __init__(self, provider: str = "auto", enable_retry: bool = True):
        self.provider = provider
        self.enable_retry = enable_retry
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Union[LLMClient, RetryableLLMClient]:
        """Initialize the appropriate LLM client with optional retry wrapper"""
        base_client = None
        
        if self.provider == "auto":
            # Auto-detect available API keys
            if os.environ.get("OPENAI_API_KEY"):
                base_client = OpenAIClient(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model_name=os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
                )
            elif os.environ.get("ANTHROPIC_API_KEY"):
                base_client = AnthropicClient(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
                )
            else:
                # Fallback to dummy for testing
                base_client = DummyLLMClient()
        
        # Specific provider requested
        elif self.provider == "openai":
            base_client = OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model_name=os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
            )
        elif self.provider == "anthropic":
            base_client = AnthropicClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model_name=os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
            )
        else:
            base_client = DummyLLMClient()
        
        # Wrap with retry logic if enabled
        if self.enable_retry and not isinstance(base_client, type(None)):
            return RetryableLLMClient(base_client)
        
        return base_client
    
    def generate_thoughts(self, context: str, count: int = 3) -> List[str]:
        """Generate multiple thoughts for exploration"""
        prompt = f"""Given the context: "{context}"
        
Generate {count} diverse and creative thoughts or approaches to explore this further.
Each thought should be a different perspective or direction.

Format your response as a JSON array of strings:
["thought 1", "thought 2", "thought 3"]"""
        
        response = self.client.generate(prompt, temperature=0.8)
        
        # Check if response is an error
        if response.startswith("Error after"):
            return [f"Failed to generate thought {i+1}" for i in range(count)]
        
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
        
        # Check if response is an error
        if response.startswith("Error after"):
            return 0.5  # Default score on error
        
        try:
            score = float(response.strip())
            return max(0, min(1, score))  # Clamp to [0, 1]
        except:
            return 0.5  # Default middle score
    
    def generate_alternatives(self, prompt: str, count: int = 3) -> List[str]:
        """Generate alternative responses"""
        full_prompt = f"""Generate {count} different alternative responses to this request:

{prompt}

Provide {count} distinct approaches, perspectives, or solutions.
Format as a JSON array of strings."""
        
        response = self.client.generate(full_prompt, temperature=0.9)
        
        # Check if response is an error
        if response.startswith("Error after"):
            return [f"Alternative {i+1}" for i in range(count)]
        
        try:
            alternatives = json.loads(response)
            if isinstance(alternatives, list):
                return alternatives[:count]
        except:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:count]
        
        return [f"Alternative {i+1}" for i in range(count)]
    
    def refine_response(self, original: str, feedback: str) -> str:
        """Refine a response based on feedback"""
        prompt = f"""Original response:
{original}

Feedback for improvement:
{feedback}

Provide an improved version that addresses the feedback while maintaining the core value of the original."""
        
        return self.client.generate(prompt, temperature=0.5)
    
    def generate_perspectives(self, topic: str, count: int = 3) -> List[Dict[str, str]]:
        """Generate different perspectives for debate mode"""
        prompt = f"""Generate {count} different perspectives on this topic: "{topic}"

Each perspective should have:
- A clear stance or viewpoint
- Supporting arguments
- Potential concerns or weaknesses

Format as a JSON array of objects with 'stance', 'arguments', and 'concerns' fields."""
        
        response = self.client.generate(prompt, temperature=0.8)
        
        # Check if response is an error
        if response.startswith("Error after"):
            return [{"stance": f"Perspective {i+1}", "arguments": "N/A", "concerns": "N/A"} 
                    for i in range(count)]
        
        try:
            perspectives = json.loads(response)
            if isinstance(perspectives, list):
                return perspectives[:count]
        except:
            # Fallback perspectives
            return [
                {"stance": "Supportive", "arguments": f"Yes, because {topic}", "concerns": "May be too optimistic"},
                {"stance": "Opposing", "arguments": f"No, because {topic}", "concerns": "May be too pessimistic"},
                {"stance": "Neutral", "arguments": f"It depends on {topic}", "concerns": "May lack conviction"}
            ][:count]
    
    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """Analyze a problem for solve mode"""
        prompt = f"""Analyze this problem: "{problem}"

Provide a structured analysis with:
1. Root causes
2. Key constraints
3. Success metrics
4. Potential approaches

Format as JSON with these fields."""
        
        response = self.client.generate(prompt, temperature=0.5)
        
        # Check if response is an error
        if response.startswith("Error after"):
            return {
                "root_causes": ["Unable to analyze"],
                "constraints": ["Analysis failed"],
                "metrics": ["N/A"],
                "approaches": ["Retry analysis"]
            }
        
        try:
            analysis = json.loads(response)
            return analysis
        except:
            # Fallback analysis
            return {
                "root_causes": ["Complexity", "Resource limitations"],
                "constraints": ["Time", "Budget", "Technical feasibility"],
                "metrics": ["Performance improvement", "Cost reduction"],
                "approaches": ["Incremental optimization", "Complete redesign"]
            }
    
    def generate_solution(self, problem: str, approach: str) -> Dict[str, Any]:
        """Generate a solution based on problem and approach"""
        prompt = f"""Problem: {problem}
Approach: {approach}

Generate a detailed solution with:
1. Implementation steps
2. Expected outcomes
3. Potential risks
4. Success metrics

Format as JSON."""
        
        response = self.client.generate(prompt, temperature=0.6)
        
        # Check if response is an error
        if response.startswith("Error after"):
            return {
                "steps": ["Unable to generate solution"],
                "outcomes": ["N/A"],
                "risks": ["Generation failed"],
                "metrics": ["N/A"]
            }
        
        try:
            solution = json.loads(response)
            return solution
        except:
            # Fallback solution
            return {
                "steps": ["Analyze current state", "Design solution", "Implement changes", "Monitor results"],
                "outcomes": ["Improved efficiency", "Better user experience"],
                "risks": ["Implementation complexity", "User adoption"],
                "metrics": ["Performance metrics", "User satisfaction"]
            }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Direct generation with retry logic"""
        return self.client.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs):
        """Stream generation with retry logic"""
        yield from self.client.generate_stream(prompt, **kwargs)
    
    def get_cost_estimate(self) -> float:
        """Estimate cost of operations performed"""
        # Simple estimation - can be enhanced with actual token counting
        return 0.001  # Default small cost per operation
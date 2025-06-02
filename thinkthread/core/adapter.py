"""
Adapter layer that bridges the new ThinkThread API to the robust old SDK implementation.

This allows us to keep the beautiful new API while leveraging the battle-tested
core components from the old SDK.
"""

from typing import Dict, Any, Optional, List, Union
import os
from dataclasses import dataclass

# Import old SDK components from core
from .session import ThinkThreadSession
from .tree_thinker import TreeThinker
from .config import ThinkThreadConfig
from .evaluation import EvaluationStrategy, DefaultEvaluationStrategy
from .monitoring import PerformanceMonitor

# Import LLM clients
from ..llm import OpenAIClient, AnthropicClient, HuggingFaceClient, DummyLLMClient
from ..llm.base import LLMClient


@dataclass
class AdapterConfig:
    """Configuration for the adapter layer"""
    provider: str = "auto"
    model: Optional[str] = None
    test_mode: bool = False
    enable_monitoring: bool = True
    enable_caching: bool = True
    max_retries: int = 3
    
    # New features
    enable_visualization: bool = False
    enable_debugging: bool = False
    enable_profiling: bool = False


class SDKAdapter:
    """
    Bridges the new simplified API to the old robust SDK implementation.
    
    This adapter:
    1. Translates new API calls to old SDK calls
    2. Adds new features (visualization, debugging) on top
    3. Maintains backward compatibility
    4. Preserves all production features (retry, caching, monitoring)
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self._session_cache: Dict[str, ThinkThreadSession] = {}
        self._monitor = PerformanceMonitor() if self.config.enable_monitoring else None
        self._llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self) -> LLMClient:
        """Initialize the appropriate LLM client with all robustness features"""
        if self.config.test_mode:
            return DummyLLMClient()
            
        if self.config.provider == "auto":
            # Auto-detect available API keys
            if os.environ.get("OPENAI_API_KEY"):
                return OpenAIClient(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model_name=self.config.model or os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
                )
            elif os.environ.get("ANTHROPIC_API_KEY"):
                return AnthropicClient(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=self.config.model or os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
                )
            else:
                # Fallback to dummy for testing
                return DummyLLMClient()
        
        # Specific provider requested
        if self.config.provider == "openai":
            return OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model_name=self.config.model or "gpt-4-turbo-preview"
            )
        elif self.config.provider == "anthropic":
            return AnthropicClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model_name=self.config.model or "claude-3-opus-20240229"
            )
        elif self.config.provider == "huggingface":
            return HuggingFaceClient(
                api_token=os.environ.get("HF_API_TOKEN", ""),
                model_name=self.config.model or "meta-llama/Llama-2-70b-chat-hf"
            )
        else:
            return DummyLLMClient()
    
    def create_session(self, **kwargs) -> ThinkThreadSession:
        """Create a ThinkThreadSession with merged configuration"""
        # Create config with evaluation settings
        from .config import ThinkThreadConfig
        # Map 'auto' to a valid provider
        provider = self.config.provider
        if provider == 'auto':
            if os.environ.get("OPENAI_API_KEY"):
                provider = 'openai'
            elif os.environ.get("ANTHROPIC_API_KEY"):
                provider = 'anthropic'
            else:
                provider = 'dummy'
        
        config = ThinkThreadConfig(
            use_self_evaluation=kwargs.get("use_self_evaluation", False),
            use_pairwise_evaluation=kwargs.get("use_pairwise_evaluation", False),
            provider=provider,
        )
        
        # Create session with proper parameters
        session = ThinkThreadSession(
            llm_client=self._llm_client,
            alternatives=kwargs.get("alternatives", 3),
            rounds=kwargs.get("rounds", 2),
            evaluation_strategy=kwargs.get("evaluation_strategy", DefaultEvaluationStrategy()),
            config=config
        )
        
        return session
    
    def create_tree_thinker(self, **kwargs) -> TreeThinker:
        """Create a TreeThinker for exploration mode"""
        from .config import ThinkThreadConfig
        # Map 'auto' to a valid provider
        provider = self.config.provider
        if provider == 'auto':
            if os.environ.get("OPENAI_API_KEY"):
                provider = 'openai'
            elif os.environ.get("ANTHROPIC_API_KEY"):
                provider = 'anthropic'
            else:
                provider = 'dummy'
                
        config = ThinkThreadConfig(
            beam_width=kwargs.get("beam_width", 3),
            max_tree_depth=kwargs.get("max_tree_depth", 3),
            branching_factor=kwargs.get("branching_factor", 3),
            provider=provider,
        )
        
        return TreeThinker(
            llm_client=self._llm_client,
            config=config
        )
    
    def reason(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Main reasoning method that auto-selects the best approach.
        Uses the old SDK's ThinkThreadSession with smart defaults.
        """
        # Auto-detect mode based on prompt patterns
        mode = self._detect_mode(prompt)
        
        if mode == "explore":
            return self.explore(prompt, **kwargs)
        elif mode == "debate":
            return self.debate(prompt, **kwargs)
        elif mode == "solve":
            return self.solve(prompt, **kwargs)
        else:
            # Default to refine mode
            return self.refine(prompt, **kwargs)
    
    def explore(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Use TreeThinker for exploration"""
        tree_thinker = self.create_tree_thinker(**kwargs)
        
        # Add exploration-specific prompting
        exploration_prompt = f"Explore different approaches and ideas for: {prompt}"
        
        result = tree_thinker.solve(exploration_prompt)
        
        # Convert to new result format
        return self._convert_result(result, mode="explore")
    
    def refine(self, prompt: str, initial_text: str = "", **kwargs) -> Dict[str, Any]:
        """Use ThinkThreadSession for refinement"""
        session = self.create_session(
            alternatives=kwargs.get("alternatives", 3),
            rounds=kwargs.get("rounds", 3),
            use_self_evaluation=True
        )
        
        if initial_text:
            refinement_prompt = f"Refine and improve the following: {initial_text}\n\nContext: {prompt}"
        else:
            refinement_prompt = f"Provide a refined and thoughtful response to: {prompt}"
        
        result = session.run(refinement_prompt)
        
        return self._convert_result(result, mode="refine")
    
    def debate(self, prompt: str, perspectives: int = 3, **kwargs) -> Dict[str, Any]:
        """Create multiple sessions for different perspectives"""
        perspectives_prompts = [
            f"Argue in favor of: {prompt}",
            f"Argue against: {prompt}",
            f"Provide a balanced view on: {prompt}"
        ][:perspectives]
        
        results = []
        for perspective_prompt in perspectives_prompts:
            session = self.create_session(alternatives=2, rounds=2)
            result = session.run(perspective_prompt)
            results.append(result)
        
        # Synthesize perspectives
        synthesis_session = self.create_session(alternatives=1, rounds=1)
        synthesis_prompt = f"Synthesize these perspectives on '{prompt}':\n\n"
        for i, result in enumerate(results):
            synthesis_prompt += f"Perspective {i+1}: {result}\n\n"
        
        final_result = synthesis_session.run(synthesis_prompt)
        
        return self._convert_result(final_result, mode="debate", metadata={"perspectives": results})
    
    def solve(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Use combination of TreeThinker and Session for problem-solving"""
        # First explore solution space
        tree_thinker = self.create_tree_thinker(beam_width=4, max_tree_depth=2)
        exploration = tree_thinker.solve(f"Identify potential solutions for: {prompt}")
        
        # Then refine the best solution
        session = self.create_session(alternatives=2, rounds=3, use_self_evaluation=True)
        solution_prompt = f"""Based on this exploration: {exploration}
        
        Provide a detailed, actionable solution for: {prompt}
        Include implementation steps, expected outcomes, and potential risks."""
        
        result = session.run(solution_prompt)
        
        return self._convert_result(result, mode="solve", metadata={"exploration": exploration})
    
    def _detect_mode(self, prompt: str) -> str:
        """Auto-detect the best reasoning mode based on prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["explore", "brainstorm", "ideas", "options", "possibilities"]):
            return "explore"
        elif any(word in prompt_lower for word in ["debate", "pros and cons", "compare", "versus", "vs"]):
            return "debate"
        elif any(word in prompt_lower for word in ["solve", "fix", "problem", "issue", "how to"]):
            return "solve"
        else:
            return "refine"
    
    def _convert_result(self, old_result: Any, mode: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert old SDK result to new format"""
        # Extract cost if available
        cost = 0.0
        if hasattr(old_result, 'metadata') and 'cost' in old_result.metadata:
            cost = old_result.metadata['cost']
        elif self._monitor and hasattr(self._monitor, 'get_last_cost'):
            cost = self._monitor.get_last_cost()
        
        # Extract confidence
        confidence = 1.0
        if hasattr(old_result, 'confidence'):
            confidence = old_result.confidence
        elif metadata and 'confidence' in metadata:
            confidence = metadata['confidence']
        
        # Create result in new format
        result = {
            "answer": str(old_result),
            "mode": mode,
            "cost": cost,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        # Add monitoring data if available (placeholder for now)
        # TODO: Integrate proper monitoring from old SDK
        
        return result
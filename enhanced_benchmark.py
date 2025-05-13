"""Enhanced benchmark script for ThinkThread SDK optimizations.

This script tests the performance of the ThinkThread SDK with different
optimization configurations and more complex scenarios.
"""

import os
import time
import asyncio
from typing import Dict, Any, List, AsyncIterator, Optional

from thinkthread_sdk.config import ThinkThreadConfig
from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm.openai_client import OpenAIClient
from thinkthread_sdk.monitoring import GLOBAL_MONITOR


class PatchedOpenAIClient(OpenAIClient):
    """Patched OpenAI client that implements advanced optimization methods."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the patched OpenAI client.
        
        Args:
            **kwargs: Arguments to pass to the OpenAIClient constructor
        """
        super().__init__(**kwargs)
        self._embedding_cache = {}
        self._semantic_cache = {}
        self._use_semantic_cache = False
        self._semantic_similarity_threshold = 0.9
    
    def enable_semantic_cache(self, enabled: bool = True, similarity_threshold: float = 0.9) -> None:
        """Enable or disable semantic caching.
        
        Args:
            enabled: Whether to enable semantic caching
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
        """
        self._use_semantic_cache = enabled
        self._semantic_similarity_threshold = similarity_threshold
    
    def _generate_uncached(self, prompt: str, **kwargs: Any) -> str:
        """Generate text without using the cache.
        
        This method implements the abstract method from LLMClient.
        It simply calls the original generate method implementation.
        
        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters
            
        Returns:
            The generated text response from the model
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            time.sleep(0.5 - time_since_last_call)

        self._last_call_time = time.time()

        options = self.opts.copy()
        options.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )

            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""

        except Exception as e:
            error_message = f"Error when calling OpenAI API: {str(e)}"
            return error_message
    
    async def aembed(self, text: str) -> List[float]:
        """Get embeddings for a text using OpenAI's Embeddings API.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 0.5:  # 500ms minimum between calls
            await asyncio.sleep(0.5 - time_since_last_call)
            
        self._last_call_time = time.time()
        
        try:
            response = await self.async_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            
            embeddings = response.data[0].embedding
            self._embedding_cache[text] = embeddings
            return embeddings
            
        except Exception as e:
            print(f"Error when creating embeddings: {str(e)}")
            return [0.0] * 10
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)
    
    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream text generation from the language model.
        
        This is a minimal implementation to satisfy the abstract method.
        
        Args:
            prompt: The input text to send to the model
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated text response
        """
        response = await self.acomplete(prompt, **kwargs)
        yield response


async def run_benchmark(config_name: str, config: ThinkThreadConfig, question: str) -> Dict[str, Any]:
    """Run a benchmark with the given configuration.
    
    Args:
        config_name: Name of the configuration for reporting
        config: The configuration to use
        question: The question to ask
        
    Returns:
        A dictionary with benchmark results
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}
    
    client = PatchedOpenAIClient(api_key=api_key, model_name="gpt-3.5-turbo")
    
    if hasattr(config, "use_caching") and config.use_caching:
        client.enable_cache(True)
    
    if hasattr(config, "use_semantic_cache") and config.use_semantic_cache:
        client.enable_semantic_cache(True, 0.85)  # Use 0.85 as default similarity threshold
    
    if hasattr(config, "concurrency_limit") and config.concurrency_limit > 0:
        client.set_concurrency_limit(config.concurrency_limit)
    
    GLOBAL_MONITOR.reset()
    GLOBAL_MONITOR.enable(config.enable_monitoring if hasattr(config, "enable_monitoring") else True)
    
    session = ThinkThreadSession(
        llm_client=client,
        alternatives=config.alternatives,
        max_rounds=config.max_rounds,
        config=config,
    )
    
    print(f"\nRunning benchmark: {config_name}")
    start_time = time.time()
    
    try:
        result = await session.run_async(question)
        elapsed = time.time() - start_time
        
        stats = GLOBAL_MONITOR.get_stats()
        
        print(f"  Elapsed time: {elapsed:.2f}s")
        for operation, op_stats in stats.items():
            print(f"  {operation}: avg={op_stats['avg']:.2f}s, total={op_stats['total']:.2f}s, count={op_stats['count']}")
        
        if hasattr(config, "use_batched_requests") and config.use_batched_requests:
            print(f"  Batched requests enabled")
        
        if hasattr(config, "use_fast_similarity") and config.use_fast_similarity:
            print(f"  Fast similarity enabled")
        
        if hasattr(config, "use_adaptive_temperature") and config.use_adaptive_temperature:
            print(f"  Adaptive temperature enabled (initial: {config.initial_temperature}, min: {config.min_generation_temperature})")
        
        if hasattr(config, "use_semantic_cache") and config.use_semantic_cache:
            print(f"  Semantic caching enabled")
        
        return {
            "name": config_name,
            "elapsed": elapsed,
            "result": result,
            "stats": stats,
        }
    except Exception as e:
        print(f"  Error: {str(e)}")
        return {
            "name": config_name,
            "error": str(e),
        }


async def main():
    """Run benchmarks with different optimization configurations."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    print(f"Using OpenAI API key: {api_key[:10]}...")
    
    questions = [
        {
            "name": "Simple Question",
            "text": "What are three key benefits of exercise?",
        },
        {
            "name": "Medium Complexity",
            "text": "Compare and contrast supervised, unsupervised, and reinforcement learning approaches in machine learning. What are the strengths and weaknesses of each?",
        },
        {
            "name": "Complex Question",
            "text": "Analyze the potential economic and social impacts of widespread adoption of artificial general intelligence. Consider both positive and negative consequences, and discuss potential policy approaches to maximize benefits while minimizing risks.",
        },
    ]
    
    base_configs = [
        {
            "name": "Baseline (no optimizations)",
            "config": {
                "parallel_alternatives": False,
                "parallel_evaluation": False,
                "use_caching": False,
                "early_termination": False,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Parallel alternatives only",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": False,
                "use_caching": False,
                "early_termination": False,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Parallel evaluation only",
            "config": {
                "parallel_alternatives": False,
                "parallel_evaluation": True,
                "use_caching": False,
                "early_termination": False,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Caching only",
            "config": {
                "parallel_alternatives": False,
                "parallel_evaluation": False,
                "use_caching": True,
                "early_termination": False,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Early termination only",
            "config": {
                "parallel_alternatives": False,
                "parallel_evaluation": False,
                "use_caching": False,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "enable_monitoring": True,
            },
        },
        {
            "name": "All basic optimizations",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Batched requests",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "use_batched_requests": True,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Fast similarity",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "use_fast_similarity": True,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Adaptive temperature",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "use_adaptive_temperature": True,
                "initial_temperature": 0.7,
                "generation_temperature": 0.9,
                "min_generation_temperature": 0.5,
                "temperature_decay_rate": 0.8,
                "enable_monitoring": True,
            },
        },
        {
            "name": "Semantic caching",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "use_semantic_cache": True,
                "enable_monitoring": True,
            },
        },
        {
            "name": "All advanced optimizations",
            "config": {
                "parallel_alternatives": True,
                "parallel_evaluation": True,
                "use_caching": True,
                "early_termination": True,
                "early_termination_threshold": 0.95,
                "concurrency_limit": 5,
                "use_batched_requests": True,
                "use_fast_similarity": True,
                "use_adaptive_temperature": True,
                "initial_temperature": 0.7,
                "generation_temperature": 0.9,
                "min_generation_temperature": 0.5,
                "temperature_decay_rate": 0.8,
                "use_semantic_cache": True,
                "enable_monitoring": True,
            },
        },
    ]
    
    scale_configs = [
        {
            "name": "Small Scale",
            "alternatives": 2,
            "max_rounds": 2,
        },
        {
            "name": "Medium Scale",
            "alternatives": 3,
            "max_rounds": 3,
        },
        {
            "name": "Large Scale",
            "alternatives": 4,
            "max_rounds": 3,
        },
    ]
    
    selected_question = questions[1]
    # 2. All basic optimizations
    # 4. All advanced optimizations
    selected_configs = [
        base_configs[0],   # Baseline
        base_configs[5],   # All basic optimizations
        base_configs[6],   # Batched requests
        base_configs[7],   # Fast similarity
        base_configs[8],   # Adaptive temperature
        base_configs[9],   # Semantic caching
        base_configs[10],  # All advanced optimizations
    ]
    selected_scales = scale_configs[:2]  # Small and Medium scale
    
    print(f"Running benchmarks for question: {selected_question['name']}")
    print(f"Question: {selected_question['text']}")
    
    all_results = []
    
    for scale in selected_scales:
        print(f"\n=== Scale: {scale['name']} ===")
        scale_results = []
        
        for base_config in selected_configs:
            config_dict = base_config["config"].copy()
            config_dict["alternatives"] = scale["alternatives"]
            config_dict["max_rounds"] = scale["max_rounds"]
            
            config = ThinkThreadConfig(**config_dict)
            
            config_name = f"{base_config['name']} ({scale['name']})"
            try:
                result = await run_benchmark(
                    config_name,
                    config,
                    selected_question["text"],
                )
                if "elapsed" in result:
                    scale_results.append(result)
            except Exception as e:
                print(f"Error running benchmark {config_name}: {str(e)}")
        
        if len(scale_results) > 1:
            print(f"\nSummary for {scale['name']}:")
            baseline = scale_results[0]["elapsed"]
            for result in scale_results:
                speedup = baseline / result["elapsed"] if result["elapsed"] > 0 else 0
                print(f"{result['name']}: {result['elapsed']:.2f}s ({speedup:.2f}x speedup)")
        
        all_results.extend(scale_results)
    
    print("\n=== Overall Summary ===")
    for scale in selected_scales:
        scale_results = [r for r in all_results if scale["name"] in r["name"]]
        if len(scale_results) > 1:
            baseline = next((r for r in scale_results if "Baseline" in r["name"]), None)
            optimized = next((r for r in scale_results if "All optimizations" in r["name"]), None)
            
            if baseline and optimized:
                speedup = baseline["elapsed"] / optimized["elapsed"] if optimized["elapsed"] > 0 else 0
                print(f"{scale['name']}: {speedup:.2f}x speedup with all optimizations")


if __name__ == "__main__":
    asyncio.run(main())

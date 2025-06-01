"""
Reasoning Profiler - Performance profiling for reasoning operations
"""

import time
import functools
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class ProfileData:
    """Data collected during profiling"""
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    token_counts: Dict[str, int] = field(default_factory=dict)
    api_calls: int = 0
    cache_hits: int = 0
    memory_usage: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class ReasoningProfiler:
    """Performance profiler for reasoning operations"""
    
    def __init__(self):
        self.current_profile = None
        self.stage_stack = []
        self.stage_start_times = {}
        
    def profile(self, reasoning_func: Callable) -> 'Profile':
        """Profile a reasoning operation"""
        self.current_profile = ProfileData()
        start_time = time.time()
        
        try:
            # Execute the function
            result = reasoning_func()
            
            # Calculate total time
            self.current_profile.total_time = time.time() - start_time
            
            # Analyze performance
            self._analyze_performance()
            
            # Create profile result
            return Profile(self.current_profile)
            
        finally:
            # Reset state
            self.current_profile = None
            self.stage_stack = []
            self.stage_start_times = {}
            
    def start_stage(self, stage_name: str):
        """Start timing a stage"""
        if self.current_profile:
            self.stage_stack.append(stage_name)
            self.stage_start_times[stage_name] = time.time()
            
    def end_stage(self, stage_name: str):
        """End timing a stage"""
        if self.current_profile and stage_name in self.stage_start_times:
            elapsed = time.time() - self.stage_start_times[stage_name]
            self.current_profile.stage_times[stage_name] = elapsed
            
            if stage_name in self.stage_stack:
                self.stage_stack.remove(stage_name)
                
    def record_tokens(self, stage: str, count: int):
        """Record token usage for a stage"""
        if self.current_profile:
            self.current_profile.token_counts[stage] = count
            
    def record_api_call(self):
        """Record an API call"""
        if self.current_profile:
            self.current_profile.api_calls += 1
            
    def record_cache_hit(self):
        """Record a cache hit"""
        if self.current_profile:
            self.current_profile.cache_hits += 1
            
    def _analyze_performance(self):
        """Analyze performance and identify bottlenecks"""
        if not self.current_profile:
            return
            
        profile = self.current_profile
        
        # Identify bottlenecks
        if profile.stage_times:
            total_stage_time = sum(profile.stage_times.values())
            
            for stage, time_spent in profile.stage_times.items():
                percentage = (time_spent / total_stage_time) * 100
                if percentage > 50:
                    profile.bottlenecks.append(
                        f"{stage} took {percentage:.0f}% of total time"
                    )
                    
        # Token efficiency
        total_tokens = sum(profile.token_counts.values())
        if total_tokens > 0:
            tokens_per_second = total_tokens / profile.total_time
            if tokens_per_second < 100:
                profile.bottlenecks.append(
                    f"Low token processing rate: {tokens_per_second:.0f} tokens/s"
                )
                
        # Cache effectiveness
        if profile.api_calls > 0:
            cache_rate = profile.cache_hits / (profile.api_calls + profile.cache_hits)
            if cache_rate < 0.3:
                profile.optimization_suggestions.append(
                    f"Enable caching - current hit rate: {cache_rate:.0%}"
                )
                
        # Stage-specific suggestions
        if 'exploration' in profile.stage_times and profile.stage_times['exploration'] > 10:
            profile.optimization_suggestions.append(
                "Consider reducing branching_factor or max_depth for exploration"
            )
            
        if 'evaluation' in profile.stage_times:
            eval_time = profile.stage_times['evaluation']
            if eval_time > profile.total_time * 0.5:
                profile.optimization_suggestions.append(
                    "Evaluation is taking too long - consider simpler evaluation strategy"
                )
                
    def stage(self, stage_name: str):
        """Context manager for profiling a stage"""
        return ProfileStage(self, stage_name)


class ProfileStage:
    """Context manager for profiling a stage"""
    
    def __init__(self, profiler: ReasoningProfiler, stage_name: str):
        self.profiler = profiler
        self.stage_name = stage_name
        
    def __enter__(self):
        self.profiler.start_stage(self.stage_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_stage(self.stage_name)


class Profile:
    """Profile result with analysis methods"""
    
    def __init__(self, data: ProfileData):
        self.data = data
        
    @property
    def total_time(self) -> float:
        return self.data.total_time
        
    @property
    def time_by_stage(self) -> Dict[str, float]:
        return self.data.stage_times
        
    @property
    def bottlenecks(self) -> List[str]:
        return self.data.bottlenecks
        
    @property
    def optimization_suggestions(self) -> List[str]:
        return self.data.optimization_suggestions
        
    def token_efficiency(self) -> float:
        """Calculate tokens per insight or result quality"""
        total_tokens = sum(self.data.token_counts.values())
        if total_tokens == 0:
            return 0.0
            
        # Simplified metric - in real implementation would measure actual insights
        return 1000 / total_tokens  # Inverse of tokens used
        
    def cost_estimate(self, cost_per_token: float = 0.00001) -> float:
        """Estimate cost based on token usage"""
        total_tokens = sum(self.data.token_counts.values())
        return total_tokens * cost_per_token
        
    def __str__(self) -> str:
        """String representation of profile"""
        output = f"=== Reasoning Profile ===\n"
        output += f"Total time: {self.total_time:.2f}s\n"
        output += f"API calls: {self.data.api_calls}\n"
        output += f"Cache hits: {self.data.cache_hits}\n"
        
        if self.time_by_stage:
            output += "\nTime by stage:\n"
            for stage, time_spent in sorted(
                self.time_by_stage.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                percentage = (time_spent / self.total_time) * 100
                output += f"  {stage}: {time_spent:.2f}s ({percentage:.0f}%)\n"
                
        if self.bottlenecks:
            output += "\nBottlenecks:\n"
            for bottleneck in self.bottlenecks:
                output += f"  - {bottleneck}\n"
                
        if self.optimization_suggestions:
            output += "\nOptimization suggestions:\n"
            for suggestion in self.optimization_suggestions:
                output += f"  - {suggestion}\n"
                
        return output
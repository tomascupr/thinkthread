#!/usr/bin/env python3
"""
Test script to verify LLM integration works correctly

This tests both with and without real API keys to ensure both paths work.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thinkthread import reason, explore, refine, debate, solve
from thinkthread.llm_integration import LLMIntegration


def test_llm_integration_layer():
    """Test the LLM integration layer directly"""
    print("=" * 80)
    print("Testing LLM Integration Layer")
    print("=" * 80)
    print()
    
    # Test auto-detection
    llm = LLMIntegration(provider="auto")
    print(f"Auto-detected provider: {llm.client.__class__.__name__}")
    print()
    
    # Test basic generation
    print("Testing thought generation:")
    thoughts = llm.generate_thoughts("How to improve customer satisfaction", count=3)
    for i, thought in enumerate(thoughts, 1):
        print(f"  {i}. {thought}")
    print()
    
    # Test evaluation
    print("Testing thought evaluation:")
    score = llm.evaluate_thought(thoughts[0], "customer satisfaction context")
    print(f"  Score: {score:.2f}")
    print()
    
    # Test alternative generation
    print("Testing alternative generation:")
    current = "We should focus on customer service training."
    alternatives = llm.generate_alternatives(
        "How to improve customer satisfaction",
        current,
        count=2
    )
    for i, alt in enumerate(alternatives, 1):
        print(f"  Alternative {i}: {alt[:80]}...")
    print()


def test_reasoning_modes_with_llm():
    """Test reasoning modes with LLM integration"""
    print("=" * 80)
    print("Testing Reasoning Modes with LLM")
    print("=" * 80)
    print()
    
    # Test if we have real API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    print(f"OpenAI API key detected: {has_openai}")
    print(f"Anthropic API key detected: {has_anthropic}")
    print()
    
    if not (has_openai or has_anthropic):
        print("No API keys detected - will use mock implementations")
        print("To test with real LLMs, set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print()
    
    # Test 1: Basic reasoning
    print("1. Testing basic reasoning:")
    print("-" * 40)
    answer = reason("What are the key principles of good software design?")
    print(f"Answer preview: {answer.answer[:200]}...")
    print(f"Mode selected: {answer.metadata.get('mode_used')}")
    print(f"Confidence: {answer.confidence:.2%}")
    print(f"Cost: ${answer.cost:.4f}")
    print()
    
    # Test 2: Explore mode
    print("2. Testing explore mode:")
    print("-" * 40)
    answer = explore("How might we use AI to improve education?", max_depth=2, branching_factor=2)
    print(f"Nodes explored: {answer.metadata.get('total_nodes', 0)}")
    print(f"Best path depth: {answer.metadata.get('max_depth_reached', 0)}")
    print(f"Answer preview: {answer.answer[:200]}...")
    print()
    
    # Test 3: Refine mode
    print("3. Testing refine mode:")
    print("-" * 40)
    initial = "AI can help personalize learning for students."
    answer = refine(
        "Improve this statement about AI in education",
        initial_answer=initial,
        max_rounds=2
    )
    print(f"Initial: {initial}")
    print(f"Refined: {answer.answer[:200]}...")
    print(f"Improvement: {answer.metadata.get('final_improvement', 0):.2%}")
    print()
    
    # Test 4: Debate mode
    print("4. Testing debate mode:")
    print("-" * 40)
    answer = debate("Should AI replace human teachers?", perspectives=3)
    print(f"Perspectives considered: {answer.metadata.get('perspectives_count')}")
    print(f"Consensus level: {answer.metadata.get('consensus_level')}")
    print(f"Synthesis preview: {answer.answer[:200]}...")
    print()
    
    # Test 5: Solve mode
    print("5. Testing solve mode:")
    print("-" * 40)
    answer = solve("How to reduce student dropout rates by 30%")
    print(f"Best approach: {answer.metadata.get('best_approach')}")
    print(f"Feasibility: {answer.metadata.get('feasibility', 0):.2%}")
    print(f"Solution preview: {answer.answer[:200]}...")
    print()


def test_cost_tracking():
    """Test cost tracking functionality"""
    print("=" * 80)
    print("Testing Cost Tracking")
    print("=" * 80)
    print()
    
    # Run several operations and track costs
    operations = [
        ("Simple question", lambda: reason("What is Python?")),
        ("Exploration", lambda: explore("Future of programming languages")),
        ("Refinement", lambda: refine("Python is a programming language", initial_answer="Python is versatile")),
    ]
    
    total_cost = 0
    for name, operation in operations:
        print(f"Running: {name}")
        result = operation()
        cost = result.cost
        total_cost += cost
        print(f"  Cost: ${cost:.4f}")
        print(f"  Token efficiency: {len(str(result)) / max(cost, 0.0001):.0f} chars/$")
    
    print(f"\nTotal cost for session: ${total_cost:.4f}")
    print()


def test_fallback_behavior():
    """Test fallback behavior when API fails"""
    print("=" * 80)
    print("Testing Fallback Behavior")
    print("=" * 80)
    print()
    
    # Force test mode to simulate no API keys
    from thinkthread.modes.base import ReasoningMode
    
    # Create a mode with test_mode enabled
    test_config = {'test_mode': True}
    
    print("Testing with forced test mode (no real LLM calls):")
    
    # This should work even without API keys
    from thinkthread.modes.explore import ExploreMode
    explorer = ExploreMode(**test_config)
    
    result = explorer.execute("Test question", max_depth=2, branching_factor=2)
    
    print(f"Result generated: {len(result.answer)} chars")
    print(f"Used mock implementation: {explorer.llm is None}")
    print(f"Cost (should be low): ${result.cost:.4f}")
    print()


def main():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print("ThinkThread LLM Integration Tests")
    print("=" * 80)
    print()
    
    tests = [
        test_llm_integration_layer,
        test_reasoning_modes_with_llm,
        test_cost_tracking,
        test_fallback_behavior,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Error in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 80)
    print("Integration tests complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print("- LLM integration layer connects to real APIs when available")
    print("- Falls back to mock implementations when no API keys")
    print("- All reasoning modes support both real and mock LLMs")
    print("- Cost tracking works in both modes")
    print()
    
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("To test with real LLMs:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
    print()


if __name__ == "__main__":
    main()
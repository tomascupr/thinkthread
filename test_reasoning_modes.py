#!/usr/bin/env python3
"""
Test script to verify the new reasoning modes and transparency features work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thinkthread import reason, explore, refine, debate, solve
from thinkthread import available, search
from thinkthread.modes.base import ReasoningResult


def test_basic_reasoning():
    """Test basic reasoning with auto mode selection"""
    print("=== Test 1: Basic Reasoning ===")
    
    # Test auto mode selection
    answer = reason("What are the benefits of renewable energy?")
    print(f"Question: What are the benefits of renewable energy?")
    print(f"Answer: {answer}")
    print(f"Mode used: {answer.metadata.get('mode_used', 'unknown')}")
    print(f"Confidence: {answer.confidence:.2%}")
    print(f"Cost: ${answer.cost:.4f}")
    print()
    
    assert isinstance(answer, ReasoningResult)
    assert answer.answer is not None
    assert 0 <= answer.confidence <= 1
    print("✓ Basic reasoning test passed")
    print()


def test_explore_mode():
    """Test explore mode (Tree of Thoughts)"""
    print("=== Test 2: Explore Mode ===")
    
    answer = explore("Design a sustainable city for 2050")
    print(f"Question: Design a sustainable city for 2050")
    print(f"Answer preview: {answer.answer[:200]}...")
    print(f"Total nodes explored: {answer.metadata.get('total_nodes', 0)}")
    print(f"Best path: {answer.metadata.get('best_path', [])[:2]}")
    print(f"Alternatives: {len(answer.alternatives)}")
    print()
    
    assert answer.mode == 'explore'
    assert answer.metadata.get('total_nodes', 0) > 0
    print("✓ Explore mode test passed")
    print()


def test_refine_mode():
    """Test refine mode (Chain of Recursive Thoughts)"""
    print("=== Test 3: Refine Mode ===")
    
    initial = "Renewable energy is good for the environment."
    answer = refine("Improve this statement about renewable energy", initial_answer=initial)
    print(f"Initial: {initial}")
    print(f"Refined: {answer}")
    print(f"Rounds completed: {answer.metadata.get('rounds_completed', 0)}")
    print(f"Final improvement: {answer.metadata.get('final_improvement', 0):.2%}")
    print()
    
    assert answer.mode == 'refine'
    assert answer.metadata.get('rounds_completed', 0) > 0
    print("✓ Refine mode test passed")
    print()


def test_debate_mode():
    """Test debate mode"""
    print("=== Test 4: Debate Mode ===")
    
    answer = debate("Is artificial intelligence a threat to humanity?")
    print(f"Question: Is artificial intelligence a threat to humanity?")
    print(f"Synthesis preview: {answer.answer[:300]}...")
    print(f"Perspectives considered: {answer.metadata.get('perspectives_count', 0)}")
    print(f"Strongest stance: {answer.metadata.get('strongest_stance', 'unknown')}")
    print(f"Consensus level: {answer.metadata.get('consensus_level', 'unknown')}")
    print()
    
    assert answer.mode == 'debate'
    assert len(answer.alternatives) > 0
    print("✓ Debate mode test passed")
    print()


def test_solve_mode():
    """Test solve mode"""
    print("=== Test 5: Solve Mode ===")
    
    answer = solve("How to reduce software bugs by 50%")
    print(f"Question: How to reduce software bugs by 50%")
    print(f"Best approach: {answer.metadata.get('best_approach', 'unknown')}")
    print(f"Feasibility: {answer.metadata.get('feasibility', 0):.2%}")
    print(f"Expected impact: {answer.metadata.get('expected_impact', 0):.2%}")
    print(f"Action steps: {answer.metadata.get('action_steps', 0)}")
    print()
    
    # Test answer formatting
    print("Solution preview:")
    print(answer.answer[:400])
    print()
    
    assert answer.mode == 'solve'
    assert answer.metadata.get('solutions_generated', 0) > 0
    print("✓ Solve mode test passed")
    print()


def test_mode_composition():
    """Test mode composition with operators"""
    print("=== Test 6: Mode Composition ===")
    
    # Test chaining with | operator
    from thinkthread.modes import ExploreMode, RefineMode, SolveMode
    chained = ExploreMode() | RefineMode()
    print(f"Chained mode: {chained}")
    
    # Test parallel with & operator  
    parallel = ExploreMode() & SolveMode()
    print(f"Parallel mode: {parallel}")
    
    # Execute chained mode
    answer = chained("What are innovative solutions for urban transportation?")
    print(f"Chained result mode: {answer.mode}")
    print(f"Answer preview: {answer.answer[:200]}...")
    print()
    
    assert "explore | refine" in answer.mode or "ChainedMode" in str(type(chained))
    print("✓ Mode composition test passed")
    print()


def test_mode_discovery():
    """Test mode discovery functions"""
    print("=== Test 7: Mode Discovery ===")
    
    # Test available modes
    modes = available()
    print(f"Available modes: {modes}")
    assert len(modes) >= 4
    assert 'explore' in modes
    assert 'refine' in modes
    
    # Test search
    creative_modes = search("creative")
    print(f"Creative modes: {creative_modes}")
    assert 'explore' in creative_modes
    
    problem_modes = search("problem")
    print(f"Problem-solving modes: {problem_modes}")
    assert 'solve' in problem_modes
    
    print("✓ Mode discovery test passed")
    print()


def test_result_methods():
    """Test ReasoningResult methods"""
    print("=== Test 8: Result Methods ===")
    
    answer = reason("What is machine learning?")
    
    # Test string representation
    print(f"String repr: {str(answer)[:50]}...")
    
    # Test explanation
    explanation = answer.explain()
    print(f"Explanation: {explanation}")
    
    # Test cost breakdown
    costs = answer.cost_breakdown
    print(f"Cost breakdown: {costs}")
    
    # Test export
    markdown = answer.export_markdown()
    print(f"Markdown export length: {len(markdown)} chars")
    
    assert isinstance(str(answer), str)
    assert isinstance(explanation, str)
    assert 'total' in costs
    print("✓ Result methods test passed")
    print()


def test_debugging_features():
    """Test debugging and transparency features"""
    print("=== Test 9: Debugging Features ===")
    
    from thinkthread import ReasoningDebugger
    
    # Create debugger
    debugger = ReasoningDebugger()
    
    # Add watch
    debugger.watch("confidence")
    
    # Simulate some steps without breakpoints to avoid input prompt
    debugger.step("Initial analysis", {"confidence": 0.7, "thought": "Starting"})
    debugger.step("Second step", {"confidence": 0.8, "thought": "Continuing"})
    
    # Check trace
    trace = debugger.get_trace()
    print(f"Trace length: {len(trace)}")
    print(f"First step: {trace[0].description}")
    print(f"Second step confidence: {trace[1].confidence:.2%}")
    
    # Test trace display
    debugger.show_trace()
    
    assert len(trace) == 2
    print("✓ Debugging features test passed")
    print()


def test_profiling():
    """Test profiling features"""
    print("=== Test 10: Profiling ===")
    
    # Profile a reasoning operation
    profile = reason.profile(lambda: explore("Test question for profiling"))
    
    print(f"Profile summary:\n{profile}")
    
    assert profile.total_time > 0
    print("✓ Profiling test passed")
    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("ThinkThread Reasoning Modes Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_reasoning,
        test_explore_mode,
        test_refine_mode,
        test_debate_mode,
        test_solve_mode,
        test_mode_composition,
        test_mode_discovery,
        test_result_methods,
        test_debugging_features,
        test_profiling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
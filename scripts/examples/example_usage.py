#!/usr/bin/env python3
"""
Example usage of the new ThinkThread reasoning modes and transparency features
"""

from thinkthread import reason, explore, refine, debate, solve

def example_basic_reasoning():
    """Example: Basic reasoning with automatic mode selection"""
    print("=" * 60)
    print("Example 1: Basic Reasoning")
    print("=" * 60)
    
    # Simple usage - just ask a question
    answer = reason("What are the main causes of climate change?")
    
    print(f"Answer: {answer}")
    print(f"\nMode used: {answer.metadata['mode_used']}")
    print(f"Confidence: {answer.confidence:.2%}")
    print()


def example_explore_mode():
    """Example: Using explore mode for creative tasks"""
    print("=" * 60)
    print("Example 2: Explore Mode - Creative Exploration")
    print("=" * 60)
    
    # Use explore mode for open-ended creative questions
    answer = explore("Design an education system for the future", max_depth=3)
    
    print(f"Creative exploration result:")
    print(answer.answer[:500] + "...")
    print(f"\nNodes explored: {answer.metadata['total_nodes']}")
    print(f"Best path depth: {answer.metadata['max_depth_reached']}")
    print()


def example_refine_mode():
    """Example: Using refine mode to improve content"""
    print("=" * 60)
    print("Example 3: Refine Mode - Iterative Improvement")
    print("=" * 60)
    
    draft = """
    AI is computers that think. They can do many things like humans.
    AI is used in many places today.
    """
    
    # Refine the draft
    improved = refine(
        "Improve this explanation of AI for a general audience",
        initial_answer=draft,
        max_rounds=2
    )
    
    print(f"Original draft: {draft.strip()}")
    print(f"\nImproved version: {improved}")
    print(f"\nImprovement: {improved.metadata['final_improvement']:.2%}")
    print()


def example_debate_mode():
    """Example: Using debate mode for balanced analysis"""
    print("=" * 60)
    print("Example 4: Debate Mode - Multi-perspective Analysis")
    print("=" * 60)
    
    # Get balanced perspective on controversial topic
    analysis = debate("Should we implement universal basic income?")
    
    print("Debate synthesis:")
    print(analysis.answer[:600] + "...")
    print(f"\nPerspectives considered: {analysis.metadata['perspectives_count']}")
    print(f"Consensus level: {analysis.metadata['consensus_level']}")
    print()


def example_solve_mode():
    """Example: Using solve mode for practical problems"""
    print("=" * 60)
    print("Example 5: Solve Mode - Actionable Solutions")
    print("=" * 60)
    
    # Get actionable solution for specific problem
    solution = solve("How to improve team productivity in remote work")
    
    print("Solution:")
    print(solution)
    print(f"\nFeasibility: {solution.metadata['feasibility']:.2%}")
    print(f"Expected impact: {solution.metadata['expected_impact']:.2%}")
    print()


def example_mode_composition():
    """Example: Composing multiple reasoning modes"""
    print("=" * 60)
    print("Example 6: Mode Composition - Combining Approaches")
    print("=" * 60)
    
    # First explore possibilities, then refine the best one
    question = "How can we make cities more livable?"
    
    # Method 1: Using the convenience functions
    exploration = explore(question)
    refined = refine(question, initial_answer=str(exploration))
    
    print(f"Explored {exploration.metadata['total_nodes']} possibilities")
    print(f"Then refined the best approach over {refined.metadata['rounds_completed']} rounds")
    print(f"\nFinal answer preview: {refined.answer[:400]}...")
    print()


def example_transparency_features():
    """Example: Using transparency and debugging features"""
    print("=" * 60)
    print("Example 7: Reasoning Transparency")
    print("=" * 60)
    
    # Enable profiling to see performance
    profile = reason.profile(
        lambda: solve("Reduce customer support response time by 50%")
    )
    
    print("Performance Profile:")
    print(profile)
    
    # Compare different approaches
    answer1 = explore("How to teach programming to beginners")
    answer2 = solve("How to teach programming to beginners")
    
    comparison = reason.compare(answer1, answer2)
    comparison.visualize(comparison)
    print()


def example_result_methods():
    """Example: Working with reasoning results"""
    print("=" * 60)
    print("Example 8: Result Methods")
    print("=" * 60)
    
    answer = reason("What are the benefits of meditation?")
    
    # Different ways to access the result
    print(f"Simple string: {answer}")  # Just the answer
    print(f"\nConfidence: {answer.confidence:.2%}")
    print(f"Cost: ${answer.cost:.4f}")
    
    # Get detailed explanation
    print(f"\nExplanation: {answer.explain()}")
    
    # Export for documentation
    markdown = answer.export_markdown()
    print(f"\nMarkdown export preview: {markdown[:200]}...")
    
    # Save for later
    answer.save("meditation_benefits.json")
    print("\nSaved to meditation_benefits.json")
    print()


def main():
    """Run all examples"""
    print("ThinkThread 10x Reasoning Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_basic_reasoning,
        example_explore_mode,
        example_refine_mode,
        example_debate_mode,
        example_solve_mode,
        example_mode_composition,
        example_transparency_features,
        example_result_methods,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
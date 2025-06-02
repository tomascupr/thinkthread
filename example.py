#!/usr/bin/env python3
"""Example usage of ThinkThread."""

from thinkthread import reason, explore, solve, debate, refine

def main():
    print("🧵 ThinkThread Example")
    print("=" * 40)
    
    # Test mode examples (no API calls)
    print("\n1. Basic reasoning (test mode):")
    answer = reason("What makes a good software engineer?", test_mode=True)
    print(f"Answer: {answer[:100]}...")
    
    print("\n2. Problem solving (test mode):")
    solution = solve("Our API is slow", test_mode=True)
    print(f"Solution: {solution[:100]}...")
    
    print("\n3. Exploration (test mode):")
    ideas = explore("Creative team building ideas", test_mode=True)
    print(f"Ideas: {ideas[:100]}...")
    
    print("\n4. Debate analysis (test mode):")
    analysis = debate("Remote work vs office work", test_mode=True)
    print(f"Analysis: {analysis[:100]}...")
    
    print("\n5. Text refinement (test mode):")
    better_text = refine("fix bug pls", "Make it professional", test_mode=True)
    print(f"Refined: {better_text[:100]}...")
    
    print("\n✅ All examples completed successfully!")

if __name__ == "__main__":
    main()
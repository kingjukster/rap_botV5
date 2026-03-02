#!/usr/bin/env python
"""
Quick test script to verify the iterative testing system works.
Tests ChatGPT integration without full iteration loop.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.chatgpt_tester import ChatGPTTester, TestCase


def test_chatgpt_connection():
    """Test basic ChatGPT connection."""
    print("Testing ChatGPT connection...")
    try:
        chatgpt = ChatGPTTester()
        print("[OK] ChatGPT client initialized")
        return chatgpt
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return None


def test_generate_test_cases(chatgpt: ChatGPTTester):
    """Test test case generation."""
    print("\nTesting test case generation...")
    try:
        test_cases = chatgpt.generate_test_cases(num_cases=5)
        print(f"[OK] Generated {len(test_cases)} test cases")
        for i, tc in enumerate(test_cases[:3], 1):
            print(f"  {i}. {tc.line1} / {tc.line2} -> {tc.expected_end_rhyme}")
        return test_cases
    except Exception as e:
        print(f"[ERROR] Failed to generate test cases: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    print("=" * 70)
    print("ITERATIVE TESTING SYSTEM - QUICK TEST")
    print("=" * 70)
    
    # Test connection
    chatgpt = test_chatgpt_connection()
    if not chatgpt:
        return 1
    
    # Test generation
    test_cases = test_generate_test_cases(chatgpt)
    if not test_cases:
        return 1
    
    print("\n[SUCCESS] Basic functionality works!")
    print("\nTo run full iterative testing:")
    print("  python scripts/tools/iterative_tester.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
rerun_saved_tests.py

Rerun tests on saved test files without generating new ones.
Uses minimal ChatGPT API calls (only for evaluation, not generation).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rapbot.rhyme_detector import RhymeDetector, RhymeType
from scripts.tools.chatgpt_tester import ChatGPTTester, TestCase, TestResults
from scripts.tools.code_modifier import CodeModifier


def load_saved_test_file(file_path: Path) -> List[TestCase]:
    """Load test cases from a saved JSONL file."""
    test_cases = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    test_case = TestCase(
                        line1=data.get("line1", ""),
                        line2=data.get("line2", ""),
                        expected_end_rhyme=data.get("expected_end_rhyme", "NONE"),
                        expected_internal_rhymes=data.get("expected_internal_rhymes"),
                        notes=data.get("notes"),
                    )
                    if test_case.line1 and test_case.line2:
                        test_cases.append(test_case)
    except Exception as e:
        logging.warning(f"Failed to load {file_path}: {e}")
    return test_cases


def run_tests(test_cases: List[TestCase], detector: RhymeDetector) -> TestResults:
    """Run detector on test cases and collect results."""
    total = len(test_cases)
    correct = 0
    failures = []
    
    per_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for test_case in test_cases:
        # Detect end rhyme
        result = detector.detect_end_rhyme(test_case.line1, test_case.line2)
        predicted = result.rhyme_type.value if result.rhyme_type != RhymeType.NONE else "NONE"
        expected = test_case.expected_end_rhyme
        
        # Check if correct (allow SLANT for EXACT and vice versa)
        is_correct = False
        if expected == "EXACT" and predicted in ["EXACT", "SLANT"]:
            is_correct = True
        elif expected == "SLANT" and predicted in ["EXACT", "SLANT"]:
            is_correct = True
        elif expected == "NONE" and predicted == "NONE":
            is_correct = True
        elif expected != "NONE" and predicted != "NONE":
            # Both detected some rhyme, check if types match
            is_correct = (expected == predicted)
        
        if is_correct:
            correct += 1
            if expected != "NONE":
                per_type_stats[expected]["tp"] += 1
        else:
            failures.append({
                "test_case": asdict(test_case),
                "expected": expected,
                "predicted": predicted,
                "similarity": result.similarity,
            })
            if expected != "NONE":
                per_type_stats[expected]["fn"] += 1
            if predicted != "NONE":
                per_type_stats[predicted]["fp"] += 1
    
    # Calculate per-type metrics
    per_type = {}
    for rhyme_type, stats in per_type_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_type[rhyme_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    accuracy = correct / total if total > 0 else 0.0
    
    return TestResults(
        total=total,
        correct=correct,
        accuracy=accuracy,
        per_type=per_type,
        failures=failures,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rerun tests on saved test files (minimal ChatGPT API calls)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/iterative_tests",
        help="Directory containing saved test files",
    )
    parser.add_argument(
        "--use_chatgpt",
        action="store_true",
        help="Use ChatGPT for evaluation (default: just run tests and report)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results summary (default: print to console)",
    )
    parser.add_argument(
        "--min_file_num",
        type=int,
        default=1,
        help="Minimum test file number to process",
    )
    parser.add_argument(
        "--max_file_num",
        type=int,
        default=None,
        help="Maximum test file number to process (default: all)",
    )
    
    args = parser.parse_args()
    
    # Setup
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # Find all test files
    test_files = sorted(test_dir.glob("test_file_*_iter_0.jsonl"))
    
    if not test_files:
        print(f"[ERROR] No test files found in {test_dir}")
        return 1
    
    # Filter by file number range
    def get_file_num(path: Path) -> int:
        """Extract file number from path like test_file_5_iter_0.jsonl"""
        try:
            parts = path.stem.split("_")
            return int(parts[2])  # test_file_5_iter_0 -> 5
        except:
            return 0
    
    test_files = [f for f in test_files if get_file_num(f) >= args.min_file_num]
    if args.max_file_num:
        test_files = [f for f in test_files if get_file_num(f) <= args.max_file_num]
    
    print("=" * 70)
    print("RERUNNING SAVED TESTS")
    print("=" * 70)
    print(f"Found {len(test_files)} test files")
    print(f"Test directory: {test_dir}")
    print(f"Using ChatGPT for evaluation: {args.use_chatgpt}")
    print()
    
    # Initialize detector
    detector = RhymeDetector()
    
    # Initialize ChatGPT (only if needed)
    chatgpt = None
    if args.use_chatgpt:
        try:
            chatgpt = ChatGPTTester()
            print("[INFO] ChatGPT initialized for evaluation")
        except Exception as e:
            print(f"[WARN] Failed to initialize ChatGPT: {e}")
            print("[INFO] Continuing without ChatGPT evaluation")
    
    # Process each test file
    all_results = []
    overall_stats = {
        "total_files": len(test_files),
        "total_test_cases": 0,
        "total_correct": 0,
        "per_file_accuracy": [],
    }
    
    for test_file in test_files:
        file_num = get_file_num(test_file)
        print(f"\n{'=' * 70}")
        print(f"Processing Test File #{file_num}: {test_file.name}")
        print(f"{'=' * 70}")
        
        # Load test cases
        test_cases = load_saved_test_file(test_file)
        if not test_cases:
            print(f"[WARN] No test cases loaded from {test_file.name}")
            continue
        
        print(f"[INFO] Loaded {len(test_cases)} test cases")
        overall_stats["total_test_cases"] += len(test_cases)
        
        # Run tests
        print("[INFO] Running tests...")
        results = run_tests(test_cases, detector)
        
        print(f"[RESULTS] Accuracy: {results.accuracy:.1%} ({results.correct}/{results.total})")
        print(f"  Per-type performance:")
        for rhyme_type, metrics in results.per_type.items():
            print(f"    {rhyme_type}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
        
        overall_stats["total_correct"] += results.correct
        overall_stats["per_file_accuracy"].append({
            "file_num": file_num,
            "accuracy": results.accuracy,
            "total": results.total,
            "correct": results.correct,
        })
        
        all_results.append({
            "file_num": file_num,
            "file_name": test_file.name,
            "results": asdict(results),
        })
        
        # Optional: Get ChatGPT evaluation
        if chatgpt and args.use_chatgpt:
            print("[INFO] Getting ChatGPT evaluation...")
            try:
                current_code = (ROOT / "rapbot" / "rhyme_detector.py").read_text(encoding="utf-8")
                evaluation = chatgpt.evaluate_results(results, [], current_code)
                print(f"[EVALUATION] {evaluation.get('analysis', 'No analysis')[:200]}...")
            except Exception as e:
                print(f"[WARN] Failed to get ChatGPT evaluation: {e}")
    
    # Overall summary
    overall_accuracy = overall_stats["total_correct"] / overall_stats["total_test_cases"] if overall_stats["total_test_cases"] > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total test files processed: {overall_stats['total_files']}")
    print(f"Total test cases: {overall_stats['total_test_cases']}")
    print(f"Total correct: {overall_stats['total_correct']}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print()
    print("Per-file accuracy:")
    for file_stat in sorted(overall_stats["per_file_accuracy"], key=lambda x: x["file_num"]):
        print(f"  File #{file_stat['file_num']:2d}: {file_stat['accuracy']:.1%} ({file_stat['correct']}/{file_stat['total']})")
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_stats": overall_stats,
                "overall_accuracy": overall_accuracy,
                "per_file_results": all_results,
            }, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Analyze failures from test results to understand what needs improvement."""

import json
from pathlib import Path
from collections import defaultdict

def analyze_failures():
    """Analyze failures from the performance report."""
    report_file = Path("data/iterative_tests/full_performance_report.json")
    
    if not report_file.exists():
        print(f"Report file not found: {report_file}")
        return
    
    with open(report_file, "r") as f:
        data = json.load(f)
    
    # Collect all failures
    all_failures = []
    for file_result in data["per_file_results"]:
        failures = file_result["results"].get("failures", [])
        for failure in failures:
            failure["file_num"] = file_result["file_num"]
            all_failures.append(failure)
    
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    print(f"Total failures: {len(all_failures)}")
    print()
    
    # Group by expected type
    by_expected = defaultdict(list)
    for failure in all_failures:
        by_expected[failure["expected"]].append(failure)
    
    print("Failures by Expected Type:")
    for expected_type, failures in sorted(by_expected.items()):
        print(f"  {expected_type}: {len(failures)} failures")
        # Show what they were predicted as
        predicted_counts = defaultdict(int)
        for f in failures:
            predicted_counts[f["predicted"]] += 1
        for pred, count in sorted(predicted_counts.items(), key=lambda x: -x[1]):
            print(f"    -> Predicted as {pred}: {count}")
    
    print()
    print("Sample Assonance Failures:")
    assonance_fails = by_expected.get("ASSONANCE", [])[:5]
    for i, f in enumerate(assonance_fails, 1):
        line1_words = f["test_case"]["line1"].split()
        line2_words = f["test_case"]["line2"].split()
        word1 = line1_words[-1] if line1_words else "?"
        word2 = line2_words[-1] if line2_words else "?"
        print(f"  {i}. '{word1}' / '{word2}'")
        print(f"     Expected: {f['expected']}, Got: {f['predicted']}, Sim: {f['similarity']:.2f}")
    
    print()
    print("Sample Consonance Failures:")
    consonance_fails = by_expected.get("CONSONANCE", [])[:5]
    for i, f in enumerate(consonance_fails, 1):
        line1_words = f["test_case"]["line1"].split()
        line2_words = f["test_case"]["line2"].split()
        word1 = line1_words[-1] if line1_words else "?"
        word2 = line2_words[-1] if line2_words else "?"
        print(f"  {i}. '{word1}' / '{word2}'")
        print(f"     Expected: {f['expected']}, Got: {f['predicted']}, Sim: {f['similarity']:.2f}")
    
    print()
    print("Sample Multi-syllable Failures:")
    multisyllable_fails = by_expected.get("MULTISYLLABLE", [])[:5]
    for i, f in enumerate(multisyllable_fails, 1):
        line1_words = f["test_case"]["line1"].split()
        line2_words = f["test_case"]["line2"].split()
        word1 = line1_words[-1] if line1_words else "?"
        word2 = line2_words[-1] if line2_words else "?"
        print(f"  {i}. '{word1}' / '{word2}'")
        print(f"     Expected: {f['expected']}, Got: {f['predicted']}, Sim: {f['similarity']:.2f}")

if __name__ == "__main__":
    analyze_failures()

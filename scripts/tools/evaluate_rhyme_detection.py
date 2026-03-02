#!/usr/bin/env python
"""
evaluate_rhyme_detection.py

Evaluation metrics (precision/recall/F1) for rhyme detection system.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

from rapbot.rhyme_detector import RhymeDetector, RhymeType


def calculate_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_on_dataset(
    test_data_path: Path,
    detector: RhymeDetector,
) -> Dict[str, Dict[str, float]]:
    """Evaluate detector on test dataset."""
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
    
    # Load test cases
    test_cases = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))
    
    # Metrics by rhyme type
    metrics_by_type = defaultdict(lambda: {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    })
    
    total_cases = 0
    
    for case in test_cases:
        line1 = case.get("line1", "")
        line2 = case.get("line2", "")
        expected = case.get("expected", {})
        expected_end_rhyme = expected.get("end_rhyme", "NONE")
        
        if not line1 or not line2:
            continue
        
        total_cases += 1
        
        # Detect end rhyme
        result = detector.detect_end_rhyme(line1, line2)
        predicted_type = result.rhyme_type.value if result.rhyme_type != RhymeType.NONE else "NONE"
        
        # Map expected to our types
        if expected_end_rhyme == "EXACT":
            expected_types = ["EXACT", "SLANT"]  # Accept slant as correct for exact
        elif expected_end_rhyme == "SLANT":
            expected_types = ["SLANT", "EXACT"]  # Accept exact as correct for slant
        else:
            expected_types = ["NONE"]
        
        # Update metrics
        if predicted_type in expected_types:
            if expected_end_rhyme != "NONE":
                metrics_by_type[expected_end_rhyme]["true_positives"] += 1
            else:
                metrics_by_type["NONE"]["true_positives"] += 1
        else:
            if expected_end_rhyme != "NONE":
                metrics_by_type[expected_end_rhyme]["false_negatives"] += 1
                if predicted_type != "NONE":
                    metrics_by_type[predicted_type]["false_positives"] += 1
            else:
                metrics_by_type["NONE"]["false_negatives"] += 1
                if predicted_type != "NONE":
                    metrics_by_type[predicted_type]["false_positives"] += 1
    
    # Calculate final metrics
    final_metrics = {}
    for rhyme_type, counts in metrics_by_type.items():
        final_metrics[rhyme_type] = calculate_metrics(
            counts["true_positives"],
            counts["false_positives"],
            counts["false_negatives"],
        )
    
    # Overall metrics
    overall_tp = sum(m["true_positives"] for m in metrics_by_type.values())
    overall_fp = sum(m["false_positives"] for m in metrics_by_type.values())
    overall_fn = sum(m["false_negatives"] for m in metrics_by_type.values())
    
    final_metrics["OVERALL"] = calculate_metrics(overall_tp, overall_fp, overall_fn)
    final_metrics["total_cases"] = {"count": total_cases}
    
    return final_metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    
    total = metrics.get("total_cases", {}).get("count", 0)
    print(f"\nTotal test cases: {total}\n")
    
    # Print per-type metrics
    for rhyme_type in ["EXACT", "SLANT", "NONE"]:
        if rhyme_type in metrics:
            m = metrics[rhyme_type]
            print(f"{rhyme_type} Rhymes:")
            print(f"  Precision: {m['precision']:.3f}")
            print(f"  Recall:    {m['recall']:.3f}")
            print(f"  F1 Score:  {m['f1']:.3f}")
            print(f"  TP: {m['true_positives']}, FP: {m['false_positives']}, FN: {m['false_negatives']}")
            print()
    
    # Overall metrics
    if "OVERALL" in metrics:
        m = metrics["OVERALL"]
        print("OVERALL:")
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall:    {m['recall']:.3f}")
        print(f"  F1 Score:  {m['f1']:.3f}")
        print(f"  TP: {m['true_positives']}, FP: {m['false_positives']}, FN: {m['false_negatives']}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rhyme detection system with precision/recall/F1 metrics"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test_rhymes.jsonl (default: data/test_rhymes.jsonl)",
    )
    parser.add_argument(
        "--rhyme_csv",
        type=str,
        default=None,
        help="Path to rhymes_grouped.csv (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for metrics",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = ROOT / "data" / "test_rhymes.jsonl"
    
    rhyme_csv = Path(args.rhyme_csv) if args.rhyme_csv else None
    
    # Initialize detector
    detector = RhymeDetector(rhyme_groups_csv=rhyme_csv)
    
    # Evaluate
    try:
        metrics = evaluate_on_dataset(test_data_path, detector)
        print_metrics(metrics)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

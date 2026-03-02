#!/usr/bin/env python
"""
iterative_tester.py

Main iterative testing loop that:
1. Generates test cases with ChatGPT
2. Runs tests
3. Evaluates results with ChatGPT
4. Modifies code based on feedback
5. Iterates until convergence
6. Requests new test file and repeats
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import subprocess
import logging
import signal
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
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

# Setup logging for unattended operation
def setup_logging(log_dir: Path):
    """Setup file logging for unattended operation."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"iterative_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

# Global state for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logging.info("Shutdown signal received. Finishing current iteration...")
    shutdown_requested = True


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


def run_unit_tests() -> bool:
    """Run pytest unit tests to ensure code still works."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_rhyme_detector.py", "-v", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[WARN] Failed to run unit tests: {e}")
        return False


def save_progress(test_dir: Path, test_file_num: int, iteration: int, results: TestResults, state: Dict):
    """Save progress to resume later if needed."""
    progress_file = test_dir / "progress.json"
    progress = {
        "test_file_num": test_file_num,
        "iteration": iteration,
        "last_accuracy": results.accuracy,
        "timestamp": datetime.now().isoformat(),
        "state": state,
    }
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    logging.info(f"Progress saved: test_file={test_file_num}, iteration={iteration}, accuracy={results.accuracy:.1%}")

def load_progress(test_dir: Path) -> Optional[Dict]:
    """Load previous progress if exists."""
    progress_file = test_dir / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load progress: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Iterative testing with ChatGPT (can run unattended)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env or .env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="ChatGPT model to use",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum iterations per test file",
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.85,
        help="Target accuracy to achieve",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/iterative_tests",
        help="Directory to store test files",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default="rapbot/rhyme_detector.py",
        help="Code file to modify",
    )
    parser.add_argument(
        "--num_test_files",
        type=int,
        default=3,
        help="Number of test files to process",
    )
    parser.add_argument(
        "--apply_priority",
        type=str,
        default="high",
        help="Only apply suggestions with this priority (high/medium/low/all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress",
    )
    parser.add_argument(
        "--max_runtime_hours",
        type=float,
        default=8.0,
        help="Maximum runtime in hours (default: 8 for overnight run)",
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup
    test_dir = Path(args.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    log_dir = test_dir / "logs"
    
    # Setup logging
    log_file = setup_logging(log_dir)
    start_time = datetime.now()
    
    target_file = ROOT / args.target_file
    if not target_file.exists():
        logging.error(f"Target file not found: {target_file}")
        return 1
    
    # Initialize components
    try:
        chatgpt = ChatGPTTester(api_key=args.api_key, model=args.model)
        code_modifier = CodeModifier(target_file)
        detector = RhymeDetector()
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return 1
    
    logging.info("=" * 70)
    logging.info("ITERATIVE CHATGPT TESTING (UNATTENDED MODE)")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model}")
    logging.info(f"Target accuracy: {args.accuracy_threshold:.1%}")
    logging.info(f"Max iterations per file: {args.max_iterations}")
    logging.info(f"Number of test files: {args.num_test_files}")
    logging.info(f"Max runtime: {args.max_runtime_hours} hours")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    
    # Check for resume
    start_test_file = 1
    if args.resume:
        progress = load_progress(test_dir)
        if progress:
            start_test_file = progress.get("test_file_num", 1)
            logging.info(f"Resuming from test file {start_test_file}, iteration {progress.get('iteration', 0)}")
    
    # Process multiple test files
    for test_file_num in range(start_test_file, args.num_test_files + 1):
        # Check runtime limit
        runtime = (datetime.now() - start_time).total_seconds() / 3600
        if runtime >= args.max_runtime_hours:
            logging.info(f"Max runtime ({args.max_runtime_hours} hours) reached. Stopping.")
            break
        
        if shutdown_requested:
            logging.info("Shutdown requested. Saving progress and exiting.")
            break
        
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"TEST FILE #{test_file_num}")
        logging.info("=" * 70)
        
        # Generate initial test cases
        logging.info("[STEP 1] Generating test cases with ChatGPT...")
        try:
            test_cases = chatgpt.generate_test_cases(
                num_cases=25,
                iteration=1,
            )
        except Exception as e:
            logging.error(f"Failed to generate test cases: {e}")
            logging.info("Waiting 60 seconds before retry...")
            import time
            time.sleep(60)
            continue
        
        if not test_cases:
            logging.error("Failed to generate test cases")
            continue
        
        # Save test cases
        test_file_path = test_dir / f"test_file_{test_file_num}_iter_0.jsonl"
        with open(test_file_path, "w", encoding="utf-8") as f:
            for tc in test_cases:
                f.write(json.dumps(asdict(tc)) + "\n")
        logging.info(f"Saved {len(test_cases)} test cases to {test_file_path}")
        
        # Iteration loop
        iteration = 0
        best_accuracy = 0.0
        no_improvement_count = 0
        
        while iteration < args.max_iterations:
            if shutdown_requested:
                logging.info("Shutdown requested. Saving progress...")
                save_progress(test_dir, test_file_num, iteration, results if 'results' in locals() else TestResults(0, 0, 0.0, {}, []), {})
                break
            
            # Check runtime
            runtime = (datetime.now() - start_time).total_seconds() / 3600
            if runtime >= args.max_runtime_hours:
                logging.info(f"Max runtime reached. Saving progress...")
                save_progress(test_dir, test_file_num, iteration, results if 'results' in locals() else TestResults(0, 0, 0.0, {}, []), {})
                break
            
            iteration += 1
            logging.info("")
            logging.info(f"--- Iteration {iteration}/{args.max_iterations} ---")
            logging.info(f"Runtime: {runtime:.1f} hours")
            
            # Run tests
            logging.info("[STEP 2] Running tests...")
            try:
                results = run_tests(test_cases, detector)
                logging.info(f"Accuracy: {results.accuracy:.1%} ({results.correct}/{results.total})")
            except Exception as e:
                logging.error(f"Error running tests: {e}")
                import traceback
                logging.error(traceback.format_exc())
                break
            
            # Check if we've improved
            if results.accuracy > best_accuracy:
                best_accuracy = results.accuracy
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Save progress after each iteration
            save_progress(test_dir, test_file_num, iteration, results, {
                "best_accuracy": best_accuracy,
                "no_improvement_count": no_improvement_count,
            })
            
            # Check convergence
            if results.accuracy >= args.accuracy_threshold:
                logging.info(f"[SUCCESS] Achieved target accuracy: {results.accuracy:.1%} >= {args.accuracy_threshold:.1%}")
                break
            
            if no_improvement_count >= 3:
                logging.info(f"[STOP] No improvement for 3 iterations. Best accuracy: {best_accuracy:.1%}")
                break
            
            # Get current code
            try:
                current_code = target_file.read_text(encoding="utf-8")
            except Exception as e:
                logging.error(f"Failed to read code file: {e}")
                break
            
            # Evaluate with ChatGPT
            logging.info("[STEP 3] Evaluating results with ChatGPT...")
            try:
                evaluation = chatgpt.evaluate_results(results, [], current_code)
            except Exception as e:
                logging.error(f"Failed to get evaluation from ChatGPT: {e}")
                logging.info("Waiting 30 seconds before retry...")
                import time
                time.sleep(30)
                continue
            
            suggestions = evaluation.get("suggestions", [])
            if not suggestions:
                logging.info("No suggestions from ChatGPT")
                break
            
            logging.info(f"Received {len(suggestions)} suggestions")
            for sug in suggestions[:3]:  # Show first 3
                logging.info(f"  - [{sug.get('priority', '?')}] {sug.get('issue', '?')}")
            
            # Apply suggestions
            logging.info("[STEP 4] Applying code modifications...")
            priority_filter = None if args.apply_priority == "all" else args.apply_priority
            try:
                success, modified_code, errors = code_modifier.apply_suggestions(
                    suggestions,
                    priority_filter=priority_filter,
                )
            except Exception as e:
                logging.error(f"Error applying suggestions: {e}")
                break
            
            if not success:
                logging.warning("Failed to apply suggestions")
                if errors:
                    for error in errors:
                        logging.warning(f"  Error: {error}")
                break
            
            # Reload detector with new code
            logging.info("[STEP 5] Reloading detector...")
            try:
                import importlib
                import rapbot.rhyme_detector
                importlib.reload(rapbot.rhyme_detector)
                detector = rapbot.rhyme_detector.RhymeDetector()
            except Exception as e:
                logging.error(f"Failed to reload detector: {e}")
                break
            
            # Run unit tests to verify
            logging.info("[STEP 6] Running unit tests...")
            if not run_unit_tests():
                logging.warning("Unit tests failed after modification. Consider rollback.")
                # Could add rollback logic here
            
            # Save iteration results
            iteration_file = test_dir / f"test_file_{test_file_num}_iter_{iteration}_results.json"
            try:
                with open(iteration_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "iteration": iteration,
                        "accuracy": results.accuracy,
                        "results": asdict(results),
                        "evaluation": evaluation,
                        "suggestions_applied": len(suggestions),
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save iteration results: {e}")
        
        logging.info(f"[COMPLETE] Test file #{test_file_num} finished. Best accuracy: {best_accuracy:.1%}")
        
        # Request new test file for next iteration
        if test_file_num < args.num_test_files and not shutdown_requested:
            logging.info("[STEP 7] Requesting new test file from ChatGPT...")
            try:
                # Ensure we pass a dict, not a bool
                perf_dict = {"accuracy": best_accuracy} if isinstance(best_accuracy, (int, float)) else None
                test_cases = chatgpt.generate_new_test_file(previous_performance=perf_dict)
            except Exception as e:
                logging.error(f"Failed to generate new test file: {e}")
                import traceback
                logging.error(traceback.format_exc())
                break
    
    total_runtime = (datetime.now() - start_time).total_seconds() / 3600
    logging.info("")
    logging.info("=" * 70)
    logging.info("ITERATIVE TESTING COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Total runtime: {total_runtime:.1f} hours")
    logging.info(f"Log file: {log_file}")
    logging.info("Check data/iterative_tests/ for results and test files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

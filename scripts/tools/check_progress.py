#!/usr/bin/env python
"""Quick script to check progress of overnight iterative testing."""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "data" / "iterative_tests"

def check_progress():
    """Check current progress."""
    progress_file = TEST_DIR / "progress.json"
    
    print("=" * 70)
    print("ITERATIVE TESTING PROGRESS")
    print("=" * 70)
    
    if not progress_file.exists():
        print("No progress file found. Testing may not have started yet.")
        return
    
    with open(progress_file, "r") as f:
        progress = json.load(f)
    
    print(f"Test File: #{progress.get('test_file_num', '?')}")
    print(f"Iteration: {progress.get('iteration', '?')}")
    print(f"Last Accuracy: {progress.get('last_accuracy', 0):.1%}")
    print(f"Last Updated: {progress.get('timestamp', '?')}")
    
    # Check log files
    log_dir = TEST_DIR / "logs"
    if log_dir.exists():
        log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            print(f"\nLatest log: {latest_log.name}")
            print(f"Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
            
            # Show last few lines
            print("\nLast 10 lines of log:")
            print("-" * 70)
            with open(latest_log, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())
    
    print("=" * 70)

if __name__ == "__main__":
    check_progress()

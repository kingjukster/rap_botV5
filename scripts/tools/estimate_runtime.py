#!/usr/bin/env python
"""Estimate runtime for iterative testing."""

def estimate_runtime(max_iterations, num_test_files):
    """
    Estimate runtime based on typical operation times.
    
    Per iteration:
    - Generate test cases (ChatGPT): ~15-30 seconds
    - Run tests: ~2-5 seconds
    - Evaluate results (ChatGPT): ~15-30 seconds
    - Get suggestions (ChatGPT): ~15-30 seconds
    - Apply code changes: ~2-5 seconds
    - Reload detector: ~1-2 seconds
    - Run unit tests: ~5-15 seconds
    - Save results: ~1 second
    
    Total per iteration: ~60-120 seconds (1-2 minutes)
    """
    
    # Time per iteration (in minutes)
    min_time_per_iter = 1.0  # Fast path
    avg_time_per_iter = 1.5  # Average
    max_time_per_iter = 2.5  # Slow path (API delays, etc.)
    
    # Additional time per test file
    time_per_file_setup = 0.5  # Generate initial test cases
    
    # Calculate
    total_iterations = max_iterations * num_test_files
    total_file_setups = num_test_files
    
    min_total = (total_iterations * min_time_per_iter) + (total_file_setups * time_per_file_setup)
    avg_total = (total_iterations * avg_time_per_iter) + (total_file_setups * time_per_file_setup)
    max_total = (total_iterations * max_time_per_iter) + (total_file_setups * time_per_file_setup)
    
    return {
        "total_iterations": total_iterations,
        "min_hours": min_total / 60,
        "avg_hours": avg_total / 60,
        "max_hours": max_total / 60,
        "min_minutes": min_total,
        "avg_minutes": avg_total,
        "max_minutes": max_total,
    }

if __name__ == "__main__":
    import sys
    
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    est = estimate_runtime(max_iter, num_files)
    
    print("=" * 70)
    print("RUNTIME ESTIMATE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Iterations per file: {max_iter}")
    print(f"  Number of test files: {num_files}")
    print(f"  Total iterations: {est['total_iterations']}")
    print()
    print(f"Estimated Runtime:")
    print(f"  Best case:  {est['min_hours']:.1f} hours ({est['min_minutes']:.0f} minutes)")
    print(f"  Average:    {est['avg_hours']:.1f} hours ({est['avg_minutes']:.0f} minutes)")
    print(f"  Worst case: {est['max_hours']:.1f} hours ({est['max_minutes']:.0f} minutes)")
    print()
    print("Note: Times can vary based on:")
    print("  - ChatGPT API response times")
    print("  - Network latency")
    print("  - Code complexity of suggestions")
    print("  - Rate limiting delays")
    print("=" * 70)

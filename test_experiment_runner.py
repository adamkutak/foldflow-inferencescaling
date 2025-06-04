#!/usr/bin/env python3
"""
Simple test script for the experiment runner.
"""

import os
import sys
import subprocess


def test_experiment_runner():
    """Test the experiment runner with minimal parameters."""
    print("Testing experiment runner...")

    # Run with minimal parameters for quick testing
    cmd = [
        sys.executable,
        "experiment_runner.py",
        "--num_samples",
        "2",  # Very small for testing
        "--sample_length",
        "30",  # Small protein
        "--scoring_function",
        "tm_score",
        "--noise_scale",
        "0.05",
        "--lambda_div",
        "0.2",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 minute timeout

        if result.returncode == 0:
            print("✓ Experiment runner completed successfully!")
            print("STDOUT:")
            print(result.stdout)
        else:
            print("✗ Experiment runner failed!")
            print("STDERR:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)

    except subprocess.TimeoutExpired:
        print("✗ Experiment runner timed out!")
    except Exception as e:
        print(f"✗ Error running experiment runner: {e}")


if __name__ == "__main__":
    test_experiment_runner()

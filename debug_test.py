#!/usr/bin/env python3
"""
Quick test script to debug inference methods.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Now run the experiment
if __name__ == "__main__":
    print("Running debug test with verbose logging...")
    print("This will show detailed branching and evaluation information.")
    print()

    # Import after logging setup
    from experiment_runner import main

    # Override sys.argv to run with our test parameters
    sys.argv = [
        "experiment_runner.py",
        "--num_samples",
        "1",
        "--sample_length",
        "30",
        "--branch_counts",
        "2",
        "--branch_interval",
        "0.1",  # This should give us ~10 branching events
    ]

    main()

#!/usr/bin/env python3
"""
Example script for running multi-GPU inference scaling experiments.

This script demonstrates how to use the MultiGPUExperimentRunner with different
configurations for concurrent experiments on multiple GPUs.
"""

import subprocess
import sys
import os


def run_experiment(config_name, gpu_ids, num_samples=4, sample_length=50):
    """Run a multi-GPU experiment with the specified configuration."""

    print(f"\n{'='*60}")
    print(f"Running {config_name} experiment")
    print(f"GPUs: {gpu_ids}")
    print(f"Samples per method: {num_samples}")
    print(f"Sample length: {sample_length}")
    print(f"{'='*60}\n")

    # Build the command
    cmd = [
        "python",
        "experiment_runner_multi_gpu.py",
        "--num_samples",
        str(num_samples),
        "--sample_length",
        str(sample_length),
        "--gpu_ids",
    ] + [str(gpu_id) for gpu_id in gpu_ids]

    # Add additional parameters for different experiment types
    if config_name == "quick_test":
        cmd.extend(["--branch_counts", "2", "4", "--num_samples", "2"])
    elif config_name == "comprehensive":
        cmd.extend(
            [
                "--branch_counts",
                "2",
                "4",
                "8",
                "16",
                "--num_samples",
                "8",
                "--sample_length",
                "100",
            ]
        )
    elif config_name == "sde_focused":
        cmd.extend(
            [
                "--branch_counts",
                "2",
                "4",
                "8",
                "--noise_scale",
                "0.1",
                "--lambda_div",
                "0.3",
            ]
        )
    elif config_name == "ode_focused":
        cmd.extend(
            [
                "--branch_counts",
                "2",
                "4",
                "8",
                "--noise_scale",
                "0.2",
                "--lambda_div",
                "0.8",
            ]
        )

    print(f"Command: {' '.join(cmd)}\n")

    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {config_name} experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {config_name} experiment failed with error code {e.returncode}")
        return False


def main():
    """Run different multi-GPU experiment configurations."""

    print("Multi-GPU Inference Scaling Experiments")
    print("=" * 50)

    # Check if the multi-GPU runner exists
    if not os.path.exists("experiment_runner_multi_gpu.py"):
        print("❌ Error: experiment_runner_multi_gpu.py not found!")
        print(
            "Please make sure the multi-GPU experiment runner is in the current directory."
        )
        sys.exit(1)

    # Example configurations
    experiments = [
        {
            "name": "quick_test",
            "gpu_ids": [0, 1],
            "description": "Quick test with 2 GPUs, minimal samples",
        },
        {
            "name": "comprehensive",
            "gpu_ids": [0, 1, 2],
            "description": "Comprehensive test with 3 GPUs, more samples",
        },
        {
            "name": "sde_focused",
            "gpu_ids": [0, 1, 2],
            "description": "SDE-focused experiments with optimized parameters",
        },
        {
            "name": "ode_focused",
            "gpu_ids": [0, 1, 2],
            "description": "ODE-focused experiments with optimized parameters",
        },
    ]

    print("Available experiment configurations:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
        print(f"     GPUs: {exp['gpu_ids']}")

    print("\nTo run a specific experiment, use:")
    print("  python run_multi_gpu_experiments.py --experiment <name>")
    print("\nOr run all experiments sequentially:")
    print("  python run_multi_gpu_experiments.py --run-all")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--run-all":
            print("\nRunning all experiments sequentially...")
            for exp in experiments:
                success = run_experiment(exp["name"], exp["gpu_ids"])
                if not success:
                    print(f"Stopping due to failure in {exp['name']}")
                    break
        elif sys.argv[1] == "--experiment" and len(sys.argv) > 2:
            exp_name = sys.argv[2]
            exp_config = next(
                (exp for exp in experiments if exp["name"] == exp_name), None
            )
            if exp_config:
                run_experiment(exp_config["name"], exp_config["gpu_ids"])
            else:
                print(f"❌ Unknown experiment: {exp_name}")
                print("Available experiments:", [exp["name"] for exp in experiments])
        else:
            print("❌ Invalid arguments. Use --run-all or --experiment <name>")
    else:
        print("\nNo arguments provided. Use --run-all or --experiment <name>")


if __name__ == "__main__":
    main()

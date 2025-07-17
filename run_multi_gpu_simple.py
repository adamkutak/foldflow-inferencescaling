#!/usr/bin/env python3
"""
Simple multi-GPU experiment runner that spawns multiple instances of experiment_runner.py
on different GPUs.

This approach avoids the complexity of trying to replicate the single-GPU runner's behavior
in a multi-GPU wrapper by simply running multiple instances of the original runner.
"""

import argparse
import logging
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import queue
import subprocess

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    if mp.get_start_method() != "spawn":
        raise RuntimeError(
            "Multiprocessing start method must be 'spawn' for CUDA compatibility"
        )


class SimpleMultiGPUExperimentRunner:
    """Simple multi-GPU runner that spawns multiple experiment_runner.py processes."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.setup_logging()
        self.setup_gpu_assignment()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def setup_gpu_assignment(self):
        """Setup GPU assignment for concurrent experiments."""
        self.available_gpus = self.args.gpu_ids
        self.num_gpus = len(self.available_gpus)
        self.logger.info(f"Using GPUs: {self.available_gpus}")

        # Debug: Check available GPUs
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            self.logger.info(f"Total CUDA devices available: {total_gpus}")
            for i in range(total_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                self.logger.info(f"  GPU {i}: {gpu_name}")
        else:
            self.logger.warning("CUDA not available!")

        # Create a queue for GPU assignment
        self.gpu_queue = Queue()
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)

    def run_single_experiment_worker(self, experiment_data: tuple) -> Dict[str, Any]:
        """Worker function to run a single experiment on a specific GPU."""
        method_config, gpu_id, result_queue = experiment_data

        # Calculate samples per GPU
        samples_per_gpu = self.args.num_samples // len(self.available_gpus)
        if gpu_id == self.available_gpus[0]:  # First GPU gets any remainder
            samples_per_gpu += self.args.num_samples % len(self.available_gpus)

        # Create command line arguments for experiment_runner.py
        cmd = [
            sys.executable,
            "experiment_runner.py",
            "--num_samples",
            str(samples_per_gpu),  # Use samples_per_gpu instead of total
            "--sample_length",
            str(self.args.sample_length),
            "--scoring_function",
            self.args.scoring_function,
            "--noise_scale",
            str(self.args.noise_scale),
            "--lambda_div",
            str(self.args.lambda_div),
            "--branch_interval",
            str(self.args.branch_interval),
            "--gpu_id",
            str(gpu_id),
            "--branch_counts",
        ] + [str(x) for x in self.args.branch_counts]

        # Add method-specific arguments
        method_name = method_config["method"]
        config = method_config.get("config", {})

        self.logger.info(
            f"Running experiment on GPU {gpu_id} with {samples_per_gpu} samples per method"
        )
        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run the experiment_runner.py process
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                self.logger.info(f"Experiment on GPU {gpu_id} completed successfully")
                # Parse results from stdout or log files
                # For now, we'll just return basic info
                return {
                    "method": method_name,
                    "gpu_id": gpu_id,
                    "samples_per_gpu": samples_per_gpu,
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                self.logger.error(
                    f"Experiment on GPU {gpu_id} failed with return code {result.returncode}"
                )
                return {
                    "method": method_name,
                    "gpu_id": gpu_id,
                    "samples_per_gpu": samples_per_gpu,
                    "status": "failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

        except subprocess.TimeoutExpired:
            self.logger.error(f"Experiment on GPU {gpu_id} timed out")
            return {
                "method": method_name,
                "gpu_id": gpu_id,
                "samples_per_gpu": samples_per_gpu,
                "status": "timeout",
            }
        except Exception as e:
            self.logger.error(f"Exception running experiment on GPU {gpu_id}: {e}")
            return {
                "method": method_name,
                "gpu_id": gpu_id,
                "samples_per_gpu": samples_per_gpu,
                "status": "exception",
                "error": str(e),
            }

    def run_all_experiments(self):
        """Run all experiments by spawning multiple experiment_runner.py processes."""
        # For simplicity, we'll run one experiment per GPU
        # Each experiment will run all methods and branch counts
        experiments = []

        # Create one experiment per GPU
        for gpu_id in self.available_gpus:
            experiments.append(
                {
                    "method": "all_methods",  # This will run all methods
                    "config": {},
                    "gpu_id": gpu_id,
                }
            )

        # Create a manager for shared result queue
        manager = Manager()
        result_queue = manager.Queue()

        # Run experiments concurrently
        processes = []
        process_gpu_map = {}

        for exp_config in experiments:
            gpu_id = exp_config["gpu_id"]
            experiment_data = (exp_config, gpu_id, result_queue)

            p = Process(
                target=self.run_single_experiment_worker, args=(experiment_data,)
            )
            p.start()
            processes.append(p)
            process_gpu_map[p] = gpu_id

            self.logger.info(f"Started experiment on GPU {gpu_id}")

        # Wait for all processes to complete
        for p in processes:
            p.join()
            gpu_id = process_gpu_map[p]
            self.logger.info(f"Process on GPU {gpu_id} finished")

        # Collect all results
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                self.results.append(result)
            except queue.Empty:
                break

        # Save results
        self.save_results()
        self.analyze_results()

    def save_results(self):
        """Save experiment results to files."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join("experiments", f"multi_gpu_simple_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        # Save detailed results as JSON
        results_file = os.path.join(experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("SIMPLE MULTI-GPU EXPERIMENT RESULTS")
        print("=" * 80)

        print(f"Total experiments run: {len(self.results)}")
        print(f"GPUs used: {self.available_gpus}")
        print(f"Total samples requested: {self.args.num_samples}")
        print(f"Sample distribution:")

        total_samples_allocated = 0
        for result in self.results:
            samples_per_gpu = result.get("samples_per_gpu", 0)
            total_samples_allocated += samples_per_gpu
            print(f"  GPU {result['gpu_id']}: {samples_per_gpu} samples")

        print(f"Total samples allocated: {total_samples_allocated}")
        print()

        # Print results for each GPU
        for result in self.results:
            print(f"GPU {result['gpu_id']}: {result['status']}")
            if result["status"] == "success":
                print(f"  Samples: {result.get('samples_per_gpu', 0)}")
                print(f"  Output length: {len(result.get('stdout', ''))} characters")
            elif result["status"] == "failed":
                print(f"  Samples: {result.get('samples_per_gpu', 0)}")
                print(f"  Error: {result.get('stderr', 'No error message')}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Run simple multi-GPU experiments using experiment_runner.py"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate per method",
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=50,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--scoring_function",
        type=str,
        default="tm_score",
        choices=["tm_score", "rmsd"],
        help="Scoring function to use for evaluation",
    )

    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.3,
        help="Noise scale for SDE path exploration",
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0.6,
        help="Lambda for divergence-free vector fields",
    )

    parser.add_argument(
        "--branch_interval",
        type=float,
        default=0.1,
        help="Time interval between branches",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of GPU IDs to use for concurrent experiments",
    )

    parser.add_argument(
        "--branch_counts",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="List of branch counts to use for experiments",
    )

    args = parser.parse_args()

    print("Starting Simple Multi-GPU Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Noise scale (SDE): {args.noise_scale}")
    print(f"  Lambda div (ODE): {args.lambda_div}")
    print(f"  Branch interval: {args.branch_interval}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print(f"  Branch counts: {args.branch_counts}")
    print()

    # Create and run experiments
    runner = SimpleMultiGPUExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

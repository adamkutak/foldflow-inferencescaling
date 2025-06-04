#!/usr/bin/env python3
"""
Experiment runner for comparing inference scaling methods in protein design.

This script runs experiments to compare different inference methods:
- Standard sampling (baseline)
- Best-of-N sampling
- SDE path exploration
- Divergence-free ODE path exploration

Each method is tested with different computational budgets (number of branches)
to evaluate inference time scaling performance.
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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runner.inference_methods import get_inference_method
from runner.inference import Sampler
from omegaconf import DictConfig, OmegaConf


class ExperimentRunner:
    """Runner for inference scaling experiments."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.setup_logging()
        self.setup_config()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def setup_config(self):
        """Setup configuration for experiments."""
        # Load base configuration
        config_path = "runner/config/inference.yaml"
        self.base_conf = OmegaConf.load(config_path)

        # Override with experiment parameters
        self.base_conf.inference.samples.samples_per_length = (
            1  # We'll control this manually
        )
        self.base_conf.inference.samples.min_length = self.args.sample_length
        self.base_conf.inference.samples.max_length = self.args.sample_length
        self.base_conf.inference.samples.length_step = 1

        # Create output directory for experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "experiments", f"inference_scaling_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def create_sampler(self, method_config: Dict[str, Any]) -> Sampler:
        """Create a sampler with specific method configuration."""
        conf = OmegaConf.create(self.base_conf)

        # Update method configuration
        conf.inference.samples.inference_method = method_config["method"]
        conf.inference.samples.method_config = method_config.get("config", {})

        # Set output directory for this experiment
        method_name = method_config["method"]
        branches = method_config.get("config", {}).get("num_branches", 1)
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_branches_{branches}"
        )

        return Sampler(conf)

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given method configuration."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        branches = config.get("num_branches", 1)

        self.logger.info(f"Running experiment: {method_name} with {branches} branches")

        # Create sampler
        sampler = self.create_sampler(method_config)

        # Track timing and results
        start_time = time.time()
        scores = []

        for sample_idx in range(self.args.num_samples):
            self.logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

            try:
                # Generate sample
                sample_start = time.time()
                sample_result = sampler.inference_method.sample(self.args.sample_length)
                sample_time = time.time() - sample_start

                # Extract sample and score
                if isinstance(sample_result, dict) and "sample" in sample_result:
                    sample_output = sample_result["sample"]
                    # Use the score from the method if available
                    if "score" in sample_result:
                        score = sample_result["score"]
                    else:
                        # Evaluate manually
                        score_fn = sampler.inference_method.get_score_function(
                            self.args.scoring_function
                        )
                        score = score_fn(sample_output, self.args.sample_length)
                else:
                    sample_output = sample_result
                    # Evaluate manually
                    score_fn = sampler.inference_method.get_score_function(
                        self.args.scoring_function
                    )
                    score = score_fn(sample_output, self.args.sample_length)

                scores.append(score)
                self.logger.info(f"    Score: {score:.4f}, Time: {sample_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Error in sample {sample_idx}: {e}")
                scores.append(float("nan"))

        total_time = time.time() - start_time

        # Calculate statistics
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            max_score = np.max(valid_scores)
            min_score = np.min(valid_scores)
        else:
            mean_score = std_score = max_score = min_score = float("nan")

        result = {
            "method": method_name,
            "num_branches": branches,
            "config": config,
            "num_samples": len(valid_scores),
            "mean_score": mean_score,
            "std_score": std_score,
            "max_score": max_score,
            "min_score": min_score,
            "total_time": total_time,
            "time_per_sample": total_time / self.args.num_samples,
            "scores": scores,
            "scoring_function": self.args.scoring_function,
        }

        self.logger.info(
            f"  Results: Mean={mean_score:.4f}±{std_score:.4f}, Time={total_time:.2f}s"
        )
        return result

    def run_all_experiments(self):
        """Run all experiments with different methods and branch counts."""
        experiments = []

        # 1. Baseline: Standard sampling
        experiments.append({"method": "standard", "config": {}})

        # 2. Best-of-N with different N values
        for n_branches in [2, 4, 8]:
            experiments.append(
                {
                    "method": "best_of_n",
                    "config": {
                        "n_samples": n_branches,
                        "selector": self.args.scoring_function,
                    },
                }
            )

        # 3. SDE path exploration with different branch counts
        for n_branches in [2, 4, 8]:
            experiments.append(
                {
                    "method": "sde_path_exploration",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "noise_scale": self.args.noise_scale,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                    },
                }
            )

        # 4. Divergence-free ODE with different branch counts
        for n_branches in [2, 4, 8]:
            experiments.append(
                {
                    "method": "divergence_free_ode",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                    },
                }
            )

        # Run all experiments
        for exp_config in experiments:
            try:
                result = self.run_single_experiment(exp_config)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Failed experiment {exp_config}: {e}")

        # Save results
        self.save_results()
        self.analyze_results()

    def save_results(self):
        """Save experiment results to files."""
        # Save detailed results as JSON
        results_file = os.path.join(self.experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []
        for result in self.results:
            summary_data.append(
                {
                    "method": result["method"],
                    "num_branches": result["num_branches"],
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                    "max_score": result["max_score"],
                    "min_score": result["min_score"],
                    "total_time": result["total_time"],
                    "time_per_sample": result["time_per_sample"],
                    "num_samples": result["num_samples"],
                    "scoring_function": result["scoring_function"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("INFERENCE SCALING EXPERIMENT RESULTS")
        print("=" * 80)

        # Find baseline (standard method)
        baseline = None
        for result in self.results:
            if result["method"] == "standard":
                baseline = result
                break

        if baseline is None:
            print("Warning: No baseline (standard) method found!")
            return

        baseline_score = baseline["mean_score"]
        baseline_time = baseline["time_per_sample"]

        print(f"Baseline (Standard): {baseline_score:.4f}±{baseline['std_score']:.4f}")
        print(f"Scoring Function: {self.args.scoring_function}")
        print(f"Sample Length: {self.args.sample_length}")
        print(f"Samples per Method: {self.args.num_samples}")
        print()

        # Group results by method
        methods = {}
        for result in self.results:
            method = result["method"]
            if method not in methods:
                methods[method] = []
            methods[method].append(result)

        # Print results for each method
        for method_name, method_results in methods.items():
            if method_name == "standard":
                continue

            print(f"{method_name.upper()}:")
            print(
                f"{'Branches':<10} {'Mean Score':<12} {'Improvement':<12} {'Time (s)':<10} {'Speedup':<10}"
            )
            print("-" * 60)

            for result in sorted(method_results, key=lambda x: x["num_branches"]):
                branches = result["num_branches"]
                mean_score = result["mean_score"]
                improvement = (
                    ((mean_score - baseline_score) / baseline_score * 100)
                    if not np.isnan(mean_score)
                    else float("nan")
                )
                time_per_sample = result["time_per_sample"]
                speedup = (
                    baseline_time / time_per_sample
                    if time_per_sample > 0
                    else float("inf")
                )

                print(
                    f"{branches:<10} {mean_score:<12.4f} {improvement:<12.2f}% {time_per_sample:<10.2f} {speedup:<10.2f}x"
                )
            print()

        # Find best overall result
        best_result = max(
            self.results,
            key=lambda x: (
                x["mean_score"] if not np.isnan(x["mean_score"]) else float("-inf")
            ),
        )
        improvement = (
            (best_result["mean_score"] - baseline_score) / baseline_score * 100
        )

        print(f"BEST RESULT:")
        print(
            f"Method: {best_result['method']} (branches: {best_result['num_branches']})"
        )
        print(f"Score: {best_result['mean_score']:.4f}±{best_result['std_score']:.4f}")
        print(f"Improvement: {improvement:.2f}% over baseline")
        print(f"Time per sample: {best_result['time_per_sample']:.2f}s")
        print()


def create_sampler(method_config: Dict[str, Any]) -> Sampler:
    """Create a sampler with specific method configuration."""
    # Load base configuration
    config_path = "runner/config/inference.yaml"
    conf = OmegaConf.load(config_path)

    # Update method configuration
    conf.inference.samples.inference_method = method_config["method"]
    conf.inference.samples.method_config = method_config.get("config", {})

    # Set minimal sampling parameters for experiments
    conf.inference.samples.samples_per_length = 1
    conf.inference.samples.min_length = 50  # Small for testing
    conf.inference.samples.max_length = 50
    conf.inference.samples.length_step = 1

    return Sampler(conf)


def main():
    parser = argparse.ArgumentParser(description="Run inference scaling experiments")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
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
        default=0.05,
        help="Noise scale for SDE path exploration",
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0.2,
        help="Lambda parameter for divergence-free ODE",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs",
    )

    args = parser.parse_args()

    print("Starting Inference Scaling Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Noise scale (SDE): {args.noise_scale}")
    print(f"  Lambda div (ODE): {args.lambda_div}")
    print()

    # Create and run experiments
    runner = ExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

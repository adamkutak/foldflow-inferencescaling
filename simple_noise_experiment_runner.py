#!/usr/bin/env python3
"""
Simple noise experiment runner for testing SDE and divergence-free methods without branching.

This script tests whether adding noise during sampling (without branching) can improve
protein generation quality. It compares:
- Standard sampling (baseline)
- Simple SDE sampling (adds Gaussian noise at each step)
- Simple divergence-free sampling (adds divergence-free noise at each step)

Each method is tested with different noise levels to find optimal parameters.
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


class SimpleNoiseExperimentRunner:
    """Runner for simple noise experiments."""

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
        # Load multiple configuration files and merge them properly
        base_config_path = "runner/config/base.yaml"
        model_config_path = "runner/config/model/ff2.yaml"
        inference_config_path = "runner/config/inference.yaml"
        flow_matcher_config_path = "runner/config/flow_matcher/default.yaml"
        data_config_path = "runner/config/data/default.yaml"
        wandb_config_path = "runner/config/wandb/default.yaml"
        experiment_config_path = "runner/config/experiment/baseline.yaml"

        # Load each config
        base_conf = OmegaConf.load(base_config_path)
        model_conf = OmegaConf.load(model_config_path)
        inference_conf = OmegaConf.load(inference_config_path)
        flow_matcher_conf = OmegaConf.load(flow_matcher_config_path)
        data_conf = OmegaConf.load(data_config_path)
        wandb_conf = OmegaConf.load(wandb_config_path)
        experiment_conf = OmegaConf.load(experiment_config_path)

        # Create merged configuration
        self.base_conf = OmegaConf.create({})

        # Merge configurations in the right order
        self.base_conf = OmegaConf.merge(
            self.base_conf,
            base_conf,
            {"model": model_conf},
            {"flow_matcher": flow_matcher_conf},
            {"data": data_conf},
            {"wandb": wandb_conf},
            {"experiment": experiment_conf},
            inference_conf,
        )

        # Override with experiment parameters
        self.base_conf.inference.samples.samples_per_length = 1
        self.base_conf.inference.samples.min_length = self.args.sample_length
        self.base_conf.inference.samples.max_length = self.args.sample_length
        self.base_conf.inference.samples.length_step = 1

        # Override GPU if specified
        if self.args.gpu_id is not None:
            self.base_conf.inference.gpu_id = self.args.gpu_id
            self.logger.info(f"Using GPU {self.args.gpu_id}")

        # Create output directory for experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join("experiments", f"simple_noise_{timestamp}")
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
        noise_param = method_config.get("config", {}).get(
            "noise_scale", method_config.get("config", {}).get("lambda_div", 0)
        )
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_noise_{noise_param}"
        )

        return Sampler(conf)

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given method configuration."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        noise_param = config.get("noise_scale", config.get("lambda_div", 0))

        self.logger.info(
            f"Running experiment: {method_name} with noise parameter {noise_param}"
        )

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
                    if "score" in sample_result:
                        score = sample_result["score"]
                    else:
                        score_fn = sampler.inference_method.get_score_function(
                            self.args.scoring_function
                        )
                        score = score_fn(sample_output, self.args.sample_length)
                else:
                    sample_output = sample_result
                    score_fn = sampler.inference_method.get_score_function(
                        self.args.scoring_function
                    )
                    score = score_fn(sample_output, self.args.sample_length)

                scores.append(score)
                self.logger.info(f"    Score: {score:.4f}, Time: {sample_time:.2f}s")

                # Clear GPU memory after each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error in sample {sample_idx}: {e}")
                scores.append(float("nan"))

                # Clear GPU memory even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
            "noise_parameter": noise_param,
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
        """Run all experiments with different methods and noise levels."""
        experiments = []

        # 1. Baseline: Standard sampling
        experiments.append({"method": "standard", "config": {}})

        # 2. Simple SDE with different noise scales
        for noise_scale in self.args.noise_scales:
            experiments.append(
                {
                    "method": "sde_simple",
                    "config": {
                        "noise_scale": noise_scale,
                    },
                }
            )

        # 3. Simple divergence-free with different lambda values
        for lambda_div in self.args.lambda_divs:
            experiments.append(
                {
                    "method": "divergence_free_simple",
                    "config": {
                        "lambda_div": lambda_div,
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
                    "noise_parameter": result["noise_parameter"],
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
        print("SIMPLE NOISE EXPERIMENT RESULTS")
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
            if method_name == "sde_simple":
                param_name = "Noise Scale"
            else:
                param_name = "Lambda Div"

            print(
                f"{param_name:<12} {'Mean Score':<12} {'Improvement':<12} {'Time (s)':<10} {'Speedup':<10}"
            )
            print("-" * 60)

            for result in sorted(method_results, key=lambda x: x["noise_parameter"]):
                noise_param = result["noise_parameter"]
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
                    f"{noise_param:<12.3f} {mean_score:<12.4f} {improvement:<12.2f}% {time_per_sample:<10.2f} {speedup:<10.2f}x"
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
            f"Method: {best_result['method']} (noise parameter: {best_result['noise_parameter']})"
        )
        print(f"Score: {best_result['mean_score']:.4f}±{best_result['std_score']:.4f}")
        print(f"Improvement: {improvement:.2f}% over baseline")
        print(f"Time per sample: {best_result['time_per_sample']:.2f}s")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run simple noise experiments")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
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
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2],
        help="List of noise scales to test for SDE method (default: [0.01, 0.02, 0.05, 0.1, 0.2])",
    )

    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8, 1.0],
        help="List of lambda values to test for divergence-free method (default: [0.1, 0.2, 0.4, 0.8, 1.0])",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="GPU ID to use for inference (overrides config file)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs",
    )

    args = parser.parse_args()

    print("Starting Simple Noise Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  SDE noise scales: {args.noise_scales}")
    print(f"  Divergence-free lambdas: {args.lambda_divs}")
    print(f"  GPU ID: {args.gpu_id}")
    print()

    # Create and run experiments
    runner = SimpleNoiseExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

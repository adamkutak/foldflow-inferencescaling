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
        self.base_conf.inference.samples.samples_per_length = (
            1  # We'll control this manually
        )
        self.base_conf.inference.samples.min_length = self.args.sample_length
        self.base_conf.inference.samples.max_length = self.args.sample_length
        self.base_conf.inference.samples.length_step = 1

        # Override GPU if specified
        if self.args.gpu_id is not None:
            self.base_conf.inference.gpu_id = self.args.gpu_id
            self.logger.info(f"Using GPU {self.args.gpu_id}")

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

    def cleanup_sampler(self, sampler, method_name: str):
        """Explicit cleanup of sampler to prevent memory leaks."""
        try:
            self.logger.debug(f"Cleaning up sampler for {method_name}")

            # Move models to CPU and delete them explicitly
            if hasattr(sampler, "model") and sampler.model is not None:
                sampler.model = sampler.model.cpu()
                del sampler.model

            if (
                hasattr(sampler, "_folding_model")
                and sampler._folding_model is not None
            ):
                sampler._folding_model = sampler._folding_model.cpu()
                del sampler._folding_model

            # Clean up experiment object and its model
            if hasattr(sampler, "exp") and sampler.exp is not None:
                if hasattr(sampler.exp, "model") and sampler.exp.model is not None:
                    sampler.exp.model = sampler.exp.model.cpu()
                    del sampler.exp.model
                if hasattr(sampler.exp, "_model") and sampler.exp._model is not None:
                    sampler.exp._model = sampler.exp._model.cpu()
                    del sampler.exp._model
                del sampler.exp

            # Clean up flow matcher
            if hasattr(sampler, "flow_matcher"):
                del sampler.flow_matcher

            # Clean up inference method
            if hasattr(sampler, "inference_method"):
                del sampler.inference_method

            # Delete the sampler object itself
            del sampler

            # Force garbage collection
            import gc

            gc.collect()

            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as cleanup_error:
            self.logger.warning(f"Error during sampler cleanup: {cleanup_error}")

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given method configuration."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        branches = config.get("num_branches", 1)

        self.logger.info(f"Running experiment: {method_name} with {branches} branches")

        # Create sampler
        sampler = self.create_sampler(method_config)

        try:
            # Track timing and results
            start_time = time.time()
            scores = []
            tm_scores = []
            rmsd_scores = []

            for sample_idx in range(self.args.num_samples):
                self.logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

                try:
                    # Generate sample
                    sample_start = time.time()
                    sample_result = sampler.inference_method.sample(
                        self.args.sample_length
                    )
                    sample_time = time.time() - sample_start

                    # Extract sample
                    if isinstance(sample_result, dict) and "sample" in sample_result:
                        sample_output = sample_result["sample"]
                        # Use the score from the method if available (for selector optimization)
                        if "score" in sample_result:
                            score = sample_result["score"]
                        else:
                            # Evaluate with the selector function
                            score_fn = sampler.inference_method.get_score_function(
                                self.args.scoring_function
                            )
                            score = score_fn(sample_output, self.args.sample_length)
                    else:
                        sample_output = sample_result
                        # Evaluate with the selector function
                        score_fn = sampler.inference_method.get_score_function(
                            self.args.scoring_function
                        )
                        score = score_fn(sample_output, self.args.sample_length)

                    # Always calculate both TM-score and RMSD for comprehensive analysis
                    dual_scores = sampler.inference_method._dual_score_function(
                        sample_output, self.args.sample_length
                    )
                    tm_score = dual_scores["tm_score"]
                    rmsd_score = dual_scores["rmsd"]

                    scores.append(score)
                    tm_scores.append(tm_score)
                    rmsd_scores.append(rmsd_score)

                    self.logger.info(
                        f"    Selector Score: {score:.4f}, TM-Score: {tm_score:.4f}, RMSD: {rmsd_score:.4f}, Time: {sample_time:.2f}s"
                    )

                except Exception as e:
                    self.logger.error(f"Error in sample {sample_idx}: {e}")
                    scores.append(float("nan"))
                    tm_scores.append(float("nan"))
                    rmsd_scores.append(float("nan"))

            total_time = time.time() - start_time

            # Calculate statistics for selector score
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                max_score = np.max(valid_scores)
                min_score = np.min(valid_scores)
            else:
                mean_score = std_score = max_score = min_score = float("nan")

            # Calculate statistics for TM-scores
            valid_tm_scores = [s for s in tm_scores if not np.isnan(s)]
            if valid_tm_scores:
                mean_tm_score = np.mean(valid_tm_scores)
                std_tm_score = np.std(valid_tm_scores)
                max_tm_score = np.max(valid_tm_scores)
                min_tm_score = np.min(valid_tm_scores)
            else:
                mean_tm_score = std_tm_score = max_tm_score = min_tm_score = float(
                    "nan"
                )

            # Calculate statistics for RMSD scores
            valid_rmsd_scores = [s for s in rmsd_scores if not np.isnan(s)]
            if valid_rmsd_scores:
                mean_rmsd_score = np.mean(valid_rmsd_scores)
                std_rmsd_score = np.std(valid_rmsd_scores)
                max_rmsd_score = np.max(valid_rmsd_scores)
                min_rmsd_score = np.min(valid_rmsd_scores)

                # Calculate designability metrics (percentage of samples below RMSD thresholds)
                designability_2 = np.mean(np.array(valid_rmsd_scores) < 2.0) * 100
                designability_1_5 = np.mean(np.array(valid_rmsd_scores) < 1.5) * 100
                designability_1 = np.mean(np.array(valid_rmsd_scores) < 1.0) * 100
            else:
                mean_rmsd_score = std_rmsd_score = max_rmsd_score = min_rmsd_score = (
                    float("nan")
                )
                designability_2 = designability_1_5 = designability_1 = float("nan")

            result = {
                "method": method_name,
                "num_branches": branches,
                "config": config,
                "num_samples": len(valid_scores),
                # Selector score (for optimization)
                "mean_score": mean_score,
                "std_score": std_score,
                "max_score": max_score,
                "min_score": min_score,
                "scores": scores,
                "scoring_function": self.args.scoring_function,
                # TM-score metrics
                "mean_tm_score": mean_tm_score,
                "std_tm_score": std_tm_score,
                "max_tm_score": max_tm_score,
                "min_tm_score": min_tm_score,
                "tm_scores": tm_scores,
                # RMSD metrics
                "mean_rmsd_score": mean_rmsd_score,
                "std_rmsd_score": std_rmsd_score,
                "max_rmsd_score": max_rmsd_score,
                "min_rmsd_score": min_rmsd_score,
                "rmsd_scores": rmsd_scores,
                # Designability metrics
                "designability_2": designability_2,
                "designability_1_5": designability_1_5,
                "designability_1": designability_1,
                # Timing
                "total_time": total_time,
                "time_per_sample": total_time / self.args.num_samples,
            }

            self.logger.info(
                f"  Results: Selector={mean_score:.4f}±{std_score:.4f}, TM={mean_tm_score:.4f}±{std_tm_score:.4f}, RMSD={mean_rmsd_score:.4f}±{std_rmsd_score:.4f}, Time={total_time:.2f}s"
            )
            self.logger.info(
                f"  Designability: <2Å={designability_2:.1f}%, <1.5Å={designability_1_5:.1f}%, <1Å={designability_1:.1f}%"
            )

            return result

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def run_all_experiments(self):
        """Run all experiments with different methods and branch counts."""
        experiments = []

        # 1. Baseline: Standard sampling
        experiments.append({"method": "standard", "config": {}})

        # Random Search + Divergence-free ODE with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "random_search_divfree",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # SDE path exploration with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "sde_path_exploration",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "noise_scale": self.args.noise_scale,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # Divergence-free ODE with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "divergence_free_ode",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # Divergence-free Max (single sample with linear noise schedule and particle repulsion)
        experiments.append(
            {
                "method": "divergence_free_max",
                "config": {
                    "lambda_div": self.args.lambda_div,
                    "noise_schedule_end_factor": 0.7,
                    "particle_repulsion_factor": 0.02,
                },
            }
        )

        # Best-of-N with different N values
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "best_of_n",
                    "config": {
                        "num_branches": n_branches,
                        "selector": self.args.scoring_function,
                    },
                }
            )

        # Noise Search SDE with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "noise_search_sde",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": min(
                            2, n_branches // 2
                        ),  # Keep fewer candidates for noise search
                        "noise_scale": self.args.noise_scale,
                        "selector": self.args.scoring_function,
                        "num_rounds": self.args.noise_search_rounds,
                    },
                }
            )

        # Noise Search DivFree with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "noise_search_divfree",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": min(
                            2, n_branches // 2
                        ),  # Keep fewer candidates for noise search
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "num_rounds": self.args.noise_search_rounds,
                    },
                }
            )

        # Noise Search DivFree Max with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "noise_search_divfree_max",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": min(
                            2, n_branches // 2
                        ),  # Keep fewer candidates for noise search
                        "lambda_div": self.args.lambda_div,
                        "noise_schedule_end_factor": 0.7,
                        "particle_repulsion_factor": 0.02,
                        "selector": self.args.scoring_function,
                        "num_rounds": self.args.noise_search_rounds,
                    },
                }
            )

        # Run all experiments
        for i, exp_config in enumerate(experiments):
            self.logger.info(
                f"Starting experiment {i+1}/{len(experiments)}: {exp_config['method']}"
            )

            try:
                result = self.run_single_experiment(exp_config)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Failed experiment {exp_config}: {e}")

            # Force garbage collection between experiments
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                    "scoring_function": result["scoring_function"],
                    "num_samples": result["num_samples"],
                    # Selector score (used for optimization)
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                    "max_score": result["max_score"],
                    "min_score": result["min_score"],
                    # TM-score metrics
                    "mean_tm_score": result["mean_tm_score"],
                    "std_tm_score": result["std_tm_score"],
                    "max_tm_score": result["max_tm_score"],
                    "min_tm_score": result["min_tm_score"],
                    # RMSD metrics
                    "mean_rmsd_score": result["mean_rmsd_score"],
                    "std_rmsd_score": result["std_rmsd_score"],
                    "max_rmsd_score": result["max_rmsd_score"],
                    "min_rmsd_score": result["min_rmsd_score"],
                    # Designability metrics
                    "designability_2": result["designability_2"],
                    "designability_1_5": result["designability_1_5"],
                    "designability_1": result["designability_1"],
                    # Timing
                    "total_time": result["total_time"],
                    "time_per_sample": result["time_per_sample"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 100)
        print("INFERENCE SCALING EXPERIMENT RESULTS")
        print("=" * 100)

        # Find baseline (standard method)
        baseline = None
        for result in self.results:
            if result["method"] == "standard":
                baseline = result
                break

        if baseline is None:
            print("Warning: No baseline (standard) method found!")
            return

        baseline_selector_score = baseline["mean_score"]
        baseline_tm_score = baseline["mean_tm_score"]
        baseline_rmsd_score = baseline["mean_rmsd_score"]
        baseline_time = baseline["time_per_sample"]

        print(f"Baseline (Standard):")
        print(
            f"  Selector Score ({self.args.scoring_function}): {baseline_selector_score:.4f}±{baseline['std_score']:.4f}"
        )
        print(f"  TM-Score: {baseline_tm_score:.4f}±{baseline['std_tm_score']:.4f}")
        print(f"  RMSD: {baseline_rmsd_score:.4f}±{baseline['std_rmsd_score']:.4f}")
        print(
            f"  Designability: <2Å={baseline['designability_2']:.1f}%, <1.5Å={baseline['designability_1_5']:.1f}%, <1Å={baseline['designability_1']:.1f}%"
        )
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
                f"{'Branches':<9} {'TM-Score':<13} {'TM-Improv':<10} {'RMSD':<10} {'RMSD-Improv':<12} {'<2Å%':<8} {'<1.5Å%':<8} {'<1Å%':<8} {'Time(s)':<8} {'Speedup':<8}"
            )
            print("-" * 100)

            for result in sorted(method_results, key=lambda x: x["num_branches"]):
                branches = result["num_branches"]
                tm_score = result["mean_tm_score"]
                rmsd_score = result["mean_rmsd_score"]

                # Calculate improvements
                tm_improvement = (
                    ((tm_score - baseline_tm_score) / baseline_tm_score * 100)
                    if not np.isnan(tm_score) and baseline_tm_score > 0
                    else float("nan")
                )
                rmsd_improvement = (
                    ((baseline_rmsd_score - rmsd_score) / baseline_rmsd_score * 100)
                    if not np.isnan(rmsd_score) and baseline_rmsd_score > 0
                    else float("nan")
                )

                time_per_sample = result["time_per_sample"]
                speedup = (
                    baseline_time / time_per_sample
                    if time_per_sample > 0
                    else float("inf")
                )

                designability_2 = result["designability_2"]
                designability_1_5 = result["designability_1_5"]
                designability_1 = result["designability_1"]

                print(
                    f"{branches:<9} {tm_score:<13.4f} {tm_improvement:<10.2f}% {rmsd_score:<10.4f} {rmsd_improvement:<12.2f}% {designability_2:<8.1f} {designability_1_5:<8.1f} {designability_1:<8.1f} {time_per_sample:<8.2f} {speedup:<8.2f}x"
                )
            print()

        # Find best results for different metrics
        best_tm_result = max(
            self.results,
            key=lambda x: (
                x["mean_tm_score"]
                if not np.isnan(x["mean_tm_score"])
                else float("-inf")
            ),
        )

        best_rmsd_result = min(
            self.results,
            key=lambda x: (
                x["mean_rmsd_score"]
                if not np.isnan(x["mean_rmsd_score"])
                else float("inf")
            ),
        )

        best_designability_result = max(
            self.results,
            key=lambda x: (
                x["designability_1"]
                if not np.isnan(x["designability_1"])
                else float("-inf")
            ),
        )

        tm_improvement = (
            (best_tm_result["mean_tm_score"] - baseline_tm_score)
            / baseline_tm_score
            * 100
        )
        rmsd_improvement = (
            (baseline_rmsd_score - best_rmsd_result["mean_rmsd_score"])
            / baseline_rmsd_score
            * 100
        )

        print(f"BEST RESULTS:")
        print(
            f"Best TM-Score: {best_tm_result['method']} (branches: {best_tm_result['num_branches']})"
        )
        print(
            f"  TM-Score: {best_tm_result['mean_tm_score']:.4f}±{best_tm_result['std_tm_score']:.4f}"
        )
        print(f"  Improvement: {tm_improvement:.2f}% over baseline")
        print()

        print(
            f"Best RMSD: {best_rmsd_result['method']} (branches: {best_rmsd_result['num_branches']})"
        )
        print(
            f"  RMSD: {best_rmsd_result['mean_rmsd_score']:.4f}±{best_rmsd_result['std_rmsd_score']:.4f}"
        )
        print(f"  Improvement: {rmsd_improvement:.2f}% over baseline")
        print()

        print(
            f"Best Designability (<1Å): {best_designability_result['method']} (branches: {best_designability_result['num_branches']})"
        )
        print(
            f"  <1Å designability: {best_designability_result['designability_1']:.1f}%"
        )
        print(f"  Baseline <1Å: {baseline['designability_1']:.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run inference scaling experiments")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate per method",
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=100,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--scoring_function",
        type=str,
        default="tm_score",
        choices=["tm_score", "rmsd", "geometric", "tm_score_4seq", "dual_score"],
        help="Scoring function to use for method optimization (both scTM-score and scRMSD are always calculated for analysis)",
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
        help="Time interval between branches (0.0 = every timestep, 0.1 = every 0.1 time units)",
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

    parser.add_argument(
        "--branch_counts",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="List of branch counts to use for experiments (default: [2, 4, 8])",
    )

    parser.add_argument(
        "--noise_search_rounds",
        type=int,
        default=9,
        help="Number of rounds for noise search methods (default: 9)",
    )

    args = parser.parse_args()

    print("Starting Inference Scaling Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Noise scale (SDE): {args.noise_scale}")
    print(f"  Lambda div (ODE): {args.lambda_div}")
    print(f"  Branch interval: {args.branch_interval}")
    print(
        f"  GPU ID: {args.gpu_id if args.gpu_id is not None else 'from config (default: 1)'}"
    )
    print(f"  Branch counts: {args.branch_counts}")
    print(f"  Noise search rounds: {args.noise_search_rounds}")
    print()

    # Create and run experiments
    runner = ExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

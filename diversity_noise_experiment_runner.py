#!/usr/bin/env python3
"""
Diversity noise experiment runner for testing how noise injection affects sample diversity.

This script tests how different noise injection methods affect the diversity of generated samples
by starting from the same initial random noise and measuring pairwise distances between final samples.

For each method and noise level:
1. Generate a batch of samples starting from the same initial noise
2. Apply noise injection during sampling (SDE, divergence-free, etc.)
3. Compute pairwise distances between all final samples in the batch
4. Average distances across multiple batches for statistical significance
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
from typing import Dict, List, Any, Tuple
import torch
import copy
import tree

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runner.inference_methods import get_inference_method
from runner.inference import Sampler
from omegaconf import DictConfig, OmegaConf


class DiversityNoiseExperimentRunner:
    """Runner for diversity noise experiments."""

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
        self.experiment_dir = os.path.join(
            "experiments", f"diversity_noise_{timestamp}"
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
        noise_param = method_config.get("config", {}).get(
            "noise_scale", method_config.get("config", {}).get("lambda_div", 0)
        )
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_noise_{noise_param}"
        )

        return Sampler(conf)

    def generate_initial_features(self, sample_length: int, seed: int):
        """Generate initial features for sampling with a specific seed."""
        # Set random seed for reproducible initial conditions
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)
        aatype = torch.zeros(sample_length, dtype=torch.int32)
        chain_idx = torch.zeros_like(aatype)

        # Create a temporary sampler to access flow_matcher
        temp_config = {"method": "standard", "config": {}}
        temp_sampler = self.create_sampler(temp_config)

        try:
            # Initialize data with the seeded random state
            ref_sample = temp_sampler.flow_matcher.sample_ref(
                n_samples=sample_length,
                as_tensor_7=True,
            )
            res_idx = torch.arange(1, sample_length + 1)

            init_feats = {
                "res_mask": res_mask,
                "seq_idx": res_idx,
                "fixed_mask": fixed_mask,
                "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
                "sc_ca_t": np.zeros((sample_length, 3)),
                "aatype": aatype,
                "chain_idx": chain_idx,
                **ref_sample,
            }

            # Add batch dimension and move to GPU.
            init_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
            )
            init_feats = tree.map_structure(
                lambda x: x[None].to(temp_sampler.device), init_feats
            )

            return init_feats

        finally:
            # Clean up temporary sampler
            self.cleanup_sampler(temp_sampler, "temp")

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

    def compute_pairwise_distances(self, samples: List[Dict[str, Any]]) -> List[float]:
        """Compute pairwise RMSD distances between all samples in a batch."""
        from tools.analysis import metrics

        distances = []
        n_samples = len(samples)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Extract protein trajectories (final structures)
                prot_i = samples[i]["prot_traj"][-1]  # Final frame
                prot_j = samples[j]["prot_traj"][-1]  # Final frame

                # Compute RMSD between the two structures
                try:
                    # Use CA atoms for RMSD calculation
                    ca_i = prot_i[:, 1, :]  # CA atoms (index 1)
                    ca_j = prot_j[:, 1, :]  # CA atoms (index 1)

                    # Compute RMSD
                    rmsd = metrics.calc_aligned_rmsd(ca_i, ca_j)
                    distances.append(rmsd)

                except Exception as e:
                    self.logger.warning(
                        f"Error computing RMSD between samples {i} and {j}: {e}"
                    )

        return distances

    def run_single_batch(
        self, method_config: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Run a single batch of samples starting from the same initial noise."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        noise_param = config.get("noise_scale", config.get("lambda_div", 0))

        self.logger.info(
            f"Running batch {batch_idx + 1}: {method_name} with noise parameter {noise_param}, batch size {self.args.batch_size}"
        )

        # Generate initial features (same for all samples in this batch)
        batch_seed = batch_idx * 1000  # Different seed for each batch
        init_feats = self.generate_initial_features(self.args.sample_length, batch_seed)

        # Create sampler
        sampler = self.create_sampler(method_config)

        try:
            # Generate batch_size samples starting from the same initial features
            samples = []
            start_time = time.time()

            for sample_idx in range(self.args.batch_size):
                self.logger.info(f"  Sample {sample_idx + 1}/{self.args.batch_size}")

                try:
                    # Use the same initial features for all samples
                    sample_init_feats = copy.deepcopy(init_feats)

                    # Generate sample using the inference method
                    if hasattr(sampler.inference_method, "_simple_sde_inference"):
                        # For simple methods, use the internal method with initial features
                        sample_result = sampler.inference_method._simple_sde_inference(
                            sample_init_feats, config.get("noise_scale", 0.05), None
                        )
                    elif hasattr(
                        sampler.inference_method, "_simple_divergence_free_inference"
                    ):
                        # For divergence-free methods
                        sample_result = (
                            sampler.inference_method._simple_divergence_free_inference(
                                sample_init_feats, config.get("lambda_div", 0.2), None
                            )
                        )
                    else:
                        # For standard method, use base_sample but with our initial features
                        # We need to temporarily replace the sampler's _base_sample method
                        original_base_sample = sampler._base_sample

                        def custom_base_sample(sample_length, context=None):
                            # Use our pre-generated initial features
                            sample_out = sampler.exp.inference_fn(
                                sample_init_feats,
                                num_t=sampler._fm_conf.num_t,
                                min_t=sampler._fm_conf.min_t,
                                aux_traj=True,
                                noise_scale=sampler._fm_conf.noise_scale,
                                context=context,
                            )
                            return tree.map_structure(lambda x: x[:, 0], sample_out)

                        sampler._base_sample = custom_base_sample
                        sample_result = sampler.inference_method.sample(
                            self.args.sample_length
                        )
                        sampler._base_sample = original_base_sample

                    samples.append(sample_result)

                except Exception as e:
                    self.logger.error(f"Error generating sample {sample_idx}: {e}")

            total_time = time.time() - start_time

            # Compute pairwise distances
            if len(samples) >= 2:
                distances = self.compute_pairwise_distances(samples)
                mean_distance = np.mean(distances) if distances else 0.0
                std_distance = np.std(distances) if distances else 0.0
                n_pairs = len(distances)
            else:
                distances = []
                mean_distance = 0.0
                std_distance = 0.0
                n_pairs = 0

            result = {
                "method": method_name,
                "noise_parameter": noise_param,
                "config": config,
                "batch_idx": batch_idx,
                "batch_size": len(samples),
                "n_pairs": n_pairs,
                "mean_distance": mean_distance,
                "std_distance": std_distance,
                "all_distances": distances,
                "total_time": total_time,
                "time_per_sample": total_time / len(samples) if samples else 0.0,
            }

            self.logger.info(
                f"  Batch {batch_idx + 1} results: {len(samples)} samples, {n_pairs} pairs, "
                f"mean distance={mean_distance:.4f}±{std_distance:.4f}, time={total_time:.2f}s"
            )

            return result

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with multiple batches."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        noise_param = config.get("noise_scale", config.get("lambda_div", 0))

        self.logger.info(
            f"Running experiment: {method_name} with noise parameter {noise_param} "
            f"({self.args.num_batches} batches of {self.args.batch_size} samples each)"
        )

        # Run multiple batches
        batch_results = []
        for batch_idx in range(self.args.num_batches):
            try:
                batch_result = self.run_single_batch(method_config, batch_idx)
                batch_results.append(batch_result)
            except Exception as e:
                self.logger.error(f"Failed batch {batch_idx}: {e}")

        # Aggregate results across batches
        if batch_results:
            all_distances = []
            total_pairs = 0
            total_time = 0.0

            for batch in batch_results:
                all_distances.extend(batch["all_distances"])
                total_pairs += batch["n_pairs"]
                total_time += batch["total_time"]

            if all_distances:
                overall_mean = np.mean(all_distances)
                overall_std = np.std(all_distances)
            else:
                overall_mean = 0.0
                overall_std = 0.0
        else:
            all_distances = []
            total_pairs = 0
            total_time = 0.0
            overall_mean = 0.0
            overall_std = 0.0

        result = {
            "method": method_name,
            "noise_parameter": noise_param,
            "config": config,
            "num_batches": len(batch_results),
            "total_samples": (
                len(batch_results) * self.args.batch_size if batch_results else 0
            ),
            "total_pairs": total_pairs,
            "overall_mean_distance": overall_mean,
            "overall_std_distance": overall_std,
            "batch_results": batch_results,
            "total_time": total_time,
            "avg_time_per_batch": (
                total_time / len(batch_results) if batch_results else 0.0
            ),
        }

        self.logger.info(
            f"  Overall results: {len(batch_results)} batches, {total_pairs} total pairs, "
            f"overall mean distance={overall_mean:.4f}±{overall_std:.4f}"
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
                    "noise_parameter": result["noise_parameter"],
                    "overall_mean_distance": result["overall_mean_distance"],
                    "overall_std_distance": result["overall_std_distance"],
                    "num_batches": result["num_batches"],
                    "total_samples": result["total_samples"],
                    "total_pairs": result["total_pairs"],
                    "total_time": result["total_time"],
                    "avg_time_per_batch": result["avg_time_per_batch"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("DIVERSITY NOISE EXPERIMENT RESULTS")
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

        baseline_distance = baseline["overall_mean_distance"]

        print(
            f"Baseline (Standard): {baseline_distance:.4f}±{baseline['overall_std_distance']:.4f}"
        )
        print(f"Sample Length: {self.args.sample_length}")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Number of Batches: {self.args.num_batches}")
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
                f"{param_name:<12} {'Mean Distance':<15} {'Diversity Change':<15} {'Total Pairs':<12} {'Time (s)':<10}"
            )
            print("-" * 70)

            for result in sorted(method_results, key=lambda x: x["noise_parameter"]):
                noise_param = result["noise_parameter"]
                mean_distance = result["overall_mean_distance"]
                diversity_change = (
                    ((mean_distance - baseline_distance) / baseline_distance * 100)
                    if baseline_distance > 0 and not np.isnan(mean_distance)
                    else float("nan")
                )
                total_pairs = result["total_pairs"]
                total_time = result["total_time"]

                print(
                    f"{noise_param:<12.3f} {mean_distance:<15.4f} {diversity_change:<15.2f}% {total_pairs:<12} {total_time:<10.2f}"
                )
            print()

        # Find method with highest diversity
        best_result = max(
            self.results,
            key=lambda x: (
                x["overall_mean_distance"]
                if not np.isnan(x["overall_mean_distance"])
                else 0.0
            ),
        )
        diversity_increase = (
            (best_result["overall_mean_distance"] - baseline_distance)
            / baseline_distance
            * 100
        )

        print(f"HIGHEST DIVERSITY:")
        print(
            f"Method: {best_result['method']} (noise parameter: {best_result['noise_parameter']})"
        )
        print(
            f"Mean Distance: {best_result['overall_mean_distance']:.4f}±{best_result['overall_std_distance']:.4f}"
        )
        print(f"Diversity Increase: {diversity_increase:.2f}% over baseline")
        print(f"Total Time: {best_result['total_time']:.2f}s")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run diversity noise experiments")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples per batch (all starting from same initial noise)",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=8,
        help="Number of batches to average over",
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=50,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2],
        help="List of noise scales to test for SDE method",
    )

    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8],
        help="List of lambda values to test for divergence-free method",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="GPU ID to use for inference",
    )

    args = parser.parse_args()

    print("Starting Diversity Noise Experiments")
    print(f"Parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches: {args.num_batches}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  SDE noise scales: {args.noise_scales}")
    print(f"  Divergence-free lambdas: {args.lambda_divs}")
    print(f"  GPU ID: {args.gpu_id}")
    print()

    # Create and run experiments
    runner = DiversityNoiseExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Noise Quality Assessment Script

This script systematically tests how different levels of noise injection during inference
affect the quality of folded proteins. It uses the comprehensive quality metrics available
in the FoldFlow repository to assess structural quality degradation.

Usage:
    python noise_quality_assessment.py --sample_length 50 --num_samples 10 --gpu_id 0
"""

import argparse
import logging
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from tools.analysis import metrics
from foldflow.data import utils as du


class NoiseQualityAssessment:
    """Systematic assessment of how noise affects protein folding quality."""

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
            "experiments", f"noise_quality_assessment_{timestamp}"
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

    def generate_and_assess_sample(
        self, method_config: Dict[str, Any], sample_idx: int
    ) -> Dict[str, Any]:
        """Generate a single sample and assess its quality."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        noise_param = config.get("noise_scale", config.get("lambda_div", 0))

        self.logger.info(
            f"Generating sample {sample_idx + 1}: {method_name} with noise parameter {noise_param}"
        )

        # Create sampler
        sampler = self.create_sampler(method_config)

        try:
            start_time = time.time()

            # Generate sample
            sample_result = sampler.inference_method.sample(self.args.sample_length)

            generation_time = time.time() - start_time

            # Extract final structure for quality assessment
            prot_traj = sample_result["prot_traj"]
            final_structure = prot_traj[-1]  # Final frame

            # Remove batch dimension if present
            if final_structure.ndim == 4:  # (1, N_residues, 37, 3)
                final_structure = final_structure[0]  # (N_residues, 37, 3)

            # Save structure to PDB for analysis
            sample_dir = os.path.join(
                self.experiment_dir, f"{method_name}_noise_{noise_param}"
            )
            os.makedirs(sample_dir, exist_ok=True)
            pdb_path = os.path.join(sample_dir, f"sample_{sample_idx}.pdb")

            # Convert to PDB and save
            self.save_structure_to_pdb(final_structure, pdb_path)

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(
                final_structure, pdb_path, sample_result
            )

            result = {
                "method": method_name,
                "noise_parameter": noise_param,
                "config": config,
                "sample_idx": sample_idx,
                "generation_time": generation_time,
                "pdb_path": pdb_path,
                **quality_metrics,
            }

            self.logger.info(
                f"  Sample {sample_idx + 1} completed: "
                f"TM-score={quality_metrics.get('tm_score', 'N/A'):.4f}, "
                f"RMSD={quality_metrics.get('rmsd', 'N/A'):.4f}, "
                f"time={generation_time:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error generating sample {sample_idx}: {e}")
            return {
                "method": method_name,
                "noise_parameter": noise_param,
                "config": config,
                "sample_idx": sample_idx,
                "error": str(e),
            }

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def save_structure_to_pdb(self, structure: np.ndarray, pdb_path: str):
        """Save structure to PDB file for analysis."""
        try:
            # Create a simple PDB file with CA atoms
            from foldflow.data import residue_constants

            CA_IDX = residue_constants.atom_order["CA"]
            ca_positions = structure[:, CA_IDX, :]  # Extract CA positions

            with open(pdb_path, "w") as f:
                f.write("HEADER    GENERATED STRUCTURE\n")
                for i, pos in enumerate(ca_positions):
                    f.write(
                        f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00           C\n"
                    )
                f.write("END\n")

        except Exception as e:
            self.logger.warning(f"Could not save PDB file {pdb_path}: {e}")

    def calculate_quality_metrics(
        self, structure: np.ndarray, pdb_path: str, sample_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for a structure."""
        quality_metrics = {}

        try:
            # Basic geometric metrics
            from foldflow.data import residue_constants

            CA_IDX = residue_constants.atom_order["CA"]
            ca_positions = structure[:, CA_IDX, :]

            # CA-CA bond length analysis
            ca_ca_bond_dev, ca_ca_valid_percent = metrics.ca_ca_distance(ca_positions)
            quality_metrics["ca_ca_bond_dev"] = ca_ca_bond_dev
            quality_metrics["ca_ca_valid_percent"] = ca_ca_valid_percent

            # Steric clash analysis
            num_ca_steric_clashes, ca_steric_clash_percent = metrics.ca_ca_clashes(
                ca_positions
            )
            quality_metrics["num_ca_steric_clashes"] = num_ca_steric_clashes
            quality_metrics["ca_steric_clash_percent"] = ca_steric_clash_percent

            # Radius of gyration
            rg = np.sqrt(
                np.mean(
                    np.sum((ca_positions - np.mean(ca_positions, axis=0)) ** 2, axis=1)
                )
            )
            quality_metrics["radius_of_gyration"] = rg

            # Secondary structure analysis (if MDTraj is available)
            try:
                mdtraj_metrics = metrics.calc_mdtraj_metrics(pdb_path)
                quality_metrics.update(mdtraj_metrics)
            except Exception as e:
                self.logger.warning(f"Could not calculate MDTraj metrics: {e}")

            # For reference comparison, we would need ground truth structures
            # For now, we'll focus on absolute quality metrics

        except Exception as e:
            self.logger.warning(f"Error calculating quality metrics: {e}")

        return quality_metrics

    def run_noise_quality_experiment(self):
        """Run systematic noise quality assessment."""
        experiments = []

        # Baseline: Standard sampling (no noise)
        experiments.append({"method": "standard", "config": {}})

        # SDE with increasing noise levels
        for noise_scale in self.args.noise_scales:
            experiments.append(
                {
                    "method": "sde_simple",
                    "config": {"noise_scale": noise_scale},
                }
            )

        # Divergence-free with increasing lambda values
        for lambda_div in self.args.lambda_divs:
            experiments.append(
                {
                    "method": "divergence_free_simple",
                    "config": {"lambda_div": lambda_div},
                }
            )

        # Run all experiments
        for exp_config in experiments:
            method_name = exp_config["method"]
            config = exp_config.get("config", {})
            noise_param = config.get("noise_scale", config.get("lambda_div", 0))

            self.logger.info(
                f"Starting experiment: {method_name} with noise parameter {noise_param}"
            )

            # Generate multiple samples for statistical significance
            for sample_idx in range(self.args.num_samples):
                try:
                    result = self.generate_and_assess_sample(exp_config, sample_idx)
                    self.results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Failed sample {sample_idx} for {exp_config}: {e}"
                    )

            # Force garbage collection between experiments
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save and analyze results
        self.save_results()
        self.analyze_results()
        self.create_visualizations()

    def save_results(self):
        """Save experiment results to files."""
        # Save detailed results as JSON
        results_file = os.path.join(self.experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []
        for result in self.results:
            if "error" not in result:
                summary_data.append(result)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("NOISE QUALITY ASSESSMENT RESULTS")
        print("=" * 80)

        # Filter out failed results
        valid_results = [r for r in self.results if "error" not in r]

        if not valid_results:
            print("No valid results to analyze!")
            return

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(valid_results)

        print(f"Sample Length: {self.args.sample_length}")
        print(f"Number of Samples per Method: {self.args.num_samples}")
        print(f"Total Valid Results: {len(valid_results)}")
        print()

        # Group by method and noise parameter
        grouped = df.groupby(["method", "noise_parameter"])

        # Calculate statistics for each group
        print("QUALITY METRICS BY METHOD AND NOISE LEVEL:")
        print("-" * 60)
        print(
            f"{'Method':<20} {'Noise':<8} {'CA Bond Dev':<12} {'Valid %':<8} {'Clashes %':<10} {'Rg':<8}"
        )
        print("-" * 60)

        baseline_metrics = None
        for (method, noise_param), group in grouped:
            ca_bond_dev = group["ca_ca_bond_dev"].mean()
            ca_valid_pct = group["ca_ca_valid_percent"].mean() * 100
            clash_pct = group["ca_steric_clash_percent"].mean() * 100
            rg = group["radius_of_gyration"].mean()

            print(
                f"{method:<20} {noise_param:<8.3f} {ca_bond_dev:<12.4f} {ca_valid_pct:<8.1f} {clash_pct:<10.2f} {rg:<8.2f}"
            )

            # Store baseline for comparison
            if method == "standard" and noise_param == 0:
                baseline_metrics = {
                    "ca_bond_dev": ca_bond_dev,
                    "ca_valid_pct": ca_valid_pct,
                    "clash_pct": clash_pct,
                    "rg": rg,
                }

        print()

        # Quality degradation analysis
        if baseline_metrics:
            print("QUALITY DEGRADATION ANALYSIS:")
            print("-" * 40)
            print("Methods showing significant quality degradation:")
            print()

            degradation_threshold = {
                "ca_bond_dev": 0.05,  # Increase > 0.05 Å is concerning
                "ca_valid_pct": 5.0,  # Decrease > 5% is concerning
                "clash_pct": 2.0,  # Increase > 2% is concerning
            }

            for (method, noise_param), group in grouped:
                if method == "standard" and noise_param == 0:
                    continue

                ca_bond_dev = group["ca_ca_bond_dev"].mean()
                ca_valid_pct = group["ca_ca_valid_percent"].mean() * 100
                clash_pct = group["ca_steric_clash_percent"].mean() * 100

                # Check for degradation
                bond_degraded = (
                    ca_bond_dev - baseline_metrics["ca_bond_dev"]
                ) > degradation_threshold["ca_bond_dev"]
                valid_degraded = (
                    baseline_metrics["ca_valid_pct"] - ca_valid_pct
                ) > degradation_threshold["ca_valid_pct"]
                clash_degraded = (
                    clash_pct - baseline_metrics["clash_pct"]
                ) > degradation_threshold["clash_pct"]

                if bond_degraded or valid_degraded or clash_degraded:
                    print(f"{method} (noise={noise_param:.3f}):")
                    if bond_degraded:
                        print(
                            f"  ✗ CA bond deviation: {ca_bond_dev:.4f} vs {baseline_metrics['ca_bond_dev']:.4f} baseline"
                        )
                    if valid_degraded:
                        print(
                            f"  ✗ Valid CA bonds: {ca_valid_pct:.1f}% vs {baseline_metrics['ca_valid_pct']:.1f}% baseline"
                        )
                    if clash_degraded:
                        print(
                            f"  ✗ Steric clashes: {clash_pct:.2f}% vs {baseline_metrics['clash_pct']:.2f}% baseline"
                        )
                    print()

        # Find optimal noise levels
        print("RECOMMENDED NOISE LEVELS:")
        print("-" * 30)
        print("Methods that maintain quality while adding diversity:")
        print()

        for method in df["method"].unique():
            if method == "standard":
                continue

            method_data = df[df["method"] == method]
            if baseline_metrics:
                # Find highest noise level that doesn't significantly degrade quality
                acceptable_levels = []
                for noise_param in sorted(method_data["noise_parameter"].unique()):
                    group = method_data[method_data["noise_parameter"] == noise_param]
                    ca_bond_dev = group["ca_ca_bond_dev"].mean()
                    ca_valid_pct = group["ca_ca_valid_percent"].mean() * 100
                    clash_pct = group["ca_steric_clash_percent"].mean() * 100

                    bond_ok = (
                        ca_bond_dev - baseline_metrics["ca_bond_dev"]
                    ) <= degradation_threshold["ca_bond_dev"]
                    valid_ok = (
                        baseline_metrics["ca_valid_pct"] - ca_valid_pct
                    ) <= degradation_threshold["ca_valid_pct"]
                    clash_ok = (
                        clash_pct - baseline_metrics["clash_pct"]
                    ) <= degradation_threshold["clash_pct"]

                    if bond_ok and valid_ok and clash_ok:
                        acceptable_levels.append(noise_param)

                if acceptable_levels:
                    max_acceptable = max(acceptable_levels)
                    print(f"{method}: Up to {max_acceptable:.3f} noise parameter")
                else:
                    print(f"{method}: No acceptable noise levels found")

    def create_visualizations(self):
        """Create visualizations of the results."""
        try:
            # Filter out failed results
            valid_results = [r for r in self.results if "error" not in r]
            if not valid_results:
                return

            df = pd.DataFrame(valid_results)

            # Set up the plotting style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                f"Noise Quality Assessment (Length={self.args.sample_length})",
                fontsize=16,
            )

            # Plot 1: CA bond deviation vs noise
            ax1 = axes[0, 0]
            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                ax1.plot(
                    method_data["noise_parameter"],
                    method_data["ca_ca_bond_dev"],
                    "o-",
                    label=method,
                    alpha=0.7,
                )
            ax1.set_xlabel("Noise Parameter")
            ax1.set_ylabel("CA Bond Deviation (Å)")
            ax1.set_title("CA Bond Quality vs Noise")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Valid CA bonds percentage vs noise
            ax2 = axes[0, 1]
            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                ax2.plot(
                    method_data["noise_parameter"],
                    method_data["ca_ca_valid_percent"] * 100,
                    "o-",
                    label=method,
                    alpha=0.7,
                )
            ax2.set_xlabel("Noise Parameter")
            ax2.set_ylabel("Valid CA Bonds (%)")
            ax2.set_title("CA Bond Validity vs Noise")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Steric clashes vs noise
            ax3 = axes[1, 0]
            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                ax3.plot(
                    method_data["noise_parameter"],
                    method_data["ca_steric_clash_percent"] * 100,
                    "o-",
                    label=method,
                    alpha=0.7,
                )
            ax3.set_xlabel("Noise Parameter")
            ax3.set_ylabel("Steric Clashes (%)")
            ax3.set_title("Steric Clashes vs Noise")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Radius of gyration vs noise
            ax4 = axes[1, 1]
            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                ax4.plot(
                    method_data["noise_parameter"],
                    method_data["radius_of_gyration"],
                    "o-",
                    label=method,
                    alpha=0.7,
                )
            ax4.set_xlabel("Noise Parameter")
            ax4.set_ylabel("Radius of Gyration (Å)")
            ax4.set_title("Compactness vs Noise")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.experiment_dir, "quality_vs_noise.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Visualization saved to {plot_path}")

        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Assess protein quality vs noise levels"
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=50,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per noise level for statistical significance",
    )

    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        help="List of noise scales to test for SDE method",
    )

    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
        help="List of lambda values to test for divergence-free method",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=5,
        help="GPU ID to use for inference",
    )

    args = parser.parse_args()

    print("Starting Noise Quality Assessment")
    print(f"Parameters:")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  SDE noise scales: {args.noise_scales}")
    print(f"  Divergence-free lambdas: {args.lambda_divs}")
    print(f"  GPU ID: {args.gpu_id}")
    print()

    # Create and run assessment
    assessment = NoiseQualityAssessment(args)
    assessment.run_noise_quality_experiment()

    print("Quality assessment completed!")


if __name__ == "__main__":
    main()

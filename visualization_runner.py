#!/usr/bin/env python3
"""
Visualization-focused experiment runner for protein design inference scaling.

This script runs inference scaling experiments (1x, 2x, 4x, 8x branches) with just 2 proteins
per configuration, calculates self-consistency, and generates 3D protein structure plots
showing the FoldFlow generated protein (in color) vs the self-consistency refolded structure (in grey).
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
from typing import Dict, List, Any, Tuple, Optional
import torch
import shutil
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variable for geomstats backend
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from omegaconf import DictConfig, OmegaConf
from runner.inference import Sampler
from runner.inference_methods import get_inference_method
from tools.analysis import plotting, utils as au
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


class VisualizationRunner:
    """Streamlined runner for inference scaling experiments with 3D visualization."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.setup_logging()
        self.setup_config()

        # Apply PyTorch settings
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def setup_config(self):
        """Setup configuration for experiments."""
        # Load configuration files
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
        self.base_conf = OmegaConf.merge(
            {},
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
        self.base_conf.inference.gpu_id = self.args.gpu_id

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "experiments", f"visualization_scaling_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Create subdirectories for plots and structures
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.structures_dir = os.path.join(self.experiment_dir, "structures")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.structures_dir, exist_ok=True)

        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def create_sampler(self, method_config: Dict[str, Any]):
        """Create a sampler with specific method configuration."""
        conf = OmegaConf.create(self.base_conf)

        # Update method configuration
        conf.inference.samples.inference_method = method_config["method"]
        conf.inference.samples.method_config = method_config.get("config", {})

        # Set output directory for this experiment
        method_name = method_config["method"]
        branches = method_config.get("config", {}).get("num_branches", 1)
        conf.inference.output_dir = os.path.join(
            self.structures_dir, f"{method_name}_branches_{branches}"
        )

        return Sampler(conf)

    def extract_ca_coordinates(self, pdb_path: str) -> np.ndarray:
        """Extract C-alpha coordinates from a PDB file."""
        try:
            from foldflow.data import utils as du

            # Read PDB file and extract CA coordinates
            pdb_feats = du.parse_pdb_feats(pdb_path)
            ca_positions = pdb_feats["atom_positions"][:, 1, :]  # CA is index 1
            return ca_positions
        except Exception as e:
            self.logger.warning(
                f"Failed to extract CA coordinates from {pdb_path}: {e}"
            )
            return None

    def create_3d_protein_plot(
        self,
        original_coords: np.ndarray,
        sc_coords: np.ndarray,
        title: str,
        save_path: str,
    ):
        """Create a 3D plot comparing original and self-consistency structures."""

        fig = go.Figure()

        # Add original structure (colored)
        original_trace = plotting.create_scatter(
            original_coords,
            mode="lines+markers",
            marker_size=4,
            name="FoldFlow Generated",
            opacity=0.8,
            color="blue",
        )
        fig.add_trace(original_trace)

        # Add self-consistency structure (grey)
        sc_trace = plotting.create_scatter(
            sc_coords,
            mode="lines+markers",
            marker_size=4,
            name="Self-Consistency",
            opacity=0.6,
            color="grey",
        )
        fig.add_trace(sc_trace)

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode="cube",
            ),
            width=800,
            height=600,
            showlegend=True,
        )

        # Save plot
        fig.write_html(save_path)
        self.logger.info(f"Saved 3D plot: {save_path}")

        return fig

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with visualization."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        branches = config.get("num_branches", 1)

        self.logger.info(f"Running experiment: {method_name} with {branches} branches")

        # Create sampler
        sampler = self.create_sampler(method_config)

        try:
            start_time = time.time()
            sample_results = []

            for sample_idx in range(self.args.num_samples):
                self.logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

                try:
                    # Generate sample
                    sample_start = time.time()
                    sample_result = sampler.inference_method.sample(
                        self.args.sample_length
                    )
                    sample_time = time.time() - sample_start

                    # Extract sample output
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

                    # Save original structure
                    sample_dir = os.path.join(
                        sampler._output_dir, f"sample_{sample_idx}"
                    )
                    os.makedirs(sample_dir, exist_ok=True)

                    traj_paths = sampler.save_traj(
                        sample_output["prot_traj"],
                        sample_output["rigid_0_traj"],
                        np.ones(self.args.sample_length),
                        output_dir=sample_dir,
                    )
                    original_pdb_path = traj_paths["sample_path"]

                    # Run self-consistency
                    sc_start = time.time()
                    sc_results = sampler.run_self_consistency(
                        sample_dir, original_pdb_path, motif_mask=None
                    )
                    sc_time = time.time() - sc_start

                    # Get self-consistency metrics
                    mean_tm_score = sc_results["tm_score"].mean()
                    mean_rmsd = sc_results["rmsd"].mean()

                    # Find the best self-consistency structure for visualization
                    best_sc_idx = sc_results["tm_score"].idxmax()
                    sc_pdb_dir = os.path.join(sample_dir, "esmf")
                    sc_pdb_files = [
                        f for f in os.listdir(sc_pdb_dir) if f.endswith(".pdb")
                    ]

                    if sc_pdb_files and best_sc_idx < len(sc_pdb_files):
                        best_sc_pdb = os.path.join(
                            sc_pdb_dir, sc_pdb_files[best_sc_idx]
                        )

                        # Extract coordinates for visualization
                        original_coords = self.extract_ca_coordinates(original_pdb_path)
                        sc_coords = self.extract_ca_coordinates(best_sc_pdb)

                        if original_coords is not None and sc_coords is not None:
                            # Create 3D visualization
                            plot_title = f"{method_name} (branches={branches}) - Sample {sample_idx+1}"
                            plot_filename = (
                                f"{method_name}_b{branches}_s{sample_idx+1}.html"
                            )
                            plot_path = os.path.join(self.plots_dir, plot_filename)

                            self.create_3d_protein_plot(
                                original_coords, sc_coords, plot_title, plot_path
                            )

                            # Store sample result
                            sample_results.append(
                                {
                                    "sample_idx": sample_idx,
                                    "score": score,
                                    "tm_score": mean_tm_score,
                                    "rmsd": mean_rmsd,
                                    "sample_time": sample_time,
                                    "sc_time": sc_time,
                                    "original_pdb": original_pdb_path,
                                    "best_sc_pdb": best_sc_pdb,
                                    "plot_path": plot_path,
                                    "original_coords": original_coords.tolist(),
                                    "sc_coords": sc_coords.tolist(),
                                }
                            )

                            self.logger.info(
                                f"    Score: {score:.4f}, TM: {mean_tm_score:.4f}, "
                                f"RMSD: {mean_rmsd:.3f}Å, Time: {sample_time:.2f}s"
                            )
                        else:
                            self.logger.warning(
                                f"Failed to extract coordinates for sample {sample_idx}"
                            )
                    else:
                        self.logger.warning(
                            f"No self-consistency structures found for sample {sample_idx}"
                        )

                except Exception as e:
                    self.logger.error(f"Error in sample {sample_idx}: {e}")
                    continue

            total_time = time.time() - start_time

            # Calculate summary statistics
            if sample_results:
                scores = [r["score"] for r in sample_results]
                tm_scores = [r["tm_score"] for r in sample_results]
                rmsds = [r["rmsd"] for r in sample_results]

                result = {
                    "method": method_name,
                    "num_branches": branches,
                    "config": config,
                    "num_samples": len(sample_results),
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "mean_tm_score": np.mean(tm_scores),
                    "std_tm_score": np.std(tm_scores),
                    "mean_rmsd": np.mean(rmsds),
                    "std_rmsd": np.std(rmsds),
                    "total_time": total_time,
                    "time_per_sample": total_time / len(sample_results),
                    "sample_results": sample_results,
                    "scoring_function": self.args.scoring_function,
                }

                self.logger.info(
                    f"  Results: Score={result['mean_score']:.4f}±{result['std_score']:.4f}, "
                    f"TM={result['mean_tm_score']:.4f}±{result['std_tm_score']:.4f}, "
                    f"RMSD={result['mean_rmsd']:.3f}±{result['std_rmsd']:.3f}Å, "
                    f"Time={total_time:.2f}s"
                )

                return result
            else:
                self.logger.error(f"No successful samples for {method_name}")
                return None

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def cleanup_sampler(self, sampler, method_name: str):
        """Clean up sampler to prevent memory leaks."""
        try:
            if hasattr(sampler, "model") and sampler.model is not None:
                sampler.model = sampler.model.cpu()
                del sampler.model

            if hasattr(sampler, "exp") and sampler.exp is not None:
                if hasattr(sampler.exp, "model") and sampler.exp.model is not None:
                    sampler.exp.model = sampler.exp.model.cpu()
                    del sampler.exp.model
                del sampler.exp

            del sampler

            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def create_comparison_grid(self):
        """Create a grid plot comparing all methods and branch counts."""
        if not self.results:
            return

        # Create a grid of subplots
        methods = list(set(r["method"] for r in self.results))
        branch_counts = sorted(list(set(r["num_branches"] for r in self.results)))

        fig = make_subplots(
            rows=len(methods),
            cols=len(branch_counts),
            subplot_titles=[
                f"{method} - {branches} branches"
                for method in methods
                for branches in branch_counts
            ],
            specs=[[{"type": "scatter3d"} for _ in branch_counts] for _ in methods],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        for i, method in enumerate(methods):
            for j, branches in enumerate(branch_counts):
                # Find matching result
                matching_results = [
                    r
                    for r in self.results
                    if r["method"] == method and r["num_branches"] == branches
                ]

                if matching_results:
                    result = matching_results[0]
                    if "sample_results" in result and result["sample_results"]:
                        # Use the first sample for the grid
                        sample = result["sample_results"][0]

                        original_coords = np.array(sample["original_coords"])
                        sc_coords = np.array(sample["sc_coords"])

                        # Add original structure
                        fig.add_trace(
                            plotting.create_scatter(
                                original_coords,
                                mode="lines+markers",
                                marker_size=3,
                                name=f"{method}_orig",
                                opacity=0.8,
                                color="blue",
                            ),
                            row=i + 1,
                            col=j + 1,
                        )

                        # Add self-consistency structure
                        fig.add_trace(
                            plotting.create_scatter(
                                sc_coords,
                                mode="lines+markers",
                                marker_size=3,
                                name=f"{method}_sc",
                                opacity=0.6,
                                color="grey",
                            ),
                            row=i + 1,
                            col=j + 1,
                        )

        fig.update_layout(
            title="Inference Scaling Comparison Grid",
            height=300 * len(methods),
            width=300 * len(branch_counts),
            showlegend=False,
        )

        grid_path = os.path.join(self.plots_dir, "comparison_grid.html")
        fig.write_html(grid_path)
        self.logger.info(f"Saved comparison grid: {grid_path}")

    def run_all_experiments(self):
        """Run all experiments with different methods and branch counts."""
        experiments = []

        # Standard inference
        experiments.append({"method": "standard", "config": {}})

        # Best-of-N with different branch counts
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

        # Noise search methods (skip 1 branch as it's inefficient)
        for n_branches in self.args.branch_counts:
            if n_branches == 1:
                continue

            # Divergence-free max
            experiments.append(
                {
                    "method": "noise_search_divfree_max",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,
                        "num_rounds": self.args.num_rounds,
                        "lambda_div": self.args.lambda_div,
                        "particle_repulsion_factor": self.args.particle_repulsion_factor,
                        "noise_schedule_end_factor": self.args.noise_schedule_end_factor,
                        "selector": self.args.scoring_function,
                        "massage_steps": self.args.massage_steps,
                    },
                }
            )

        # Run experiments
        for exp_config in experiments:
            result = self.run_single_experiment(exp_config)
            if result:
                self.results.append(result)

        # Save results and create visualizations
        self.save_results()
        self.create_comparison_grid()
        self.analyze_results()

    def save_results(self):
        """Save experiment results."""
        # Save detailed results as JSON
        results_file = os.path.join(self.experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []
        for result in self.results:
            summary_row = {
                "method": result["method"],
                "num_branches": result["num_branches"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "mean_tm_score": result["mean_tm_score"],
                "std_tm_score": result["std_tm_score"],
                "mean_rmsd": result["mean_rmsd"],
                "std_rmsd": result["std_rmsd"],
                "total_time": result["total_time"],
                "time_per_sample": result["time_per_sample"],
                "num_samples": result["num_samples"],
                "scoring_function": result["scoring_function"],
            }
            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("VISUALIZATION INFERENCE SCALING EXPERIMENT RESULTS")
        print("=" * 80)

        # Find baseline
        baseline = None
        for result in self.results:
            if result["method"] == "standard":
                baseline = result
                break

        if baseline:
            print(
                f"Baseline (Standard): {baseline['mean_score']:.4f}±{baseline['std_score']:.4f}"
            )

        print(f"Scoring Function: {self.args.scoring_function}")
        print(f"Sample Length: {self.args.sample_length}")
        print(f"Samples per Method: {self.args.num_samples}")
        print(f"GPU: {self.args.gpu_id}")
        print(f"Plots Directory: {self.plots_dir}")
        print()

        # Print results table
        print(
            f"{'Method':<25} {'Branches':<8} {'Score':<12} {'TM Score':<12} {'RMSD (Å)':<10} {'Time (s)':<8}"
        )
        print("-" * 85)

        for result in sorted(
            self.results, key=lambda x: (x["method"], x["num_branches"])
        ):
            print(
                f"{result['method']:<25} {result['num_branches']:<8} "
                f"{result['mean_score']:<12.4f} {result['mean_tm_score']:<12.4f} "
                f"{result['mean_rmsd']:<10.3f} {result['time_per_sample']:<8.2f}"
            )

        print()
        print(f"3D visualizations saved in: {self.plots_dir}")
        print(
            f"Individual plots: {len([r for r in self.results for s in r.get('sample_results', [])])} files"
        )
        print(f"Comparison grid: comparison_grid.html")


def main():
    parser = argparse.ArgumentParser(
        description="Run visualization-focused inference scaling experiments"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to generate per method (default: 2)",
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
        help="Scoring function to use for evaluation",
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0.3,
        help="Lambda for divergence-free vector fields",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=9,
        help="Number of rounds for noise search methods",
    )

    parser.add_argument(
        "--particle_repulsion_factor",
        type=float,
        default=0,
        help="Particle repulsion factor for divergence-free max methods",
    )

    parser.add_argument(
        "--noise_schedule_end_factor",
        type=float,
        default=0.7,
        help="Noise schedule end factor for divergence-free max methods",
    )

    parser.add_argument(
        "--massage_steps",
        type=int,
        default=0,
        help="Number of massage steps for sample cleanup",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for experiments",
    )

    parser.add_argument(
        "--branch_counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of branch counts to use for experiments",
    )

    args = parser.parse_args()

    print("Starting Visualization Inference Scaling Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Lambda div: {args.lambda_div}")
    print(f"  Num rounds: {args.num_rounds}")
    print(f"  GPU ID: {args.gpu_id}")
    print(f"  Branch counts: {args.branch_counts}")
    print()

    # Create and run experiments
    runner = VisualizationRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()


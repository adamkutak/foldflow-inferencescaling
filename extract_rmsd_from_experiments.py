#!/usr/bin/env python3
"""
Extract RMSD values from previous multi-GPU experiment runs.

This script searches through experiment directories and extracts self-consistency
results (including RMSD values) from saved CSV files, then reconstructs the
comprehensive metrics that would have been calculated by the updated runner.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob
import logging
from datetime import datetime


class RMSDExtractor:
    """Extract RMSD values from previous experiment runs."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def find_experiment_directories(self) -> List[str]:
        """Find all experiment directories that match the multi-GPU pattern."""
        pattern = os.path.join(self.experiment_dir, "inference_scaling_multi_gpu_*")
        experiment_dirs = glob.glob(pattern)
        experiment_dirs.sort()

        self.logger.info(f"Found {len(experiment_dirs)} experiment directories")
        for exp_dir in experiment_dirs:
            self.logger.info(f"  - {os.path.basename(exp_dir)}")

        return experiment_dirs

    def extract_self_consistency_results(self, method_dir: str) -> List[Dict[str, Any]]:
        """Extract self-consistency results from a method directory."""
        sample_results = []

        # Look for length directories
        length_dirs = glob.glob(os.path.join(method_dir, "length_*"))

        for length_dir in length_dirs:
            length = int(os.path.basename(length_dir).split("_")[1])

            # Look for sample directories
            sample_dirs = glob.glob(os.path.join(length_dir, "sample_*"))

            for sample_dir in sample_dirs:
                sample_idx = int(os.path.basename(sample_dir).split("_")[1])

                # Look for self-consistency results
                sc_results_path = os.path.join(
                    sample_dir, "self_consistency", "sc_results.csv"
                )

                if os.path.exists(sc_results_path):
                    try:
                        sc_df = pd.read_csv(sc_results_path)

                        # Extract metrics
                        tm_scores = sc_df["tm_score"].values
                        rmsd_scores = sc_df["rmsd"].values

                        sample_result = {
                            "length": length,
                            "sample_idx": sample_idx,
                            "sc_tm_scores": tm_scores.tolist(),
                            "sc_rmsd_scores": rmsd_scores.tolist(),
                            "sc_mean_tm": tm_scores.mean(),
                            "sc_mean_rmsd": rmsd_scores.mean(),
                            "rmsd_lt_2A_percent": (rmsd_scores < 2.0).mean() * 100,
                            "rmsd_lt_1_5A_percent": (rmsd_scores < 1.5).mean() * 100,
                            "rmsd_lt_1A_percent": (rmsd_scores < 1.0).mean() * 100,
                            "num_sequences": len(rmsd_scores),
                        }

                        sample_results.append(sample_result)

                    except Exception as e:
                        self.logger.warning(f"Failed to read {sc_results_path}: {e}")
                else:
                    self.logger.debug(
                        f"No self-consistency results found in {sample_dir}"
                    )

        return sample_results

    def parse_method_info(self, method_dir_name: str) -> Dict[str, Any]:
        """Parse method information from directory name."""
        # Expected format: {method}_branches_{branches}_gpu_{gpu_id}
        parts = method_dir_name.split("_")

        # Find branches and gpu parts
        method_parts = []
        branches = 1
        gpu_id = 0

        i = 0
        while i < len(parts):
            if parts[i] == "branches" and i + 1 < len(parts):
                branches = int(parts[i + 1])
                i += 2
            elif parts[i] == "gpu" and i + 1 < len(parts):
                gpu_id = int(parts[i + 1])
                i += 2
            else:
                method_parts.append(parts[i])
                i += 1

        method = "_".join(method_parts)

        return {
            "method": method,
            "num_branches": branches,
            "gpu_id": gpu_id,
        }

    def calculate_method_statistics(
        self, sample_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate method-level statistics from sample results."""
        if not sample_results:
            return {}

        # Extract values for statistics
        tm_scores = [r["sc_mean_tm"] for r in sample_results]
        rmsd_scores = [r["sc_mean_rmsd"] for r in sample_results]
        rmsd_2a_percentages = [r["rmsd_lt_2A_percent"] for r in sample_results]
        rmsd_1_5a_percentages = [r["rmsd_lt_1_5A_percent"] for r in sample_results]
        rmsd_1a_percentages = [r["rmsd_lt_1A_percent"] for r in sample_results]

        # Filter out NaN values
        valid_tm = [x for x in tm_scores if not np.isnan(x)]
        valid_rmsd = [x for x in rmsd_scores if not np.isnan(x)]
        valid_2a = [x for x in rmsd_2a_percentages if not np.isnan(x)]
        valid_1_5a = [x for x in rmsd_1_5a_percentages if not np.isnan(x)]
        valid_1a = [x for x in rmsd_1a_percentages if not np.isnan(x)]

        stats = {
            "num_samples": len(sample_results),
            "num_valid_samples": len(valid_tm),
        }

        # TM-score statistics
        if valid_tm:
            stats.update(
                {
                    "mean_score": np.mean(valid_tm),
                    "std_score": np.std(valid_tm),
                    "max_score": np.max(valid_tm),
                    "min_score": np.min(valid_tm),
                }
            )

        # RMSD statistics
        if valid_rmsd:
            stats.update(
                {
                    "mean_rmsd": np.mean(valid_rmsd),
                    "std_rmsd": np.std(valid_rmsd),
                    "max_rmsd": np.max(valid_rmsd),
                    "min_rmsd": np.min(valid_rmsd),
                }
            )

        # Percentage statistics
        if valid_2a:
            stats.update(
                {
                    "mean_rmsd_lt_2A_percent": np.mean(valid_2a),
                    "std_rmsd_lt_2A_percent": np.std(valid_2a),
                }
            )

        if valid_1_5a:
            stats.update(
                {
                    "mean_rmsd_lt_1_5A_percent": np.mean(valid_1_5a),
                    "std_rmsd_lt_1_5A_percent": np.std(valid_1_5a),
                }
            )

        if valid_1a:
            stats.update(
                {
                    "mean_rmsd_lt_1A_percent": np.mean(valid_1a),
                    "std_rmsd_lt_1A_percent": np.std(valid_1a),
                }
            )

        return stats

    def process_experiment_directory(self, exp_dir: str) -> Dict[str, Any]:
        """Process a single experiment directory."""
        self.logger.info(f"Processing experiment: {os.path.basename(exp_dir)}")

        # Look for method directories
        method_dirs = [
            d
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")
        ]

        experiment_results = []

        for method_dir_name in method_dirs:
            method_dir_path = os.path.join(exp_dir, method_dir_name)

            # Parse method information
            method_info = self.parse_method_info(method_dir_name)
            self.logger.info(
                f"  Processing method: {method_info['method']} "
                f"(branches: {method_info['num_branches']}, gpu: {method_info['gpu_id']})"
            )

            # Extract self-consistency results
            sample_results = self.extract_self_consistency_results(method_dir_path)

            if sample_results:
                # Calculate method statistics
                method_stats = self.calculate_method_statistics(sample_results)

                # Combine method info with statistics
                result = {
                    **method_info,
                    **method_stats,
                    "sample_results": sample_results,
                }

                experiment_results.append(result)

                self.logger.info(
                    f"    Found {len(sample_results)} samples with self-consistency results"
                )
                if "mean_rmsd" in method_stats:
                    self.logger.info(
                        f"    RMSD: {method_stats['mean_rmsd']:.3f}±{method_stats['std_rmsd']:.3f}Å"
                    )
                    self.logger.info(
                        f"    <2Å: {method_stats['mean_rmsd_lt_2A_percent']:.1f}%"
                    )
            else:
                self.logger.warning(
                    f"    No self-consistency results found for {method_dir_name}"
                )

        return {
            "experiment_dir": exp_dir,
            "timestamp": os.path.basename(exp_dir),
            "results": experiment_results,
        }

    def save_extracted_results(
        self, all_experiments: List[Dict[str, Any]], output_dir: str
    ):
        """Save extracted results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as JSON
        detailed_file = os.path.join(output_dir, "extracted_detailed_results.json")
        with open(detailed_file, "w") as f:
            json.dump(all_experiments, f, indent=2, default=str)
        self.logger.info(f"Saved detailed results to {detailed_file}")

        # Create summary CSV
        summary_data = []
        for exp in all_experiments:
            exp_timestamp = exp["timestamp"]
            for result in exp["results"]:
                summary_row = {
                    "experiment": exp_timestamp,
                    "method": result["method"],
                    "num_branches": result["num_branches"],
                    "gpu_id": result["gpu_id"],
                    "num_samples": result.get("num_samples", 0),
                    "num_valid_samples": result.get("num_valid_samples", 0),
                }

                # Add TM-score metrics
                for key in ["mean_score", "std_score", "max_score", "min_score"]:
                    summary_row[key] = result.get(key, float("nan"))

                # Add RMSD metrics
                for key in ["mean_rmsd", "std_rmsd", "max_rmsd", "min_rmsd"]:
                    summary_row[key] = result.get(key, float("nan"))

                # Add percentage metrics
                for key in [
                    "mean_rmsd_lt_2A_percent",
                    "std_rmsd_lt_2A_percent",
                    "mean_rmsd_lt_1_5A_percent",
                    "std_rmsd_lt_1_5A_percent",
                    "mean_rmsd_lt_1A_percent",
                    "std_rmsd_lt_1A_percent",
                ]:
                    summary_row[key] = result.get(key, float("nan"))

                summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "extracted_summary_results.csv")
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Saved summary results to {summary_file}")

        return detailed_file, summary_file

    def display_summary(self, all_experiments: List[Dict[str, Any]]):
        """Display a summary of extracted results."""
        print("\n" + "=" * 80)
        print("EXTRACTED RMSD RESULTS SUMMARY")
        print("=" * 80)

        total_experiments = len(all_experiments)
        total_methods = sum(len(exp["results"]) for exp in all_experiments)
        total_samples = sum(
            sum(r.get("num_samples", 0) for r in exp["results"])
            for exp in all_experiments
        )

        print(f"Total experiments processed: {total_experiments}")
        print(f"Total method configurations: {total_methods}")
        print(f"Total samples with self-consistency results: {total_samples}")
        print()

        # Show results by experiment
        for exp in all_experiments:
            print(f"Experiment: {exp['timestamp']}")

            if not exp["results"]:
                print("  No results found")
                continue

            # Group by method
            methods = {}
            for result in exp["results"]:
                method = result["method"]
                if method not in methods:
                    methods[method] = []
                methods[method].append(result)

            for method_name, method_results in methods.items():
                print(f"  {method_name.upper()}:")

                # Check if we have RMSD data
                has_rmsd = any("mean_rmsd" in r for r in method_results)

                if has_rmsd:
                    print(
                        f"    {'Branches':<8} {'TM Score':<10} {'RMSD (Å)':<10} {'<2Å %':<8} {'Samples':<8}"
                    )
                    print("    " + "-" * 50)

                    for result in sorted(
                        method_results, key=lambda x: x["num_branches"]
                    ):
                        branches = result["num_branches"]
                        tm_score = result.get("mean_score", float("nan"))
                        rmsd = result.get("mean_rmsd", float("nan"))
                        rmsd_2a = result.get("mean_rmsd_lt_2A_percent", float("nan"))
                        num_samples = result.get("num_samples", 0)

                        print(
                            f"    {branches:<8} {tm_score:<10.4f} {rmsd:<10.3f} {rmsd_2a:<8.1f} {num_samples:<8}"
                        )
                else:
                    print(f"    {'Branches':<8} {'Samples':<8}")
                    print("    " + "-" * 20)

                    for result in sorted(
                        method_results, key=lambda x: x["num_branches"]
                    ):
                        branches = result["num_branches"]
                        num_samples = result.get("num_samples", 0)
                        print(f"    {branches:<8} {num_samples:<8}")

            print()

    def run_extraction(self, output_dir: Optional[str] = None) -> tuple:
        """Run the full extraction process."""
        # Find experiment directories
        experiment_dirs = self.find_experiment_directories()

        if not experiment_dirs:
            self.logger.error("No experiment directories found!")
            return None, None

        # Process each experiment
        all_experiments = []
        for exp_dir in experiment_dirs:
            exp_results = self.process_experiment_directory(exp_dir)
            all_experiments.append(exp_results)

        # Set default output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"extracted_rmsd_results_{timestamp}"

        # Save results
        detailed_file, summary_file = self.save_extracted_results(
            all_experiments, output_dir
        )

        # Display summary
        self.display_summary(all_experiments)

        return detailed_file, summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract RMSD values from previous multi-GPU experiment runs"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Base directory containing experiment subdirectories (default: experiments)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for extracted results (default: auto-generated)",
    )

    parser.add_argument(
        "--specific_experiment",
        type=str,
        default=None,
        help="Process only a specific experiment directory (provide full path)",
    )

    args = parser.parse_args()

    if args.specific_experiment:
        # Process only the specific experiment
        if not os.path.exists(args.specific_experiment):
            print(
                f"Error: Experiment directory {args.specific_experiment} does not exist"
            )
            return

        extractor = RMSDExtractor(os.path.dirname(args.specific_experiment))
        exp_results = extractor.process_experiment_directory(args.specific_experiment)
        all_experiments = [exp_results]

        # Set output directory
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = os.path.basename(args.specific_experiment)
            args.output_dir = f"extracted_rmsd_{exp_name}_{timestamp}"

        # Save and display results
        extractor.save_extracted_results(all_experiments, args.output_dir)
        extractor.display_summary(all_experiments)

    else:
        # Process all experiments in the base directory
        extractor = RMSDExtractor(args.experiment_dir)
        detailed_file, summary_file = extractor.run_extraction(args.output_dir)

        if detailed_file and summary_file:
            print(f"\nExtraction completed successfully!")
            print(f"Detailed results: {detailed_file}")
            print(f"Summary results: {summary_file}")
        else:
            print("No results extracted.")


if __name__ == "__main__":
    main()

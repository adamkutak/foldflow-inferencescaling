#!/usr/bin/env python3
"""
Simple example script demonstrating protein quality assessment with noise injection.

This script shows how to:
1. Generate proteins with different noise levels
2. Assess their quality using built-in metrics
3. Compare results to identify quality degradation thresholds

Usage:
    python example_quality_test.py
"""

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runner.inference import Sampler
from tools.analysis import metrics
from foldflow.data import residue_constants


def create_sampler_config(method="standard", noise_param=0.0):
    """Create a basic sampler configuration."""
    # Load base configurations
    base_conf = OmegaConf.load("runner/config/base.yaml")
    model_conf = OmegaConf.load("runner/config/model/ff2.yaml")
    inference_conf = OmegaConf.load("runner/config/inference.yaml")
    flow_matcher_conf = OmegaConf.load("runner/config/flow_matcher/default.yaml")
    data_conf = OmegaConf.load("runner/config/data/default.yaml")
    wandb_conf = OmegaConf.load("runner/config/wandb/default.yaml")
    experiment_conf = OmegaConf.load("runner/config/experiment/baseline.yaml")

    # Merge configurations
    conf = OmegaConf.merge(
        base_conf,
        {"model": model_conf},
        {"flow_matcher": flow_matcher_conf},
        {"data": data_conf},
        {"wandb": wandb_conf},
        {"experiment": experiment_conf},
        inference_conf,
    )

    # Set sampling parameters
    conf.inference.samples.samples_per_length = 1
    conf.inference.samples.min_length = 100
    conf.inference.samples.max_length = 100
    conf.inference.samples.length_step = 1
    conf.inference.output_dir = "example_output"

    # Set inference method
    if method == "sde_simple":
        conf.inference.samples.inference_method = "sde_simple"
        conf.inference.samples.method_config = {"noise_scale": noise_param}
    elif method == "divergence_free_simple":
        conf.inference.samples.inference_method = "divergence_free_simple"
        conf.inference.samples.method_config = {"lambda_div": noise_param}
    else:
        conf.inference.samples.inference_method = "standard"
        conf.inference.samples.method_config = {}

    return conf


def calculate_basic_quality_metrics(structure):
    """Calculate basic quality metrics for a protein structure."""
    CA_IDX = residue_constants.atom_order["CA"]

    # Remove batch dimension if present
    if structure.ndim == 4:
        structure = structure[0]

    # Extract CA positions
    ca_positions = structure[:, CA_IDX, :]

    # Calculate metrics
    ca_bond_dev, ca_valid_percent = metrics.ca_ca_distance(ca_positions)
    num_clashes, clash_percent = metrics.ca_ca_clashes(ca_positions)

    # Radius of gyration
    center = np.mean(ca_positions, axis=0)
    rg = np.sqrt(np.mean(np.sum((ca_positions - center) ** 2, axis=1)))

    return {
        "ca_bond_deviation": ca_bond_dev,
        "ca_valid_percent": ca_valid_percent * 100,
        "clash_percent": clash_percent * 100,
        "num_clashes": num_clashes,
        "radius_of_gyration": rg,
    }


def run_quality_comparison():
    """Run a simple quality comparison with different noise levels."""
    print("=" * 60)
    print("PROTEIN QUALITY ASSESSMENT EXAMPLE")
    print("=" * 60)
    print()

    # Test configurations
    test_configs = [
        ("Standard (No Noise)", "standard", 0.0),
        ("SDE Low Noise", "sde_simple", 0.2),
        ("SDE Medium Noise", "sde_simple", 0.5),
        ("SDE High Noise", "sde_simple", 1.0),
        ("DivFree Low", "divergence_free_simple", 0.2),
        ("DivFree Medium", "divergence_free_simple", 0.5),
        ("DivFree High", "divergence_free_simple", 1.0),
    ]

    results = []

    for name, method, noise_param in test_configs:
        print(f"Testing: {name}")

        try:
            # Create sampler
            conf = create_sampler_config(method, noise_param)
            sampler = Sampler(conf)

            # Generate sample
            sample_result = sampler.inference_method.sample(20)

            # Extract final structure
            final_structure = sample_result["prot_traj"][-1]

            # Calculate quality metrics
            quality_metrics = calculate_basic_quality_metrics(final_structure)
            quality_metrics["method"] = name
            quality_metrics["noise_param"] = noise_param

            results.append(quality_metrics)

            print(f"  ✓ Generated successfully")
            print(f"    CA bond dev: {quality_metrics['ca_bond_deviation']:.4f} Å")
            print(f"    Valid bonds: {quality_metrics['ca_valid_percent']:.1f}%")
            print(f"    Clashes: {quality_metrics['clash_percent']:.2f}%")
            print(
                f"    Radius of gyration: {quality_metrics['radius_of_gyration']:.2f} Å"
            )
            print()

            # Cleanup
            del sampler
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()

    # Analysis
    print("=" * 60)
    print("QUALITY ANALYSIS")
    print("=" * 60)

    if not results:
        print("No successful results to analyze!")
        return

    # Find baseline (standard method)
    baseline = next((r for r in results if "Standard" in r["method"]), None)

    if baseline:
        print("Quality degradation analysis (compared to standard):")
        print()
        print(
            f"{'Method':<20} {'Bond Dev Δ':<12} {'Valid % Δ':<10} {'Clash % Δ':<10} {'Status':<10}"
        )
        print("-" * 65)

        for result in results:
            if result["method"] == baseline["method"]:
                continue

            bond_delta = result["ca_bond_deviation"] - baseline["ca_bond_deviation"]
            valid_delta = result["ca_valid_percent"] - baseline["ca_valid_percent"]
            clash_delta = result["clash_percent"] - baseline["clash_percent"]

            # Quality assessment
            bond_ok = bond_delta < 0.05  # Less than 0.05 Å increase
            valid_ok = valid_delta > -5.0  # Less than 5% decrease
            clash_ok = clash_delta < 2.0  # Less than 2% increase

            status = "✓ Good" if (bond_ok and valid_ok and clash_ok) else "✗ Degraded"

            print(
                f"{result['method']:<20} {bond_delta:+8.4f}    {valid_delta:+7.1f}    {clash_delta:+7.2f}    {status}"
            )

    print()
    print("Recommendations:")
    print("- Bond deviation increase < 0.05 Å: Acceptable")
    print("- Valid bonds decrease < 5%: Acceptable")
    print("- Clash increase < 2%: Acceptable")
    print()

    # Find best noise levels
    acceptable_methods = []
    if baseline:
        for result in results:
            if result["method"] == baseline["method"]:
                continue

            bond_delta = result["ca_bond_deviation"] - baseline["ca_bond_deviation"]
            valid_delta = result["ca_valid_percent"] - baseline["ca_valid_percent"]
            clash_delta = result["clash_percent"] - baseline["clash_percent"]

            if bond_delta < 0.05 and valid_delta > -5.0 and clash_delta < 2.0:
                acceptable_methods.append((result["method"], result["noise_param"]))

    if acceptable_methods:
        print("Recommended noise levels that maintain quality:")
        for method, noise_param in acceptable_methods:
            print(f"  - {method}: {noise_param}")
    else:
        print("No noise levels found that maintain acceptable quality.")
        print("Consider using lower noise parameters.")


if __name__ == "__main__":
    print("Starting protein quality assessment example...")
    print(
        "This will generate proteins with different noise levels and compare quality."
    )
    print()

    # Check if we have the required files
    required_files = [
        "runner/config/base.yaml",
        "runner/config/model/ff2.yaml",
        "runner/config/inference.yaml",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required configuration files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run this script from the FoldFlow root directory.")
        sys.exit(1)

    try:
        run_quality_comparison()
        print("\nExample completed successfully!")
        print("\nFor more comprehensive analysis, use:")
        print("  python noise_quality_assessment.py --sample_length 50 --num_samples 5")

    except Exception as e:
        print(f"\nError running example: {e}")

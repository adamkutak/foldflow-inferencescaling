#!/usr/bin/env python3
"""
Debug script to investigate quality assessment issues.
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


def create_sampler_config():
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
    conf.inference.output_dir = "debug_output"
    conf.inference.samples.inference_method = "standard"
    conf.inference.samples.method_config = {}

    return conf


def debug_structure_analysis(structure):
    """Debug the structure and quality metrics calculation."""
    print("=== STRUCTURE DEBUG ===")
    print(f"Structure shape: {structure.shape}")
    print(f"Structure dtype: {structure.dtype}")
    print(
        f"Structure device: {structure.device if hasattr(structure, 'device') else 'numpy'}"
    )

    # Convert to numpy if needed
    if torch.is_tensor(structure):
        structure_np = structure.detach().cpu().numpy()
    else:
        structure_np = structure

    # Remove batch dimension if present
    if structure_np.ndim == 4:
        print("Removing batch dimension")
        structure_np = structure_np[0]

    print(f"Final structure shape: {structure_np.shape}")
    print(f"Structure min/max: {structure_np.min():.4f} / {structure_np.max():.4f}")

    # Check for NaN or inf values
    has_nan = np.isnan(structure_np).any()
    has_inf = np.isinf(structure_np).any()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    if has_nan or has_inf:
        print("ERROR: Structure contains NaN or Inf values!")
        return None

    # Extract CA positions
    CA_IDX = residue_constants.atom_order["CA"]
    print(f"CA index: {CA_IDX}")

    ca_positions = structure_np[:, CA_IDX, :]
    print(f"CA positions shape: {ca_positions.shape}")
    print(f"CA positions min/max: {ca_positions.min():.4f} / {ca_positions.max():.4f}")

    # Check first few CA positions
    print("First 5 CA positions:")
    for i in range(min(5, len(ca_positions))):
        print(
            f"  CA {i}: [{ca_positions[i, 0]:.3f}, {ca_positions[i, 1]:.3f}, {ca_positions[i, 2]:.3f}]"
        )

    # Calculate consecutive CA-CA distances
    ca_dists = []
    for i in range(1, len(ca_positions)):
        dist = np.linalg.norm(ca_positions[i] - ca_positions[i - 1])
        ca_dists.append(dist)

    ca_dists = np.array(ca_dists)
    print(f"\nCA-CA distances:")
    print(f"  Number of bonds: {len(ca_dists)}")
    print(f"  Mean distance: {ca_dists.mean():.4f} Å")
    print(f"  Std distance: {ca_dists.std():.4f} Å")
    print(f"  Min distance: {ca_dists.min():.4f} Å")
    print(f"  Max distance: {ca_dists.max():.4f} Å")
    print(f"  Expected CA-CA distance: {residue_constants.ca_ca:.4f} Å")

    # Show distribution of distances
    print(f"  Distances < 2.0 Å: {np.sum(ca_dists < 2.0)}")
    print(f"  Distances 2.0-5.0 Å: {np.sum((ca_dists >= 2.0) & (ca_dists <= 5.0))}")
    print(f"  Distances > 5.0 Å: {np.sum(ca_dists > 5.0)}")

    # Calculate quality metrics step by step
    print("\n=== QUALITY METRICS DEBUG ===")

    # CA-CA bond analysis
    expected_ca_ca = residue_constants.ca_ca  # Should be ~3.8 Å
    tolerance = 0.1

    deviations = np.abs(ca_dists - expected_ca_ca)
    mean_deviation = np.mean(deviations)

    valid_bonds = ca_dists < (expected_ca_ca + tolerance)
    valid_percentage = np.mean(valid_bonds)

    print(f"CA-CA bond analysis:")
    print(f"  Expected distance: {expected_ca_ca:.4f} Å")
    print(f"  Tolerance: {tolerance:.4f} Å")
    print(f"  Mean deviation: {mean_deviation:.4f} Å")
    print(
        f"  Valid bonds: {np.sum(valid_bonds)}/{len(ca_dists)} ({valid_percentage*100:.1f}%)"
    )

    # Show which bonds are invalid
    invalid_indices = np.where(~valid_bonds)[0]
    if len(invalid_indices) > 0:
        print(f"  Invalid bonds (indices): {invalid_indices[:10]}...")  # Show first 10
        for idx in invalid_indices[:5]:  # Show details for first 5
            print(
                f"    Bond {idx}-{idx+1}: {ca_dists[idx]:.4f} Å (deviation: {deviations[idx]:.4f} Å)"
            )

    # Steric clash analysis
    print(f"\nSteric clash analysis:")
    clash_tolerance = 1.5  # Å

    # Calculate all pairwise distances
    n_residues = len(ca_positions)
    clash_count = 0
    total_pairs = 0

    for i in range(n_residues):
        for j in range(i + 2, n_residues):  # Skip adjacent residues
            dist = np.linalg.norm(ca_positions[i] - ca_positions[j])
            total_pairs += 1
            if dist < clash_tolerance:
                clash_count += 1
                if clash_count <= 5:  # Show first 5 clashes
                    print(f"    Clash: residues {i}-{j}, distance: {dist:.4f} Å")

    clash_percentage = clash_count / total_pairs if total_pairs > 0 else 0
    print(f"  Total pairs checked: {total_pairs}")
    print(f"  Clashes found: {clash_count}")
    print(f"  Clash percentage: {clash_percentage*100:.2f}%")

    # Radius of gyration
    center = np.mean(ca_positions, axis=0)
    distances_from_center = np.linalg.norm(ca_positions - center, axis=1)
    rg = np.sqrt(np.mean(distances_from_center**2))

    print(f"\nRadius of gyration:")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  RG: {rg:.4f} Å")
    print(f"  Max distance from center: {distances_from_center.max():.4f} Å")

    return {
        "ca_bond_deviation": mean_deviation,
        "ca_valid_percent": valid_percentage * 100,
        "clash_count": clash_count,
        "clash_percent": clash_percentage * 100,
        "radius_of_gyration": rg,
        "structure_valid": not (has_nan or has_inf),
    }


def main():
    print("=== DEBUG QUALITY ASSESSMENT ===")
    print()

    try:
        # Create sampler
        print("Creating sampler...")
        conf = create_sampler_config()
        sampler = Sampler(conf)

        # Generate sample
        print("Generating sample...")
        sample_result = sampler.inference_method.sample(100)

        print("Sample generation complete!")
        print(f"Sample result keys: {list(sample_result.keys())}")

        # Extract final structure
        if "prot_traj" in sample_result:
            prot_traj = sample_result["prot_traj"]
            print(f"Protein trajectory length: {len(prot_traj)}")
            print(
                f"Trajectory shape: {[frame.shape for frame in prot_traj[:3]]}"
            )  # First 3 frames

            final_structure = prot_traj[-1]
            print(f"Final structure shape: {final_structure.shape}")

            # Debug the structure
            quality_metrics = debug_structure_analysis(final_structure)

            if quality_metrics and quality_metrics["structure_valid"]:
                print("\n=== FINAL QUALITY SUMMARY ===")
                print(
                    f"CA bond deviation: {quality_metrics['ca_bond_deviation']:.4f} Å"
                )
                print(f"Valid bonds: {quality_metrics['ca_valid_percent']:.1f}%")
                print(f"Clashes: {quality_metrics['clash_percent']:.2f}%")
                print(
                    f"Radius of gyration: {quality_metrics['radius_of_gyration']:.2f} Å"
                )

                # Provide interpretation
                print("\n=== INTERPRETATION ===")
                if quality_metrics["ca_bond_deviation"] < 0.5:
                    print("✓ CA bond deviation looks reasonable")
                else:
                    print(
                        "✗ CA bond deviation is very high - structure may be malformed"
                    )

                if quality_metrics["ca_valid_percent"] > 80:
                    print("✓ Most CA bonds are within expected range")
                else:
                    print("✗ Many CA bonds are outside expected range")

                if quality_metrics["clash_percent"] < 5:
                    print("✓ Clash rate is acceptable")
                else:
                    print("✗ High clash rate - structure may have packing issues")
            else:
                print("✗ Structure contains invalid values (NaN/Inf)")
        else:
            print("ERROR: No 'prot_traj' found in sample result")
            print(f"Available keys: {list(sample_result.keys())}")

        # Cleanup
        del sampler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

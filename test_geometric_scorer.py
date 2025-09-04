#!/usr/bin/env python3
"""
Test script to generate samples using the geometric scorer with detailed diagnostics.
"""

import os
import sys
import logging
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runner.inference import Sampler


def setup_logging():
    """Setup detailed logging to see geometric scorer diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_sampler_config():
    """Create a sampler configuration that uses geometric scorer."""
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

    # Set sampling parameters for testing
    conf.inference.samples.samples_per_length = 1
    conf.inference.samples.min_length = 20
    conf.inference.samples.max_length = 20
    conf.inference.samples.length_step = 1
    conf.inference.output_dir = "geometric_scorer_test"

    # Use a method that will trigger the geometric scorer
    conf.inference.samples.inference_method = "best_of_n"
    conf.inference.samples.method_config = {
        "n_samples": 3,  # Generate 3 samples
        "selector": "geometric_score",  # Use geometric scorer
    }

    return conf


def main():
    print("=== TESTING GEOMETRIC SCORER WITH DIAGNOSTICS ===")
    print()

    # Setup logging to see detailed diagnostics
    setup_logging()

    try:
        # Create sampler
        print("Creating sampler with geometric scorer...")
        conf = create_sampler_config()
        sampler = Sampler(conf)

        print(f"Using inference method: {conf.inference.samples.inference_method}")
        print(f"Method config: {conf.inference.samples.method_config}")
        print()

        # Generate sample - this will trigger the geometric scorer multiple times
        print("Generating samples (this will show geometric scorer diagnostics)...")
        print("=" * 80)

        sample_result = sampler.inference_method.sample(20)

        print("=" * 80)
        print("Sample generation complete!")
        print()

        # Show final result
        if isinstance(sample_result, dict):
            print(f"Sample result keys: {list(sample_result.keys())}")
            if "score" in sample_result:
                print(f"Final geometric score: {sample_result['score']:.4f}")
            if "method" in sample_result:
                print(f"Method used: {sample_result['method']}")

        print(
            "\nLook at the detailed diagnostics above to understand structure quality!"
        )

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

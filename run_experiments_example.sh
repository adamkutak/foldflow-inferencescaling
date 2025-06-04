#!/bin/bash

# Example commands for running inference scaling experiments

echo "=== Inference Scaling Experiment Examples ==="
echo

# Quick test with minimal parameters
echo "1. Quick test (2 samples, length 30):"
echo "python experiment_runner.py --num_samples 2 --sample_length 30"
echo

# Standard experiment
echo "2. Standard experiment (10 samples, length 50):"
echo "python experiment_runner.py --num_samples 10 --sample_length 50"
echo

# Comprehensive experiment with custom parameters
echo "3. Comprehensive experiment with custom parameters:"
echo "python experiment_runner.py \\"
echo "    --num_samples 20 \\"
echo "    --sample_length 100 \\"
echo "    --scoring_function tm_score \\"
echo "    --noise_scale 0.1 \\"
echo "    --lambda_div 0.3"
echo

# RMSD-based experiment
echo "4. RMSD-based experiment:"
echo "python experiment_runner.py \\"
echo "    --num_samples 15 \\"
echo "    --sample_length 75 \\"
echo "    --scoring_function rmsd"
echo

# High-quality experiment (more samples, longer proteins)
echo "5. High-quality experiment (warning: computationally expensive):"
echo "python experiment_runner.py \\"
echo "    --num_samples 50 \\"
echo "    --sample_length 150 \\"
echo "    --scoring_function tm_score \\"
echo "    --noise_scale 0.05 \\"
echo "    --lambda_div 0.2"
echo

echo "=== Parameter Guidelines ==="
echo "- num_samples: 2-5 for testing, 10-20 for experiments, 50+ for publication"
echo "- sample_length: 30-50 for testing, 50-100 for experiments, 100-200 for realistic proteins"
echo "- noise_scale: 0.01-0.1 (lower = less exploration, higher = more stochastic)"
echo "- lambda_div: 0.1-0.5 (lower = less divergence-free effect, higher = more exploration)"
echo
echo "Results will be saved in experiments/inference_scaling_TIMESTAMP/" 
# Advanced Protein Design Inference Methods

This document explains how to use the advanced inference methods for improved protein design in FoldFlow.

## Overview

FoldFlow now supports multiple inference strategies that can significantly improve the quality of generated protein designs:

1. **Standard Sampling**: Basic single-sample generation
2. **Best-of-N Sampling**: Generate N candidates and select the best one
3. **SDE Path Exploration**: Use stochastic differential equations with Euler-Maruyama sampling for path exploration
4. **Divergence-Free ODE**: Use divergence-free vector fields for deterministic path exploration

## Inference Methods

### 1. Standard Sampling
The default method that generates a single sample using the trained flow matching model.

### 2. Best-of-N Sampling
Generates multiple protein design candidates and selects the best one based on a quality metric (TM-score or RMSD).

**How it works:**
1. Generate N independent samples using the trained flow matching model
2. Evaluate each sample using the self-consistency pipeline
3. Select the sample with the highest score

### 3. SDE Path Exploration
Uses stochastic differential equations with branching to explore multiple paths during generation.

**How it works:**
1. Start with standard flow matching until a specified branching time
2. At each branching step, create multiple branches by adding different noise samples
3. Simulate each branch to completion deterministically
4. Evaluate and select the best branches to continue
5. Repeat until generation is complete

### 4. Divergence-Free ODE Path Exploration
Similar to SDE but uses divergence-free vector fields instead of noise for exploration.

**How it works:**
1. Start with standard flow matching until branching time
2. Add divergence-free vector fields to create different exploration paths
3. Simulate branches deterministically and select the best ones
4. Continue until generation is complete

## Configuration

### Basic Configuration

Edit `runner/config/inference.yaml`:

```yaml
inference:
  samples:
    # Choose inference method
    inference_method: "best_of_n"  # Options: "standard", "best_of_n", "sde_path_exploration", "divergence_free_ode"
    
    # Method-specific configuration
    method_config:
      # Best-of-N settings
      n_samples: 5
      selector: "tm_score"  # "tm_score" or "rmsd"
```

### SDE Path Exploration Configuration

```yaml
inference:
  samples:
    inference_method: "sde_path_exploration"
    method_config:
      num_branches: 4      # Number of branches per step
      num_keep: 2          # Number of branches to keep
      noise_scale: 0.05    # Scale of noise for SDE
      selector: "tm_score" # Scoring function
      branch_start_time: 0.0  # When to start branching (0.0 to 1.0)
```

### Divergence-Free ODE Configuration

```yaml
inference:
  samples:
    inference_method: "divergence_free_ode"
    method_config:
      num_branches: 4      # Number of branches per step
      num_keep: 2          # Number of branches to keep
      lambda_div: 0.2      # Scale factor for divergence-free field
      selector: "tm_score" # Scoring function
      branch_start_time: 0.0  # When to start branching
```

## Pre-configured Files

Use the provided configuration files for quick setup:

```bash
# Best-of-N sampling (default)
python runner/inference.py

# SDE path exploration
python runner/inference.py --config-name=inference_sde

# Divergence-free ODE exploration
python runner/inference.py --config-name=inference_divfree
```

## Scoring Functions

All advanced methods support different scoring functions:

- **tm_score**: Template Modeling score (higher is better)
- **rmsd**: Root Mean Square Deviation (lower is better, automatically negated)

## Output

For each sample, you'll find:

- The final selected protein design
- An `inference_score.txt` file containing the score and method used
- Method-specific directories with candidate evaluations (for branching methods)

## Performance Considerations

- **Best-of-N**: Computational cost scales linearly with N
- **SDE/Divergence-Free**: Cost scales with `num_branches` and number of branching steps
- **Branching methods**: More expensive but can find higher quality designs
- **Scoring**: Each evaluation requires ProteinMPNN + ESMFold, which is computationally intensive

## Tips

- Start with Best-of-N (N=5) for a good balance of quality and speed
- Use SDE path exploration for challenging targets requiring more exploration
- Increase `num_branches` for better quality at higher computational cost
- Set `branch_start_time > 0.0` to reduce computational cost while maintaining quality
- Use `tm_score` for general quality assessment
- Use `rmsd` when structural accuracy is most important

## Implementation Details

The inference methods are implemented in `runner/inference_methods.py` with a modular design:

- Each method inherits from `InferenceMethod` base class
- Scoring functions are standardized and reusable
- Configuration is handled through the YAML system
- Legacy Best-of-N configuration is still supported for backward compatibility 
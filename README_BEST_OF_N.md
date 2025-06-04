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

**Configuration:**
```yaml
inference:
  samples:
    inference_method: "sde_path_exploration"
    method_config:
      num_branches: 4          # Number of branches to explore
      num_keep: 2             # Number of best branches to keep  
      noise_scale: 0.05       # SDE noise scale
      selector: "tm_score"    # Scoring function
      branch_start_time: 0.0  # When to start branching (0.0 to 1.0)
      branch_interval: 0.0    # Time interval between branches (0.0 = every timestep)

**Key Parameters:**
- **`branch_interval`**: **NEW!** Controls branching frequency:
  - `0.0`: Branch at every timestep (original behavior, highest cost)
  - `0.1`: Branch every 0.1 time units (~10 branch points, 5x cost reduction)
  - `0.2`: Branch every 0.2 time units (~5 branch points, 10x cost reduction)
- **`noise_scale`**: Controls stochastic exploration strength
- **`branch_start_time`**: Delay branching to reduce early-stage computational cost

### 4. Divergence-Free ODE Path Exploration
Similar to SDE but uses divergence-free vector fields instead of noise for exploration.

**How it works:**
1. Start with standard flow matching until branching time
2. Add divergence-free vector fields to create different exploration paths
3. Simulate branches deterministically and select the best ones
4. Continue until generation is complete

**Configuration:**
```yaml
inference:
  samples:
    inference_method: "divergence_free_ode"
    method_config:
      num_branches: 4          # Number of branches to explore
      num_keep: 2             # Number of best branches to keep
      lambda_div: 0.2         # Divergence-free field strength
      selector: "tm_score"    # Scoring function
      branch_start_time: 0.0  # When to start branching (0.0 to 1.0)
      branch_interval: 0.0    # Time interval between branches (0.0 = every timestep)

**Key Parameters:**
- **`branch_interval`**: **NEW!** Same as SDE - controls branching frequency for cost reduction
- **`lambda_div`**: Strength of divergence-free field enhancement
- Different random seeds create diverse exploration patterns at each branch point

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

## Pre-configured Files

Use the provided configuration files for quick setup:

```bash
# Best-of-N sampling (default)
python runner/inference.py

# SDE path exploration
python runner/inference.py --config-name=inference_sde

# Divergence-free ODE exploration (every timestep)
python runner/inference.py --config-name=inference_divfree

## Branch Interval Parameter - Computational Cost Control

The **`branch_interval`** parameter is a major performance optimization that controls how frequently branching occurs during the sampling process.

### **Cost Analysis:**

| Configuration | Branch Points | Relative Cost | Use Case |
|---------------|---------------|---------------|----------|
| `branch_interval: 0.0` | Every timestep (50) | **50x** | Research/benchmarking |
| `branch_interval: 0.1` | Every 0.1 units (~10) | **10x** | High-quality production |
| `branch_interval: 0.2` | Every 0.2 units (~5) | **5x** | Balanced quality/speed |
| `branch_interval: 0.5` | Every 0.5 units (~2) | **2x** | Fast exploration |

### **Computational Savings Example:**

```yaml
# High cost: Branch at all 50 timesteps
branch_interval: 0.0  # 50 × 4 branches × 1 completion = 200 forward passes

# Medium cost: Branch every 0.1 time units  
branch_interval: 0.1  # 10 × 4 branches × 1 completion = 40 forward passes

# Low cost: Branch every 0.2 time units
branch_interval: 0.2  # 5 × 4 branches × 1 completion = 20 forward passes
```

**Cost reduction: 10x improvement** with minimal quality loss!

### **Recommended Settings:**

- **Research/Benchmarking**: `branch_interval: 0.0` (maximum quality)
- **Production High-Quality**: `branch_interval: 0.1` (good balance)  
- **Production Fast**: `branch_interval: 0.2` (efficient exploration)
- **Rapid Prototyping**: `branch_interval: 0.5` (minimal cost)
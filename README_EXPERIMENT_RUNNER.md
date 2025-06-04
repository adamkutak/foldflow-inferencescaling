# Inference Scaling Experiment Runner

This experiment runner compares different inference scaling methods for protein design to evaluate how each method performs with increasing computational budgets.

## Overview

The experiment runner tests four inference methods:

1. **Standard Sampling** (baseline) - Single sample generation
2. **Best-of-N Sampling** - Generate N samples, select best
3. **SDE Path Exploration** - Stochastic differential equation with branching
4. **Divergence-Free ODE** - Deterministic exploration using divergence-free vector fields

Each method (except standard) is tested with 2, 4, and 8 branches to evaluate scaling performance.

## Usage

### Basic Usage

```bash
python experiment_runner.py --num_samples 10 --sample_length 50
```

### Full Parameter Control

```bash
python experiment_runner.py \
    --num_samples 20 \
    --sample_length 100 \
    --scoring_function tm_score \
    --noise_scale 0.05 \
    --lambda_div 0.2 \
    --output_dir experiments
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_samples` | int | 5 | Number of samples to generate per method |
| `--sample_length` | int | 50 | Length of protein samples to generate |
| `--scoring_function` | str | tm_score | Scoring function (`tm_score` or `rmsd`) |
| `--noise_scale` | float | 0.05 | Noise scale for SDE path exploration |
| `--lambda_div` | float | 0.2 | Lambda parameter for divergence-free ODE |
| `--output_dir` | str | experiments | Base directory for experiment outputs |

## Experiment Design

### Methods Tested

1. **Standard (Baseline)**
   - Single sample generation
   - No scaling

2. **Best-of-N**
   - N = 2, 4, 8 samples
   - Select best based on scoring function
   - Linear scaling with N

3. **SDE Path Exploration**
   - Branches = 2, 4, 8
   - `num_keep = 1` (always keep only best)
   - Euler-Maruyama sampling with noise
   - Branching starts at t=0.0

4. **Divergence-Free ODE**
   - Branches = 2, 4, 8
   - `num_keep = 1` (always keep only best)
   - Divergence-free vector field exploration
   - Branching starts at t=0.0

### Evaluation Metrics

- **Mean Score**: Average score across all samples
- **Standard Deviation**: Score variability
- **Max/Min Scores**: Best and worst individual results
- **Total Time**: Time for all samples in method
- **Time per Sample**: Average time per sample
- **Improvement**: Percentage improvement over baseline
- **Speedup**: Time efficiency compared to baseline

## Output Files

The experiment runner creates a timestamped directory under `experiments/` with:

### `detailed_results.json`
Complete results including individual scores and configurations:
```json
{
  "method": "best_of_n",
  "num_branches": 4,
  "mean_score": 0.8234,
  "std_score": 0.0456,
  "scores": [0.8123, 0.8345, ...],
  "total_time": 45.67,
  "config": {...}
}
```

### `summary_results.csv`
Tabular summary for easy analysis:
```csv
method,num_branches,mean_score,std_score,total_time,time_per_sample
standard,1,0.7845,0.0234,12.34,2.47
best_of_n,2,0.8012,0.0345,25.67,5.13
...
```

## Example Output

```
================================================================================
INFERENCE SCALING EXPERIMENT RESULTS
================================================================================
Baseline (Standard): 0.7845±0.0234
Scoring Function: tm_score
Sample Length: 50
Samples per Method: 10

BEST_OF_N:
Branches    Mean Score   Improvement  Time (s)   Speedup   
------------------------------------------------------------
2           0.8012       2.13%        5.13       0.48x     
4           0.8234       4.96%        10.25      0.24x     
8           0.8456       7.79%        20.50      0.12x     

SDE_PATH_EXPLORATION:
Branches    Mean Score   Improvement  Time (s)   Speedup   
------------------------------------------------------------
2           0.8123       3.54%        6.78       0.36x     
4           0.8345       6.37%        13.56      0.18x     
8           0.8567       9.21%        27.12      0.09x     

DIVERGENCE_FREE_ODE:
Branches    Mean Score   Improvement  Time (s)   Speedup   
------------------------------------------------------------
2           0.8089       3.11%        7.23       0.34x     
4           0.8312       5.95%        14.46      0.17x     
8           0.8534       8.78%        28.92      0.09x     

BEST RESULT:
Method: sde_path_exploration (branches: 8)
Score: 0.8567±0.0123
Improvement: 9.21% over baseline
Time per sample: 27.12s
```

## Interpretation

### Performance Metrics

- **Higher scores are better** for tm_score
- **Lower scores are better** for rmsd (automatically negated)
- **Improvement %** shows gain over baseline
- **Speedup** shows time efficiency (>1.0 is faster, <1.0 is slower)

### Expected Patterns

1. **Quality vs Compute Trade-off**: More branches should improve quality but increase time
2. **Method Comparison**: Different methods may excel at different computational budgets
3. **Diminishing Returns**: Quality improvements may plateau with more branches
4. **Efficiency**: Some methods may be more time-efficient than others

## Quick Test

For a quick test with minimal computational cost:

```bash
python experiment_runner.py --num_samples 2 --sample_length 30
```

## Advanced Usage

### Custom Scoring Functions

The experiment runner supports different scoring functions:

- `tm_score`: Template Modeling score (structural similarity)
- `rmsd`: Root Mean Square Deviation (structural accuracy)

### Parameter Tuning

- **noise_scale**: Controls exploration strength in SDE (0.01-0.1 typical)
- **lambda_div**: Controls divergence-free field strength (0.1-0.5 typical)
- **sample_length**: Protein length (30-200 typical, longer = more expensive)

### Computational Considerations

- Each experiment runs: 1 + 3×3 = 10 method configurations
- Total samples: `num_samples × 10`
- Time scales roughly linearly with `num_samples` and `sample_length`
- Memory usage scales with `sample_length` and number of branches

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `sample_length` or `num_samples`
2. **Long runtime**: Start with small `num_samples` for testing
3. **Import errors**: Ensure all dependencies are installed
4. **Config errors**: Check that `runner/config/inference.yaml` exists

### Performance Tips

1. Use smaller `sample_length` for initial testing
2. Start with `num_samples=2` to verify everything works
3. Monitor GPU memory usage during experiments
4. Use `tm_score` for faster evaluation than full structure analysis

## Integration

The experiment runner integrates with the existing FoldFlow inference system:

- Uses same configuration files (`runner/config/inference.yaml`)
- Compatible with all existing models and checkpoints
- Outputs standard PDB files and evaluation metrics
- Can be extended with additional inference methods 
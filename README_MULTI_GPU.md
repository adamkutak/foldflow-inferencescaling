# Multi-GPU Experiment Runner

This is a modified version of the experiment runner that can run experiments concurrently on multiple GPUs, significantly reducing the total time needed to complete all experiments.

## Overview

The multi-GPU experiment runner (`experiment_runner_multi_gpu.py`) uses Python's `multiprocessing` module to run different experiments simultaneously on different GPUs. This allows you to:

- Run experiments in parallel instead of sequentially
- Utilize multiple GPUs efficiently
- Reduce total experiment time significantly
- Scale experiments across available hardware

## Key Features

### Concurrent Execution
- Each experiment runs on a dedicated GPU
- Automatic GPU assignment and management
- Process isolation prevents memory conflicts
- Queue-based GPU allocation system

### Memory Management
- Explicit cleanup of GPU memory after each experiment
- Process-level isolation prevents memory leaks
- Automatic garbage collection between experiments

### Flexible Configuration
- Specify any number of GPU IDs
- Customizable experiment parameters
- Same interface as single-GPU runner

## Usage

### Basic Usage

```bash
# Run with 3 GPUs (default)
python experiment_runner_multi_gpu.py --gpu_ids 0 1 2

# Run with specific parameters
python experiment_runner_multi_gpu.py \
    --num_samples 10 \
    --sample_length 100 \
    --gpu_ids 0 1 2 3 \
    --branch_counts 2 4 8 16
```

### Using the Example Script

The `run_multi_gpu_experiments.py` script provides pre-configured experiment setups:

```bash
# Run a quick test with 2 GPUs
python run_multi_gpu_experiments.py --experiment quick_test

# Run comprehensive test with 3 GPUs
python run_multi_gpu_experiments.py --experiment comprehensive

# Run all experiments sequentially
python run_multi_gpu_experiments.py --run-all
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu_ids` | int list | [0, 1, 2] | List of GPU IDs to use |
| `--num_samples` | int | 4 | Number of samples per method |
| `--sample_length` | int | 50 | Length of protein samples |
| `--scoring_function` | str | tm_score | Scoring function (`tm_score` or `rmsd`) |
| `--noise_scale` | float | 0.3 | Noise scale for SDE path exploration |
| `--lambda_div` | float | 0.6 | Lambda for divergence-free ODE |
| `--branch_interval` | float | 0.1 | Time interval between branches |
| `--branch_counts` | int list | [2, 4, 8] | Branch counts to test |

## How It Works

### GPU Assignment
1. Creates a queue of available GPUs
2. Assigns GPUs to experiments as they become available
3. Each experiment runs in its own process with dedicated GPU
4. GPU is returned to queue when experiment completes

### Process Management
1. Each experiment runs in a separate Python process
2. Processes are isolated to prevent memory conflicts
3. Results are collected via shared queue
4. Automatic cleanup when processes complete

### Memory Management
1. Explicit model cleanup after each experiment
2. GPU memory cache clearing
3. Garbage collection between experiments
4. Process termination ensures complete cleanup

## Example Configurations

### Quick Test (2 GPUs)
```bash
python experiment_runner_multi_gpu.py \
    --gpu_ids 0 1 \
    --num_samples 2 \
    --branch_counts 2 4
```

### Comprehensive Test (3 GPUs)
```bash
python experiment_runner_multi_gpu.py \
    --gpu_ids 0 1 2 \
    --num_samples 8 \
    --sample_length 100 \
    --branch_counts 2 4 8 16
```

### SDE-Focused (3 GPUs)
```bash
python experiment_runner_multi_gpu.py \
    --gpu_ids 0 1 2 \
    --noise_scale 0.1 \
    --lambda_div 0.3 \
    --branch_counts 2 4 8
```

## Performance Benefits

### Time Savings
- **Sequential**: 12 experiments × 5 minutes = 60 minutes
- **Multi-GPU (3 GPUs)**: 12 experiments ÷ 3 GPUs × 5 minutes = 20 minutes
- **Speedup**: ~3x faster with 3 GPUs

### Resource Utilization
- Better GPU utilization
- Reduced idle time
- Efficient memory management
- Scalable to any number of GPUs

## Output

The multi-GPU runner produces the same outputs as the single-GPU version:

- `detailed_results.json`: Complete experiment results
- `summary_results.csv`: Summary statistics
- Console output with analysis
- GPU-specific output directories

### Additional Output Features
- GPU ID tracking in results
- Process-specific logging
- Concurrent execution status
- Memory usage monitoring

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce `num_samples` or `sample_length`
   - Use fewer GPUs simultaneously
   - Check available GPU memory

2. **Process Hanging**
   - Check GPU availability
   - Verify CUDA installation
   - Monitor system resources

3. **Results Missing**
   - Check queue collection logic
   - Verify process completion
   - Review error logs

### Debug Mode

Add debug logging to see detailed process information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison with Single-GPU Runner

| Feature | Single-GPU | Multi-GPU |
|---------|------------|-----------|
| Execution | Sequential | Concurrent |
| GPU Usage | 1 GPU | Multiple GPUs |
| Memory | Shared | Isolated |
| Speed | Baseline | 2-4x faster |
| Complexity | Simple | Moderate |
| Resource Usage | Low | High |

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- Multiple CUDA-capable GPUs
- Sufficient system memory
- Same dependencies as original runner

## Best Practices

1. **GPU Selection**: Use GPUs with similar capabilities
2. **Memory Management**: Monitor GPU memory usage
3. **Process Limits**: Don't exceed system resources
4. **Error Handling**: Check logs for failed experiments
5. **Resource Monitoring**: Use `nvidia-smi` to monitor GPUs

## Advanced Usage

### Custom GPU Assignment
```python
# In your script
runner = MultiGPUExperimentRunner(args)
runner.available_gpus = [1, 3, 5]  # Use specific GPUs
```

### Process Pool Configuration
```python
# Modify process creation in run_all_experiments
p = Process(target=self.run_single_experiment_worker, 
           args=(experiment_data,), 
           daemon=True)  # Auto-cleanup
```

### Custom Experiment Queues
```python
# Create custom experiment queue
experiments = [
    {"method": "sde_path_exploration", "config": {...}},
    {"method": "divergence_free_ode", "config": {...}},
    # ... more experiments
]
```

This multi-GPU version provides significant performance improvements while maintaining the same experimental rigor as the original single-GPU runner. 
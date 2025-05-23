# Best-of-N Protein Design Sampling

This document explains how to use the Best-of-N sampling approach for improved protein design in FoldFlow.

## Overview

Best-of-N sampling is a technique that generates multiple protein design candidates and selects the best one based on a quality metric (TM-score). This approach significantly improves the quality of the final designs by exploring a larger portion of the design space.

## How It Works

1. The system generates N independent samples using the trained flow matching model
2. Each sample is evaluated for quality using the self-consistency pipeline:
   - ProteinMPNN predicts sequences that would fold into the structure
   - ESMFold folds these sequences back into structures
   - TM-scores are calculated between the original and refolded structures
3. The sample with the highest average TM-score is selected as the final output

## Usage

To use Best-of-N sampling, you need to set the `best_of_n` parameter in the configuration file:

```yaml
# In runner/config/inference.yaml
inference:
  samples:
    # ... other parameters ...
    
    # Number of candidates to generate for Best-of-N sampling
    best_of_n: 10  # Set to 1 to disable
```

The recommended range for `best_of_n` is between 5 and 20, with higher values providing better quality at the cost of increased computational time.

## Running Inference

Run inference as usual with your Best-of-N configuration:

```bash
python runner/inference.py
```

## Output

For each sample, you'll find:

- The final selected protein design
- A `best_of_n_tm_score.txt` file containing the best TM-score and number of candidates
- A `best_of_n_candidates` directory with all candidate designs and their evaluations

## Implementation Details

The Best-of-N sampling is implemented in the `best_of_n_sample` method in the `Sampler` class in `runner/inference.py`. This method:

1. Generates N independent samples
2. Evaluates each sample using the `run_self_consistency` method
3. Selects the sample with the highest average TM-score
4. Returns the best sample along with its metrics

## Tips

- Increase `best_of_n` for higher quality designs, especially for challenging targets
- For quick exploration, use a smaller value (e.g., 5)
- For final designs, consider using a larger value (e.g., 20)
- The computational cost scales linearly with `best_of_n` 
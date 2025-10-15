<div align="center">

# FoldFlow: Inference Time Scaling for Protein Design

[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description

FoldFlow uses flow matching generative models for protein backbone generation. This fork explores methods to improve self-consistency distance by scaling inference compute for unconditional protein generation. We benchmark multiple methods, including Best-of-N (Random Search), and our novel noise search method paired with a unique noise schedule that aims to maximize the diversity-quality tradeoff, as well as standard SDE noise. We further include a two-stage sampling algorithm that optimizes the initial noise, then the trajectory for the best performance. These methods allow trading computational budget for improved protein design quality in a controlled manner. Our experiment runners provide automated comparison of these approaches across different computational budgets.

This repository is a fork of [DreamFold/FoldFlow](https://github.com/DreamFold/FoldFlow) focused on improving protein design quality through inference time scaling methods. We implement and compare several techniques for allocating additional computational budget during inference to generate higher quality protein structures.

# Installation

Clone this repository and install the dependencies using micromamba. We tested the code with Python 3.9.15 and CUDA 11.6.1. First [install micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) then run the following:

```bash
# Clone repo
git clone https://github.com/YourUsername/FoldFlow.git
cd FoldFlow

# Install dependencies and activate environment
micromamba create -f environment.yaml
micromamba activate foldflow-env
```
# Inference

## Running Inference Scaling Experiments

This repository provides experiment runners to compare different inference time scaling methods. The runners automatically test multiple methods with different computational budgets and generate comprehensive comparison results.

### Single GPU

For running experiments on a single GPU:

```bash
python experiment_runner.py --num_samples 10 --sample_length 100
```

### Multi-GPU

For parallel execution across multiple GPUs:

```bash
python experiment_runner_multi_gpu.py --num_samples 10 --sample_length 100 --gpus 0 1 2 3
```

### Key Parameters

- `--num_samples`: Number of protein samples to generate per method (default: 5)
- `--sample_length`: Length of protein backbones to generate (default: 100)
- `--scoring_function`: Metric for optimization - `tm_score`, `rmsd`, or `geometric` (default: `tm_score`)
- `--branch_counts`: Computational budgets to test (default: [2, 4, 8])

The experiment runners will compare standard sampling, best-of-N, SDE path exploration, and divergence-free ODE methods. Results are saved with detailed metrics and summary CSVs for analysis.

For more detailed documentation on parameters and output formats, see `README_EXPERIMENT_RUNNER.md`.

## Standard Inference

For standard FoldFlow inference without scaling experiments, you can use the original inference pipeline. Download pretrained weights from the [original FoldFlow releases](https://github.com/DreamFold/FoldFlow/releases) and configure `runner/config/inference.yaml` with the checkpoint path, then run:

```bash
python runner/inference.py
```

# Citation

This repository is a fork of [DreamFold/FoldFlow](https://github.com/DreamFold/FoldFlow). If this codebase is useful for your research, please cite the original FoldFlow papers:

```bibtex
@article{huguet2024sequence,
  title={Sequence-Augmented SE (3)-Flow Matching For Conditional Protein Backbone Generation},
  author={Huguet, Guillaume and Vuckovic, James and Fatras, Kilian and Thibodeau-Laufer, Eric and Lemos, Pablo and Islam, Riashat and Liu, Cheng-Hao and Rector-Brooks, Jarrid and Akhound-Sadegh, Tara and Bronstein, Michael and others},
  journal={Advances in neural information processing systems},
  volumne={38},
  year={2024}
}

@inproceedings{bose2024se3,
  title={SE (3)-Stochastic Flow Matching for Protein Backbone Generation},
  author={Bose, Joey and Akhound-Sadegh, Tara and Huguet, Guillaume and FATRAS, Kilian and Rector-Brooks, Jarrid and Liu, Cheng-Hao and Nica, Andrei Cristian and Korablyov, Maksym and Bronstein, Michael M and Tong, Alexander},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

If you use our inference time scaling methods, please also cite our work (citation will be added once our paper is published).

# Third Party Source Code

This repository contains forks of [OpenFold](https://github.com/aqlaboratory/openfold) and [ProteinMPNN](https://github.com/dauparas/ProteinMPNN). Each of these codebases are actively under development and you may want to refork. Several files in `/data/` are adapted from [AlphaFold](https://github.com/deepmind/alphafold).

# License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Dreamfold/foldflow">FoldFlow</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://dreamfold.ai">Dreamfold</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>

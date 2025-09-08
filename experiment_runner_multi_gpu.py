#!/usr/bin/env python3
"""
Multi-GPU experiment runner for comparing inference scaling methods in protein design.

This script runs experiments concurrently on multiple GPUs to compare different inference methods:
- Standard inference method (baseline)
- Random search/best-of-N sampling
- Noise search (divergence free max)
- Noise search (sde)
- Random search + noise search (divergence free max)

Each method is tested with different computational budgets (number of branches)
to evaluate inference time scaling performance.
"""

import argparse
import logging
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import queue

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set, check if it's 'spawn'
    if mp.get_start_method() != "spawn":
        raise RuntimeError(
            "Multiprocessing start method must be 'spawn' for CUDA compatibility"
        )

# Note: runner.inference_methods and runner.inference imports are now done inside worker function
# to ensure GEOMSTATS_BACKEND is set before importing
from omegaconf import DictConfig, OmegaConf


class MultiGPUExperimentRunner:
    """Multi-GPU runner for inference scaling experiments."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.setup_logging()
        self.setup_config()
        self.setup_gpu_assignment()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def setup_config(self):
        """Setup configuration for experiments."""
        # Load multiple configuration files and merge them properly
        base_config_path = "runner/config/base.yaml"
        model_config_path = "runner/config/model/ff2.yaml"
        inference_config_path = "runner/config/inference.yaml"
        flow_matcher_config_path = "runner/config/flow_matcher/default.yaml"
        data_config_path = "runner/config/data/default.yaml"
        wandb_config_path = "runner/config/wandb/default.yaml"
        experiment_config_path = "runner/config/experiment/baseline.yaml"

        # Load each config
        base_conf = OmegaConf.load(base_config_path)
        model_conf = OmegaConf.load(model_config_path)
        inference_conf = OmegaConf.load(inference_config_path)
        flow_matcher_conf = OmegaConf.load(flow_matcher_config_path)
        data_conf = OmegaConf.load(data_config_path)
        wandb_conf = OmegaConf.load(wandb_config_path)
        experiment_conf = OmegaConf.load(experiment_config_path)

        # Create merged configuration
        self.base_conf = OmegaConf.create({})

        # Merge configurations in the right order
        self.base_conf = OmegaConf.merge(
            self.base_conf,
            base_conf,
            {"model": model_conf},
            {"flow_matcher": flow_matcher_conf},
            {"data": data_conf},
            {"wandb": wandb_conf},
            {"experiment": experiment_conf},
            inference_conf,
        )

        # Override with experiment parameters
        self.base_conf.inference.samples.samples_per_length = (
            1  # We'll control this manually
        )
        self.base_conf.inference.samples.min_length = self.args.sample_length
        self.base_conf.inference.samples.max_length = self.args.sample_length
        self.base_conf.inference.samples.length_step = 1

        # Create output directory for experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "experiments", f"inference_scaling_multi_gpu_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def setup_gpu_assignment(self):
        """Setup GPU assignment for concurrent experiments."""
        self.available_gpus = self.args.gpu_ids
        self.num_gpus = len(self.available_gpus)
        self.logger.info(f"Using GPUs: {self.available_gpus}")

        # Debug: Check available GPUs
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            self.logger.info(f"Total CUDA devices available: {total_gpus}")
            for i in range(total_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                self.logger.info(f"  GPU {i}: {gpu_name}")
        else:
            self.logger.warning("CUDA not available!")

        # Create a queue for GPU assignment
        self.gpu_queue = Queue()
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)

    def create_sampler(self, method_config: Dict[str, Any], gpu_id: int):
        """Create a sampler with specific method configuration and GPU assignment."""
        # Import here since it's not imported at the top level
        from runner.inference import Sampler

        conf = OmegaConf.create(self.base_conf)

        # Update method configuration
        conf.inference.samples.inference_method = method_config["method"]
        conf.inference.samples.method_config = method_config.get("config", {})

        # Set GPU
        conf.inference.gpu_id = gpu_id

        # Set output directory for this experiment
        method_name = method_config["method"]
        branches = method_config.get("config", {}).get("num_branches", 1)
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_branches_{branches}_gpu_{gpu_id}"
        )

        return Sampler(conf)

    def cleanup_sampler(self, sampler, method_name: str):
        """Explicit cleanup of sampler to prevent memory leaks."""
        try:
            self.logger.debug(f"Cleaning up sampler for {method_name}")

            # Move models to CPU and delete them explicitly
            if hasattr(sampler, "model") and sampler.model is not None:
                sampler.model = sampler.model.cpu()
                del sampler.model

            if (
                hasattr(sampler, "_folding_model")
                and sampler._folding_model is not None
            ):
                sampler._folding_model = sampler._folding_model.cpu()
                del sampler._folding_model

            # Clean up experiment object and its model
            if hasattr(sampler, "exp") and sampler.exp is not None:
                if hasattr(sampler.exp, "model") and sampler.exp.model is not None:
                    sampler.exp.model = sampler.exp.model.cpu()
                    del sampler.exp.model
                if hasattr(sampler.exp, "_model") and sampler.exp._model is not None:
                    sampler.exp._model = sampler.exp._model.cpu()
                    del sampler.exp._model
                del sampler.exp

            # Clean up flow matcher
            if hasattr(sampler, "flow_matcher"):
                del sampler.flow_matcher

            # Clean up inference method
            if hasattr(sampler, "inference_method"):
                del sampler.inference_method

            # Delete the sampler object itself
            del sampler

            # Force garbage collection
            import gc

            gc.collect()

            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as cleanup_error:
            self.logger.warning(f"Error during sampler cleanup: {cleanup_error}")

    def run_single_experiment_worker(
        self, experiment_data: Tuple[Dict[str, Any], int, Queue]
    ) -> Dict[str, Any]:
        """Worker function to run a single experiment on a specific GPU."""
        method_config, gpu_id, result_queue = experiment_data

        # Set environment variable that affects geomstats backend BEFORE any imports
        # This must be done before importing any modules that might use geomstats
        os.environ["GEOMSTATS_BACKEND"] = "pytorch"

        # Now we can safely import the required modules
        import sys
        import logging
        import time
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from runner.inference_methods import get_inference_method
        from runner.inference import Sampler

        # Apply the same PyTorch settings as in train.py
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True

        # Set the GPU for this process using torch.cuda.set_device
        # This allows PyTorch to see all GPUs but use the specific one
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            # Verify we're on the correct device
            current_device = torch.cuda.current_device()
            if current_device != gpu_id:
                raise RuntimeError(
                    f"Failed to set GPU to {gpu_id}, current device is {current_device}"
                )

            # Debug: Log device information
            worker_logger = logging.getLogger(f"{__name__}.worker.{gpu_id}")
            worker_logger.info(f"Worker process assigned to GPU {gpu_id}")
            worker_logger.info(f"Current device: {torch.cuda.current_device()}")
            worker_logger.info(f"Device name: {torch.cuda.get_device_name(gpu_id)}")
            worker_logger.info(f"Total devices visible: {torch.cuda.device_count()}")
        else:
            worker_logger = logging.getLogger(f"{__name__}.worker.{gpu_id}")
            worker_logger.warning("CUDA not available in worker process!")

        method_name = method_config["method"]
        config = method_config.get("config", {})
        branches = config.get("num_branches", 1)

        # Setup logging for this worker (if not already set up above)
        if not "worker_logger" in locals():
            worker_logger = logging.getLogger(f"{__name__}.worker.{gpu_id}")
        worker_logger.info(
            f"Running experiment: {method_name} with {branches} branches on GPU {gpu_id}"
        )

        # Create sampler
        sampler = self.create_sampler(method_config, gpu_id)

        try:
            # Track timing and results
            start_time = time.time()
            scores = []
            sample_metrics_list = []

            for sample_idx in range(self.args.num_samples):
                worker_logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

                try:
                    # Generate sample
                    sample_start = time.time()
                    sample_result = sampler.inference_method.sample(
                        self.args.sample_length
                    )
                    sample_time = time.time() - sample_start

                    # Extract sample and score
                    if isinstance(sample_result, dict) and "sample" in sample_result:
                        sample_output = sample_result["sample"]
                        # Use the score from the method if available
                        if "score" in sample_result:
                            score = sample_result["score"]
                        else:
                            # Evaluate manually
                            score_fn = sampler.inference_method.get_score_function(
                                self.args.scoring_function
                            )
                            score = score_fn(sample_output, self.args.sample_length)
                    else:
                        sample_output = sample_result
                        # Evaluate manually
                        score_fn = sampler.inference_method.get_score_function(
                            self.args.scoring_function
                        )
                        score = score_fn(sample_output, self.args.sample_length)

                    # Always collect comprehensive metrics (TM score, RMSD, and geometric)
                    sample_metrics = {"selector_score": score}

                    try:
                        # Use the universal scoring system to get all metrics at once
                        all_scores = (
                            sampler.inference_method._universal_score_all_function(
                                sample_output, self.args.sample_length
                            )
                        )

                        # Extract the main metrics we want to track
                        sample_metrics["geometric_score"] = all_scores.get(
                            "geometric", float("nan")
                        )
                        sample_metrics["tm_score"] = all_scores.get(
                            "tm_score", float("nan")
                        )
                        sample_metrics["rmsd_score"] = -all_scores.get(
                            "rmsd", float("nan")
                        )  # Convert back to positive RMSD

                        # Also store additional useful metrics
                        sample_metrics["tm_score_4seq"] = all_scores.get(
                            "tm_score_4seq", float("nan")
                        )
                        sample_metrics["rmsd_4seq"] = -all_scores.get(
                            "rmsd_4seq", float("nan")
                        )  # Convert back to positive RMSD
                        sample_metrics["radius_of_gyration"] = all_scores.get(
                            "radius_of_gyration", float("nan")
                        )

                        # Store geometric components for detailed analysis
                        sample_metrics["ca_ca_bond_dev"] = all_scores.get(
                            "ca_ca_bond_dev", float("nan")
                        )
                        sample_metrics["ca_ca_valid_percent"] = all_scores.get(
                            "ca_ca_valid_percent", float("nan")
                        )
                        sample_metrics["ca_steric_clash_percent"] = all_scores.get(
                            "ca_steric_clash_percent", float("nan")
                        )
                        sample_metrics["non_coil_percent"] = all_scores.get(
                            "non_coil_percent", float("nan")
                        )

                        # 3. Get detailed self-consistency results for analysis
                        temp_dir = os.path.join(
                            sampler._output_dir, f"temp_sample_{sample_idx}"
                        )
                        os.makedirs(temp_dir, exist_ok=True)

                        try:
                            # Save sample for self-consistency evaluation
                            traj_paths = sampler.save_traj(
                                sample_output["prot_traj"],
                                sample_output["rigid_0_traj"],
                                np.ones(self.args.sample_length),
                                output_dir=temp_dir,
                            )
                            pdb_path = traj_paths["sample_path"]

                            # Run self-consistency to get individual sequence results
                            sc_results = sampler.run_self_consistency(
                                temp_dir, pdb_path, motif_mask=None
                            )

                            # Extract detailed metrics
                            sample_metrics["sc_tm_scores"] = sc_results[
                                "tm_score"
                            ].tolist()
                            sample_metrics["sc_rmsd_scores"] = sc_results[
                                "rmsd"
                            ].tolist()
                            sample_metrics["sc_mean_tm"] = sc_results["tm_score"].mean()
                            sample_metrics["sc_mean_rmsd"] = sc_results["rmsd"].mean()

                            # Calculate percentage of sequences with RMSD < thresholds
                            rmsd_values = sc_results["rmsd"].values
                            sample_metrics["rmsd_lt_2A_percent"] = (
                                rmsd_values < 2.0
                            ).mean() * 100
                            sample_metrics["rmsd_lt_1_5A_percent"] = (
                                rmsd_values < 1.5
                            ).mean() * 100
                            sample_metrics["rmsd_lt_1A_percent"] = (
                                rmsd_values < 1.0
                            ).mean() * 100

                        finally:
                            # Clean up temporary directory
                            if os.path.exists(temp_dir):
                                import shutil

                                shutil.rmtree(temp_dir)

                    except Exception as e:
                        worker_logger.warning(
                            f"Failed to get comprehensive metrics: {e}"
                        )
                        # Set default values if comprehensive evaluation fails
                        sample_metrics.update(
                            {
                                "geometric_score": float("nan"),
                                "tm_score": float("nan"),
                                "rmsd_score": float("nan"),
                                "tm_score_4seq": float("nan"),
                                "rmsd_4seq": float("nan"),
                                "radius_of_gyration": float("nan"),
                                "ca_ca_bond_dev": float("nan"),
                                "ca_ca_valid_percent": float("nan"),
                                "ca_steric_clash_percent": float("nan"),
                                "non_coil_percent": float("nan"),
                                "sc_mean_tm": float("nan"),
                                "sc_mean_rmsd": float("nan"),
                                "rmsd_lt_2A_percent": float("nan"),
                                "rmsd_lt_1_5A_percent": float("nan"),
                                "rmsd_lt_1A_percent": float("nan"),
                            }
                        )

                    scores.append(score)
                    sample_metrics_list.append(sample_metrics)

                    # Enhanced logging with all three metrics
                    log_parts = [f"Selector: {score:.4f}"]

                    if "geometric_score" in sample_metrics and not np.isnan(
                        sample_metrics["geometric_score"]
                    ):
                        log_parts.append(
                            f"Geom: {sample_metrics['geometric_score']:.4f}"
                        )

                    if "tm_score" in sample_metrics and not np.isnan(
                        sample_metrics["tm_score"]
                    ):
                        log_parts.append(f"TM: {sample_metrics['tm_score']:.4f}")

                    if "rmsd_score" in sample_metrics and not np.isnan(
                        sample_metrics["rmsd_score"]
                    ):
                        log_parts.append(f"RMSD: {sample_metrics['rmsd_score']:.3f}Å")

                    if "rmsd_lt_2A_percent" in sample_metrics and not np.isnan(
                        sample_metrics["rmsd_lt_2A_percent"]
                    ):
                        log_parts.append(
                            f"<2Å: {sample_metrics['rmsd_lt_2A_percent']:.1f}%"
                        )

                    log_parts.append(f"Time: {sample_time:.2f}s")
                    worker_logger.info(f"    {', '.join(log_parts)}")

                except Exception as e:
                    worker_logger.error(f"Error in sample {sample_idx}: {e}")
                    scores.append(float("nan"))
                    sample_metrics_list.append({"selector_score": float("nan")})

            total_time = time.time() - start_time

            # Calculate statistics
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                max_score = np.max(valid_scores)
                min_score = np.min(valid_scores)
            else:
                mean_score = std_score = max_score = min_score = float("nan")

            # Calculate comprehensive metrics for all three scoring types
            comprehensive_metrics = {}

            # 1. Geometric score statistics
            geometric_values = [
                m.get("geometric_score")
                for m in sample_metrics_list
                if not np.isnan(m.get("geometric_score", float("nan")))
            ]
            if geometric_values:
                comprehensive_metrics.update(
                    {
                        "mean_geometric": np.mean(geometric_values),
                        "std_geometric": np.std(geometric_values),
                        "max_geometric": np.max(geometric_values),
                        "min_geometric": np.min(geometric_values),
                    }
                )
            else:
                comprehensive_metrics.update(
                    {
                        "mean_geometric": float("nan"),
                        "std_geometric": float("nan"),
                        "max_geometric": float("nan"),
                        "min_geometric": float("nan"),
                    }
                )

            # 2. TM score statistics
            tm_values = [
                m.get("tm_score")
                for m in sample_metrics_list
                if not np.isnan(m.get("tm_score", float("nan")))
            ]
            if tm_values:
                comprehensive_metrics.update(
                    {
                        "mean_tm_score": np.mean(tm_values),
                        "std_tm_score": np.std(tm_values),
                        "max_tm_score": np.max(tm_values),
                        "min_tm_score": np.min(tm_values),
                    }
                )
            else:
                comprehensive_metrics.update(
                    {
                        "mean_tm_score": float("nan"),
                        "std_tm_score": float("nan"),
                        "max_tm_score": float("nan"),
                        "min_tm_score": float("nan"),
                    }
                )

            # 3. RMSD statistics
            rmsd_values = [
                m.get("rmsd_score")
                for m in sample_metrics_list
                if not np.isnan(m.get("rmsd_score", float("nan")))
            ]
            if rmsd_values:
                comprehensive_metrics.update(
                    {
                        "mean_rmsd": np.mean(rmsd_values),
                        "std_rmsd": np.std(rmsd_values),
                        "max_rmsd": np.max(rmsd_values),
                        "min_rmsd": np.min(rmsd_values),
                    }
                )
            else:
                comprehensive_metrics.update(
                    {
                        "mean_rmsd": float("nan"),
                        "std_rmsd": float("nan"),
                        "max_rmsd": float("nan"),
                        "min_rmsd": float("nan"),
                    }
                )

            # 4. Additional metrics from universal scorer
            # 4-sequence metrics
            tm_4seq_values = [
                m.get("tm_score_4seq")
                for m in sample_metrics_list
                if not np.isnan(m.get("tm_score_4seq", float("nan")))
            ]
            if tm_4seq_values:
                comprehensive_metrics.update(
                    {
                        "mean_tm_score_4seq": np.mean(tm_4seq_values),
                        "std_tm_score_4seq": np.std(tm_4seq_values),
                    }
                )
            else:
                comprehensive_metrics.update(
                    {
                        "mean_tm_score_4seq": float("nan"),
                        "std_tm_score_4seq": float("nan"),
                    }
                )

            rmsd_4seq_values = [
                m.get("rmsd_4seq")
                for m in sample_metrics_list
                if not np.isnan(m.get("rmsd_4seq", float("nan")))
            ]
            if rmsd_4seq_values:
                comprehensive_metrics.update(
                    {
                        "mean_rmsd_4seq": np.mean(rmsd_4seq_values),
                        "std_rmsd_4seq": np.std(rmsd_4seq_values),
                    }
                )
            else:
                comprehensive_metrics.update(
                    {
                        "mean_rmsd_4seq": float("nan"),
                        "std_rmsd_4seq": float("nan"),
                    }
                )

            # 5. Calculate percentage statistics
            for threshold_key in [
                "rmsd_lt_2A_percent",
                "rmsd_lt_1_5A_percent",
                "rmsd_lt_1A_percent",
            ]:
                threshold_values = [
                    m.get(threshold_key)
                    for m in sample_metrics_list
                    if not np.isnan(m.get(threshold_key, float("nan")))
                ]
                if threshold_values:
                    comprehensive_metrics[f"mean_{threshold_key}"] = np.mean(
                        threshold_values
                    )
                    comprehensive_metrics[f"std_{threshold_key}"] = np.std(
                        threshold_values
                    )
                else:
                    comprehensive_metrics[f"mean_{threshold_key}"] = float("nan")
                    comprehensive_metrics[f"std_{threshold_key}"] = float("nan")

            result = {
                "method": method_name,
                "num_branches": branches,
                "config": config,
                "num_samples": len(valid_scores),
                "mean_score": mean_score,
                "std_score": std_score,
                "max_score": max_score,
                "min_score": min_score,
                "total_time": total_time,
                "time_per_sample": total_time / self.args.num_samples,
                "scores": scores,
                "scoring_function": self.args.scoring_function,
                "gpu_id": gpu_id,
                "sample_metrics": sample_metrics_list,
                **comprehensive_metrics,
            }

            # Enhanced result logging with all three metrics
            log_parts = [f"Selector={mean_score:.4f}±{std_score:.4f}"]

            if "mean_geometric" in comprehensive_metrics and not np.isnan(
                comprehensive_metrics["mean_geometric"]
            ):
                log_parts.append(
                    f"Geom={comprehensive_metrics['mean_geometric']:.4f}±{comprehensive_metrics['std_geometric']:.4f}"
                )

            if "mean_tm_score" in comprehensive_metrics and not np.isnan(
                comprehensive_metrics["mean_tm_score"]
            ):
                log_parts.append(
                    f"TM={comprehensive_metrics['mean_tm_score']:.4f}±{comprehensive_metrics['std_tm_score']:.4f}"
                )

            if "mean_rmsd" in comprehensive_metrics and not np.isnan(
                comprehensive_metrics["mean_rmsd"]
            ):
                log_parts.append(
                    f"RMSD={comprehensive_metrics['mean_rmsd']:.3f}±{comprehensive_metrics['std_rmsd']:.3f}Å"
                )

            if "mean_rmsd_lt_2A_percent" in comprehensive_metrics and not np.isnan(
                comprehensive_metrics["mean_rmsd_lt_2A_percent"]
            ):
                log_parts.append(
                    f"<2Å={comprehensive_metrics['mean_rmsd_lt_2A_percent']:.1f}%"
                )

            log_parts.append(f"Time={total_time:.2f}s")
            worker_logger.info(f"  Results: {', '.join(log_parts)}")

            # Put result in queue
            result_queue.put(result)
            return result

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def run_all_experiments(self):
        """Run all experiments with different methods and branch counts using multiple GPUs."""
        experiments = []

        # 1. Standard inference method
        experiments.append({"method": "standard", "config": {}})

        # 2. Random search/best-of-N with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "best_of_n",
                    "config": {
                        "num_branches": n_branches,
                        "selector": self.args.scoring_function,
                    },
                }
            )

        # 3. Noise search (divergence free max) with different branch counts
        # Skip num_branches=1 since it's inefficient (falls back to standard inference)
        for n_branches in self.args.branch_counts:
            if n_branches == 1:
                continue
            experiments.append(
                {
                    "method": "noise_search_divfree_max",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,
                        "num_rounds": self.args.num_rounds,
                        "lambda_div": self.args.lambda_div,
                        "particle_repulsion_factor": self.args.particle_repulsion_factor,
                        "noise_schedule_end_factor": self.args.noise_schedule_end_factor,
                        "selector": self.args.scoring_function,
                        "massage_steps": self.args.massage_steps,
                    },
                }
            )

        # 4. Noise search (sde) with different branch counts
        # Skip num_branches=1 since it's inefficient (falls back to standard inference)
        for n_branches in self.args.branch_counts:
            if n_branches == 1:
                continue
            experiments.append(
                {
                    "method": "noise_search_sde",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,
                        "num_rounds": self.args.num_rounds,
                        "noise_scale": self.args.noise_scale,
                        "selector": self.args.scoring_function,
                        "massage_steps": self.args.massage_steps,
                    },
                }
            )

        # 5. Random search + noise search (divergence free max) with different branch counts
        # Skip num_branches=1 since the noise search phase would be inefficient
        for n_branches in self.args.branch_counts:
            if n_branches == 1:
                continue
            experiments.append(
                {
                    "method": "random_search_noise",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,
                        "num_rounds": self.args.num_rounds,
                        "noise_type": "divfree_max",
                        "lambda_div": self.args.lambda_div,
                        "particle_repulsion_factor": self.args.particle_repulsion_factor,
                        "noise_schedule_end_factor": self.args.noise_schedule_end_factor,
                        "selector": self.args.scoring_function,
                        "massage_steps": self.args.massage_steps,
                    },
                }
            )

        # Sort experiments to prioritize higher branch counts (longer experiments) first
        # Keep baseline (standard) first, then sort others by branch count descending
        baseline_experiments = [
            exp for exp in experiments if exp["method"] == "standard"
        ]
        branched_experiments = [
            exp for exp in experiments if exp["method"] != "standard"
        ]

        # Sort branched experiments by num_branches in descending order (highest first)
        branched_experiments.sort(
            key=lambda x: x.get("config", {}).get("num_branches", 0), reverse=True
        )

        # Combine: baseline first, then sorted branched experiments
        experiments = baseline_experiments + branched_experiments

        # Create a manager for shared result queue
        manager = Manager()
        result_queue = manager.Queue()

        # Run experiments concurrently
        processes = []
        process_gpu_map = {}  # Track which GPU each process is using
        experiment_idx = 0

        while experiment_idx < len(experiments):
            # Wait for available GPU
            try:
                gpu_id = self.gpu_queue.get(timeout=1)
            except queue.Empty:
                # Check if any processes have finished
                for p in processes[:]:
                    if not p.is_alive():
                        p.join()
                        # Return the GPU to the queue
                        if p in process_gpu_map:
                            returned_gpu = process_gpu_map[p]
                            self.gpu_queue.put(returned_gpu)
                            del process_gpu_map[p]
                            self.logger.info(
                                f"Process finished, returned GPU {returned_gpu} to queue"
                            )
                        processes.remove(p)
                continue

            # Start new experiment on this GPU
            exp_config = experiments[experiment_idx]
            experiment_data = (exp_config, gpu_id, result_queue)

            p = Process(
                target=self.run_single_experiment_worker, args=(experiment_data,)
            )
            p.start()
            processes.append(p)
            process_gpu_map[p] = gpu_id  # Track which GPU this process is using

            self.logger.info(
                f"Started experiment {experiment_idx+1}/{len(experiments)}: {exp_config['method']} on GPU {gpu_id}"
            )
            experiment_idx += 1

        # Wait for all processes to complete
        for p in processes:
            p.join()
            # Return the GPU to the queue
            if p in process_gpu_map:
                returned_gpu = process_gpu_map[p]
                self.gpu_queue.put(returned_gpu)
                del process_gpu_map[p]
                self.logger.info(
                    f"Process finished, returned GPU {returned_gpu} to queue"
                )

        # Collect all results
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                self.results.append(result)
            except queue.Empty:
                break

        # Save results
        self.save_results()
        self.analyze_results()

    def save_results(self):
        """Save experiment results to files."""
        # Save detailed results as JSON
        results_file = os.path.join(self.experiment_dir, "detailed_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []
        for result in self.results:
            summary_row = {
                "method": result["method"],
                "num_branches": result["num_branches"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "max_score": result["max_score"],
                "min_score": result["min_score"],
                "total_time": result["total_time"],
                "time_per_sample": result["time_per_sample"],
                "num_samples": result["num_samples"],
                "scoring_function": result["scoring_function"],
                "gpu_id": result["gpu_id"],
            }

            # Add comprehensive metrics for all three scoring types
            # Geometric metrics
            if "mean_geometric" in result:
                summary_row.update(
                    {
                        "mean_geometric": result["mean_geometric"],
                        "std_geometric": result["std_geometric"],
                        "max_geometric": result["max_geometric"],
                        "min_geometric": result["min_geometric"],
                    }
                )

            # TM score metrics (8-sequence and 4-sequence)
            if "mean_tm_score" in result:
                summary_row.update(
                    {
                        "mean_tm_score": result["mean_tm_score"],
                        "std_tm_score": result["std_tm_score"],
                        "max_tm_score": result["max_tm_score"],
                        "min_tm_score": result["min_tm_score"],
                    }
                )

            if "mean_tm_score_4seq" in result:
                summary_row.update(
                    {
                        "mean_tm_score_4seq": result["mean_tm_score_4seq"],
                        "std_tm_score_4seq": result["std_tm_score_4seq"],
                    }
                )

            # RMSD metrics (8-sequence and 4-sequence)
            if "mean_rmsd" in result:
                summary_row.update(
                    {
                        "mean_rmsd": result["mean_rmsd"],
                        "std_rmsd": result["std_rmsd"],
                        "max_rmsd": result["max_rmsd"],
                        "min_rmsd": result["min_rmsd"],
                        "mean_rmsd_lt_2A_percent": result.get(
                            "mean_rmsd_lt_2A_percent", float("nan")
                        ),
                        "mean_rmsd_lt_1_5A_percent": result.get(
                            "mean_rmsd_lt_1_5A_percent", float("nan")
                        ),
                        "mean_rmsd_lt_1A_percent": result.get(
                            "mean_rmsd_lt_1A_percent", float("nan")
                        ),
                    }
                )

            if "mean_rmsd_4seq" in result:
                summary_row.update(
                    {
                        "mean_rmsd_4seq": result["mean_rmsd_4seq"],
                        "std_rmsd_4seq": result["std_rmsd_4seq"],
                    }
                )

            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("MULTI-GPU INFERENCE SCALING EXPERIMENT RESULTS")
        print("=" * 80)

        # Find baseline (standard method)
        baseline = None
        for result in self.results:
            if result["method"] == "standard":
                baseline = result
                break

        if baseline is None:
            print("Warning: No baseline (standard) method found!")
            return

        baseline_score = baseline["mean_score"]
        baseline_time = baseline["time_per_sample"]

        print(f"Baseline (Standard): {baseline_score:.4f}±{baseline['std_score']:.4f}")
        print(f"Scoring Function: {self.args.scoring_function}")
        print(f"Sample Length: {self.args.sample_length}")
        print(f"Samples per Method: {self.args.num_samples}")
        print(f"GPUs Used: {self.available_gpus}")
        print()

        # Group results by method
        methods = {}
        for result in self.results:
            method = result["method"]
            if method not in methods:
                methods[method] = []
            methods[method].append(result)

        # Print results for each method
        for method_name, method_results in methods.items():
            if method_name == "standard":
                continue

            print(f"{method_name.upper()}:")

            # Check if we have comprehensive metrics
            has_comprehensive = any("mean_rmsd" in result for result in method_results)

            if has_comprehensive:
                print(
                    f"{'Branches':<8} {'Selector':<10} {'Geometric':<10} {'TM Score':<10} {'RMSD (Å)':<10} {'<2Å %':<8} {'Time (s)':<8} {'GPU':<4}"
                )
                print("-" * 90)
            else:
                print(
                    f"{'Branches':<10} {'Mean Score':<12} {'Improvement':<12} {'Time (s)':<10} {'Speedup':<10} {'GPU':<5}"
                )
                print("-" * 65)

            for result in sorted(method_results, key=lambda x: x["num_branches"]):
                branches = result["num_branches"]
                mean_score = result["mean_score"]
                improvement = (
                    ((mean_score - baseline_score) / baseline_score * 100)
                    if not np.isnan(mean_score)
                    else float("nan")
                )
                time_per_sample = result["time_per_sample"]
                speedup = (
                    baseline_time / time_per_sample
                    if time_per_sample > 0
                    else float("inf")
                )
                gpu_id = result["gpu_id"]

                if has_comprehensive and "mean_rmsd" in result:
                    mean_geometric = result.get("mean_geometric", float("nan"))
                    mean_tm = result.get("mean_tm_score", float("nan"))
                    mean_rmsd = result["mean_rmsd"]
                    rmsd_2a = result.get("mean_rmsd_lt_2A_percent", float("nan"))

                    print(
                        f"{branches:<8} {mean_score:<10.4f} {mean_geometric:<10.4f} {mean_tm:<10.4f} {mean_rmsd:<10.3f} {rmsd_2a:<8.1f} {time_per_sample:<8.2f} {gpu_id:<4}"
                    )
                else:
                    print(
                        f"{branches:<10} {mean_score:<12.4f} {improvement:<12.2f}% {time_per_sample:<10.2f} {speedup:<10.2f}x {gpu_id:<5}"
                    )
            print()

        # Find best overall result
        best_result = max(
            self.results,
            key=lambda x: (
                x["mean_score"] if not np.isnan(x["mean_score"]) else float("-inf")
            ),
        )
        improvement = (
            (best_result["mean_score"] - baseline_score) / baseline_score * 100
        )

        print(f"BEST RESULT:")
        print(
            f"Method: {best_result['method']} (branches: {best_result['num_branches']})"
        )
        print(
            f"Selector Score: {best_result['mean_score']:.4f}±{best_result['std_score']:.4f}"
        )
        print(f"Improvement: {improvement:.2f}% over baseline")

        # Show all three metric types if available
        if "mean_geometric" in best_result and not np.isnan(
            best_result["mean_geometric"]
        ):
            print(
                f"Geometric Score: {best_result['mean_geometric']:.4f}±{best_result['std_geometric']:.4f}"
            )

        if "mean_tm_score" in best_result and not np.isnan(
            best_result["mean_tm_score"]
        ):
            print(
                f"TM Score: {best_result['mean_tm_score']:.4f}±{best_result['std_tm_score']:.4f}"
            )

        if "mean_rmsd" in best_result and not np.isnan(best_result["mean_rmsd"]):
            print(
                f"RMSD: {best_result['mean_rmsd']:.3f}±{best_result['std_rmsd']:.3f}Å"
            )
            print(
                f"Designability: <2Å={best_result.get('mean_rmsd_lt_2A_percent', 0):.1f}%, "
                f"<1.5Å={best_result.get('mean_rmsd_lt_1_5A_percent', 0):.1f}%, "
                f"<1Å={best_result.get('mean_rmsd_lt_1A_percent', 0):.1f}%"
            )

        print(f"Time per sample: {best_result['time_per_sample']:.2f}s")
        print(f"GPU: {best_result['gpu_id']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-GPU inference scaling experiments"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate per method",
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=100,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--scoring_function",
        type=str,
        default="tm_score",
        choices=["tm_score", "rmsd", "geometric", "tm_score_4seq", "dual_score"],
        help="Scoring function to use for evaluation",
    )

    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.2,
        help="Noise scale for SDE path exploration",
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0.3,
        help="Lambda for divergence-free vector fields",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=9,
        help="Number of rounds for noise search methods",
    )

    parser.add_argument(
        "--particle_repulsion_factor",
        type=float,
        default=0,
        help="Particle repulsion factor for divergence-free max methods",
    )

    parser.add_argument(
        "--noise_schedule_end_factor",
        type=float,
        default=0.7,
        help="Noise schedule end factor for divergence-free max methods",
    )

    parser.add_argument(
        "--massage_steps",
        type=int,
        default=0,
        help="Number of massage steps for sample cleanup",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of GPU IDs to use for concurrent experiments (default: [2, 3, 4, 5])",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs",
    )

    parser.add_argument(
        "--branch_counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of branch counts to use for experiments (default: [2, 4, 8])",
    )

    args = parser.parse_args()

    print("Starting Multi-GPU Inference Scaling Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Noise scale (SDE): {args.noise_scale}")
    print(f"  Lambda div: {args.lambda_div}")
    print(f"  Num rounds: {args.num_rounds}")
    print(f"  Particle repulsion factor: {args.particle_repulsion_factor}")
    print(f"  Noise schedule end factor: {args.noise_schedule_end_factor}")
    print(f"  Massage steps: {args.massage_steps}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print(f"  Branch counts: {args.branch_counts}")
    print()

    # Create and run experiments
    runner = MultiGPUExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple methods validation experiment runner.

This script runs focused experiments to validate the simple inference methods:
- Standard sampling (baseline)
- SDE simple sampling (with noise injection)
- DivFree Max simple sampling (with divergence-free max noise)

The goal is to ensure that noise injection methods do not reduce overall quality
compared to standard sampling while potentially providing other benefits.
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


class SimpleMethodsRunner:
    """Multi-GPU runner for simple inference methods validation experiments."""

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
        self.base_conf.inference.samples.samples_per_length = 1
        self.base_conf.inference.samples.min_length = self.args.sample_length
        self.base_conf.inference.samples.max_length = self.args.sample_length
        self.base_conf.inference.samples.length_step = 1

        # Create output directory for experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "experiments", f"simple_methods_validation_multi_gpu_{timestamp}"
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
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_gpu_{gpu_id}"
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

        # Setup logging for this worker (if not already set up above)
        if not "worker_logger" in locals():
            worker_logger = logging.getLogger(f"{__name__}.worker.{gpu_id}")
        worker_logger.info(f"Running experiment: {method_name} on GPU {gpu_id}")

        # Create sampler
        sampler = self.create_sampler(method_config, gpu_id)

        try:
            # Track timing and results
            start_time = time.time()
            tm_scores = []
            rmsd_scores = []
            sc_times = []  # Track self-consistency evaluation times

            for sample_idx in range(self.args.num_samples):
                worker_logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

                try:
                    # Generate sample
                    sample_start = time.time()
                    sample_result = sampler.inference_method.sample(
                        self.args.sample_length
                    )
                    sample_time = time.time() - sample_start

                    # Extract sample
                    sample_output = sample_result

                    # Calculate self-consistency scores (TM-score and RMSD)
                    sc_start = time.time()
                    dual_scores = sampler.inference_method._dual_score_function(
                        sample_output, self.args.sample_length
                    )
                    sc_time = time.time() - sc_start

                    tm_score = dual_scores["tm_score"]
                    rmsd_score = dual_scores["rmsd"]

                    tm_scores.append(tm_score)
                    rmsd_scores.append(rmsd_score)
                    sc_times.append(sc_time)

                    worker_logger.info(
                        f"    TM-Score: {tm_score:.4f}, RMSD: {rmsd_score:.4f}, "
                        f"Sample Time: {sample_time:.2f}s, SC Time: {sc_time:.2f}s"
                    )

                except Exception as e:
                    worker_logger.error(f"Error in sample {sample_idx}: {e}")
                    tm_scores.append(float("nan"))
                    rmsd_scores.append(float("nan"))
                    sc_times.append(float("nan"))

            total_time = time.time() - start_time

            # Calculate statistics for TM-scores
            valid_tm_scores = [s for s in tm_scores if not np.isnan(s)]
            if valid_tm_scores:
                mean_tm_score = np.mean(valid_tm_scores)
                std_tm_score = np.std(valid_tm_scores)
                max_tm_score = np.max(valid_tm_scores)
                min_tm_score = np.min(valid_tm_scores)
                median_tm_score = np.median(valid_tm_scores)
            else:
                mean_tm_score = std_tm_score = max_tm_score = min_tm_score = (
                    median_tm_score
                ) = float("nan")

            # Calculate statistics for RMSD scores
            valid_rmsd_scores = [s for s in rmsd_scores if not np.isnan(s)]
            if valid_rmsd_scores:
                mean_rmsd_score = np.mean(valid_rmsd_scores)
                std_rmsd_score = np.std(valid_rmsd_scores)
                max_rmsd_score = np.max(valid_rmsd_scores)
                min_rmsd_score = np.min(valid_rmsd_scores)
                median_rmsd_score = np.median(valid_rmsd_scores)

                # Calculate designability metrics (percentage of samples below RMSD thresholds)
                designability_2 = np.mean(np.array(valid_rmsd_scores) < 2.0) * 100
                designability_1_5 = np.mean(np.array(valid_rmsd_scores) < 1.5) * 100
                designability_1 = np.mean(np.array(valid_rmsd_scores) < 1.0) * 100
            else:
                mean_rmsd_score = std_rmsd_score = max_rmsd_score = min_rmsd_score = (
                    median_rmsd_score
                ) = float("nan")
                designability_2 = designability_1_5 = designability_1 = float("nan")

            # Calculate timing statistics
            valid_sc_times = [t for t in sc_times if not np.isnan(t)]
            mean_sc_time = np.mean(valid_sc_times) if valid_sc_times else float("nan")

            result = {
                "method": method_name,
                "config": config,
                "num_samples": len(valid_tm_scores),
                "sample_length": self.args.sample_length,
                # TM-score metrics
                "mean_tm_score": mean_tm_score,
                "std_tm_score": std_tm_score,
                "max_tm_score": max_tm_score,
                "min_tm_score": min_tm_score,
                "median_tm_score": median_tm_score,
                "tm_scores": tm_scores,
                # RMSD metrics
                "mean_rmsd_score": mean_rmsd_score,
                "std_rmsd_score": std_rmsd_score,
                "max_rmsd_score": max_rmsd_score,
                "min_rmsd_score": min_rmsd_score,
                "median_rmsd_score": median_rmsd_score,
                "rmsd_scores": rmsd_scores,
                # Designability metrics
                "designability_2": designability_2,
                "designability_1_5": designability_1_5,
                "designability_1": designability_1,
                # Timing
                "total_time": total_time,
                "time_per_sample": total_time / self.args.num_samples,
                "mean_sc_time": mean_sc_time,
                "gpu_id": gpu_id,
            }

            worker_logger.info(
                f"  Results: TM={mean_tm_score:.4f}±{std_tm_score:.4f}, "
                f"RMSD={mean_rmsd_score:.4f}±{std_rmsd_score:.4f}, "
                f"Time={total_time:.2f}s"
            )

            # Put result in queue
            result_queue.put(result)
            return result

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def run_validation_experiments(self):
        """Run validation experiments for simple methods using multiple GPUs."""
        experiments = []

        # 1. Standard sampling (baseline)
        experiments.append({"method": "standard", "config": {}})

        # 2. SDE Simple sampling
        experiments.append(
            {
                "method": "sde_simple",
                "config": {
                    "noise_scale": self.args.sde_noise_scale,
                    "massage_steps": self.args.massage_steps,
                },
            }
        )

        # 3. DivFree Max Simple sampling
        experiments.append(
            {
                "method": "divfree_max_simple",
                "config": {
                    "lambda_div": self.args.lambda_div,
                    "particle_repulsion_factor": self.args.particle_repulsion_factor,
                    "noise_schedule_end_factor": self.args.noise_schedule_end_factor,
                    "massage_steps": self.args.massage_steps,
                },
            }
        )

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
            summary_data.append(
                {
                    "method": result["method"],
                    "num_samples": result["num_samples"],
                    "sample_length": result["sample_length"],
                    # TM-score metrics
                    "mean_tm_score": result["mean_tm_score"],
                    "std_tm_score": result["std_tm_score"],
                    "max_tm_score": result["max_tm_score"],
                    "min_tm_score": result["min_tm_score"],
                    "median_tm_score": result["median_tm_score"],
                    # RMSD metrics
                    "mean_rmsd_score": result["mean_rmsd_score"],
                    "std_rmsd_score": result["std_rmsd_score"],
                    "max_rmsd_score": result["max_rmsd_score"],
                    "min_rmsd_score": result["min_rmsd_score"],
                    "median_rmsd_score": result["median_rmsd_score"],
                    # Designability metrics
                    "designability_2": result["designability_2"],
                    "designability_1_5": result["designability_1_5"],
                    "designability_1": result["designability_1"],
                    # Timing
                    "total_time": result["total_time"],
                    "time_per_sample": result["time_per_sample"],
                    "mean_sc_time": result["mean_sc_time"],
                    "gpu_id": result["gpu_id"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print validation results."""
        print("\n" + "=" * 80)
        print("SIMPLE METHODS VALIDATION RESULTS")
        print("=" * 80)

        # Find baseline (standard method)
        baseline = None
        for result in self.results:
            if result["method"] == "standard":
                baseline = result
                break

        if baseline is None:
            print("Error: No baseline (standard) method found!")
            return

        baseline_tm_score = baseline["mean_tm_score"]
        baseline_rmsd_score = baseline["mean_rmsd_score"]
        baseline_time = baseline["time_per_sample"]

        print(f"Experimental Setup:")
        print(f"  Sample Length: {self.args.sample_length}")
        print(f"  Samples per Method: {self.args.num_samples}")
        print(f"  GPUs Used: {self.available_gpus}")
        print(f"  SDE Noise Scale: {self.args.sde_noise_scale}")
        print(f"  DivFree Lambda: {self.args.lambda_div}")
        print(f"  Particle Repulsion Factor: {self.args.particle_repulsion_factor}")
        print(f"  Noise Schedule End Factor: {self.args.noise_schedule_end_factor}")
        print()

        print(f"Baseline (Standard Sampling):")
        print(f"  TM-Score: {baseline_tm_score:.4f}±{baseline['std_tm_score']:.4f}")
        print(f"  RMSD: {baseline_rmsd_score:.4f}±{baseline['std_rmsd_score']:.4f}")
        print(
            f"  Designability: <2Å={baseline['designability_2']:.1f}%, "
            f"<1.5Å={baseline['designability_1_5']:.1f}%, <1Å={baseline['designability_1']:.1f}%"
        )
        print(f"  Time per sample: {baseline_time:.2f}s")
        print()

        # Print detailed comparison table
        print(f"METHOD COMPARISON:")
        print(
            f"{'Method':<20} {'TM-Score':<13} {'TM-Diff':<10} {'RMSD':<10} {'RMSD-Diff':<12} {'<2Å%':<8} {'<1.5Å%':<8} {'<1Å%':<8} {'Time(s)':<8} {'GPU':<5}"
        )
        print("-" * 105)

        for result in self.results:
            method = result["method"]
            tm_score = result["mean_tm_score"]
            rmsd_score = result["mean_rmsd_score"]

            # Calculate differences from baseline
            tm_diff = (
                tm_score - baseline_tm_score if not np.isnan(tm_score) else float("nan")
            )
            rmsd_diff = (
                rmsd_score - baseline_rmsd_score
                if not np.isnan(rmsd_score)
                else float("nan")
            )

            time_per_sample = result["time_per_sample"]
            designability_2 = result["designability_2"]
            designability_1_5 = result["designability_1_5"]
            designability_1 = result["designability_1"]
            gpu_id = result["gpu_id"]

            print(
                f"{method:<20} {tm_score:<13.4f} {tm_diff:<+10.4f} {rmsd_score:<10.4f} {rmsd_diff:<+12.4f} "
                f"{designability_2:<8.1f} {designability_1_5:<8.1f} {designability_1:<8.1f} {time_per_sample:<8.2f} {gpu_id:<5}"
            )

        print()

        # Quality preservation analysis
        print("QUALITY PRESERVATION ANALYSIS:")
        print("-" * 50)

        for result in self.results:
            if result["method"] == "standard":
                continue

            method = result["method"]
            tm_score = result["mean_tm_score"]
            rmsd_score = result["mean_rmsd_score"]

            # Statistical significance tests would be ideal here, but for now use practical thresholds
            tm_diff = tm_score - baseline_tm_score
            rmsd_diff = rmsd_score - baseline_rmsd_score

            # Define acceptable quality thresholds (method should not degrade quality significantly)
            tm_threshold = -0.02  # Allow up to 2% decrease in TM-score
            rmsd_threshold = 0.2  # Allow up to 0.2Å increase in RMSD

            tm_preserved = tm_diff >= tm_threshold
            rmsd_preserved = rmsd_diff <= rmsd_threshold

            print(f"{method}:")
            print(
                f"  TM-Score: {tm_score:.4f} vs {baseline_tm_score:.4f} (Δ={tm_diff:+.4f}) - {'✓ PRESERVED' if tm_preserved else '✗ DEGRADED'}"
            )
            print(
                f"  RMSD: {rmsd_score:.4f} vs {baseline_rmsd_score:.4f} (Δ={rmsd_diff:+.4f}) - {'✓ PRESERVED' if rmsd_preserved else '✗ DEGRADED'}"
            )

            overall_quality = (
                "✓ QUALITY PRESERVED"
                if (tm_preserved and rmsd_preserved)
                else "✗ QUALITY DEGRADED"
            )
            print(f"  Overall: {overall_quality}")
            print()

        # Summary recommendations
        print("SUMMARY & RECOMMENDATIONS:")
        print("-" * 30)

        quality_methods = []
        for result in self.results:
            if result["method"] == "standard":
                continue

            method = result["method"]
            tm_diff = result["mean_tm_score"] - baseline_tm_score
            rmsd_diff = result["mean_rmsd_score"] - baseline_rmsd_score

            if tm_diff >= -0.02 and rmsd_diff <= 0.2:
                quality_methods.append(method)

        if quality_methods:
            print(f"✓ Methods that preserve quality: {', '.join(quality_methods)}")
            print("  These methods can be safely used without quality degradation.")
        else:
            print("✗ No methods fully preserve baseline quality.")
            print("  Consider parameter tuning or method refinement.")

        print()
        print("Note: Quality preservation thresholds:")
        print("  TM-Score: degradation < 0.02")
        print("  RMSD: increase < 0.2Å")


def main():
    parser = argparse.ArgumentParser(description="Validate simple inference methods")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of samples to generate per method (default: 32)",
    )

    parser.add_argument(
        "--sample_length",
        type=int,
        default=100,
        help="Length of protein samples to generate (default: 100)",
    )

    parser.add_argument(
        "--sde_noise_scale",
        type=float,
        default=0.2,
        help="Noise scale for SDE simple method (default: 0.05)",
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0,
        help="Lambda for divergence-free max method (default: 0.2)",
    )

    parser.add_argument(
        "--particle_repulsion_factor",
        type=float,
        default=0,
        help="Particle repulsion factor for DivFree Max (default: 0.02)",
    )

    parser.add_argument(
        "--noise_schedule_end_factor",
        type=float,
        default=0.7,
        help="Noise schedule end factor for DivFree Max (default: 0.7)",
    )

    parser.add_argument(
        "--massage_steps",
        type=int,
        default=0,
        help="Number of massaging steps to clean up noisy samples (default: 3, 0 to disable)",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of GPU IDs to use for concurrent experiments (default: [1, 2, 3])",
    )

    args = parser.parse_args()

    print("Simple Methods Validation Experiment")
    print("=" * 40)
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  SDE noise scale: {args.sde_noise_scale}")
    print(f"  DivFree lambda: {args.lambda_div}")
    print(f"  Particle repulsion factor: {args.particle_repulsion_factor}")
    print(f"  Noise schedule end factor: {args.noise_schedule_end_factor}")
    print(f"  Massage steps: {args.massage_steps}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print()

    # Create and run experiments
    runner = SimpleMethodsRunner(args)
    runner.run_validation_experiments()

    print("Validation experiments completed!")


if __name__ == "__main__":
    main()

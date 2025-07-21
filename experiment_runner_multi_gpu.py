#!/usr/bin/env python3
"""
Multi-GPU experiment runner for comparing inference scaling methods in protein design.

This script runs experiments concurrently on multiple GPUs to compare different inference methods:
- Standard sampling (baseline)
- Best-of-N sampling
- SDE path exploration
- Divergence-free ODE path exploration

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

                    scores.append(score)
                    worker_logger.info(
                        f"    Score: {score:.4f}, Time: {sample_time:.2f}s"
                    )

                except Exception as e:
                    worker_logger.error(f"Error in sample {sample_idx}: {e}")
                    scores.append(float("nan"))

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
            }

            worker_logger.info(
                f"  Results: Mean={mean_score:.4f}±{std_score:.4f}, Time={total_time:.2f}s"
            )

            # Put result in queue
            result_queue.put(result)
            return result

        finally:
            # Cleanup sampler
            self.cleanup_sampler(sampler, method_name)

    def run_all_experiments(self):
        """Run all experiments with different methods and branch counts using multiple GPUs."""
        experiments = []

        # 1. Baseline: Standard sampling
        experiments.append({"method": "standard", "config": {}})

        # Random Search + Divergence-free ODE with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "random_search_divfree",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # SDE path exploration with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "sde_path_exploration",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "noise_scale": self.args.noise_scale,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # Divergence-free ODE with different branch counts
        for n_branches in self.args.branch_counts:
            experiments.append(
                {
                    "method": "divergence_free_ode",
                    "config": {
                        "num_branches": n_branches,
                        "num_keep": 1,  # Always keep only 1 as specified
                        "lambda_div": self.args.lambda_div,
                        "selector": self.args.scoring_function,
                        "branch_start_time": 0.0,
                        "branch_interval": self.args.branch_interval,
                    },
                }
            )

        # Best-of-N with different N values
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
            )

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
        print(f"Score: {best_result['mean_score']:.4f}±{best_result['std_score']:.4f}")
        print(f"Improvement: {improvement:.2f}% over baseline")
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
        default=50,
        help="Length of protein samples to generate",
    )

    parser.add_argument(
        "--scoring_function",
        type=str,
        default="tm_score",
        choices=["tm_score", "rmsd"],
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
        default=0.5,
        help="Lambda for divergence-free vector fields",
    )

    parser.add_argument(
        "--branch_interval",
        type=float,
        default=0.1,
        help="Time interval between branches (0.0 = every timestep, 0.1 = every 0.1 time units)",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of GPU IDs to use for concurrent experiments (default: [0, 1, 2])",
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
        default=[2, 4, 8],
        help="List of branch counts to use for experiments (default: [2, 4, 8])",
    )

    args = parser.parse_args()

    print("Starting Multi-GPU Inference Scaling Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  Noise scale (SDE): {args.noise_scale}")
    print(f"  Lambda div (ODE): {args.lambda_div}")
    print(f"  Branch interval: {args.branch_interval}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print(f"  Branch counts: {args.branch_counts}")
    print()

    # Create and run experiments
    runner = MultiGPUExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

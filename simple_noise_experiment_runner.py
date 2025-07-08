#!/usr/bin/env python3
"""
Simple noise experiment runner for testing SDE and divergence-free methods without branching.

This script tests whether adding noise during sampling (without branching) can improve
protein generation quality. It compares:
- Standard sampling (baseline)
- Simple SDE sampling (adds Gaussian noise at each step)
- Simple divergence-free sampling (adds divergence-free noise at each step)

Each method is tested with different noise levels to find optimal parameters.
The script also measures the actual relative noise injection (noise magnitude / velocity magnitude).
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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runner.inference_methods import get_inference_method
from runner.inference import Sampler
from omegaconf import DictConfig, OmegaConf


class NoiseAnalyzingSDEInference:
    """SDE inference that measures relative noise injection."""

    def __init__(self, sampler, config: Dict[str, Any]):
        self.sampler = sampler
        self.config = config
        self.noise_scale = config.get("noise_scale", 0.05)
        self._log = logging.getLogger(__name__)

        # Track noise statistics
        self.noise_measurements = []

    def sample(self, sample_length: int, context=None) -> Dict[str, Any]:
        """Generate sample with noise measurement."""
        # Initialize features (same as SDESimpleInference)
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)
        aatype = torch.zeros(sample_length, dtype=torch.int32)
        chain_idx = torch.zeros_like(aatype)

        ref_sample = self.sampler.flow_matcher.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length + 1)

        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            "aatype": aatype,
            "chain_idx": chain_idx,
            **ref_sample,
        }

        # Add batch dimension and move to GPU
        import tree

        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        return self._sde_with_noise_measurement(init_feats, context)

    def _sde_with_noise_measurement(self, data_init, context):
        """SDE sampling with noise measurement."""
        import tree
        import copy
        from foldflow.data import utils as du
        from foldflow.data import all_atom
        from openfold.utils import rigid_utils as ru

        sample_feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else x.copy(), data_init
        )
        device = sample_feats["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Initialize trajectory collection
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        # Reset noise measurements
        self.noise_measurements = []

        with torch.no_grad():
            for step_idx, t in enumerate(reverse_steps):
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                # Measure original velocity magnitudes
                rot_velocity_mag = torch.norm(rot_vectorfield, dim=-1).mean()
                trans_velocity_mag = torch.norm(trans_vectorfield, dim=-1).mean()

                # Generate SDE noise (Euler-Maruyama)
                noise_rot = (
                    torch.randn_like(rot_vectorfield) * self.noise_scale * np.sqrt(dt)
                )
                noise_trans = (
                    torch.randn_like(trans_vectorfield) * self.noise_scale * np.sqrt(dt)
                )

                # Measure noise magnitudes
                rot_noise_mag = torch.norm(noise_rot, dim=-1).mean()
                trans_noise_mag = torch.norm(noise_trans, dim=-1).mean()

                # Calculate overall relative noise for this step
                total_velocity_mag = rot_velocity_mag + trans_velocity_mag
                total_noise_mag = rot_noise_mag + trans_noise_mag
                relative_noise = (
                    (total_noise_mag / total_velocity_mag).item()
                    if total_velocity_mag > 0
                    else 0
                )

                # Store measurements
                self.noise_measurements.append(
                    {
                        "step": step_idx,
                        "time": t,
                        "rot_velocity_mag": rot_velocity_mag.item(),
                        "trans_velocity_mag": trans_velocity_mag.item(),
                        "rot_noise_mag": rot_noise_mag.item(),
                        "trans_noise_mag": trans_noise_mag.item(),
                        "relative_noise": relative_noise,
                        "noise_scale": self.noise_scale,
                        "dt": dt,
                    }
                )

                # Apply noise to vector fields
                rot_vectorfield_noisy = rot_vectorfield + noise_rot
                trans_vectorfield_noisy = trans_vectorfield + noise_trans

                rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield_noisy),
                    trans_vectorfield=du.move_to_np(trans_vectorfield_noisy),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=True,
                    noise_scale=1.0,  # Already applied noise above
                )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

                # Collect trajectory data
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )

                atom37_0 = all_atom.compute_backbone(
                    ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                )[0]
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))

                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))
                final_psi_pred = psi_pred

        # Flip trajectory so that it starts from t=0
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        sample_result = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }

        # Remove batch dimension
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
        )

    def get_score_function(self, selector: str = "tm_score"):
        """Get the scoring function based on selector."""
        from runner.inference_methods import InferenceMethod

        # Create a temporary inference method instance to access scoring functions
        base_method = InferenceMethod.__new__(InferenceMethod)
        base_method.sampler = self.sampler
        base_method.config = self.config
        base_method._log = self._log

        if selector == "tm_score":
            return base_method._tm_score_function
        elif selector == "rmsd":
            return base_method._rmsd_function
        else:
            raise ValueError(f"Unknown selector: {selector}")


class NoiseAnalyzingDivFreeInference:
    """Divergence-free inference that measures relative noise injection."""

    def __init__(self, sampler, config: Dict[str, Any]):
        self.sampler = sampler
        self.config = config
        self.lambda_div = config.get("lambda_div", 0.2)
        self._log = logging.getLogger(__name__)

        # Track noise statistics
        self.noise_measurements = []

    def sample(self, sample_length: int, context=None) -> Dict[str, Any]:
        """Generate sample with noise measurement."""
        # Initialize features (same as DivergenceFreeSimpleInference)
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)
        aatype = torch.zeros(sample_length, dtype=torch.int32)
        chain_idx = torch.zeros_like(aatype)

        ref_sample = self.sampler.flow_matcher.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length + 1)

        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            "aatype": aatype,
            "chain_idx": chain_idx,
            **ref_sample,
        }

        # Add batch dimension and move to GPU
        import tree

        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        return self._divfree_with_noise_measurement(init_feats, context)

    def _divfree_with_noise_measurement(self, data_init, context):
        """Divergence-free sampling with noise measurement."""
        import tree
        import copy
        from foldflow.data import utils as du
        from foldflow.data import all_atom
        from openfold.utils import rigid_utils as ru
        from runner.divergence_free_utils import divfree_swirl_si

        sample_feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else x.copy(), data_init
        )
        device = sample_feats["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Initialize trajectory collection
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        # Reset noise measurements
        self.noise_measurements = []

        with torch.no_grad():
            for step_idx, t in enumerate(reverse_steps):
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                # Measure original velocity magnitudes
                rot_velocity_mag = torch.norm(rot_vectorfield, dim=-1).mean()
                trans_velocity_mag = torch.norm(trans_vectorfield, dim=-1).mean()

                # Generate divergence-free noise
                rigids_tensor = sample_feats["rigids_t"]
                t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                # Extract rotation matrices and translations for divfree noise
                rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                rot_mats = rigid_obj.get_rots().get_rot_mats()  # [B, N, 3, 3]
                trans_vecs = rigid_obj.get_trans()  # [B, N, 3]

                # Generate divergence-free noise
                rot_divfree_noise = divfree_swirl_si(
                    rot_mats, t_batch, None, rot_vectorfield
                )
                trans_divfree_noise = divfree_swirl_si(
                    trans_vecs, t_batch, None, trans_vectorfield
                )

                # Apply lambda scaling
                noise_rot = rot_divfree_noise * self.lambda_div
                noise_trans = trans_divfree_noise * self.lambda_div

                # Measure noise magnitudes
                rot_noise_mag = torch.norm(noise_rot, dim=-1).mean()
                trans_noise_mag = torch.norm(noise_trans, dim=-1).mean()

                # Calculate overall relative noise for this step
                total_velocity_mag = rot_velocity_mag + trans_velocity_mag
                total_noise_mag = rot_noise_mag + trans_noise_mag
                relative_noise = (
                    (total_noise_mag / total_velocity_mag).item()
                    if total_velocity_mag > 0
                    else 0
                )

                # Store measurements
                self.noise_measurements.append(
                    {
                        "step": step_idx,
                        "time": t,
                        "rot_velocity_mag": rot_velocity_mag.item(),
                        "trans_velocity_mag": trans_velocity_mag.item(),
                        "rot_noise_mag": rot_noise_mag.item(),
                        "trans_noise_mag": trans_noise_mag.item(),
                        "relative_noise": relative_noise,
                        "lambda_div": self.lambda_div,
                        "dt": dt,
                    }
                )

                # Apply noise to vector fields
                rot_vectorfield_noisy = rot_vectorfield + noise_rot
                trans_vectorfield_noisy = trans_vectorfield + noise_trans

                rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield_noisy),
                    trans_vectorfield=du.move_to_np(trans_vectorfield_noisy),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=True,
                    noise_scale=1.0,  # No additional noise scaling
                )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

                # Collect trajectory data
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )

                atom37_0 = all_atom.compute_backbone(
                    ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                )[0]
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))

                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))
                final_psi_pred = psi_pred

        # Flip trajectory so that it starts from t=0
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        sample_result = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }

        # Remove batch dimension
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
        )

    def get_score_function(self, selector: str = "tm_score"):
        """Get the scoring function based on selector."""
        from runner.inference_methods import InferenceMethod

        # Create a temporary inference method instance to access scoring functions
        base_method = InferenceMethod.__new__(InferenceMethod)
        base_method.sampler = self.sampler
        base_method.config = self.config
        base_method._log = self._log

        if selector == "tm_score":
            return base_method._tm_score_function
        elif selector == "rmsd":
            return base_method._rmsd_function
        else:
            raise ValueError(f"Unknown selector: {selector}")


class SimpleNoiseExperimentRunner:
    """Runner for simple noise experiments."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.setup_logging()
        self.setup_config()

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

        # Override GPU if specified
        if self.args.gpu_id is not None:
            self.base_conf.inference.gpu_id = self.args.gpu_id
            self.logger.info(f"Using GPU {self.args.gpu_id}")

        # Create output directory for experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join("experiments", f"simple_noise_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def create_sampler(self, method_config: Dict[str, Any]) -> Sampler:
        """Create a sampler with specific method configuration."""
        conf = OmegaConf.create(self.base_conf)

        # Update method configuration
        conf.inference.samples.inference_method = method_config["method"]
        conf.inference.samples.method_config = method_config.get("config", {})

        # Set output directory for this experiment
        method_name = method_config["method"]
        noise_param = method_config.get("config", {}).get(
            "noise_scale", method_config.get("config", {}).get("lambda_div", 0)
        )
        conf.inference.output_dir = os.path.join(
            self.experiment_dir, f"{method_name}_noise_{noise_param}"
        )

        sampler = Sampler(conf)

        # Replace the inference method with our noise-analyzing version if needed
        if method_config["method"] == "sde_simple":
            sampler.inference_method = NoiseAnalyzingSDEInference(
                sampler, method_config.get("config", {})
            )
        elif method_config["method"] == "divergence_free_simple":
            sampler.inference_method = NoiseAnalyzingDivFreeInference(
                sampler, method_config.get("config", {})
            )

        return sampler

    def run_single_experiment(self, method_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given method configuration."""
        method_name = method_config["method"]
        config = method_config.get("config", {})
        noise_param = config.get("noise_scale", config.get("lambda_div", 0))

        self.logger.info(
            f"Running experiment: {method_name} with noise parameter {noise_param}"
        )

        # Create sampler
        sampler = self.create_sampler(method_config)

        try:
            # Track timing and results
            start_time = time.time()
            scores = []
            all_noise_measurements = []

            for sample_idx in range(self.args.num_samples):
                self.logger.info(f"  Sample {sample_idx + 1}/{self.args.num_samples}")

                try:
                    # Generate sample
                    sample_start = time.time()
                    sample_result = sampler.inference_method.sample(
                        self.args.sample_length
                    )
                    sample_time = time.time() - sample_start

                    # Collect noise measurements if available
                    if hasattr(sampler.inference_method, "noise_measurements"):
                        all_noise_measurements.extend(
                            sampler.inference_method.noise_measurements
                        )

                    # Extract sample and score
                    if isinstance(sample_result, dict) and "sample" in sample_result:
                        sample_output = sample_result["sample"]
                        if "score" in sample_result:
                            score = sample_result["score"]
                        else:
                            score = sampler.inference_method.get_score_function(
                                self.args.scoring_function
                            )(sample_output, self.args.sample_length)
                    else:
                        sample_output = sample_result
                        score = sampler.inference_method.get_score_function(
                            self.args.scoring_function
                        )(sample_output, self.args.sample_length)

                    scores.append(score)
                    self.logger.info(
                        f"    Score: {score:.4f}, Time: {sample_time:.2f}s"
                    )

                except Exception as e:
                    self.logger.error(f"Error in sample {sample_idx}: {e}")
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

            # Analyze noise measurements
            noise_stats = self._analyze_noise_measurements(
                all_noise_measurements, method_name
            )

            result = {
                "method": method_name,
                "noise_parameter": noise_param,
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
                "noise_stats": noise_stats,
                "noise_measurements": all_noise_measurements,  # Raw data for detailed analysis
            }

            self.logger.info(
                f"  Results: Mean={mean_score:.4f}±{std_score:.4f}, Time={total_time:.2f}s"
            )

            # Log noise statistics
            if noise_stats:
                self.logger.info(f"  Noise Statistics:")
                for key, value in noise_stats.items():
                    if isinstance(value, float):
                        self.logger.info(f"    {key}: {value:.4f}")
                    else:
                        self.logger.info(f"    {key}: {value}")

            return result

        finally:
            # Explicit cleanup of sampler to prevent memory leaks
            try:
                self.logger.info(f"Cleaning up sampler for {method_name}")

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
                    if (
                        hasattr(sampler.exp, "_model")
                        and sampler.exp._model is not None
                    ):
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

    def _analyze_noise_measurements(
        self, measurements: List[Dict], method_name: str
    ) -> Dict[str, Any]:
        """Analyze noise measurements and return statistics."""
        if not measurements:
            return {}

        # Calculate overall relative noise for each timestep
        overall_relative_noise_per_step = []
        velocity_mags = []
        noise_mags = []

        for m in measurements:
            # Combine rotation and translation components
            total_velocity_mag = m["rot_velocity_mag"] + m["trans_velocity_mag"]
            total_noise_mag = m["rot_noise_mag"] + m["trans_noise_mag"]

            if total_velocity_mag > 0:
                relative_noise = total_noise_mag / total_velocity_mag
                overall_relative_noise_per_step.append(relative_noise)
                velocity_mags.append(total_velocity_mag)
                noise_mags.append(total_noise_mag)

        if not overall_relative_noise_per_step:
            return {}

        # Calculate statistics
        stats = {
            "num_measurements": len(measurements),
            "mean_relative_noise": np.mean(overall_relative_noise_per_step),
            "std_relative_noise": np.std(overall_relative_noise_per_step),
            "mean_velocity_mag": np.mean(velocity_mags),
            "mean_noise_mag": np.mean(noise_mags),
        }

        # Add method-specific parameters
        if measurements:
            first_measurement = measurements[0]
            if "noise_scale" in first_measurement:
                stats["noise_scale_param"] = first_measurement["noise_scale"]
                stats["dt"] = first_measurement["dt"]
                stats["theoretical_sde_scaling"] = first_measurement[
                    "noise_scale"
                ] * np.sqrt(first_measurement["dt"])
            elif "lambda_div" in first_measurement:
                stats["lambda_div_param"] = first_measurement["lambda_div"]

        return stats

    def run_all_experiments(self):
        """Run all experiments with different methods and noise levels."""
        experiments = []

        # 1. Baseline: Standard sampling
        experiments.append({"method": "standard", "config": {}})

        # 2. Simple SDE with different noise scales
        for noise_scale in self.args.noise_scales:
            experiments.append(
                {
                    "method": "sde_simple",
                    "config": {
                        "noise_scale": noise_scale,
                    },
                }
            )

        # 3. Simple divergence-free with different lambda values
        for lambda_div in self.args.lambda_divs:
            experiments.append(
                {
                    "method": "divergence_free_simple",
                    "config": {
                        "lambda_div": lambda_div,
                    },
                }
            )

        # Run all experiments
        for i, exp_config in enumerate(experiments):
            self.logger.info(
                f"Starting experiment {i+1}/{len(experiments)}: {exp_config['method']}"
            )

            try:
                result = self.run_single_experiment(exp_config)
                self.results.append(result)

            except Exception as e:
                self.logger.error(f"Failed experiment {exp_config}: {e}")

            # Force garbage collection between experiments
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"After experiment {i+1} cleanup")

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
            noise_stats = result.get("noise_stats", {})
            summary_row = {
                "method": result["method"],
                "noise_parameter": result["noise_parameter"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "max_score": result["max_score"],
                "min_score": result["min_score"],
                "total_time": result["total_time"],
                "time_per_sample": result["time_per_sample"],
                "num_samples": result["num_samples"],
                "scoring_function": result["scoring_function"],
                "relative_noise": noise_stats.get("mean_relative_noise", None),
                "relative_noise_std": noise_stats.get("std_relative_noise", None),
                "velocity_magnitude": noise_stats.get("mean_velocity_mag", None),
                "noise_magnitude": noise_stats.get("mean_noise_mag", None),
            }

            # Add method-specific parameters
            if "noise_scale_param" in noise_stats:
                summary_row["theoretical_sde_scaling"] = noise_stats.get(
                    "theoretical_sde_scaling", None
                )
            elif "lambda_div_param" in noise_stats:
                summary_row["lambda_div_param"] = noise_stats.get(
                    "lambda_div_param", None
                )

            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.experiment_dir, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)

        # Save detailed noise measurements as separate CSV for analysis
        noise_data = []
        for result in self.results:
            measurements = result.get("noise_measurements", [])
            for m in measurements:
                noise_row = {
                    "method": result["method"],
                    "noise_parameter": result["noise_parameter"],
                    **m,  # Include all measurement data
                }
                noise_data.append(noise_row)

        if noise_data:
            noise_df = pd.DataFrame(noise_data)
            noise_file = os.path.join(self.experiment_dir, "noise_measurements.csv")
            noise_df.to_csv(noise_file, index=False)

        self.logger.info(f"Results saved to {self.experiment_dir}")

    def analyze_results(self):
        """Analyze and print experiment results."""
        print("\n" + "=" * 80)
        print("SIMPLE NOISE EXPERIMENT RESULTS")
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
            if method_name == "sde_simple":
                param_name = "Noise Scale"
            else:
                param_name = "Lambda Div"

            print(
                f"{param_name:<12} {'Mean Score':<12} {'Improvement':<12} {'Time (s)':<10} {'Speedup':<10} {'Rel. Noise':<12}"
            )
            print("-" * 80)

            for result in sorted(method_results, key=lambda x: x["noise_parameter"]):
                noise_param = result["noise_parameter"]
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

                # Extract noise statistics
                noise_stats = result.get("noise_stats", {})
                relative_noise = noise_stats.get("mean_relative_noise", float("nan"))

                print(
                    f"{noise_param:<12.3f} {mean_score:<12.4f} {improvement:<12.2f}% {time_per_sample:<10.2f} {speedup:<10.2f}x {relative_noise:<12.4f}"
                )

                # Show theoretical vs actual for SDE only
                if (
                    noise_stats
                    and "theoretical_sde_scaling" in noise_stats
                    and not np.isnan(relative_noise)
                ):
                    theoretical = noise_stats["theoretical_sde_scaling"]
                    ratio = (
                        relative_noise / theoretical
                        if theoretical > 0
                        else float("inf")
                    )
                    print(
                        f"   (Theoretical SDE: {theoretical:.4f}, Actual/Theory ratio: {ratio:.2f})"
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
            f"Method: {best_result['method']} (noise parameter: {best_result['noise_parameter']})"
        )
        print(f"Score: {best_result['mean_score']:.4f}±{best_result['std_score']:.4f}")
        print(f"Improvement: {improvement:.2f}% over baseline")
        print(f"Time per sample: {best_result['time_per_sample']:.2f}s")

        # Print best result noise statistics
        best_noise_stats = best_result.get("noise_stats", {})
        if best_noise_stats:
            print(f"Noise Analysis:")
            print(
                f"  Relative noise: {best_noise_stats.get('mean_relative_noise', 'N/A'):.4f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="Run simple noise experiments")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
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
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.02, 0.1, 0.4, 0.8, 1.6],
        help="List of noise scales to test for SDE method (default: [0.01, 0.02, 0.05, 0.1, 0.2])",
    )

    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.1, 0.4, 0.8, 1.6],
        help="List of lambda values to test for divergence-free method (default: [0.1, 0.2, 0.4, 0.8, 1.0])",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="GPU ID to use for inference (overrides config file)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs",
    )

    args = parser.parse_args()

    print("Starting Simple Noise Experiments")
    print(f"Parameters:")
    print(f"  Samples per method: {args.num_samples}")
    print(f"  Sample length: {args.sample_length}")
    print(f"  Scoring function: {args.scoring_function}")
    print(f"  SDE noise scales: {args.noise_scales}")
    print(f"  Divergence-free lambdas: {args.lambda_divs}")
    print(f"  GPU ID: {args.gpu_id}")
    print()

    # Create and run experiments
    runner = SimpleNoiseExperimentRunner(args)
    runner.run_all_experiments()

    print("Experiments completed!")


if __name__ == "__main__":
    main()

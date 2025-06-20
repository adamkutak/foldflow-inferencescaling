"""
Inference methods for protein design sampling.

This module contains various inference strategies including:
- Standard sampling
- Best-of-N sampling
- SDE path exploration with Euler-Maruyama
- Divergence-free ODE path exploration
"""

import logging
import os
import shutil
import numpy as np
import torch
import tree
import copy
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from foldflow.data import utils as du
from foldflow.data import residue_constants, all_atom
from openfold.utils import rigid_utils as ru
from tools.analysis import metrics
from runner.divergence_free_utils import divfree_swirl_si


class InferenceMethod(ABC):
    """Base class for inference methods."""

    def __init__(self, sampler, config: Dict[str, Any]):
        self.sampler = sampler
        self.config = config
        self._log = logging.getLogger(__name__)

    @abstractmethod
    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate a sample using this inference method."""
        pass

    def get_score_function(self, selector: str = "tm_score") -> Callable:
        """Get the scoring function based on selector."""
        if selector == "tm_score":
            return self._tm_score_function
        elif selector == "rmsd":
            return self._rmsd_function
        else:
            raise ValueError(f"Unknown selector: {selector}")

    def _tm_score_function(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> float:
        """Evaluate sample using TM-score."""
        self._log.debug(f"        _tm_score_function: Starting evaluation")
        self._log.debug(f"        _tm_score_function: sample_length={sample_length}")

        # Log trajectory shapes for debugging
        if "prot_traj" in sample_output:
            self._log.debug(
                f"        _tm_score_function: prot_traj shape = {sample_output['prot_traj'].shape}"
            )
        if "rigid_0_traj" in sample_output:
            self._log.debug(
                f"        _tm_score_function: rigid_0_traj shape = {sample_output['rigid_0_traj'].shape}"
            )

        # Create temporary directory for evaluation
        temp_dir = os.path.join(self.sampler._output_dir, "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save trajectory
            self._log.debug(
                f"        _tm_score_function: Saving trajectory to {temp_dir}"
            )
            traj_paths = self.sampler.save_traj(
                sample_output["prot_traj"],
                sample_output["rigid_0_traj"],
                np.ones(sample_length),
                output_dir=temp_dir,
            )

            # Run evaluation
            pdb_path = traj_paths["sample_path"]
            self._log.debug(f"        _tm_score_function: PDB saved to {pdb_path}")

            sc_output_dir = os.path.join(temp_dir, "self_consistency")
            os.makedirs(sc_output_dir, exist_ok=True)
            shutil.copy(
                pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path))
            )

            self._log.debug(
                f"        _tm_score_function: Running self-consistency evaluation"
            )
            sc_results = self.sampler.run_self_consistency(
                sc_output_dir, pdb_path, motif_mask=None
            )

            tm_score = sc_results["tm_score"].mean()
            self._log.debug(f"        _tm_score_function: TM-score = {tm_score:.4f}")

            return tm_score

        except Exception as e:
            self._log.error(f"        _tm_score_function: Error during evaluation: {e}")
            return float("-inf")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _rmsd_function(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> float:
        """Evaluate sample using RMSD (lower is better, so return negative)."""
        # Similar to TM-score but return negative RMSD
        temp_dir = os.path.join(self.sampler._output_dir, "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            traj_paths = self.sampler.save_traj(
                sample_output["prot_traj"],
                sample_output["rigid_0_traj"],
                np.ones(sample_length),
                output_dir=temp_dir,
            )

            pdb_path = traj_paths["sample_path"]
            sc_output_dir = os.path.join(temp_dir, "self_consistency")
            os.makedirs(sc_output_dir, exist_ok=True)
            shutil.copy(
                pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path))
            )

            sc_results = self.sampler.run_self_consistency(
                sc_output_dir, pdb_path, motif_mask=None
            )

            return -sc_results["rmsd"].mean()  # Negative because lower RMSD is better

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _simulate_to_completion(self, feats, current_t, dt, remaining_steps, context):
        """Simulate a branch to completion deterministically from current_t to min_t."""
        device = feats["rigids_t"].device
        min_t = self.sampler._fm_conf.min_t

        self._log.debug(
            f"        _simulate_to_completion: {len(remaining_steps)} steps from t={current_t:.4f} to t={min_t:.4f}"
        )

        # Initialize trajectory collection
        all_rigids = [du.move_to_np(copy.deepcopy(feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        with torch.no_grad():
            for step_idx, t in enumerate(remaining_steps):
                self._log.debug(f"          Step {step_idx}: t={t:.4f}")

                feats = self.sampler.exp._set_t_feats(
                    feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                fixed_mask = feats["fixed_mask"] * feats["res_mask"]
                flow_mask = (1 - feats["fixed_mask"]) * feats["res_mask"]

                rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=True,
                    noise_scale=1.0,
                )

                feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

                # Collect trajectory data - keep batch dimension
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions
                gt_trans_0 = feats["rigids_t"][..., 4:]
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

        self._log.debug(
            f"        _simulate_to_completion: Completed {len(remaining_steps)} steps"
        )
        self._log.debug(
            f"        _simulate_to_completion: Collected {len(all_bb_prots)} trajectory frames"
        )

        # Flip trajectory so that it starts from t=0 (for visualization)
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        # Return final sample in proper format (matching inference_fn)
        sample_out = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }

        self._log.debug(
            f"        _simulate_to_completion: Trajectory shapes - prot_traj: {all_bb_prots.shape}, rigid_traj: {all_rigids.shape}"
        )

        # Remove batch dimension like _base_sample does
        result = tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

        self._log.debug(
            f"        _simulate_to_completion: Final result shapes - prot_traj: {result['prot_traj'].shape}, rigid_traj: {result['rigid_traj'].shape}"
        )

        return result


class StandardInference(InferenceMethod):
    """Standard inference method - single sample generation."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate a single sample using standard inference."""
        self._log.info(f"STANDARD INFERENCE START")

        # Get timestep info for debugging
        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1] if len(reverse_steps) > 1 else 0

        self._log.info(f"  num_t={num_t}, min_t={min_t:.4f}, dt={dt:.4f}")
        self._log.info(
            f"  reverse_steps: {reverse_steps[0]:.4f} -> {reverse_steps[-1]:.4f}"
        )

        result = self.sampler._base_sample(sample_length, context)

        self._log.info(f"STANDARD INFERENCE COMPLETE")
        return result


class BestOfNInference(InferenceMethod):
    """Best-of-N inference method."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate N samples and return the best one."""
        n_samples = self.config.get("n_samples", 5)
        selector = self.config.get("selector", "tm_score")
        temp_dir = self.config.get("temp_dir", None)

        self._log.info(f"Running Best-of-N sampling with N={n_samples}")

        if temp_dir is None:
            temp_dir = os.path.join(
                self.sampler._output_dir, f"best_of_{n_samples}_temp"
            )
            os.makedirs(temp_dir, exist_ok=True)

        score_fn = self.get_score_function(selector)

        best_sample = None
        best_score = float("-inf")
        best_metrics = None

        for i in range(n_samples):
            self._log.info(f"Generating sample {i+1}/{n_samples}")
            sample_output = self.sampler._base_sample(sample_length, context)

            # Evaluate the sample
            score = score_fn(sample_output, sample_length)
            self._log.info(f"Sample {i+1} score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_sample = sample_output
                self._log.info(f"New best sample found: score = {best_score:.4f}")

        self._log.info(
            f"Best-of-{n_samples} sampling complete. Best score: {best_score:.4f}"
        )

        return {"sample": best_sample, "score": best_score, "method": "best_of_n"}


class SDEPathExplorationInference(InferenceMethod):
    """SDE path exploration inference with Euler-Maruyama sampling."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using SDE path exploration."""
        num_branches = self.config.get("num_branches", 4)
        num_keep = self.config.get("num_keep", 2)
        noise_scale = self.config.get("noise_scale", 0.05)
        selector = self.config.get("selector", "tm_score")
        branch_start_time = self.config.get("branch_start_time", 0.0)
        branch_interval = self.config.get("branch_interval", 0.0)

        self._log.info(
            f"Running SDE path exploration with {num_branches} branches, keeping {num_keep}"
        )

        if num_branches == 1 and num_keep == 1:
            return self.sampler._base_sample(sample_length, context)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"
        assert branch_interval >= 0.0, "branch_interval must be >= 0.0"

        score_fn = self.get_score_function(selector)

        # Initialize features
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
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        # Run SDE path exploration
        sample_out = self._sde_path_exploration_inference(
            init_feats,
            num_branches,
            num_keep,
            noise_scale,
            score_fn,
            sample_length,
            branch_start_time,
            branch_interval,
            context,
        )

        return sample_out

    def _sde_path_exploration_inference(
        self,
        data_init,
        num_branches,
        num_keep,
        noise_scale,
        score_fn,
        sample_length,
        branch_start_time,
        branch_interval,
        context,
    ):
        """Core SDE path exploration logic."""
        self._log.info(f"SDE PATH EXPLORATION START")
        self._log.info(f"  num_branches={num_branches}, num_keep={num_keep}")
        self._log.info(
            f"  noise_scale={noise_scale}, branch_start_time={branch_start_time}"
        )
        self._log.info(f"  branch_interval={branch_interval}")

        sample_feats = data_init.copy()
        device = sample_feats["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t

        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        self._log.info(f"  num_t={num_t}, min_t={min_t:.4f}, dt={dt:.4f}")
        self._log.info(
            f"  reverse_steps: {reverse_steps[0]:.4f} -> {reverse_steps[-1]:.4f}"
        )

        current_samples = [sample_feats]

        # Initialize trajectory collection for final sample
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        with torch.no_grad():
            branching_steps = []

            # Calculate branching step interval based on step indices
            if branch_interval > 0.0:
                branching_step_interval = max(1, int(num_t * branch_interval))
                self._log.info(
                    f"  Branching every {branching_step_interval} steps (branch_interval={branch_interval})"
                )
            else:
                branching_step_interval = 1  # Branch at every step
                self._log.info(
                    f"  Branching at every step (branch_interval={branch_interval})"
                )

            for step_idx, t in enumerate(reverse_steps):
                # Fixed branching condition: branch if t >= branch_start_time and step_idx is a multiple of branching_step_interval
                should_branch = False

                if t >= branch_start_time:
                    if branch_interval <= 0.0:
                        # Branch at every timestep if branch_interval is 0
                        should_branch = True
                    else:
                        # Branch at regular step intervals
                        should_branch = step_idx % branching_step_interval == 0

                if should_branch:
                    branching_steps.append((step_idx, t))

                self._log.debug(
                    f"Step {step_idx}: t={t:.4f}, should_branch={should_branch} (step_idx % {branching_step_interval} = {step_idx % branching_step_interval})"
                )

                if not should_branch:
                    # Regular flow with SDE noise at every step
                    for i, feats in enumerate(current_samples):
                        feats = self.sampler.exp._set_t_feats(
                            feats, t, torch.ones((1,)).to(device)
                        )
                        model_out = self.sampler.model(feats)

                        rot_vectorfield = model_out["rot_vectorfield"]
                        trans_vectorfield = model_out["trans_vectorfield"]
                        rigid_pred = model_out["rigids"]
                        psi_pred = model_out["psi"]

                        # Add SDE noise at every step
                        noise_rot = (
                            torch.randn_like(rot_vectorfield)
                            * noise_scale
                            * np.sqrt(dt)
                        )
                        noise_trans = (
                            torch.randn_like(trans_vectorfield)
                            * noise_scale
                            * np.sqrt(dt)
                        )

                        rot_vectorfield = rot_vectorfield + noise_rot
                        trans_vectorfield = trans_vectorfield + noise_trans

                        fixed_mask = feats["fixed_mask"] * feats["res_mask"]
                        flow_mask = (1 - feats["fixed_mask"]) * feats["res_mask"]

                        rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                            rigid_t=ru.Rigid.from_tensor_7(feats["rigids_t"]),
                            rot_vectorfield=du.move_to_np(rot_vectorfield),
                            trans_vectorfield=du.move_to_np(trans_vectorfield),
                            flow_mask=du.move_to_np(flow_mask),
                            t=t,
                            dt=dt,
                            center=True,
                            noise_scale=1.0,
                        )

                        feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                        current_samples[i] = feats

                        # Collect trajectory data for the main sample (first one) - keep batch dimension
                        if i == 0:
                            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                            # Calculate x0 prediction derived from vectorfield predictions
                            gt_trans_0 = feats["rigids_t"][..., 4:]
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

                            self._log.debug(
                                f"    Non-branch step {step_idx}: collected trajectory frame, total frames = {len(all_bb_prots)}"
                            )
                else:
                    # Branching phase
                    self._log.info(
                        f"BRANCHING at timestep t={t:.4f} (step {step_idx}/{len(reverse_steps)})"
                    )
                    self._log.info(f"  Current samples: {len(current_samples)}")
                    self._log.info(
                        f"  Creating {num_branches} branches per sample, keeping {num_keep}"
                    )
                    self._log.debug(
                        f"  TRAJECTORY STATUS: Before branching, collected {len(all_bb_prots)} frames"
                    )

                    new_samples = []

                    for sample_idx, feats in enumerate(current_samples):
                        self._log.info(f"  Processing sample {sample_idx}")

                        # Create branches
                        branches = []
                        for branch_idx in range(num_branches):
                            branch_feats = tree.map_structure(
                                lambda x: x.clone(), feats
                            )

                            # Apply SDE step with noise
                            branch_feats = self.sampler.exp._set_t_feats(
                                branch_feats, t, torch.ones((1,)).to(device)
                            )
                            model_out = self.sampler.model(branch_feats)

                            rot_vectorfield = model_out["rot_vectorfield"]
                            trans_vectorfield = model_out["trans_vectorfield"]

                            # Add noise to vector fields for SDE
                            noise_rot = (
                                torch.randn_like(rot_vectorfield)
                                * noise_scale
                                * np.sqrt(dt)
                            )
                            noise_trans = (
                                torch.randn_like(trans_vectorfield)
                                * noise_scale
                                * np.sqrt(dt)
                            )

                            rot_vectorfield = rot_vectorfield + noise_rot
                            trans_vectorfield = trans_vectorfield + noise_trans

                            fixed_mask = (
                                branch_feats["fixed_mask"] * branch_feats["res_mask"]
                            )
                            flow_mask = (1 - branch_feats["fixed_mask"]) * branch_feats[
                                "res_mask"
                            ]

                            rots_t, trans_t, rigids_t = (
                                self.sampler.flow_matcher.reverse(
                                    rigid_t=ru.Rigid.from_tensor_7(
                                        branch_feats["rigids_t"]
                                    ),
                                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                                    flow_mask=du.move_to_np(flow_mask),
                                    t=t,
                                    dt=dt,
                                    center=True,
                                    noise_scale=1.0,
                                )
                            )

                            branch_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                            branches.append(branch_feats)

                        self._log.info(f"    Created {len(branches)} branches")

                        # Simulate branches to completion and evaluate
                        remaining_steps = reverse_steps[step_idx + 1 :]
                        self._log.info(
                            f"    Remaining steps: {len(remaining_steps)} (from t={t:.4f} to t={remaining_steps[-1]:.4f})"
                        )

                        branch_scores = []
                        for branch_idx, branch_feats in enumerate(branches):
                            try:
                                # Log the branch state before simulation
                                self._log.debug(
                                    f"      Branch {branch_idx} state at t={t:.4f}: rigids_t shape = {branch_feats['rigids_t'].shape}"
                                )

                                # Simulate to completion deterministically
                                self._log.debug(
                                    f"      Simulating branch {branch_idx} to completion..."
                                )
                                completed_sample = self._simulate_to_completion(
                                    branch_feats,
                                    t,
                                    dt,
                                    remaining_steps,
                                    context,
                                )

                                # Log the completed trajectory
                                self._log.debug(
                                    f"      Branch {branch_idx} completed: prot_traj shape = {completed_sample['prot_traj'].shape}"
                                )

                                # Evaluate
                                self._log.debug(
                                    f"      Evaluating branch {branch_idx}..."
                                )
                                score = score_fn(completed_sample, sample_length)
                                branch_scores.append(score)
                                self._log.info(
                                    f"      Branch {branch_idx}: score = {score:.4f}"
                                )

                            except Exception as e:
                                self._log.error(
                                    f"      Branch {branch_idx} failed: {e}"
                                )
                                branch_scores.append(float("-inf"))

                        # Log all branch scores
                        self._log.info(
                            f"    Branch scores: {[f'{s:.4f}' for s in branch_scores]}"
                        )

                        # Select best branches and continue with their states
                        branch_scores = torch.tensor(branch_scores)
                        top_k_indices = torch.topk(
                            branch_scores, k=min(num_keep, len(branches))
                        )[1]

                        self._log.info(
                            f"    Selected branch indices: {top_k_indices.tolist()}"
                        )
                        self._log.info(
                            f"    Selected branch scores: {[f'{branch_scores[i]:.4f}' for i in top_k_indices]}"
                        )

                        # Use the actual branch states for continuing
                        for idx in top_k_indices:
                            new_samples.append(branches[idx])

                        # CRITICAL FIX: Collect trajectory data for the selected branch during branching steps
                        if len(top_k_indices) > 0:
                            # Use the first selected branch for trajectory collection
                            selected_branch = branches[top_k_indices[0]]

                            # Apply the same model step to get the predictions for trajectory collection
                            selected_branch_copy = tree.map_structure(
                                lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                                selected_branch,
                            )
                            selected_branch_copy = self.sampler.exp._set_t_feats(
                                selected_branch_copy, t, torch.ones((1,)).to(device)
                            )
                            model_out = self.sampler.model(selected_branch_copy)

                            rigid_pred = model_out["rigids"]
                            psi_pred = model_out["psi"]

                            # Collect trajectory data for this branching step
                            all_rigids.append(
                                du.move_to_np(selected_branch["rigids_t"])
                            )

                            # Calculate x0 prediction
                            fixed_mask = (
                                selected_branch["fixed_mask"]
                                * selected_branch["res_mask"]
                            )
                            flow_mask = (
                                1 - selected_branch["fixed_mask"]
                            ) * selected_branch["res_mask"]

                            gt_trans_0 = selected_branch["rigids_t"][..., 4:]
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

                            atom37_t = all_atom.compute_backbone(
                                ru.Rigid.from_tensor_7(selected_branch["rigids_t"]),
                                psi_pred,
                            )[0]
                            all_bb_prots.append(du.move_to_np(atom37_t))
                            final_psi_pred = psi_pred

                            self._log.debug(
                                f"    Branching step {step_idx}: collected trajectory frame from selected branch, total frames = {len(all_bb_prots)}"
                            )

                    current_samples = new_samples

            self._log.info(f"SDE PATH EXPLORATION COMPLETE")
            self._log.info(f"  Total branching steps: {len(branching_steps)}")
            if branching_steps:
                self._log.info(
                    f"  Branching occurred at: {[(idx, f'{t:.4f}') for idx, t in branching_steps]}"
                )
            else:
                self._log.info(
                    f"  No branching occurred (branch_start_time={branch_start_time}, branch_interval={branch_interval})"
                )

            # Return best final sample
            if len(current_samples) == 1:
                final_sample = current_samples[0]
                self._log.info(f"  Using single remaining sample")
            else:
                # Evaluate all final samples and pick best
                self._log.info(f"  Evaluating {len(current_samples)} final samples...")
                final_scores = []
                for i, feats in enumerate(current_samples):
                    # Create a simple trajectory for evaluation
                    sample_out = {
                        "prot_traj": feats["rigids_t"],
                        "rigid_0_traj": feats["rigids_t"],
                    }
                    score = score_fn(sample_out, sample_length)
                    final_scores.append(score)
                    self._log.info(f"    Final sample {i}: score = {score:.4f}")

                best_idx = np.argmax(final_scores)
                final_sample = current_samples[best_idx]
                self._log.info(
                    f"  Selected final sample {best_idx} with score {final_scores[best_idx]:.4f}"
                )

            # Flip trajectory so that it starts from t=0 (for visualization)
            flip = lambda x: np.flip(np.stack(x), (0,))
            all_bb_prots = flip(all_bb_prots)
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

            self._log.info(
                f"  Final trajectory shapes: prot_traj={all_bb_prots.shape}, rigid_traj={all_rigids.shape}"
            )
            self._log.info(f"  Expected trajectory length: {len(reverse_steps)} steps")
            self._log.info(f"  Actual trajectory length: {len(all_bb_prots)} frames")
            self._log.info(
                f"  Missing frames: {len(reverse_steps) - len(all_bb_prots)}"
            )
            self._log.info(f"  Branching steps: {len(branching_steps)}")

            # Return final sample in proper format (matching inference_fn)
            sample_out = {
                "prot_traj": all_bb_prots,
                "rigid_traj": all_rigids,
                "trans_traj": all_trans_0_pred,
                "psi_pred": (
                    final_psi_pred[None] if final_psi_pred is not None else None
                ),
                "rigid_0_traj": all_bb_0_pred,
            }

            # Remove batch dimension like _base_sample does
            result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
            )

            return result


class DivergenceFreeODEInference(InferenceMethod):
    """Divergence-free ODE path exploration inference."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using divergence-free ODE path exploration."""
        num_branches = self.config.get("num_branches", 4)
        num_keep = self.config.get("num_keep", 2)
        lambda_div = self.config.get("lambda_div", 0.2)
        selector = self.config.get("selector", "tm_score")
        branch_start_time = self.config.get("branch_start_time", 0.0)
        branch_interval = self.config.get("branch_interval", 0.0)

        self._log.info(
            f"Running divergence-free ODE path exploration with {num_branches} branches, keeping {num_keep}"
        )

        if num_branches == 1 and num_keep == 1:
            return self.sampler._base_sample(sample_length, context)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"
        assert branch_interval >= 0.0, "branch_interval must be >= 0.0"

        score_fn = self.get_score_function(selector)

        # Initialize features
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
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        # Run divergence-free ODE path exploration
        sample_out = self._divergence_free_path_exploration_inference(
            init_feats,
            num_branches,
            num_keep,
            lambda_div,
            score_fn,
            sample_length,
            branch_start_time,
            branch_interval,
            context,
        )

        return sample_out

    def _divergence_free_path_exploration_inference(
        self,
        data_init,
        num_branches,
        num_keep,
        lambda_div,
        score_fn,
        sample_length,
        branch_start_time,
        branch_interval,
        context,
    ):
        """Core divergence-free ODE path exploration logic."""
        sample_feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else x.copy(), data_init
        )
        device = sample_feats["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t

        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        current_samples = [sample_feats]

        # Initialize trajectory collection for final sample
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        with torch.no_grad():
            branching_steps = []

            # Calculate branching step interval based on step indices
            if branch_interval > 0.0:
                branching_step_interval = max(1, int(num_t * branch_interval))
                self._log.info(
                    f"  Branching every {branching_step_interval} steps (branch_interval={branch_interval})"
                )
            else:
                branching_step_interval = 1  # Branch at every step
                self._log.info(
                    f"  Branching at every step (branch_interval={branch_interval})"
                )

            for step_idx, t in enumerate(reverse_steps):
                # Fixed branching condition: branch if t >= branch_start_time and step_idx is a multiple of branching_step_interval
                should_branch = False

                if t >= branch_start_time:
                    if branch_interval <= 0.0:
                        # Branch at every timestep if branch_interval is 0
                        should_branch = True
                    else:
                        # Branch at regular step intervals
                        should_branch = step_idx % branching_step_interval == 0

                if should_branch:
                    branching_steps.append((step_idx, t))

                self._log.debug(
                    f"Step {step_idx}: t={t:.4f}, should_branch={should_branch} (step_idx % {branching_step_interval} = {step_idx % branching_step_interval})"
                )

                if not should_branch:
                    # Regular flow with divergence-free noise at every step
                    for i, feats in enumerate(current_samples):
                        feats = self.sampler.exp._set_t_feats(
                            feats, t, torch.ones((1,)).to(device)
                        )
                        model_out = self.sampler.model(feats)

                        rot_vectorfield = model_out["rot_vectorfield"]
                        trans_vectorfield = model_out["trans_vectorfield"]
                        rigid_pred = model_out["rigids"]
                        psi_pred = model_out["psi"]

                        # Add divergence-free noise at every step
                        rigids_tensor = feats["rigids_t"]
                        t_batch = torch.full(
                            (rigids_tensor.shape[0],), t, device=device
                        )

                        # Extract rotation matrices and translations directly as torch tensors (stay on GPU)
                        rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                        rot_mats = rigid_obj.get_rots().get_rot_mats()  # [B, N, 3, 3]
                        trans_vecs = rigid_obj.get_trans()  # [B, N, 3]

                        # Generate divergence-free noise for rotation field
                        rot_divfree_noise = divfree_swirl_si(
                            rot_mats,  # [B, N, 3, 3] rotation matrices
                            t_batch,
                            None,  # y not used
                            rot_vectorfield,
                        )

                        # Generate divergence-free noise for translation field
                        trans_divfree_noise = divfree_swirl_si(
                            trans_vecs,  # [B, N, 3] translation vectors
                            t_batch,
                            None,  # y not used
                            trans_vectorfield,
                        )

                        # Add divergence-free noise to vector fields (no sqrt(dt) scaling)
                        rot_vectorfield = (
                            rot_vectorfield + lambda_div * rot_divfree_noise
                        )
                        trans_vectorfield = (
                            trans_vectorfield + lambda_div * trans_divfree_noise
                        )

                        fixed_mask = feats["fixed_mask"] * feats["res_mask"]
                        flow_mask = (1 - feats["fixed_mask"]) * feats["res_mask"]

                        rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                            rigid_t=ru.Rigid.from_tensor_7(feats["rigids_t"]),
                            rot_vectorfield=du.move_to_np(rot_vectorfield),
                            trans_vectorfield=du.move_to_np(trans_vectorfield),
                            flow_mask=du.move_to_np(flow_mask),
                            t=t,
                            dt=dt,
                            center=True,
                            noise_scale=1.0,
                        )

                        feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                        current_samples[i] = feats

                        # Collect trajectory data for the main sample (first one) - keep batch dimension
                        if i == 0:
                            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                            # Calculate x0 prediction derived from vectorfield predictions
                            gt_trans_0 = feats["rigids_t"][..., 4:]
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

                            self._log.debug(
                                f"    Non-branch step {step_idx}: collected trajectory frame, total frames = {len(all_bb_prots)}"
                            )
                else:
                    # Branching phase with divergence-free exploration
                    new_samples = []

                    for feats in current_samples:
                        # Create branches with different divergence-free fields
                        branches = []
                        for branch_idx in range(num_branches):
                            branch_feats = tree.map_structure(
                                lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                                feats,
                            )

                            # Apply divergence-free ODE step
                            branch_feats = self.sampler.exp._set_t_feats(
                                branch_feats, t, torch.ones((1,)).to(device)
                            )
                            model_out = self.sampler.model(branch_feats)

                            rot_vectorfield = model_out["rot_vectorfield"]
                            trans_vectorfield = model_out["trans_vectorfield"]

                            rigids_tensor = branch_feats["rigids_t"]
                            t_batch = torch.full(
                                (rigids_tensor.shape[0],), t, device=device
                            )

                            # Extract rotation matrices and translations directly as torch tensors (stay on GPU)
                            rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                            rot_mats = (
                                rigid_obj.get_rots().get_rot_mats()
                            )  # [B, N, 3, 3]
                            trans_vecs = rigid_obj.get_trans()  # [B, N, 3]

                            # Generate divergence-free noise for rotation field
                            rot_divfree_noise = divfree_swirl_si(
                                rot_mats,  # [B, N, 3, 3] rotation matrices
                                t_batch,
                                None,  # y not used
                                rot_vectorfield,
                            )

                            # Generate divergence-free noise for translation field
                            trans_divfree_noise = divfree_swirl_si(
                                trans_vecs,  # [B, N, 3] translation vectors
                                t_batch,
                                None,  # y not used
                                trans_vectorfield,
                            )

                            # Add divergence-free noise to vector fields (no sqrt(dt) scaling)
                            rot_vectorfield = (
                                rot_vectorfield + lambda_div * rot_divfree_noise
                            )
                            trans_vectorfield = (
                                trans_vectorfield + lambda_div * trans_divfree_noise
                            )

                            fixed_mask = (
                                branch_feats["fixed_mask"] * branch_feats["res_mask"]
                            )
                            flow_mask = (1 - branch_feats["fixed_mask"]) * branch_feats[
                                "res_mask"
                            ]

                            rots_t, trans_t, rigids_t = (
                                self.sampler.flow_matcher.reverse(
                                    rigid_t=ru.Rigid.from_tensor_7(
                                        branch_feats["rigids_t"]
                                    ),
                                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                                    flow_mask=du.move_to_np(flow_mask),
                                    t=t,
                                    dt=dt,
                                    center=True,
                                    noise_scale=1.0,
                                )
                            )

                            branch_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

                            # Simulate this branch to completion
                            remaining_steps = reverse_steps[step_idx + 1 :]
                            if len(remaining_steps) > 0:
                                branch_result = self._simulate_to_completion(
                                    branch_feats, t, dt, remaining_steps, context
                                )
                                branches.append((branch_result, branch_feats))
                            else:
                                # This is the final step
                                rigid_pred = model_out["rigids"]
                                psi_pred = model_out["psi"]

                                fixed_mask = (
                                    branch_feats["fixed_mask"]
                                    * branch_feats["res_mask"]
                                )
                                flow_mask = (
                                    1 - branch_feats["fixed_mask"]
                                ) * branch_feats["res_mask"]

                                gt_trans_0 = branch_feats["rigids_t"][..., 4:]
                                pred_trans_0 = rigid_pred[..., 4:]
                                trans_pred_0 = (
                                    flow_mask[..., None] * pred_trans_0
                                    + fixed_mask[..., None] * gt_trans_0
                                )

                                atom37_0 = all_atom.compute_backbone(
                                    ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                                )[0]

                                final_result = {
                                    "prot_traj": np.array([du.move_to_np(atom37_0)]),
                                    "rigid_traj": np.array(
                                        [du.move_to_np(rigids_t.to_tensor_7())]
                                    ),
                                    "trans_traj": np.array(
                                        [du.move_to_np(trans_pred_0)]
                                    ),
                                    "psi_pred": (
                                        psi_pred[None] if psi_pred is not None else None
                                    ),
                                    "rigid_0_traj": np.array([du.move_to_np(atom37_0)]),
                                }

                                # Remove batch dimension like _base_sample does
                                final_result = tree.map_structure(
                                    lambda x: (
                                        x[:, 0] if x is not None and x.ndim > 1 else x
                                    ),
                                    final_result,
                                )
                                branches.append((final_result, branch_feats))

                        # Score all branches and keep the best ones
                        if branches:
                            scored_branches = []
                            for branch_result, branch_feats in branches:
                                try:
                                    score = score_fn(branch_result, sample_length)
                                    scored_branches.append(
                                        (score, branch_result, branch_feats)
                                    )
                                except Exception as e:
                                    self._log.warning(f"Failed to score branch: {e}")
                                    scored_branches.append(
                                        (-float("inf"), branch_result, branch_feats)
                                    )

                            # Sort by score (descending) and keep the best
                            scored_branches.sort(key=lambda x: x[0], reverse=True)
                            best_branches = scored_branches[:num_keep]

                            # Update current samples with the best branches
                            for _, branch_result, branch_feats in best_branches:
                                new_samples.append(branch_feats)

                    current_samples = new_samples

        # Return the final sample from the best branch
        if current_samples:
            final_feats = current_samples[0]
        else:
            final_feats = sample_feats

        # Complete the simulation if we haven't reached the end
        remaining_steps = [t for t in reverse_steps if t < branch_start_time]
        if remaining_steps and len(all_bb_prots) == 0:
            # We branched early, need to complete the trajectory
            final_result = self._simulate_to_completion(
                final_feats, reverse_steps[0], dt, remaining_steps, context
            )
            return final_result

        # Flip trajectory so that it starts from t=0 (for visualization)
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        # Return final sample in proper format (matching inference_fn)
        sample_out = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": (final_psi_pred[None] if final_psi_pred is not None else None),
            "rigid_0_traj": all_bb_0_pred,
        }

        # Remove batch dimension like _base_sample does
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )


class SDESimpleInference(InferenceMethod):
    """Simple SDE inference with noise but no branching."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using SDE with noise but no branching."""
        noise_scale = self.config.get("noise_scale", 0.05)

        self._log.info(f"Running simple SDE sampling with noise_scale={noise_scale}")

        # Initialize features (same as standard method)
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
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        # Run simple SDE sampling
        sample_out = self._simple_sde_inference(init_feats, noise_scale, context)

        # Remove batch dimension like _base_sample does
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

    def _simple_sde_inference(self, data_init, noise_scale, context):
        """Simple SDE sampling with noise at every step."""
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

        with torch.no_grad():
            for t in reverse_steps:
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Add SDE noise
                noise_rot = (
                    torch.randn_like(rot_vectorfield) * noise_scale * np.sqrt(dt)
                )
                noise_trans = (
                    torch.randn_like(trans_vectorfield) * noise_scale * np.sqrt(dt)
                )

                rot_vectorfield = rot_vectorfield + noise_rot
                trans_vectorfield = trans_vectorfield + noise_trans

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=True,
                    noise_scale=1.0,
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

        return {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }


class DivergenceFreeSimpleInference(InferenceMethod):
    """Simple divergence-free inference with noise but no branching."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using divergence-free noise but no branching."""
        lambda_div = self.config.get("lambda_div", 0.2)

        self._log.info(
            f"Running simple divergence-free sampling with lambda_div={lambda_div}"
        )

        # Initialize features (same as standard method)
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
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.sampler.device), init_feats
        )

        # Run simple divergence-free sampling
        sample_out = self._simple_divergence_free_inference(
            init_feats, lambda_div, context
        )

        # Remove batch dimension like _base_sample does
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

    def _simple_divergence_free_inference(self, data_init, lambda_div, context):
        """Simple divergence-free sampling with noise at every step."""
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

        with torch.no_grad():
            for t in reverse_steps:
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Add divergence-free noise
                rigids_tensor = sample_feats["rigids_t"]
                t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                # Extract rotation matrices and translations
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

                # Add divergence-free noise to vector fields
                rot_vectorfield = rot_vectorfield + lambda_div * rot_divfree_noise
                trans_vectorfield = trans_vectorfield + lambda_div * trans_divfree_noise

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=True,
                    noise_scale=1.0,
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

        return {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }


def get_inference_method(
    method_name: str, sampler, config: Dict[str, Any]
) -> InferenceMethod:
    """Factory function to get inference method by name."""
    methods = {
        "standard": StandardInference,
        "best_of_n": BestOfNInference,
        "sde_path_exploration": SDEPathExplorationInference,
        "divergence_free_ode": DivergenceFreeODEInference,
        "sde_simple": SDESimpleInference,
        "divergence_free_simple": DivergenceFreeSimpleInference,
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown inference method: {method_name}. Available: {list(methods.keys())}"
        )

    return methods[method_name](sampler, config)

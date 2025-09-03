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
from runner.divergence_free_utils import divfree_swirl_si, divfree_max_noise


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
        elif selector == "geometric":
            return self._geometric_score_function
        elif selector == "tm_score_4seq":
            return self._tm_score_4seq_function
        elif selector == "dual_score":
            return self._dual_score_function
        else:
            raise ValueError(f"Unknown selector: {selector}")

    def get_score_functions(self) -> Dict[str, Callable]:
        """Get all scoring functions for comprehensive evaluation."""
        return {
            "tm_score": self._tm_score_function,
            "rmsd": self._rmsd_function,
            "geometric": self._geometric_score_function,
            "tm_score_4seq": self._tm_score_4seq_function,
        }

    def _calculate_and_log_self_consistency(
        self, sample_output: Dict[str, Any], sample_length: int, method_name: str
    ) -> float:
        """Calculate and log self-consistency score for simple inference methods."""
        try:
            tm_score = self._tm_score_function(sample_output, sample_length)
            self._log.info(f"{method_name} self-consistency TM-score: {tm_score:.4f}")
            return tm_score
        except Exception as e:
            self._log.warning(
                f"Failed to calculate self-consistency score for {method_name}: {e}"
            )
            return float("-inf")

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

    def _geometric_score_function(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> float:
        """Evaluate sample using fast geometric validation metrics only.

        This scorer analyzes the generated backbone structure directly without
        any sequence design or refolding steps, making it very fast.

        Returns a composite score where higher values indicate better geometry.
        """
        from tools.analysis import metrics
        import tempfile

        self._log.debug(f"        _geometric_score_function: Starting fast evaluation")

        try:
            # Create temporary file for PDB
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                tmp_pdb_path = tmp_pdb.name

            # Save the backbone structure to PDB
            traj_paths = self.sampler.save_traj(
                sample_output["prot_traj"],
                sample_output["rigid_0_traj"],
                np.ones(sample_length),
                output_dir=os.path.dirname(tmp_pdb_path),
            )
            pdb_path = traj_paths["sample_path"]

            # Extract backbone coordinates for geometric analysis
            final_coords = sample_output["prot_traj"][-1]  # Final structure
            ca_pos = final_coords[:sample_length, 1, :]  # C-alpha coordinates

            # Calculate geometric metrics
            ca_ca_bond_dev, ca_ca_valid_percent = metrics.ca_ca_distance(ca_pos)
            num_ca_steric_clashes, ca_steric_clash_percent = metrics.ca_ca_clashes(
                ca_pos
            )

            # Secondary structure analysis
            try:
                ss_metrics = metrics.calc_mdtraj_metrics(pdb_path)
                non_coil_percent = ss_metrics["non_coil_percent"]
                rg = ss_metrics["radius_of_gyration"]
            except Exception as e:
                self._log.warning(
                    f"        _geometric_score_function: Secondary structure analysis failed: {e}"
                )
                non_coil_percent = 0.0
                rg = np.linalg.norm(ca_pos.std(axis=0))  # Fallback RG calculation

            # Composite score calculation (higher is better)
            # Penalize bad geometry, reward good secondary structure
            score = (
                ca_ca_valid_percent * 2.0  # Reward valid C-alpha distances (0-2)
                + (1.0 - ca_steric_clash_percent) * 1.5  # Penalize clashes (0-1.5)
                + non_coil_percent * 1.0  # Reward secondary structure (0-1)
                + max(0, 1.0 - ca_ca_bond_dev) * 0.5  # Penalize bond deviations (0-0.5)
            )

            self._log.debug(
                f"        _geometric_score_function: Geometric score = {score:.4f}"
            )
            self._log.debug(
                f"          ca_valid_percent={ca_ca_valid_percent:.3f}, clash_percent={ca_steric_clash_percent:.3f}"
            )
            self._log.debug(
                f"          non_coil_percent={non_coil_percent:.3f}, bond_dev={ca_ca_bond_dev:.3f}"
            )

            return score

        except Exception as e:
            self._log.error(
                f"        _geometric_score_function: Error during evaluation: {e}"
            )
            return float("-inf")
        finally:
            # Clean up temporary files
            if "tmp_pdb_path" in locals() and os.path.exists(tmp_pdb_path):
                os.unlink(tmp_pdb_path)
            if "pdb_path" in locals() and os.path.exists(pdb_path):
                try:
                    os.unlink(pdb_path)
                except:
                    pass

    def _tm_score_4seq_function(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> float:
        """Evaluate sample using TM-score with only 4 sequence refolds instead of 8.

        This is faster than the full self-consistency while maintaining most accuracy.
        """
        self._log.debug(
            f"        _tm_score_4seq_function: Starting evaluation with 4 sequences"
        )

        # Create temporary directory for evaluation
        temp_dir = os.path.join(self.sampler._output_dir, "temp_eval_4seq")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save trajectory
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

            # Temporarily modify seq_per_sample to 4
            original_seq_per_sample = self.sampler._sample_conf.seq_per_sample
            self.sampler._sample_conf.seq_per_sample = 4

            try:
                sc_results = self.sampler.run_self_consistency(
                    sc_output_dir, pdb_path, motif_mask=None
                )
                tm_score = sc_results["tm_score"].mean()
                self._log.debug(
                    f"        _tm_score_4seq_function: TM-score = {tm_score:.4f} (4 sequences)"
                )
                return tm_score
            finally:
                # Restore original seq_per_sample
                self.sampler._sample_conf.seq_per_sample = original_seq_per_sample

        except Exception as e:
            self._log.error(
                f"        _tm_score_4seq_function: Error during evaluation: {e}"
            )
            return float("-inf")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _dual_score_function(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> Dict[str, float]:
        """Evaluate sample using both TM-score and RMSD."""
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

            return {
                "tm_score": sc_results["tm_score"].mean(),
                "rmsd": sc_results["rmsd"].mean(),
            }

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def get_rmsd_value(
        self, sample_output: Dict[str, Any], sample_length: int
    ) -> float:
        """Get the actual RMSD value (positive) for analysis."""
        scores = self._dual_score_function(sample_output, sample_length)
        return scores["rmsd"]

    def _simulate_to_completion(self, feats, current_t, dt, remaining_steps, context):
        """Simulate a branch to completion deterministically from current_t to min_t."""
        # Clone the features to avoid modifying the original branch in place
        feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else copy.deepcopy(x), feats
        )

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
            "psi_pred": (final_psi_pred[None] if final_psi_pred is not None else None),
            "rigid_0_traj": all_bb_0_pred,
            "feature_states_traj": all_feature_states[
                ::-1
            ],  # Complete feature states for proper intermediate extraction
        }

        # Remove batch dimension like _base_sample does
        result = tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
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
        num_branches = self.config.get("num_branches", 5)
        selector = self.config.get("selector", "tm_score")
        temp_dir = self.config.get("temp_dir", None)

        self._log.info(f"Running Best-of-N sampling with N={num_branches}")

        if temp_dir is None:
            temp_dir = os.path.join(
                self.sampler._output_dir, f"best_of_{num_branches}_temp"
            )
            os.makedirs(temp_dir, exist_ok=True)

        score_fn = self.get_score_function(selector)

        best_sample = None
        best_score = float("-inf")
        best_metrics = None

        for i in range(num_branches):
            self._log.info(f"Generating sample {i+1}/{num_branches}")
            sample_output = self.sampler._base_sample(sample_length, context)

            # Evaluate the sample
            score = score_fn(sample_output, sample_length)
            self._log.info(f"Sample {i+1} score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_sample = sample_output
                self._log.info(f"New best sample found: score = {best_score:.4f}")

        self._log.info(
            f"Best-of-{num_branches} sampling complete. Best score: {best_score:.4f}"
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

        # Track best intermediate sample across all simulations
        best_intermediate_score = float("-inf")
        best_intermediate_sample = None

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
                    # Regular deterministic ODE flow (no noise except during branching)
                    for i, feats in enumerate(current_samples):
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
                                lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                                feats,
                            )

                            # Apply SDE step with noise
                            branch_feats = self.sampler.exp._set_t_feats(
                                branch_feats, t, torch.ones((1,)).to(device)
                            )
                            model_out = self.sampler.model(branch_feats)

                            rot_vectorfield = model_out["rot_vectorfield"]
                            trans_vectorfield = model_out["trans_vectorfield"]

                            # Add noise to vector fields for SDE branching
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

                                # Track best intermediate sample
                                if score > best_intermediate_score:
                                    best_intermediate_score = score
                                    best_intermediate_sample = completed_sample
                                    self._log.info(
                                        f"      New best intermediate sample: score = {score:.4f}"
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
            self._log.info(
                f"  Branching occurred at: {[(idx, f'{t:.4f}') for idx, t in branching_steps]}"
            )

        # After branching, simulate remaining samples to completion and compare with best intermediate
        final_samples = []
        for sample_idx, feats in enumerate(current_samples):
            self._log.info(f"Finalizing sample {sample_idx + 1}/{len(current_samples)}")

            # Simulate this sample to completion
            final_sample = self._simulate_to_completion(feats, min_t, dt, [], context)
            final_samples.append(final_sample)

            # Score the final sample
            final_score = score_fn(final_sample, sample_length)
            self._log.info(f"  Final sample {sample_idx + 1} score: {final_score:.4f}")

            # Compare with best intermediate found during branching
            if final_score > best_intermediate_score:
                best_intermediate_score = final_score
                best_intermediate_sample = final_sample
                self._log.info(
                    f"  New global best found in final samples: {final_score:.4f}"
                )

        self._log.info(f"SDE PATH EXPLORATION COMPLETE")
        self._log.info(f"  Total branching steps: {len(branching_steps)}")
        self._log.info(
            f"  Branching occurred at: {[(idx, f'{t:.4f}') for idx, t in branching_steps]}"
        )
        self._log.info(f"  Global best score: {best_intermediate_score:.4f}")

        return {
            "sample": best_intermediate_sample,
            "score": best_intermediate_score,
            "method": "sde_path_exploration",
        }


class NoiseSearchInference(InferenceMethod):
    """Base noise search inference with multi-round refinement.

    Can use different noise functions (SDE, DivFree, DivFreeMax) based on noise_type.
    """

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using noise search with configurable noise type."""
        num_branches = self.config.get("num_branches", 8)
        num_keep = self.config.get("num_keep", 2)
        selector = self.config.get("selector", "tm_score")
        num_rounds = self.config.get("num_rounds", 3)
        noise_type = self.config.get("noise_type", "sde")  # sde, divfree, divfree_max

        # Get noise-specific parameters
        if noise_type == "sde":
            noise_scale = self.config.get("noise_scale", 0.05)
            self._log.info(
                f"Running Noise Search SDE with {num_branches} branches, keeping {num_keep}, {num_rounds} rounds"
            )
        elif noise_type == "divfree":
            lambda_div = self.config.get("lambda_div", 0.2)
            self._log.info(
                f"Running Noise Search DivFree with {num_branches} branches, keeping {num_keep}, {num_rounds} rounds"
            )
        elif noise_type == "divfree_max":
            lambda_div = self.config.get("lambda_div", 0.2)
            particle_repulsion_factor = self.config.get(
                "particle_repulsion_factor", 0.02
            )
            noise_schedule_end_factor = self.config.get(
                "noise_schedule_end_factor", 0.7
            )
            self._log.info(
                f"Running Noise Search DivFreeMax with {num_branches} branches, keeping {num_keep}, {num_rounds} rounds"
            )
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self._log.info(f"  Noise type: {noise_type}")

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

        # Run noise search with multiple rounds based on noise type
        if noise_type == "sde":
            sample_out = self._noise_search_sde(
                init_feats,
                num_branches,
                num_keep,
                noise_scale,
                selector,
                sample_length,
                num_rounds,
                context,
            )
        elif noise_type == "divfree":
            sample_out = self._noise_search_divfree_unified(
                init_feats,
                num_branches,
                num_keep,
                lambda_div,
                None,
                None,
                selector,
                sample_length,
                num_rounds,
                context,
                noise_type,
            )
        elif noise_type == "divfree_max":
            sample_out = self._noise_search_divfree_unified(
                init_feats,
                num_branches,
                num_keep,
                lambda_div,
                particle_repulsion_factor,
                noise_schedule_end_factor,
                selector,
                sample_length,
                num_rounds,
                context,
                noise_type,
            )

        # Extract the actual sample from the result dict and remove batch dimension
        if isinstance(sample_out, dict) and "sample" in sample_out:
            return sample_out  # Return the full dict with sample, score, method
        else:
            return sample_out

    def _noise_search_sde(
        self,
        data_init,
        num_branches,
        num_keep,
        noise_scale,
        selector,
        sample_length,
        num_rounds,
        context,
    ):
        """Core noise search SDE logic with multi-round refinement."""
        self._log.info(f"NOISE SEARCH SDE START")
        self._log.info(f"  num_branches={num_branches}, num_keep={num_keep}")
        self._log.info(f"  noise_scale={noise_scale}, num_rounds={num_rounds}")

        score_fn = self.get_score_function(selector)
        device = data_init["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Calculate round start times - new 9-round schedule
        predefined_schedule = [0.0, 0.2, 0.4, 0.6, 0.74, 0.8, 0.86, 0.92, 0.96]

        round_start_times = []
        for round_idx in range(num_rounds):
            if round_idx < len(predefined_schedule):
                # Use predefined schedule (note: these are progress values, need to convert to t values)
                progress = predefined_schedule[round_idx]
                # Convert progress to t: progress 0 = t=1.0, progress 1 = t=min_t
                start_t = 1.0 - progress * (1.0 - min_t)
            else:
                # Fallback for additional rounds beyond predefined schedule
                start_t = max(min_t, min_t + 0.01 * (num_rounds - round_idx))
            round_start_times.append(start_t)

        self._log.info(f"  Round start times: {round_start_times}")

        # Track the best samples across rounds
        current_candidates = [data_init]
        best_overall_sample = None
        best_overall_score = float("-inf")

        for round_idx in range(num_rounds):
            start_t = round_start_times[round_idx]
            self._log.info(
                f"ROUND {round_idx + 1}/{num_rounds}: Starting from t={start_t:.4f}"
            )

            round_samples = []
            round_scores = []

            for candidate_idx, candidate_feats in enumerate(current_candidates):
                self._log.info(
                    f"  Processing candidate {candidate_idx + 1}/{len(current_candidates)}"
                )

                # Generate multiple samples from this candidate
                for branch_idx in range(num_branches):
                    # Create a copy of the candidate features
                    branch_feats = tree.map_structure(
                        lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                        candidate_feats,
                    )

                    # Run SDE sampling from start_t to min_t
                    completed_sample = self._sde_simulate_from_time(
                        branch_feats, start_t, dt, reverse_steps, noise_scale, context
                    )

                    # Evaluate the completed sample
                    score = score_fn(completed_sample, sample_length)
                    round_samples.append(completed_sample)
                    round_scores.append(score)

                    self._log.debug(f"    Branch {branch_idx}: score = {score:.4f}")

                    # Track overall best
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_sample = completed_sample

            # Select top-k samples for next round (if not the last round)
            if round_idx < num_rounds - 1:
                round_scores_tensor = torch.tensor(round_scores)
                top_k_indices = torch.topk(
                    round_scores_tensor, k=min(num_keep, len(round_samples))
                )[1]

                # Extract intermediate states for selected samples at the next round's start time
                next_start_t = round_start_times[round_idx + 1]
                current_candidates = []

                for idx in top_k_indices:
                    # Get the intermediate state from the sample at next_start_t
                    intermediate_state = self._extract_intermediate_state(
                        round_samples[idx], next_start_t, reverse_steps
                    )
                    current_candidates.append(intermediate_state)

            else:
                # Last round - just find the best sample
                best_round_idx = np.argmax(round_scores)
                best_round_score = round_scores[best_round_idx]
                self._log.info(
                    f"  Best sample in final round: score = {best_round_score:.4f}"
                )

        self._log.info(f"NOISE SEARCH SDE COMPLETE")
        self._log.info(f"  Best overall score: {best_overall_score:.4f}")

        return {
            "sample": best_overall_sample,
            "score": best_overall_score,
            "method": "noise_search_sde",
        }

    def _sde_simulate_from_time(
        self, init_feats, start_t, dt, reverse_steps, noise_scale, context
    ):
        """Simulate SDE from a given start time to completion."""
        device = init_feats["rigids_t"].device
        min_t = self.sampler._fm_conf.min_t

        # Find the step index for start_t
        start_step_idx = 0
        for i, t in enumerate(reverse_steps):
            if t <= start_t:
                start_step_idx = i
                break

        # Get the steps from start_t to min_t
        simulation_steps = reverse_steps[start_step_idx:]

        # Initialize trajectory collection
        all_rigids = []
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        all_feature_states = (
            []
        )  # Store complete feature states for proper intermediate state extraction
        final_psi_pred = None

        sample_feats = init_feats.copy()

        with torch.no_grad():
            for step_idx, t in enumerate(simulation_steps):
                # Apply SDE step with noise
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Add noise for SDE
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

                # Store complete feature state for proper intermediate state extraction
                # Deep copy to ensure independence between timesteps
                feature_state_copy = {}
                for key, value in sample_feats.items():
                    if torch.is_tensor(value):
                        feature_state_copy[key] = value.clone().detach()
                    else:
                        feature_state_copy[key] = (
                            value.copy() if hasattr(value, "copy") else value
                        )
                all_feature_states.append(feature_state_copy)

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
            "feature_states_traj": all_feature_states[
                ::-1
            ],  # Complete feature states for proper intermediate extraction
        }

        # Remove batch dimension from trajectory data, but preserve feature_states_traj dimensions
        if "feature_states_traj" in sample_result:
            # Special handling: preserve feature_states_traj, apply batch removal to others
            feature_states = sample_result.pop("feature_states_traj")
            processed_result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )
            processed_result["feature_states_traj"] = feature_states

            return processed_result
        else:
            # Standard batch dimension removal
            result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )
            return result

    def _noise_search_divfree_unified(
        self,
        data_init,
        num_branches,
        num_keep,
        lambda_div,
        particle_repulsion_factor,
        noise_schedule_end_factor,
        selector,
        sample_length,
        num_rounds,
        context,
        noise_type,
    ):
        """Unified noise search divergence-free logic that works for both divfree and divfree_max."""
        method_name = f"NOISE SEARCH {noise_type.upper()}"
        self._log.info(f"{method_name} START")
        self._log.info(f"  num_branches={num_branches}, num_keep={num_keep}")

        if noise_type == "divfree":
            self._log.info(f"  lambda_div={lambda_div}, num_rounds={num_rounds}")
        elif noise_type == "divfree_max":
            self._log.info(
                f"  lambda_div={lambda_div}, repulsion={particle_repulsion_factor}, end_factor={noise_schedule_end_factor}, num_rounds={num_rounds}"
            )

        score_fn = self.get_score_function(selector)
        device = data_init["rigids_t"].device

        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Calculate round start times - new 9-round schedule
        predefined_schedule = [0.0, 0.2, 0.4, 0.6, 0.74, 0.8, 0.86, 0.92, 0.96]

        round_start_times = []
        for round_idx in range(num_rounds):
            if round_idx < len(predefined_schedule):
                # Use predefined schedule (note: these are progress values, need to convert to t values)
                progress = predefined_schedule[round_idx]
                # Convert progress to t: progress 0 = t=1.0, progress 1 = t=min_t
                start_t = 1.0 - progress * (1.0 - min_t)
            else:
                # Fallback for additional rounds beyond predefined schedule
                start_t = max(min_t, min_t + 0.01 * (num_rounds - round_idx))
            round_start_times.append(start_t)

        self._log.info(f"  Round start times: {round_start_times}")

        # Track the best samples across rounds
        current_candidates = [data_init]
        best_overall_sample = None
        best_overall_score = float("-inf")

        for round_idx in range(num_rounds):
            start_t = round_start_times[round_idx]
            self._log.info(
                f"ROUND {round_idx + 1}/{num_rounds}: Starting from t={start_t:.4f}"
            )

            round_samples = []
            round_scores = []

            for candidate_idx, candidate_feats in enumerate(current_candidates):
                self._log.info(
                    f"  Processing candidate {candidate_idx + 1}/{len(current_candidates)}"
                )

                if noise_type == "divfree_max":
                    # Use synchronized timestep approach for divfree_max (enables particle repulsion)
                    completed_samples = self._divfree_max_simulate_synchronized(
                        candidate_feats,
                        start_t,
                        dt,
                        reverse_steps,
                        lambda_div,
                        particle_repulsion_factor,
                        noise_schedule_end_factor,
                        context,
                        num_branches,
                    )

                    # Evaluate all completed samples
                    for branch_idx, completed_sample in enumerate(completed_samples):
                        score = score_fn(completed_sample, sample_length)
                        round_samples.append(completed_sample)
                        round_scores.append(score)

                        self._log.debug(f"    Branch {branch_idx}: score = {score:.4f}")

                        # Track overall best
                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_sample = completed_sample

                else:
                    # Use sequential approach for divfree (proven working method)
                    for branch_idx in range(num_branches):
                        # Create a copy of the candidate features
                        branch_feats = tree.map_structure(
                            lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                            candidate_feats,
                        )

                        # Run divergence-free sampling from start_t to min_t (sequential method)
                        completed_sample = self._divfree_simulate_from_time_unified(
                            branch_feats,
                            start_t,
                            dt,
                            reverse_steps,
                            lambda_div,
                            particle_repulsion_factor,
                            noise_schedule_end_factor,
                            context,
                            noise_type,
                        )

                        # Evaluate the completed sample
                        score = score_fn(completed_sample, sample_length)
                        round_samples.append(completed_sample)
                        round_scores.append(score)

                        self._log.debug(f"    Branch {branch_idx}: score = {score:.4f}")

                        # Track overall best
                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_sample = completed_sample

            # Select top-k samples for next round (if not the last round)
            if round_idx < num_rounds - 1:
                round_scores_tensor = torch.tensor(round_scores)
                top_k_indices = torch.topk(
                    round_scores_tensor, k=min(num_keep, len(round_samples))
                )[1]

                # Extract intermediate states for selected samples at the next round's start time
                next_start_t = round_start_times[round_idx + 1]
                current_candidates = []

                for idx in top_k_indices:
                    # Get the intermediate state from the sample at next_start_t
                    intermediate_state = self._extract_intermediate_state(
                        round_samples[idx], next_start_t, reverse_steps
                    )
                    current_candidates.append(intermediate_state)

            else:
                # Last round - just find the best sample
                best_round_idx = np.argmax(round_scores)
                best_round_score = round_scores[best_round_idx]
                self._log.info(
                    f"  Best sample in final round: score = {best_round_score:.4f}"
                )

        self._log.info(f"{method_name} COMPLETE")
        self._log.info(f"  Best overall score: {best_overall_score:.4f}")

        return {
            "sample": best_overall_sample,
            "score": best_overall_score,
            "method": f"noise_search_{noise_type}",
        }

    def _divfree_simulate_from_time_unified(
        self,
        init_feats,
        start_t,
        dt,
        reverse_steps,
        lambda_div,
        particle_repulsion_factor,
        noise_schedule_end_factor,
        context,
        noise_type,
    ):
        """Unified simulate divergence-free from a given start time to completion.

        Works for both 'divfree' and 'divfree_max' noise types.
        """
        device = init_feats["rigids_t"].device
        min_t = self.sampler._fm_conf.min_t

        # Find the step index for start_t
        start_step_idx = 0
        for i, t in enumerate(reverse_steps):
            if t <= start_t:
                start_step_idx = i
                break

        # Get the steps from start_t to min_t
        simulation_steps = reverse_steps[start_step_idx:]

        # Initialize trajectory collection
        all_rigids = []
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        all_feature_states = (
            []
        )  # Store complete feature states for proper intermediate state extraction
        final_psi_pred = None

        sample_feats = init_feats.copy()

        with torch.no_grad():
            for step_idx, t in enumerate(simulation_steps):
                # Apply divergence-free step
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Generate divergence-free noise based on noise_type
                rigids_tensor = sample_feats["rigids_t"]
                t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                # Extract rotation matrices and translations
                rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                rot_mats = rigid_obj.get_rots().get_rot_mats()
                trans_vecs = rigid_obj.get_trans()

                if noise_type == "divfree":
                    # Standard divergence-free noise
                    rot_divfree_noise = divfree_swirl_si(
                        rot_mats, t_batch, None, rot_vectorfield
                    )
                    trans_divfree_noise = divfree_swirl_si(
                        trans_vecs, t_batch, None, trans_vectorfield
                    )

                    # Add divergence-free noise to vector fields
                    rot_vectorfield = rot_vectorfield + lambda_div * rot_divfree_noise
                    trans_vectorfield = (
                        trans_vectorfield + lambda_div * trans_divfree_noise
                    )

                elif noise_type == "divfree_max":
                    # Divergence-free max noise with particle repulsion and linear schedule
                    rot_divfree_noise = divfree_max_noise(
                        rot_mats,
                        t_batch,
                        rot_vectorfield,
                        lambda_div=lambda_div,
                        repulsion_strength=particle_repulsion_factor,
                        noise_schedule_end_factor=noise_schedule_end_factor,
                        min_t=min_t,
                    )
                    trans_divfree_noise = divfree_max_noise(
                        trans_vecs,
                        t_batch,
                        trans_vectorfield,
                        lambda_div=lambda_div,
                        repulsion_strength=particle_repulsion_factor,
                        noise_schedule_end_factor=noise_schedule_end_factor,
                        min_t=min_t,
                    )

                    # Add divergence-free max noise to vector fields (already scaled in utility)
                    rot_vectorfield = rot_vectorfield + rot_divfree_noise
                    trans_vectorfield = trans_vectorfield + trans_divfree_noise

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                # Use the proven, working reverse function approach from original divfree code
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

                # Collect trajectory data using proven working approach
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Store complete feature state for proper intermediate state extraction
                # Deep copy to ensure independence between timesteps
                feature_state_copy = {}
                for key, value in sample_feats.items():
                    if torch.is_tensor(value):
                        feature_state_copy[key] = value.clone().detach()
                    else:
                        feature_state_copy[key] = (
                            value.copy() if hasattr(value, "copy") else value
                        )
                all_feature_states.append(feature_state_copy)

                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]

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
                all_bb_prots.append(du.move_to_np(atom37_t))
                final_psi_pred = psi_pred

        # Flip trajectories to correct time order
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        # Flip feature states trajectory - need to reverse the list, not stack since these are dicts
        all_feature_states = all_feature_states[::-1]

        sample_result = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
            "feature_states_traj": all_feature_states,  # Complete feature states for proper intermediate extraction
        }

        # Use proven working batch dimension handling from original divfree code, but preserve feature_states_traj dimensions
        if "feature_states_traj" in sample_result:
            # Special handling: preserve feature_states_traj, apply batch removal to others
            feature_states = sample_result.pop("feature_states_traj")
            processed_result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )
            processed_result["feature_states_traj"] = feature_states
            return processed_result
        else:
            # Standard batch dimension removal
            return tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )

    def _divfree_max_simulate_synchronized(
        self,
        candidate_feats,
        start_t,
        dt,
        reverse_steps,
        lambda_div,
        particle_repulsion_factor,
        noise_schedule_end_factor,
        context,
        num_branches,
    ):
        """Synchronized timestep simulation for divfree_max with particle repulsion between branches.

        This enables particle repulsion by processing all branches at each timestep simultaneously,
        allowing repulsion forces to be calculated between the N samples in the virtual batch.
        """
        device = candidate_feats["rigids_t"].device
        min_t = self.sampler._fm_conf.min_t

        # Find the step index for start_t
        start_step_idx = 0
        for i, t in enumerate(reverse_steps):
            if t <= start_t:
                start_step_idx = i
                break

        # Get the steps from start_t to min_t
        simulation_steps = reverse_steps[start_step_idx:]

        # Initialize all branch samples
        all_branch_feats = []
        for branch_idx in range(num_branches):
            branch_feats = tree.map_structure(
                lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                candidate_feats,
            )
            all_branch_feats.append(branch_feats)

        # Initialize trajectory collection for all branches
        all_trajectories = {
            i: {
                "rigids": [],
                "bb_prots": [],
                "trans_0_pred": [],
                "bb_0_pred": [],
                "feature_states": [],
            }
            for i in range(num_branches)
        }
        final_psi_preds = [None] * num_branches

        self._log.debug(
            f"    Starting synchronized simulation for {num_branches} branches"
        )

        with torch.no_grad():
            for step_idx, t in enumerate(simulation_steps):
                # Extract positions from all branches for repulsion calculation
                branch_positions = []
                for branch_feats in all_branch_feats:
                    rigid_obj = ru.Rigid.from_tensor_7(branch_feats["rigids_t"])
                    positions = rigid_obj.get_trans()  # [1, seq_len, 3]
                    branch_positions.append(
                        positions[0]
                    )  # Remove batch dim -> [seq_len, 3]

                # Stack into virtual batch for repulsion calculation
                batch_positions = torch.stack(
                    branch_positions, dim=0
                )  # [num_branches, seq_len, 3]

                # Calculate repulsion forces between all branches
                repulsion_forces = self._calculate_synchronized_repulsion(
                    batch_positions, particle_repulsion_factor
                )  # [num_branches, seq_len, 3]

                # Process each branch with its repulsion force
                for branch_idx, branch_feats in enumerate(all_branch_feats):
                    # Set time features
                    branch_feats = self.sampler.exp._set_t_feats(
                        branch_feats, t, torch.ones((1,)).to(device)
                    )

                    # Get model prediction
                    model_out = self.sampler.model(branch_feats)
                    rot_vectorfield = model_out["rot_vectorfield"]
                    trans_vectorfield = model_out["trans_vectorfield"]
                    rigid_pred = model_out["rigids"]
                    psi_pred = model_out["psi"]

                    # Store complete feature state for proper intermediate state extraction
                    # Deep copy to ensure independence between timesteps
                    feature_state_copy = {}
                    for key, value in branch_feats.items():
                        if torch.is_tensor(value):
                            feature_state_copy[key] = value.clone().detach()
                        else:
                            feature_state_copy[key] = (
                                value.copy() if hasattr(value, "copy") else value
                            )
                    all_trajectories[branch_idx]["feature_states"].append(
                        feature_state_copy
                    )

                    # Generate divergence-free max noise with precomputed repulsion
                    rigids_tensor = branch_feats["rigids_t"]
                    t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                    rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                    rot_mats = rigid_obj.get_rots().get_rot_mats()
                    trans_vecs = rigid_obj.get_trans()

                    # Use synchronized repulsion for translation (repulsion acts on positions)
                    trans_repulsion = repulsion_forces[
                        branch_idx : branch_idx + 1
                    ]  # [1, seq_len, 3] - add batch dim back

                    # Generate divergence-free max noise with external repulsion
                    rot_divfree_noise = self._divfree_max_noise_with_repulsion(
                        rot_mats,
                        t_batch,
                        rot_vectorfield,
                        None,  # No repulsion for rotations
                        lambda_div,
                        particle_repulsion_factor,
                        noise_schedule_end_factor,
                        min_t,
                    )
                    trans_divfree_noise = self._divfree_max_noise_with_repulsion(
                        trans_vecs,
                        t_batch,
                        trans_vectorfield,
                        trans_repulsion,
                        lambda_div,
                        particle_repulsion_factor,
                        noise_schedule_end_factor,
                        min_t,
                    )

                    # Add noise to vector fields
                    rot_vectorfield = rot_vectorfield + rot_divfree_noise
                    trans_vectorfield = trans_vectorfield + trans_divfree_noise

                    # Apply reverse step
                    fixed_mask = branch_feats["fixed_mask"] * branch_feats["res_mask"]
                    flow_mask = (1 - branch_feats["fixed_mask"]) * branch_feats[
                        "res_mask"
                    ]

                    rots_t, trans_t, rigids_t = self.sampler.flow_matcher.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(branch_feats["rigids_t"]),
                        rot_vectorfield=du.move_to_np(rot_vectorfield),
                        trans_vectorfield=du.move_to_np(trans_vectorfield),
                        flow_mask=du.move_to_np(flow_mask),
                        t=t,
                        dt=dt,
                        center=True,
                        noise_scale=1.0,
                    )
                    branch_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

                    # Collect trajectory data for this branch
                    all_trajectories[branch_idx]["rigids"].append(
                        du.move_to_np(rigids_t.to_tensor_7())
                    )
                    atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                    all_trajectories[branch_idx]["bb_prots"].append(
                        du.move_to_np(atom37_t)
                    )

                    # Calculate x0 prediction
                    gt_trans_0 = branch_feats["rigids_t"][..., 4:]
                    pred_trans_0 = rigid_pred[..., 4:]
                    trans_pred_0 = (
                        flow_mask[..., None] * pred_trans_0
                        + fixed_mask[..., None] * gt_trans_0
                    )

                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_trajectories[branch_idx]["bb_0_pred"].append(
                        du.move_to_np(atom37_0)
                    )
                    all_trajectories[branch_idx]["trans_0_pred"].append(
                        du.move_to_np(trans_pred_0)
                    )
                    final_psi_preds[branch_idx] = psi_pred

                    # Update the branch features for next timestep
                    all_branch_feats[branch_idx] = branch_feats

        # Build completed samples for all branches
        completed_samples = []
        flip = lambda x: np.flip(np.stack(x), (0,))

        for branch_idx in range(num_branches):
            traj = all_trajectories[branch_idx]

            sample_result = {
                "prot_traj": flip(traj["bb_prots"]),
                "rigid_traj": flip(traj["rigids"]),
                "trans_traj": flip(traj["trans_0_pred"]),
                "psi_pred": (
                    final_psi_preds[branch_idx][None]
                    if final_psi_preds[branch_idx] is not None
                    else None
                ),
                "rigid_0_traj": flip(traj["bb_0_pred"]),
                "feature_states_traj": list(
                    reversed(traj["feature_states"])
                ),  # Complete feature states for proper intermediate extraction
            }

            # Use proven working batch dimension handling, but preserve feature_states_traj
            if "feature_states_traj" in sample_result:
                # Special handling: preserve feature_states_traj, apply batch removal to others
                feature_states = sample_result.pop("feature_states_traj")
                processed_result = tree.map_structure(
                    lambda x: x[:, 0] if x is not None and x.ndim > 1 else x,
                    sample_result,
                )
                processed_result["feature_states_traj"] = feature_states
                completed_samples.append(processed_result)
            else:
                # Standard batch dimension removal
                sample_result = tree.map_structure(
                    lambda x: x[:, 0] if x is not None and x.ndim > 1 else x,
                    sample_result,
                )
                completed_samples.append(sample_result)

        self._log.debug(
            f"    Completed synchronized simulation for {num_branches} branches"
        )
        return completed_samples

    def _calculate_synchronized_repulsion(self, batch_positions, repulsion_strength):
        """Calculate repulsion forces between all samples in the synchronized batch.

        Args:
            batch_positions: [num_branches, seq_len, 3] - positions for all branches
            repulsion_strength: Strength factor for repulsion

        Returns:
            repulsion_forces: [num_branches, seq_len, 3] - repulsion forces for each branch
        """
        from runner.divergence_free_utils import calculate_euclidean_repulsion_forces

        # Flatten sequence dimension for repulsion calculation
        # [num_branches, seq_len, 3] -> [num_branches, seq_len*3]
        num_branches, seq_len, _ = batch_positions.shape
        flattened_positions = batch_positions.view(num_branches, -1)

        # Calculate repulsion between branches (not residues)
        repulsion_flat = calculate_euclidean_repulsion_forces(flattened_positions)

        # Reshape back to position space and scale
        repulsion_forces = repulsion_flat.view(num_branches, seq_len, 3)
        return repulsion_forces * repulsion_strength

    def _divfree_max_noise_with_repulsion(
        self,
        x,
        t_batch,
        u_t,
        external_repulsion,
        lambda_div,
        repulsion_strength,
        noise_schedule_end_factor,
        min_t,
    ):
        """Generate divergence-free max noise with optional external repulsion forces.

        This is a modified version of divfree_max_noise that can use precomputed repulsion forces
        instead of calculating them from x (which would fail for batch_size=1).
        """
        from runner.divergence_free_utils import divfree_swirl_si, make_divergence_free

        # Calculate time-dependent noise scaling
        t_scalar = t_batch[0].item()
        normalized_time = (1.0 - t_scalar) / (1.0 - min_t)
        noise_scale_factor = 1.0 + (noise_schedule_end_factor - 1.0) * normalized_time

        # Generate standard divergence-free noise
        w_divfree = divfree_swirl_si(x, t_batch, None, u_t)

        if external_repulsion is not None:
            # Use external repulsion (already computed between branches)
            # Make it divergence-free
            repulsion_divfree = make_divergence_free(
                external_repulsion, x, t_batch, u_t
            )
            total_divfree_noise = w_divfree + repulsion_divfree
        else:
            # No repulsion (e.g., for rotations)
            total_divfree_noise = w_divfree

        # Apply time-dependent scaling and lambda scaling
        return lambda_div * noise_scale_factor * total_divfree_noise

    def _extract_intermediate_state(self, completed_sample, target_t, reverse_steps):
        """Extract intermediate state from a completed sample at a specific time."""
        # Find the closest time step to target_t
        target_step_idx = 0
        for i, t in enumerate(reverse_steps):
            if t <= target_t:
                target_step_idx = i
                break

        # Extract the state at that time step from the trajectory
        # Note: trajectory is flipped, so we need to reverse the index
        trajectory_idx = len(reverse_steps) - 1 - target_step_idx

        # Check if we have complete feature states trajectory (new approach)
        if (
            "feature_states_traj" in completed_sample
            and completed_sample["feature_states_traj"] is not None
        ):
            # Use the stored complete feature states - this preserves all original features!
            if trajectory_idx >= len(completed_sample["feature_states_traj"]):
                trajectory_idx = len(completed_sample["feature_states_traj"]) - 1

            # Get the complete feature state at this timestep
            intermediate_feats = completed_sample["feature_states_traj"][trajectory_idx]

            # Ensure tensors are on correct device and cloned for independence
            device_feats = {}
            for key, value in intermediate_feats.items():
                if torch.is_tensor(value):
                    device_feats[key] = value.clone().to(self.sampler.device)
                else:
                    device_feats[key] = (
                        value.copy() if hasattr(value, "copy") else value
                    )

            return device_feats
        else:
            # Fallback to old approach (for compatibility with existing samples)
            if trajectory_idx >= len(completed_sample["rigid_traj"]):
                trajectory_idx = len(completed_sample["rigid_traj"]) - 1

            # Create features from the trajectory state
            rigids_t = torch.tensor(completed_sample["rigid_traj"][trajectory_idx]).to(
                self.sampler.device
            )

            # Add batch dimension if needed
            if rigids_t.ndim == 2:
                rigids_t = rigids_t[None]

            # Create minimal features needed for continuation
            intermediate_feats = {
                "rigids_t": rigids_t,
                "res_mask": torch.ones(rigids_t.shape[1]).to(self.sampler.device)[None],
                "fixed_mask": torch.zeros(rigids_t.shape[1]).to(self.sampler.device)[
                    None
                ],
            }

            return intermediate_feats


class NoiseSearchSDEInference(NoiseSearchInference):
    """Noise search inference with SDE using multi-round refinement."""

    def __init__(self, sampler, config):
        super().__init__(sampler, config)
        # Force noise_type to sde
        self.config["noise_type"] = "sde"


class NoiseSearchDivFreeInference(NoiseSearchInference):
    """Noise search inference with divergence-free ODE using multi-round refinement."""

    def __init__(self, sampler, config):
        super().__init__(sampler, config)
        # Force noise_type to divfree
        self.config["noise_type"] = "divfree"


class NoiseSearchDivFreeMaxInference(NoiseSearchInference):
    """Noise search inference with divergence-free max using multi-round refinement."""

    def __init__(self, sampler, config):
        super().__init__(sampler, config)
        # Force noise_type to divfree_max
        self.config["noise_type"] = "divfree_max"


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
        self._log.info(f"DIVERGENCE-FREE ODE PATH EXPLORATION START")
        self._log.info(f"  num_branches={num_branches}, num_keep={num_keep}")
        self._log.info(
            f"  lambda_div={lambda_div}, branch_start_time={branch_start_time}"
        )
        self._log.info(f"  branch_interval={branch_interval}")

        sample_feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else x.copy(), data_init
        )
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

        # Track best intermediate sample across all simulations
        best_intermediate_score = float("-inf")
        best_intermediate_sample = None

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
                    # Regular deterministic ODE flow (no noise except during branching)
                    for i, feats in enumerate(current_samples):
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

                            # Add divergence-free noise to vector fields for branching
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
                                self._log.debug(
                                    f"      Simulating branch {branch_idx} to completion..."
                                )
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

                        self._log.info(f"    Created {len(branches)} branches")

                        # Simulate branches to completion and evaluate
                        remaining_steps = reverse_steps[step_idx + 1 :]
                        self._log.info(
                            f"    Remaining steps: {len(remaining_steps)} (from t={t:.4f} to t={remaining_steps[-1]:.4f})"
                        )

                        # Score all branches and keep the best ones
                        if branches:
                            scored_branches = []
                            for branch_idx, (branch_result, branch_feats) in enumerate(
                                branches
                            ):
                                try:
                                    self._log.debug(
                                        f"      Evaluating branch {branch_idx}..."
                                    )
                                    score = score_fn(branch_result, sample_length)
                                    scored_branches.append(
                                        (score, branch_result, branch_feats)
                                    )
                                    self._log.info(
                                        f"      Branch {branch_idx}: score = {score:.4f}"
                                    )

                                    # Track best intermediate sample
                                    if score > best_intermediate_score:
                                        best_intermediate_score = score
                                        best_intermediate_sample = branch_result
                                        self._log.info(
                                            f"      New best intermediate sample: score = {score:.4f}"
                                        )

                                except Exception as e:
                                    self._log.error(
                                        f"      Branch {branch_idx} failed: {e}"
                                    )
                                    scored_branches.append(
                                        (-float("inf"), branch_result, branch_feats)
                                    )

                            # Log all branch scores
                            branch_scores = [score for score, _, _ in scored_branches]
                            self._log.info(
                                f"    Branch scores: {[f'{s:.4f}' for s in branch_scores]}"
                            )

                            # Sort by score (descending) and keep the best
                            scored_branches.sort(key=lambda x: x[0], reverse=True)
                            best_branches = scored_branches[:num_keep]

                            # Log selected branches
                            selected_indices = []
                            selected_scores = []
                            for i, (score, _, _) in enumerate(best_branches):
                                selected_indices.append(i)
                                selected_scores.append(score)

                            self._log.info(
                                f"    Selected branch indices: {selected_indices}"
                            )
                            self._log.info(
                                f"    Selected branch scores: {[f'{s:.4f}' for s in selected_scores]}"
                            )

                            # Update current samples with the best branches
                            for score, branch_result, branch_feats in best_branches:
                                new_samples.append(branch_feats)
                                self._log.debug(
                                    f"    Added branch with score {score:.4f} to continue from"
                                )

                            # CRITICAL FIX: Collect trajectory data for the selected branch during branching steps
                            if best_branches:
                                # Use the first selected branch for trajectory collection
                                _, _, selected_branch = best_branches[0]

                                # Apply the same model step to get the predictions for trajectory collection
                                selected_branch_copy = tree.map_structure(
                                    lambda x: (
                                        x.clone() if torch.is_tensor(x) else x.copy()
                                    ),
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

            self._log.info(f"DIVERGENCE-FREE ODE PATH EXPLORATION COMPLETE")
            self._log.info(f"  Total branching steps: {len(branching_steps)}")
            self._log.info(
                f"  Branching occurred at: {[(idx, f'{t:.4f}') for idx, t in branching_steps]}"
            )

        # Return best intermediate sample found during branching
        return best_intermediate_sample

    def _extract_final_sample(self, feats):
        """Extract final sample from features."""
        with torch.no_grad():
            # Get final structure prediction
            model_out = self.sampler.model(feats)
            rigid_pred = model_out["rigids"]
            psi_pred = model_out["psi"]

            # Compute backbone structure
            atom37_0 = all_atom.compute_backbone(
                ru.Rigid.from_tensor_7(rigid_pred), psi_pred
            )[0]

            # Create trajectory-like output (simplified)
            prot_traj = du.move_to_np(atom37_0)[None]  # Add time dimension
            rigid_traj = du.move_to_np(rigid_pred)[None]

            return {
                "prot_traj": prot_traj,
                "rigid_traj": rigid_traj,
                "trans_traj": du.move_to_np(rigid_pred[..., 4:])[None],
                "psi_pred": psi_pred[None] if psi_pred is not None else None,
                "rigid_0_traj": prot_traj,
            }


class DivergenceFreeMaxInference(InferenceMethod):
    """Divergence-free max inference with linear noise schedule and particle repulsion."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using divergence-free max noise with linear schedule."""
        lambda_div = self.config.get("lambda_div", 0.2)
        noise_schedule_end_factor = self.config.get("noise_schedule_end_factor", 0.7)
        particle_repulsion_factor = self.config.get("particle_repulsion_factor", 0.02)

        self._log.info(
            f"Running divergence-free max sampling with lambda_div={lambda_div}, "
            f"noise_schedule_end_factor={noise_schedule_end_factor}, "
            f"particle_repulsion_factor={particle_repulsion_factor}"
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

        # Run divergence-free max sampling
        sample_out = self._divergence_free_max_inference(
            init_feats,
            lambda_div,
            noise_schedule_end_factor,
            particle_repulsion_factor,
            context,
        )

        # Remove batch dimension like _base_sample does
        return tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

    def _divergence_free_max_inference(
        self,
        data_init,
        lambda_div,
        noise_schedule_end_factor,
        particle_repulsion_factor,
        context,
    ):
        """Core divergence-free max logic with linear noise schedule and particle repulsion."""
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
            for step_idx, t in enumerate(reverse_steps):
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Generate divergence-free max noise using the utility function
                rigids_tensor = sample_feats["rigids_t"]
                t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                # Extract rotation matrices and translations directly as torch tensors (stay on GPU)
                rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                rot_mats = rigid_obj.get_rots().get_rot_mats()  # [B, N, 3, 3]
                trans_vecs = rigid_obj.get_trans()  # [B, N, 3]

                # Generate divergence-free max noise for rotation field
                rot_divfree_noise = divfree_max_noise(
                    rot_mats,  # x
                    t_batch,  # t_batch
                    rot_vectorfield,  # u_t
                    lambda_div=lambda_div,
                    repulsion_strength=particle_repulsion_factor,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    min_t=min_t,
                )

                # Generate divergence-free max noise for translation field
                trans_divfree_noise = divfree_max_noise(
                    trans_vecs,  # x
                    t_batch,  # t_batch
                    trans_vectorfield,  # u_t
                    lambda_div=lambda_div,
                    repulsion_strength=particle_repulsion_factor,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    min_t=min_t,
                )

                # Add divergence-free max noise to vector fields
                rot_vectorfield = rot_vectorfield + rot_divfree_noise
                trans_vectorfield = trans_vectorfield + trans_divfree_noise

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

                # Calculate x0 prediction derived from vectorfield predictions
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
            "feature_states_traj": all_feature_states[
                ::-1
            ],  # Complete feature states for proper intermediate extraction
        }

        return sample_result


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
        sample_result = tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

        # Calculate and log self-consistency score
        self._calculate_and_log_self_consistency(
            sample_result, sample_length, "SDE Simple"
        )

        return sample_result

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
        sample_result = tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

        # Calculate and log self-consistency score
        self._calculate_and_log_self_consistency(
            sample_result, sample_length, "DivergenceFree Simple"
        )

        return sample_result

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


class DivFreeMaxSimpleInference(InferenceMethod):
    """Simple divergence-free max inference with noise but no branching."""

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using simple divergence-free max (no path exploration)."""
        lambda_div = self.config.get("lambda_div", 0.2)
        particle_repulsion_factor = self.config.get("particle_repulsion_factor", 0.02)
        noise_schedule_end_factor = self.config.get("noise_schedule_end_factor", 0.7)

        self._log.info(
            f"Running Simple DivFree Max with lambda_div={lambda_div}, repulsion={particle_repulsion_factor}, end_factor={noise_schedule_end_factor}"
        )

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

        # Run simple divergence-free max sampling
        sample_out = self._base_sample_divfree_max(
            init_feats,
            lambda_div,
            particle_repulsion_factor,
            noise_schedule_end_factor,
            context,
        )

        # Remove batch dimension like _base_sample does
        sample_result = tree.map_structure(
            lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_out
        )

        # Calculate and log self-consistency score
        self._calculate_and_log_self_consistency(
            sample_result, sample_length, "DivFreeMax Simple"
        )

        return sample_result

    def _base_sample_divfree_max(
        self,
        init_feats,
        lambda_div,
        particle_repulsion_factor,
        noise_schedule_end_factor,
        context=None,
    ):
        """Synchronized divergence-free max sampling for proper particle guidance."""
        device = init_feats["rigids_t"].device
        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Initialize trajectory collection
        all_rigids = [du.move_to_np(copy.deepcopy(init_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi_pred = None

        sample_feats = init_feats.copy()

        with torch.no_grad():
            for step_idx, t in enumerate(reverse_steps):
                # Set time features
                sample_feats = self.sampler.exp._set_t_feats(
                    sample_feats, t, torch.ones((1,)).to(device)
                )

                # Get model prediction
                model_out = self.sampler.model(sample_feats)

                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                psi_pred = model_out["psi"]

                # Generate divergence-free max noise using synchronized approach
                rigids_tensor = sample_feats["rigids_t"]
                t_batch = torch.full((rigids_tensor.shape[0],), t, device=device)

                # Extract rotation matrices and translations
                rigid_obj = ru.Rigid.from_tensor_7(rigids_tensor)
                rot_mats = rigid_obj.get_rots().get_rot_mats()
                trans_vecs = rigid_obj.get_trans()

                # Generate divergence-free max noise for rotation field
                rot_divfree_noise = divfree_max_noise(
                    rot_mats,
                    t_batch,
                    rot_vectorfield,
                    lambda_div=lambda_div,
                    repulsion_strength=particle_repulsion_factor,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    min_t=min_t,
                )

                # Generate divergence-free max noise for translation field
                trans_divfree_noise = divfree_max_noise(
                    trans_vecs,
                    t_batch,
                    trans_vectorfield,
                    lambda_div=lambda_div,
                    repulsion_strength=particle_repulsion_factor,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    min_t=min_t,
                )

                # Add divergence-free max noise to vector fields
                rot_vectorfield = rot_vectorfield + rot_divfree_noise
                trans_vectorfield = trans_vectorfield + trans_divfree_noise

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

        # Flip trajectories to correct time order
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        all_trans_0_pred = flip(all_trans_0_pred)
        all_bb_0_pred = flip(all_bb_0_pred)

        # Flip feature states trajectory

        sample_result = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
        }

        return sample_result


class RandomSearchDivFreeInference(DivergenceFreeODEInference):
    """Combined random search and divergence-free ODE branching method.

    First performs random search over N initial noises to select the best ones,
    then uses those selected noises as starting points for divergence-free ODE branching.
    """

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using random search + divergence-free ODE branching."""
        num_branches = self.config.get("num_branches", 4)
        num_keep = self.config.get("num_keep", 1)
        lambda_div = self.config.get("lambda_div", 0.1)
        selector = self.config.get("selector", "tm_score")
        branch_start_time = self.config.get("branch_start_time", 0.0)
        branch_interval = self.config.get("branch_interval", 0.05)

        self._log.info(
            f"Running random search + div-free ODE: {num_branches} random initial noises -> "
            f"{num_branches} div-free branches, keeping {num_keep}"
        )

        # Step 1: Random search to select best initial noises
        self._log.info(f"Phase 1: Random search over {num_branches} initial noises")

        selected_noises, best_intermediate_score, best_intermediate_sample = (
            self._random_search_phase(num_branches, selector, sample_length, context)
        )

        # Step 2: Use selected noises for divergence-free ODE branching
        self._log.info(
            f"Phase 2: Div-free ODE branching with {len(selected_noises)} selected noises"
        )

        best_sample = self._divfree_branching_phase(
            selected_noises,
            num_branches,
            num_keep,
            lambda_div,
            selector,
            sample_length,
            branch_start_time,
            branch_interval,
            context,
            best_intermediate_score,
            best_intermediate_sample,
        )

        return best_sample

    def _random_search_phase(self, num_random, selector, sample_length, context):
        """Phase 1: Random search to identify best initial noises."""
        score_fn = self.get_score_function(selector)
        candidates = []

        # Track best intermediate sample across random search
        best_intermediate_score = float("-inf")
        best_intermediate_sample = None

        for i in range(num_random):
            self._log.debug(f"  Random sample {i+1}/{num_random}")

            # Generate random initial features
            init_feats = self.generate_initial_features(sample_length)

            # Sample to completion using the specific initial features
            sample_result = self._sample_from_init_feats(init_feats, context)
            score = score_fn(sample_result, sample_length)

            candidates.append(
                {"init_feats": init_feats, "score": score, "sample": sample_result}
            )
            self._log.debug(f"    Score: {score:.4f}")

            # Track best intermediate sample
            if score > best_intermediate_score:
                best_intermediate_score = score
                best_intermediate_sample = sample_result
                self._log.info(
                    f"    New best random search sample: score = {score:.4f}"
                )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Keep top k candidates based on num_keep
        num_keep = self.config.get("num_keep", 1)
        selected = candidates[:num_keep]

        scores_str = [f"{c['score']:.4f}" for c in selected]
        self._log.info(
            f"Selected {len(selected)} best initial noises with scores: {scores_str}"
        )

        return (
            [c["init_feats"] for c in selected],
            best_intermediate_score,
            best_intermediate_sample,
        )

    def _sample_from_init_feats(self, init_feats, context):
        """Sample to completion starting from specific initial features."""
        # Use the standard method from the base class but with specific initial features
        sample_feats = tree.map_structure(
            lambda x: x.clone() if torch.is_tensor(x) else x.copy(), init_feats
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

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                # Standard ODE flow (no noise in random search phase)
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

        sample_result = {
            "prot_traj": all_bb_prots,
            "rigid_traj": all_rigids,
            "trans_traj": all_trans_0_pred,
            "psi_pred": final_psi_pred[None] if final_psi_pred is not None else None,
            "rigid_0_traj": all_bb_0_pred,
            "feature_states_traj": all_feature_states[
                ::-1
            ],  # Complete feature states for proper intermediate extraction
        }

        # Remove batch dimension from trajectory data, but preserve feature_states_traj dimensions
        if "feature_states_traj" in sample_result:
            # Special handling: preserve feature_states_traj, apply batch removal to others
            feature_states = sample_result.pop("feature_states_traj")
            processed_result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )
            processed_result["feature_states_traj"] = feature_states

            return processed_result
        else:
            # Standard batch dimension removal
            result = tree.map_structure(
                lambda x: x[:, 0] if x is not None and x.ndim > 1 else x, sample_result
            )
            return result

    def generate_initial_features(self, sample_length):
        """Generate initial features for a sample."""
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

        return init_feats

    def _divfree_branching_phase(
        self,
        selected_noises,
        num_branches,
        num_keep,
        lambda_div,
        selector,
        sample_length,
        branch_start_time,
        branch_interval,
        context,
        best_intermediate_score,
        best_intermediate_sample,
    ):
        """Phase 2: Divergence-free ODE branching from selected noises."""
        score_fn = self.get_score_function(selector)

        for i, init_noise in enumerate(selected_noises):
            self._log.debug(f"  Running div-free branching from selected noise {i+1}")

            # Run divergence-free path exploration starting from this selected noise
            # Use the parent class's method with the selected initial noise
            best_sample = self._divergence_free_path_exploration_inference(
                init_noise,
                num_branches,
                num_keep,
                lambda_div,
                score_fn,
                sample_length,
                branch_start_time,
                branch_interval,
                context,
            )

            # Update best intermediate sample if we found a better one
            if best_sample is not None:
                score = score_fn(best_sample, sample_length)
                if score > best_intermediate_score:
                    best_intermediate_score = score
                    best_intermediate_sample = best_sample

        # Return the best intermediate sample from all phases
        self._log.info(
            f"Returning best sample with score: {best_intermediate_score:.4f}"
        )
        return best_intermediate_sample


class RandomSearchNoiseInference(NoiseSearchInference):
    """Combined random search and noise search method.

    First performs random search over N initial noises to select the best ones,
    then uses those selected noises as starting points for noise search refinement.
    """

    def sample(
        self, sample_length: int, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate samples using random search + noise search."""
        num_branches = self.config.get("num_branches", 2)
        num_keep = self.config.get("num_keep", 1)
        selector = self.config.get("selector", "tm_score")
        num_rounds = self.config.get("num_rounds", 3)
        noise_type = self.config.get("noise_type", "sde")

        self._log.info(
            f"Running random search + noise search: {num_branches} random initial noises -> "
            f"noise search with {num_branches} branches, {num_rounds} rounds, keeping {num_keep}"
        )
        self._log.info(f"  Noise type: {noise_type}")

        # Step 1: Random search to select best initial noises
        self._log.info(f"Phase 1: Random search over {num_branches} initial noises")

        selected_noises, best_random_score, best_random_sample = (
            self._random_search_phase(num_branches, selector, sample_length, context)
        )

        # Step 2: Use selected noises for noise search refinement
        self._log.info(
            f"Phase 2: Noise search refinement with {len(selected_noises)} selected noises"
        )

        best_sample = self._noise_search_phase(
            selected_noises,
            num_branches,
            num_keep,
            selector,
            sample_length,
            num_rounds,
            context,
            noise_type,
            best_random_score,
            best_random_sample,
        )

        return best_sample

    def _random_search_phase(self, num_branches, selector, sample_length, context):
        """Phase 1: Random search to identify best initial noises."""
        score_fn = self.get_score_function(selector)
        candidates = []

        # Track best sample across random search
        best_random_score = float("-inf")
        best_random_sample = None

        for i in range(num_branches):
            self._log.debug(f"  Random sample {i+1}/{num_branches}")

            # Generate random initial features
            init_feats = self.generate_initial_features(sample_length)

            # Sample to completion using the specific initial features
            sample_result = self._sample_from_init_feats(init_feats, context)
            score = score_fn(sample_result, sample_length)

            candidates.append(
                {"init_feats": init_feats, "score": score, "sample": sample_result}
            )
            self._log.debug(f"    Score: {score:.4f}")

            # Track best sample
            if score > best_random_score:
                best_random_score = score
                best_random_sample = sample_result
                self._log.info(
                    f"    New best random search sample: score = {score:.4f}"
                )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Keep top k candidates based on num_keep
        selected = candidates[:num_keep]

        scores_str = [f"{c['score']:.4f}" for c in selected]
        self._log.info(
            f"Selected {len(selected)} best initial noises with scores: {scores_str}"
        )

        return (
            [c["init_feats"] for c in selected],
            best_random_score,
            best_random_sample,
        )

    def _noise_search_phase(
        self,
        selected_noises,
        num_branches,
        num_keep,
        selector,
        sample_length,
        num_rounds,
        context,
        noise_type,
        best_random_score,
        best_random_sample,
    ):
        """Phase 2: Noise search refinement from selected noises."""
        score_fn = self.get_score_function(selector)

        # Track global best across all noise search runs
        global_best_score = best_random_score
        global_best_sample = best_random_sample

        for i, init_noise in enumerate(selected_noises):
            self._log.info(
                f"  Running noise search from selected noise {i+1}/{len(selected_noises)}"
            )

            # Run noise search starting from this selected noise as the initial candidate
            if noise_type == "sde":
                noise_scale = self.config.get("noise_scale", 0.05)
                result = self._noise_search_sde_custom_init(
                    init_noise,
                    num_branches,
                    num_keep,
                    noise_scale,
                    selector,
                    sample_length,
                    num_rounds,
                    context,
                )
            elif noise_type in ["divfree", "divfree_max"]:
                lambda_div = self.config.get("lambda_div", 0.2)
                particle_repulsion_factor = self.config.get(
                    "particle_repulsion_factor", 0.02
                )
                noise_schedule_end_factor = self.config.get(
                    "noise_schedule_end_factor", 0.7
                )
                result = self._noise_search_divfree_unified(
                    init_noise,
                    num_branches,
                    num_keep,
                    lambda_div,
                    particle_repulsion_factor,
                    noise_schedule_end_factor,
                    selector,
                    sample_length,
                    num_rounds,
                    context,
                    noise_type,
                )
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

            # Update global best if we found a better one
            if result["score"] > global_best_score:
                global_best_score = result["score"]
                global_best_sample = result["sample"]
                self._log.info(
                    f"    New global best found in noise search: score = {result['score']:.4f}"
                )

        # Return the global best sample
        self._log.info(
            f"Random search + noise search complete. Global best score: {global_best_score:.4f}"
        )
        return {
            "sample": global_best_sample,
            "score": global_best_score,
            "method": f"random_search_noise_{noise_type}",
        }

    def generate_initial_features(self, sample_length):
        """Generate random initial features."""
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

        return init_feats

    def _sample_from_init_feats(self, init_feats, context):
        """Sample to completion starting from specific initial features."""
        # Use the sampler's inference function directly with the provided initial features
        sample_out = self.sampler.exp.inference_fn(
            init_feats,
            num_t=self.sampler._fm_conf.num_t,
            min_t=self.sampler._fm_conf.min_t,
            aux_traj=True,
            noise_scale=self.sampler._fm_conf.noise_scale,
            context=context,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

    def _noise_search_sde_custom_init(
        self,
        init_feats,
        num_branches,
        num_keep,
        noise_scale,
        selector,
        sample_length,
        num_rounds,
        context,
    ):
        """Modified SDE noise search that starts with custom initial features."""
        # Instead of generating random initial features, start with the provided init_feats
        score_fn = self.get_score_function(selector)

        # Set up time steps for rounds
        num_t = self.sampler._fm_conf.num_t
        min_t = self.sampler._fm_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]

        # Define round start times (same as in regular noise search)
        round_start_times = np.linspace(1.0, min_t, num_rounds + 1)[:-1]

        # Start with the provided initial features as our first candidate
        current_candidates = [init_feats]
        best_overall_score = float("-inf")
        best_overall_sample = None

        for round_idx in range(num_rounds):
            start_t = round_start_times[round_idx]
            self._log.info(
                f"ROUND {round_idx + 1}/{num_rounds}: Starting from t={start_t:.4f}"
            )

            round_samples = []
            round_scores = []

            for candidate_idx, candidate_feats in enumerate(current_candidates):
                self._log.info(
                    f"  Processing candidate {candidate_idx + 1}/{len(current_candidates)}"
                )

                for branch_idx in range(num_branches):
                    # Create a copy of the candidate features
                    branch_feats = tree.map_structure(
                        lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
                        candidate_feats,
                    )

                    # Run SDE sampling from start_t to min_t
                    completed_sample = self._sde_simulate_from_time(
                        branch_feats, start_t, dt, reverse_steps, noise_scale, context
                    )

                    # Evaluate the completed sample
                    score = score_fn(completed_sample, sample_length)
                    round_samples.append(completed_sample)
                    round_scores.append(score)

                    self._log.debug(f"    Branch {branch_idx}: score = {score:.4f}")

                    # Track overall best
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_sample = completed_sample

            # Select top-k samples for next round (if not the last round)
            if round_idx < num_rounds - 1:
                round_scores_tensor = torch.tensor(round_scores)
                top_k_indices = torch.topk(
                    round_scores_tensor, k=min(num_keep, len(round_samples))
                )[1]

                # Extract intermediate states for selected samples at the next round's start time
                next_start_t = round_start_times[round_idx + 1]
                current_candidates = []

                for idx in top_k_indices:
                    # Get the intermediate state from the sample at next_start_t
                    intermediate_state = self._extract_intermediate_state(
                        round_samples[idx], next_start_t, reverse_steps
                    )
                    current_candidates.append(intermediate_state)

        self._log.info(f"CUSTOM INIT SDE NOISE SEARCH COMPLETE")
        self._log.info(f"  Best overall score: {best_overall_score:.4f}")

        return {
            "sample": best_overall_sample,
            "score": best_overall_score,
            "method": "noise_search_sde_custom_init",
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
        "divergence_free_max": DivergenceFreeMaxInference,
        "noise_search_sde": NoiseSearchSDEInference,
        "noise_search_divfree": NoiseSearchDivFreeInference,
        "noise_search_divfree_max": NoiseSearchDivFreeMaxInference,
        "sde_simple": SDESimpleInference,
        "divergence_free_simple": DivergenceFreeSimpleInference,
        "divfree_max_simple": DivFreeMaxSimpleInference,
        "random_search_divfree": RandomSearchDivFreeInference,
        "random_search_noise": RandomSearchNoiseInference,
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown inference method: {method_name}. Available: {list(methods.keys())}"
        )

    return methods[method_name](sampler, config)

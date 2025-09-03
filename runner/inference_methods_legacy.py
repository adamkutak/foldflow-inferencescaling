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

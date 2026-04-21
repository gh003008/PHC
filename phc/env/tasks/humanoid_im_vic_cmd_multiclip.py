import json
import os
import torch

from phc.env.tasks.humanoid_im_vic_cmd_retime import HumanoidImVICCmdRetime


class HumanoidImVICCmdMultiClip(HumanoidImVICCmdRetime):
    """Multi-clip VIC with v_cmd-driven clip selection + optional retiming.

    Framework:
      At each env reset,
        1. Sample v_cmd ~ U(v_cmd_lo, v_cmd_hi) per env.
        2. Pick motion_id whose natural speed v_nat is closest to v_cmd.
        3. If multiclip_retime_enabled: per-env scale s = clamp(v_cmd / v_nat,
           s_lo, s_hi). Else: s = 1.0 (rely on clip selection alone).
        4. Effective v_cmd written back into obs: v_cmd_x = s * v_nat(clip).

    Architecture is single-network: one shared policy across all clips. Clip
    selection happens at env reset (pre-rollout); the network only ever sees
    the currently-loaded reference pose/velocity + v_cmd in its observation.
    No branching inside the network.

    AMP discriminator: positives come from the mixed 50-clip distribution via
    the standard motion_lib demo sampler. Retime-scaled velocities (per demo)
    continue to be applied via the parent HumanoidImVICCmdRetime demo path.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._multiclip_retime_enabled = env_cfg.get("multiclip_retime_enabled", True)
        self._multiclip_v_cmd_range = env_cfg.get("multiclip_v_cmd_range", [0.25, 0.85])
        # Metadata with signed v_x_mean_mid from compute_walking_direction_metadata.py.
        # Supports two schemas:
        #   (1) *_fwd_only.json      — already filtered to forward-straight clips,
        #                              each entry has v_x_mean_mid (signed).
        #   (2) *_dirmeta.json       — all clips with v_x_mean_mid + dir_class.
        #                              We filter to dir_class == "forward_straight".
        #   (3) *_v_nat.json (legacy) — magnitude-only v_mean_mid, direction-blind.
        #                              Kept as a fallback for reproducibility only.
        self._multiclip_v_nat_path = env_cfg.get(
            "multiclip_v_nat_path",
            "sample_data/amass_isaac_walking_primitive_fwd_only.json",
        )

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        meta = json.load(open(self._multiclip_v_nat_path, "r"))
        motion_keys = list(self._motion_lib.curr_motion_keys)

        # Decide the retrieval "speed key" per loaded motion, and whether the motion
        # is eligible (forward-walking) for retrieval.
        v_key = None  # float per motion_id (use |v_x| if available else legacy magnitude)
        eligible = None  # bool per motion_id
        any_sample = next(iter(meta.values())) if meta else {}
        uses_signed_vx = "v_x_mean_mid" in any_sample
        if uses_signed_vx:
            # Signed schema: eligible iff key present in filtered file (or iff dir_class is forward).
            fwd_keys = set(
                k for k, v in meta.items()
                if v.get("dir_class", "forward_straight") == "forward_straight"
            )
            v_key = []
            eligible = []
            for k in motion_keys:
                if k in meta and k in fwd_keys:
                    v_key.append(abs(meta[k]["v_x_mean_mid"]))
                    eligible.append(True)
                else:
                    v_key.append(0.0)  # dummy — masked out by eligible
                    eligible.append(False)
        else:
            # Legacy schema — direction-blind magnitude. All motions eligible, v_key = v_mean_mid.
            missing = [k for k in motion_keys if k not in meta]
            if missing:
                raise RuntimeError(
                    f"[MultiClip] {len(missing)} motion keys missing from v_nat file "
                    f"{self._multiclip_v_nat_path}: {missing[:3]}..."
                )
            v_key = [meta[k]["v_mean_mid"] for k in motion_keys]
            eligible = [True] * len(motion_keys)

        self._clip_v_nat = torch.tensor(v_key, device=self.device, dtype=torch.float32)
        self._clip_eligible = torch.tensor(eligible, device=self.device, dtype=torch.bool)

        n_eligible = int(self._clip_eligible.sum().item())
        n_total = len(motion_keys)
        if n_eligible == 0:
            raise RuntimeError(
                f"[MultiClip] No eligible (forward) clips found from {self._multiclip_v_nat_path}. "
                f"Schema uses_signed_vx={uses_signed_vx}, meta entries={len(meta)}, "
                f"motion_lib loaded {n_total}."
            )
        eligible_v = self._clip_v_nat[self._clip_eligible]
        print(
            f"[MultiClip] schema={'signed-vx' if uses_signed_vx else 'legacy-magnitude'}. "
            f"eligible clips: {n_eligible}/{n_total}. "
            f"|v_x| range=[{eligible_v.min().item():.3f}, {eligible_v.max().item():.3f}] "
            f"(p50={eligible_v.median().item():.3f}). "
            f"v_cmd_range={self._multiclip_v_cmd_range}, "
            f"retime_enabled={self._multiclip_retime_enabled}, "
            f"retime_scale_range={self._retime_scale_range}"
        )

    # ------------------------------------------------------------------
    # Override parent's random scale sampling — MultiClip derives scale from
    # (v_cmd, chosen_clip) inside _sample_ref_state. We make the parent call
    # into _resample_motion_scale (from _reset_envs) a no-op.
    # ------------------------------------------------------------------

    def _resample_motion_scale(self, env_ids):
        return

    def _resample_cmd(self, env_ids):
        # also a no-op — v_cmd is set inside _sample_ref_state
        return

    # ------------------------------------------------------------------
    # v_cmd-conditioned clip selection on reset
    # ------------------------------------------------------------------

    def _sample_ref_state(self, env_ids):
        n = env_ids.shape[0]
        if n == 0:
            return super()._sample_ref_state(env_ids)

        v_lo, v_hi = self._multiclip_v_cmd_range
        v_cmd = torch.rand(n, device=self.device) * (v_hi - v_lo) + v_lo  # [n]

        # argmin |v_nat[m] - v_cmd[i]| — restricted to eligible (forward) clips.
        # Non-eligible entries get +inf distance so they are never selected.
        v_diff = (self._clip_v_nat.unsqueeze(0) - v_cmd.unsqueeze(1)).abs()  # [n, num_clips]
        v_diff = v_diff.masked_fill(~self._clip_eligible.unsqueeze(0), float("inf"))
        chosen_ids = v_diff.argmin(dim=1)  # [n]

        # Persist chosen clip ids — _sample_ref_state of the parent reads
        # self._sampled_motion_ids[env_ids].
        self._sampled_motion_ids[env_ids] = chosen_ids

        # Per-env scale
        if self._multiclip_retime_enabled:
            s_lo, s_hi = self._retime_scale_range
            ideal_s = v_cmd / self._clip_v_nat[chosen_ids].clamp(min=1e-6)
            s = ideal_s.clamp(s_lo, s_hi)
        else:
            s = torch.ones(n, device=self.device)
        self._motion_scale[env_ids] = s

        # Observation-side v_cmd = effective speed after clip selection + retiming.
        effective_v = s * self._clip_v_nat[chosen_ids]
        self._current_cmd[env_ids, 0] = effective_v
        self._current_cmd[env_ids, 1] = 0.0
        self._current_cmd[env_ids, 2] = 0.0

        return super()._sample_ref_state(env_ids)

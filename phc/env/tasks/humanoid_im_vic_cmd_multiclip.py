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
        self._multiclip_v_cmd_range = env_cfg.get("multiclip_v_cmd_range", [0.3, 0.9])
        self._multiclip_v_nat_path = env_cfg.get(
            "multiclip_v_nat_path",
            "sample_data/amass_isaac_walking_primitive_v_nat.json",
        )

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # Build per-motion v_nat tensor aligned with motion_lib's loaded order.
        # `self._motion_lib.curr_motion_keys` is the active key-per-motion_id mapping.
        v_nat_dict = json.load(open(self._multiclip_v_nat_path, "r"))
        motion_keys = list(self._motion_lib.curr_motion_keys)
        missing = [k for k in motion_keys if k not in v_nat_dict]
        if len(missing) > 0:
            raise RuntimeError(
                f"[MultiClip] {len(missing)} motion keys missing from v_nat file "
                f"{self._multiclip_v_nat_path}: {missing[:3]}..."
            )
        self._clip_v_nat = torch.tensor(
            [v_nat_dict[k]["v_mean_mid"] for k in motion_keys],
            device=self.device,
            dtype=torch.float32,
        )
        n_clips = self._clip_v_nat.shape[0]
        print(
            f"[MultiClip] Loaded {n_clips} clips. "
            f"v_nat range=[{self._clip_v_nat.min().item():.3f}, {self._clip_v_nat.max().item():.3f}] "
            f"(p50={self._clip_v_nat.median().item():.3f}). "
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

        # argmin |v_nat[m] - v_cmd[i]| for each i
        v_diff = (self._clip_v_nat.unsqueeze(0) - v_cmd.unsqueeze(1)).abs()  # [n, num_clips]
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

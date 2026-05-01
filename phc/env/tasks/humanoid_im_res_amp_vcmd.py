"""Residual + AMP task for continuous v_cmd tracking on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
import isaacgym  # noqa: F401  # must precede torch
import torch
import numpy as np
from phc.env.tasks.humanoid_im import HumanoidIm


class HumanoidImResAMPVCmd(HumanoidIm):
    """1D v_cmd-conditioned imitation with multi-clip + a-1 hard switch.

    Adds 1-D v_cmd observation to phc_3's input, retimes the active clip
    by v_cmd/v_natural ratio, and switches active clip at midpoints with
    hysteresis (port of scripts/phc_walk_demo.py v3 logic). Reward is
    augmented with v_cmd tracking and a small imitation anchor.
    """

    # v_cmd config (spec §6)
    V_CMD_MIN = 0.76
    V_CMD_MAX = 1.23
    V_CMD_MAX_ACCEL = 0.5     # m/s² ramp rate
    V_CMD_RAMP_PROB = 0.01    # per-step prob of new target (S2)
    HYSTERESIS = 0.02

    # Reward weights (spec §5.4)
    W_TRACK = 0.5
    W_AMP = 0.3
    W_IM = 0.2
    ALPHA_TRACK = 5.0
    ALPHA_IM = 2.0

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        # v_cmd state, allocated per-env
        self._v_cmd_target = torch.full((self.num_envs,), 1.0, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), 1.0, device=self.device)
        # Natural speeds of the 3 clips loaded by motion_lib (filled in _read_dirmeta())
        self._v_natural = None  # set in _post_init
        # Active clip per env (index into _v_natural). Init = middle clip.
        self._active_clip = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._post_init()

    def _post_init(self):
        """Read clip dirmeta to populate v_natural; called once after super().__init__."""
        # Default to spec § §3 V_NATURAL if dirmeta missing
        self._v_natural = torch.tensor([0.897, 0.975, 1.068], device=self.device)

    # ------------------------------------------------------------------
    # Observation extension: +1-D v_cmd (normalized to [-1, 1])
    # ------------------------------------------------------------------

    def get_obs_size(self):
        # Original PHC obs + 1-D v_cmd (normalized to [-1, 1])
        return super().get_obs_size() + 1

    def _compute_observations(self, env_ids=None):
        """Replicate HumanoidIm._compute_observations and append v_cmd.

        We cannot call super()._compute_observations() directly because the
        parent writes `obs` (shape parent_size) into self.obs_buf whose rows
        are parent_size+1 (set by our overridden get_obs_size()), which would
        raise a PyTorch shape-mismatch error.  Instead we call the parent's
        internal sub-methods and write the augmented tensor ourselves.
        """
        from phc.utils.flags import flags  # local import mirrors parent pattern

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs

        if self.add_obs_noise and not flags.test:
            obs = obs + torch.randn_like(obs) * 0.1

        # Append normalized v_cmd as the last dimension.
        v_cmd_col = self._v_cmd_norm()[env_ids].unsqueeze(-1)   # (B, 1)
        obs_aug = torch.cat([obs, v_cmd_col], dim=-1)           # (B, parent+1)

        # obs_v == 4 uses a history buffer with a different layout; handle it
        # so we don't silently corrupt that path if someone switches configs.
        if self.obs_v == 4:
            B, N = obs_aug.shape
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs_aug[zeros], (1, self.past_track_steps))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs_aug[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs_aug

        return obs_aug

    def _v_cmd_norm(self) -> torch.Tensor:
        """Map v_cmd_ramped from [V_CMD_MIN, V_CMD_MAX] to [-1, 1]."""
        center = (self.V_CMD_MIN + self.V_CMD_MAX) * 0.5
        half = (self.V_CMD_MAX - self.V_CMD_MIN) * 0.5
        return (self._v_cmd_ramped - center) / half

    # ------------------------------------------------------------------
    # Multi-clip switching (port of scripts/phc_walk_demo.py v3 logic)
    # ------------------------------------------------------------------

    def _midpoints(self) -> tuple[float, float]:
        """Return (mid_AB, mid_BC) for the 3-clip pool."""
        v = self._v_natural
        return float((v[0] + v[1]) * 0.5), float((v[1] + v[2]) * 0.5)

    def _select_clip_for_env(self, env_id: int) -> int:
        """Return desired clip idx for env_id given its v_cmd_ramped, with hysteresis."""
        v_cmd = float(self._v_cmd_ramped[env_id].item())
        cur = int(self._active_clip[env_id].item())
        mid_ab, mid_bc = self._midpoints()
        h = self.HYSTERESIS
        if cur == 0:
            return 1 if v_cmd > mid_ab + h else 0
        if cur == 1:
            if v_cmd < mid_ab - h:
                return 0
            if v_cmd > mid_bc + h:
                return 2
            return 1
        if cur == 2:
            return 1 if v_cmd < mid_bc - h else 2
        return cur

    def _apply_clip_switches(self):
        """For each env where desired != active, atomically:
          - update _sampled_motion_ids
          - reset _motion_start_times / _motion_start_times_offset
          - reset _global_offset to align new motion's t=0 root with current humanoid root
          - clear ref_motion_cache so next query is fresh
        Mirrors scripts/phc_walk_demo.py v3 _apply_pending_clip_switch."""
        switch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_clip = self._active_clip.clone()
        for env_id in range(self.num_envs):
            desired = self._select_clip_for_env(env_id)
            if desired != int(self._active_clip[env_id].item()):
                switch_mask[env_id] = True
                new_clip[env_id] = desired
        if not switch_mask.any():
            return
        ids = switch_mask.nonzero(as_tuple=False).reshape(-1)
        self._sampled_motion_ids[ids] = new_clip[ids]
        self._motion_start_times[ids] = 0.0
        self._motion_start_times_offset[ids] = 0.0
        self.progress_buf[ids] = 0
        # Re-sync _global_offset so new motion's t=0 root aligns with current humanoid root
        times = torch.zeros_like(self._motion_start_times[ids])
        root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids[ids], times)
        new_root = root_res["root_pos"]
        self._global_offset[ids, :2] = self._humanoid_root_states[ids, :2] - new_root[:, :2]
        self._global_offset[ids, 2] = 0.0
        if hasattr(self, "ref_motion_cache"):
            self.ref_motion_cache.clear()
        self._active_clip = new_clip

    def pre_physics_step(self, actions):
        # Apply queued clip switches BEFORE the base step (so motion_lib reads
        # downstream see consistent motion_id + _global_offset).
        self._apply_clip_switches()
        return super().pre_physics_step(actions)

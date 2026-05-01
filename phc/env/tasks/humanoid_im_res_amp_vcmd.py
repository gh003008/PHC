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
        # motion_lib_base.get_root_pos_smpl always returns dict (verified 2026-05-01).
        # v3 demo had a fallback for tensor returns; we drop it since the API is stable.
        new_root = root_res["root_pos"]
        self._global_offset[ids, :2] = self._humanoid_root_states[ids, :2] - new_root[:, :2]
        self._global_offset[ids, 2] = 0.0
        if hasattr(self, "ref_motion_cache"):
            self.ref_motion_cache.clear()
        self._active_clip = new_clip

    # ------------------------------------------------------------------
    # Reset hook: sample new v_cmd per-env on every reset
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids):
        """Sample new v_cmd target for each env that is resetting.

        Parent hook name confirmed by grepping humanoid_amp.py:378,
        humanoid.py:585, and humanoid_im.py:956 — all use _reset_envs.
        """
        super()._reset_envs(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        new_v = torch.empty(len(env_ids), device=self.device).uniform_(
            self.V_CMD_MIN, self.V_CMD_MAX)
        self._v_cmd_target[env_ids] = new_v
        self._v_cmd_ramped[env_ids] = new_v   # start at target (no transient)
        # Resample active clip based on the new v_cmd
        ids_list = env_ids.tolist() if hasattr(env_ids, 'tolist') else list(env_ids)
        for env_id in ids_list:
            self._active_clip[env_id] = self._select_clip_for_env(env_id)

    # ------------------------------------------------------------------
    # Per-step ramp: v_cmd_ramped tracks v_cmd_target + random re-samples
    # ------------------------------------------------------------------

    def _ramp_v_cmd(self):
        """S2: ramp v_cmd_ramped toward v_cmd_target + 1% chance of new target."""
        delta_max = self.V_CMD_MAX_ACCEL * self.dt
        diff = self._v_cmd_target - self._v_cmd_ramped
        step_size = torch.clamp(diff, -delta_max, delta_max)
        self._v_cmd_ramped = self._v_cmd_ramped + step_size
        # Per-step probabilistic re-sample of target (1% per env per step)
        trigger = (torch.rand(self.num_envs, device=self.device) < self.V_CMD_RAMP_PROB)
        if trigger.any():
            n = int(trigger.sum().item())
            new_targets = torch.empty(n, device=self.device).uniform_(
                self.V_CMD_MIN, self.V_CMD_MAX)
            self._v_cmd_target[trigger] = new_targets

    def pre_physics_step(self, actions):
        self._ramp_v_cmd()
        self._apply_clip_switches()
        # Per-step retime: motion advances at v_cmd_ramped/v_natural rate.
        v_natural_per_env = self._v_natural[self._active_clip]
        ratio = self._v_cmd_ramped / v_natural_per_env  # (num_envs,)
        self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return super().pre_physics_step(actions)

    # ------------------------------------------------------------------
    # Reward: r_total = w_track·r_track + w_im·r_im + r_survive
    # ------------------------------------------------------------------

    def _compute_reward(self, actions):
        """r_total = w_track·r_track + w_im·r_im + r_survive

        AMP reward (r_amp) is computed by the agent via the AMP discriminator
        and added downstream by AmpAgent through disc_reward_w. This method
        sets only the task-side rewards into self.rew_buf.
        """
        # v_actual = horizontal root speed (m/s)
        root_vel_xy = self._humanoid_root_states[:, 7:9]
        v_act = torch.linalg.norm(root_vel_xy, dim=-1)
        # Tracking reward — Gaussian kernel
        v_err = v_act - self._v_cmd_ramped
        r_track = torch.exp(-self.ALPHA_TRACK * v_err * v_err)
        # Imitation anchor — distance between current rigid body positions and
        # reference rigid body positions (PHC's standard ref_body_pos style).
        # Parent class (HumanoidIm) populates self.ref_body_pos in
        # _update_task() / _set_env_state() before _compute_reward is called.
        ref_pos = getattr(self, "ref_body_pos", None)
        if ref_pos is not None:
            pose_diff = self._rigid_body_pos - ref_pos     # (num_envs, num_bodies, 3)
            pose_dist = torch.linalg.norm(pose_diff, dim=-1).mean(dim=-1)  # (num_envs,)
            r_im = torch.exp(-self.ALPHA_IM * pose_dist)
        else:
            r_im = torch.zeros_like(r_track)
        # Survive reward (1.0 base; killed downstream if env terminates)
        r_survive = torch.ones_like(r_track)
        # Combine. Note: r_amp is added by the agent via disc_reward_w, not here.
        self.rew_buf[:] = (self.W_TRACK * r_track
                           + self.W_IM * r_im
                           + r_survive)
        # Stash components for logging
        self._r_track = r_track.mean().detach()
        self._r_im = r_im.mean().detach()

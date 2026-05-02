"""V2 task class: residual + AMP + v_cmd on top of frozen phc_3.

V2 design (post 2026-05-02 v1 failure analysis):
  - Built from gates: each gate is the smallest possible addition that we
    verify with the diagnostic (PHC_ZERO_RESIDUAL=1, PHC_DIAG_DONE_PRINT=1
    → eps_len ≥ 200, ideally 1799 = full motion clip).
  - Gate 1: empty subclass (verified 2026-05-02, eps_len 1799).
  - Gate 2 (this commit): add v_cmd observation (last dim), keep
    everything else identical to Gate 1.

Subsequent gates:
  Gate 3: r_track + r_im reward (no r_survive)
  Gate 4: multi-clip switch (motion_lib level)
  Gate 5: per-step retime
  Gate 6: residual head (PHC_ZERO_RESIDUAL=1 by default)
  Gate 7: residual random init → start training

Spec: docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md
Failure analysis: 02_research_dev/260502_res_amp_vcmd_failure_analysis.md
"""
from __future__ import annotations
import os
import isaacgym  # noqa: F401  # must precede torch
import torch
from phc.env.tasks.humanoid_im import HumanoidIm

# v_cmd config — single source of truth (constants)
V_CMD_MIN = 0.76
V_CMD_MAX = 1.23
V_CMD_INIT = 1.0           # neutral, near middle clip's natural speed (0.975)
V_NATURAL = (0.897, 0.975, 1.068)  # natural speeds of clips 0/1/2 in motion file
HYSTERESIS = 0.02          # m/s deadband at midpoints to avoid clip-switch chatter

# Diagnostic: pin v_cmd to a fixed value via env var (e.g. PHC_PIN_VCMD=0.897
# pins all envs to clip 0). Used to verify Gate 3+ across all 3 clips.
_DIAG_PIN_VCMD = os.environ.get("PHC_PIN_VCMD", "")
_DIAG_PIN_VCMD_VAL = float(_DIAG_PIN_VCMD) if _DIAG_PIN_VCMD else None


class HumanoidImResAMPVCmdV2(HumanoidIm):
    """Gate 3: HumanoidIm subclass with v_cmd obs + multi-clip switching.

    Multi-clip switching: at reset and on every pre_physics_step, evaluate
    each env's v_cmd_ramped and pick the natural-speed clip closest to it.
    Switch boundaries use hysteresis to avoid chatter near midpoints.

    Network builder splits obs:
      phc_3 frozen base sees obs[:, :-1]   (original 934-D)
      residual head sees obs[:, :]         (full 935-D incl. v_cmd)

    No reward / retime / residual yet — those come in later gates.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        # v_cmd state. If PHC_PIN_VCMD is set, init to that fixed value;
        # else random uniform sample at first _reset_envs.
        init_v = _DIAG_PIN_VCMD_VAL if _DIAG_PIN_VCMD_VAL is not None else V_CMD_INIT
        self._v_cmd_target = torch.full((self.num_envs,), init_v, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), init_v, device=self.device)
        # Natural clip speeds + active clip index
        self._v_natural = torch.tensor(list(V_NATURAL), device=self.device)
        self._active_clip = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        if _DIAG_PIN_VCMD_VAL is not None:
            print(f"[V2 diag] PHC_PIN_VCMD={_DIAG_PIN_VCMD_VAL} → v_cmd pinned for all envs")

    # ------------------------------------------------------------------
    # Observation extension: +1-D v_cmd (normalized to [-1, 1])
    # ------------------------------------------------------------------

    def get_obs_size(self):
        return super().get_obs_size() + 1

    def _v_cmd_norm(self) -> torch.Tensor:
        """Map v_cmd_ramped from [V_CMD_MIN, V_CMD_MAX] to [-1, 1]."""
        center = (V_CMD_MIN + V_CMD_MAX) * 0.5
        half = (V_CMD_MAX - V_CMD_MIN) * 0.5
        return (self._v_cmd_ramped - center) / half

    def _compute_observations(self, env_ids=None):
        """Replicate HumanoidIm._compute_observations and append v_cmd.

        We cannot call super()._compute_observations() directly because the
        parent writes obs (parent_size dim) into self.obs_buf rows of size
        parent_size+1 (set by our overridden get_obs_size()), which would
        raise a PyTorch shape-mismatch error. Instead we replicate the
        parent's logic and write the augmented tensor ourselves.
        """
        from phc.utils.flags import flags

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
        v_cmd_col = self._v_cmd_norm()[env_ids].unsqueeze(-1)
        obs_aug = torch.cat([obs, v_cmd_col], dim=-1)

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

    # ------------------------------------------------------------------
    # Multi-clip switching (Gate 3)
    #   Pick clip index by v_cmd_ramped, with hysteresis at midpoints.
    #   Port of scripts/phc_walk_demo.py v3 _select_clip / _apply_pending logic.
    # ------------------------------------------------------------------

    def _midpoints(self) -> tuple[float, float]:
        v = self._v_natural
        return float((v[0] + v[1]) * 0.5), float((v[1] + v[2]) * 0.5)

    def _select_clip_for_env(self, env_id: int) -> int:
        v_cmd = float(self._v_cmd_ramped[env_id].item())
        cur = int(self._active_clip[env_id].item())
        mid_ab, mid_bc = self._midpoints()
        if cur == 0:
            return 1 if v_cmd > mid_ab + HYSTERESIS else 0
        if cur == 1:
            if v_cmd < mid_ab - HYSTERESIS:
                return 0
            if v_cmd > mid_bc + HYSTERESIS:
                return 2
            return 1
        if cur == 2:
            return 1 if v_cmd < mid_bc - HYSTERESIS else 2
        return cur

    def _apply_clip_switches(self):
        """For each env where desired clip != active, atomically:
          - update _sampled_motion_ids
          - reset _motion_start_times / _motion_start_times_offset
          - reset _global_offset to align new motion's t=0 root with current humanoid root
          - clear ref_motion_cache so next query is fresh
        """
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

    # ------------------------------------------------------------------
    # Reset hook: sample new v_cmd per-env on every reset, select clip,
    # force motion_lib's sampled_motion_ids to that clip.
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        # Sample v_cmd: pinned via env var, else uniform [V_CMD_MIN, V_CMD_MAX]
        if _DIAG_PIN_VCMD_VAL is not None:
            new_v = torch.full((len(env_ids),), _DIAG_PIN_VCMD_VAL, device=self.device)
        else:
            new_v = torch.empty(len(env_ids), device=self.device).uniform_(V_CMD_MIN, V_CMD_MAX)
        self._v_cmd_target[env_ids] = new_v
        self._v_cmd_ramped[env_ids] = new_v
        # Select active clip for each reset env based on its new v_cmd, then
        # force motion_lib's sample to that clip and re-align root offset.
        ids_list = env_ids.tolist() if hasattr(env_ids, 'tolist') else list(env_ids)
        for env_id in ids_list:
            self._active_clip[env_id] = self._select_clip_for_env(env_id)
        # Force the right clip on these envs (parent _reset_envs already sampled
        # randomly; override with our v_cmd-selected clip).
        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        self._sampled_motion_ids[env_ids_t] = self._active_clip[env_ids_t]

    def pre_physics_step(self, actions):
        # Mid-episode clip switching (with hysteresis). Skipped if pinned —
        # in pinned mode we want exactly one clip throughout.
        if _DIAG_PIN_VCMD_VAL is None:
            self._apply_clip_switches()
        # Gate 4: per-step retime. Motion advances at v_cmd_ramped/v_natural
        # rate. ratio > 1 → faster playback; ratio < 1 → slower.
        # This lets phc_3 track v_cmd values *between* the natural clip speeds.
        v_natural_per_env = self._v_natural[self._active_clip]
        ratio = self._v_cmd_ramped / v_natural_per_env
        self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return super().pre_physics_step(actions)

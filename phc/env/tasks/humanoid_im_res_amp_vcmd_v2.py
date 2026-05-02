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
V_CMD_INIT = 1.0   # neutral, near middle clip's natural speed (0.975)


class HumanoidImResAMPVCmdV2(HumanoidIm):
    """Gate 2: HumanoidIm subclass with +1-D v_cmd observation.

    Adds a single normalized v_cmd dimension to the end of the obs vector.
    Network builder is responsible for splitting:
      phc_3 frozen base sees obs[:, :-1]   (the original 934-D)
      residual head sees obs[:, :]         (full 935-D incl. v_cmd)

    No reward / multi-clip / retime / residual yet — those come in later gates.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        self._v_cmd_target = torch.full((self.num_envs,), V_CMD_INIT, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), V_CMD_INIT, device=self.device)

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

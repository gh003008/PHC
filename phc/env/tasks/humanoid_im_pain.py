"""HumanoidImPain — PHC-Pain-v0 task subclass.

Spec: docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md.
Extends HumanoidIm with an environment-side pain estimator, an optional
action guard (pd_tar pullback toward current dof_pos), and an optional
reward penalty proportional to mean per-env pain_state.

Obs dim, network, action space, and checkpoint format are unchanged by design.
"""

import torch

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.util.pain_baseline import (
    apply_pain_action_guard,
    broadcast_body_pain_to_dof,
    combine_internal_pain,
    compute_contact_pain,
    compute_joint_limit_pain,
    compute_power_pain,
    compute_torque_pain,
    update_pain_state,
)
from phc.utils.flags import flags

_VALID_MODES = {"off", "log_only", "reward_only", "guard_only", "guard_and_reward"}


class HumanoidImPain(HumanoidIm):
    """HumanoidIm with an environment-side pain proxy."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        pc = cfg["env"].get("pain", {})
        self.pain_cfg = pc
        self.pain_enabled = bool(pc.get("enabled", False))
        self.pain_mode = str(pc.get("mode", "off"))
        assert self.pain_mode in _VALID_MODES, (
            f"env.pain.mode must be one of {_VALID_MODES}; got {self.pain_mode!r}"
        )

        # Cache config into attributes to avoid dict lookup on the hot path.
        self._p_lambda = float(pc.get("lambda_p", 0.05))
        self._p_w_limit = float(pc.get("w_limit", 1.0))
        self._p_w_torque = float(pc.get("w_torque", 0.35))
        self._p_w_power = float(pc.get("w_power", 0.15))
        self._p_w_contact = float(pc.get("w_contact", 0.50))
        self._p_margin = float(pc.get("joint_limit_margin_ratio", 0.15))
        self._p_tau_scale = float(pc.get("torque_ref_scale", 0.50))
        self._p_power_ref = float(pc.get("power_ref", 5.0))
        self._p_contact_ref = float(pc.get("contact_force_ref", 150.0))
        self._p_thr = float(pc.get("pain_threshold", 0.10))
        self._p_cap = float(pc.get("pain_cap", 3.0))
        self._p_rise = float(pc.get("rise_alpha", 0.25))
        self._p_decay = float(pc.get("decay_alpha", 0.02))
        self._p_guard_gain = float(pc.get("guard_gain", 0.60))
        self._p_max_guard = float(pc.get("max_guard", 0.75))
        self._p_use_contact = bool(pc.get("use_external_contact", True))
        self._p_log_joint = bool(pc.get("log_joint_pain", True))
        self._p_log_contact = bool(pc.get("log_contact_pain", True))

        # SMPL layout assertion — v0 is SMPL-only.
        assert self.num_dof == 3 * (self.num_bodies - 1), (
            f"HumanoidImPain v0 supports SMPL layout only "
            f"(num_dof == 3*(num_bodies-1)); got num_dof={self.num_dof}, "
            f"num_bodies={self.num_bodies}."
        )

        # Per-env buffers. Allocate unconditionally so the hot path is branch-free.
        z = lambda: torch.zeros((self.num_envs, self.num_dof), device=self.device)
        zb = lambda: torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self.pain_inst = z()
        self.pain_state = z()
        self.pain_internal = z()
        self.pain_external = z()
        self.pain_contact_body = zb()
        self.pain_scalar = torch.zeros((self.num_envs,), device=self.device)

    def _update_pain_buffers(self):
        if not self.pain_enabled or self.pain_mode == "off":
            return

        q = self._dof_pos
        dq = self._dof_vel
        tau = self.dof_force_tensor

        limit_pain = compute_joint_limit_pain(
            q, self.dof_limits_lower, self.dof_limits_upper, self._p_margin
        )
        torque_pain = compute_torque_pain(tau, self.torque_limits, self._p_tau_scale)
        power_pain = compute_power_pain(tau, dq, self._p_power_ref)

        internal = combine_internal_pain(
            limit_pain, torque_pain, power_pain,
            self._p_w_limit, self._p_w_torque, self._p_w_power,
        )

        if self._p_use_contact:
            body = compute_contact_pain(
                self._contact_forces[:, : self.num_bodies, :], self._p_contact_ref
            )
            external_dof = self._p_w_contact * broadcast_body_pain_to_dof(body, self.num_dof)
            self.pain_contact_body[:] = body
        else:
            external_dof = torch.zeros_like(internal)
            self.pain_contact_body.zero_()

        inst = internal + external_dof
        self.pain_internal[:] = internal
        self.pain_external[:] = external_dof
        self.pain_inst[:] = inst
        self.pain_state[:] = update_pain_state(
            self.pain_state, inst,
            self._p_rise, self._p_decay, self._p_thr, self._p_cap,
        )
        self.pain_scalar[:] = self.pain_state.mean(dim=-1)

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        if not self.pain_enabled:
            return
        self._update_pain_buffers()

        if self.pain_mode in ("reward_only", "guard_and_reward"):
            self.rew_buf[:] = self.rew_buf[:] - self._p_lambda * self.pain_scalar

        self.extras["pain_mean"] = float(self.pain_scalar.mean().item())
        self.extras["pain_max"] = float(self.pain_scalar.max().item())
        self.extras["pain_internal_mean"] = float(self.pain_internal.mean().item())
        self.extras["pain_external_mean"] = float(self.pain_external.mean().item())

        if flags.im_eval:
            if self._p_log_joint:
                self.extras["pain_state"] = self.pain_state.detach().cpu().numpy()
            if self._p_log_contact:
                self.extras["pain_contact_body"] = self.pain_contact_body.detach().cpu().numpy()

    def _action_to_pd_targets(self, action):
        pd_tar = super()._action_to_pd_targets(action)
        if self.pain_enabled and self.pain_mode in ("guard_only", "guard_and_reward"):
            pd_tar = apply_pain_action_guard(
                pd_tar, self._dof_pos, self.pain_state,
                self._p_guard_gain, self._p_max_guard,
            )
        return pd_tar

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self.pain_enabled:
            for buf in (
                self.pain_inst, self.pain_state,
                self.pain_internal, self.pain_external,
                self.pain_contact_body,
            ):
                buf[env_ids] = 0
            self.pain_scalar[env_ids] = 0

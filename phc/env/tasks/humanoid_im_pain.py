"""HumanoidImPain — PHC-Pain-v0 task subclass.

Spec: docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md.
Extends HumanoidIm with an environment-side pain estimator, an optional
action guard (pd_tar pullback toward current dof_pos), and an optional
reward penalty proportional to mean per-env pain_state.

Obs dim, network, action space, and checkpoint format are unchanged by design.
"""

import numpy as np
import torch

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.util.pain_baseline import (
    apply_pain_action_guard,
    broadcast_body_pain_to_dof,
    combine_internal_pain,
    combine_knee_oa_load_proxy,
    compute_contact_pain,
    compute_joint_limit_pain,
    compute_knee_contact_load_proxy,
    compute_knee_moment_load_proxy,
    compute_knee_torque_load_proxy,
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
        grf_cfg = pc.get("grf_debug", {})
        self._grf_debug_enabled = bool(grf_cfg.get("enabled", False))
        self._grf_debug_scale = float(grf_cfg.get("scale", 0.0015))
        self._grf_debug_min_force = float(grf_cfg.get("min_force", 5.0))
        self._grf_debug_max_len = float(grf_cfg.get("max_len", 1.5))
        self._grf_debug_horizontal_gain = float(grf_cfg.get("horizontal_gain", 1.0))

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
        self._group_dof_idx = self._build_body_group_idx()
        self._grf_debug_body_ids = self._build_grf_debug_body_ids()

    def _build_body_group_idx(self):
        """SMPL body-name groups -> DOF index tensors. Called once in __init__.

        Groups aggregate DOF-level pain_state into 5 interpretable buckets.
        Root body (`Pelvis`) is excluded — it has no DOFs under SMPL.
        """
        groups = {
            "legs":  ["L_Hip", "R_Hip", "L_Knee", "R_Knee",
                      "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"],
            "arms":  ["L_Thorax", "R_Thorax", "L_Shoulder", "R_Shoulder",
                      "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"],
            "torso": ["Torso", "Spine", "Chest"],
            "head":  ["Neck", "Head"],
            "hands": ["L_Hand", "R_Hand"],
        }

        # SMPL DOF layout: non-root body at index i -> DOFs [3*(i-1) : 3*i].
        name_to_body_idx = {n: i for i, n in enumerate(self._body_names)}
        out = {}
        for group_name, body_names in groups.items():
            dof_idx = []
            for bn in body_names:
                bi = name_to_body_idx.get(bn)
                if bi is None or bi == 0:
                    continue
                dof_idx.extend([3 * (bi - 1), 3 * (bi - 1) + 1, 3 * (bi - 1) + 2])
            out[group_name] = torch.tensor(
                dof_idx, device=self.device, dtype=torch.long
            )
        return out

    def _build_grf_debug_body_ids(self):
        name_to_body_idx = {n: i for i, n in enumerate(self._body_names)}
        out = {}
        for side, names in {
            "left": ("L_Ankle", "L_Toe"),
            "right": ("R_Ankle", "R_Toe"),
        }.items():
            ids = [name_to_body_idx[n] for n in names if n in name_to_body_idx]
            out[side] = torch.tensor(ids, device=self.device, dtype=torch.long)
        return out

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

        for name, idx in self._group_dof_idx.items():
            self.extras[f"pain_{name}"] = float(self.pain_state[:, idx].mean().item())

        self.extras["pain_p90"] = float(torch.quantile(self.pain_scalar, 0.90).item())
        self.extras["pain_p99"] = float(torch.quantile(self.pain_scalar, 0.99).item())
        self.extras["pain_saturated_frac"] = float((self.pain_state >= 2.5).float().mean().item())

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

    def _draw_task(self):
        super()._draw_task()
        if self._grf_debug_enabled:
            self._draw_grf_debug()
        return

    def _draw_grf_debug(self):
        if not self.viewer or self.num_envs < 1:
            return

        self.gym.clear_lines(self.viewer)
        env_ptr = self.envs[0]
        colors = {
            "left": np.asarray([[0.1, 0.35, 1.0]], dtype=np.float32),
            "right": np.asarray([[1.0, 0.15, 0.1]], dtype=np.float32),
        }

        for side, body_ids in self._grf_debug_body_ids.items():
            if body_ids.numel() == 0:
                continue

            force = self._contact_forces[0, body_ids, :].sum(dim=0)
            force_norm = torch.norm(force).item()
            if force_norm < self._grf_debug_min_force:
                continue

            pos = self._rigid_body_pos[0, body_ids, :].mean(dim=0)
            draw_force = force.clone()
            draw_force[0:2] *= self._grf_debug_horizontal_gain
            draw_norm = torch.norm(draw_force).item()
            if draw_norm <= 1e-6:
                continue

            direction = draw_force / (draw_norm + 1e-6)
            length = min(force_norm * self._grf_debug_scale, self._grf_debug_max_len)
            start = pos.detach().cpu().numpy()
            end = (pos + direction * length).detach().cpu().numpy()
            color = colors[side]

            head_len = min(0.12, length * 0.25)
            direction_cpu = direction.detach().cpu()
            lateral = torch.tensor([1.0, 0.0, 0.0])
            if abs(torch.dot(direction_cpu, lateral).item()) > 0.9:
                lateral = torch.tensor([0.0, 1.0, 0.0])
            side_vec = torch.cross(direction_cpu, lateral)
            side_norm = torch.norm(side_vec).item()
            if side_norm <= 1e-6:
                continue

            shaft_verts = np.asarray([start, end], dtype=np.float32)
            self.gym.add_lines(self.viewer, env_ptr, 1, shaft_verts, color)

            if head_len <= 0.0:
                continue
            arrow_side = (side_vec / side_norm * head_len * 0.4).numpy()
            head_base = end - direction_cpu.numpy() * head_len
            head_verts = np.asarray([
                end,
                head_base + arrow_side,
                end,
                head_base - arrow_side,
            ], dtype=np.float32)
            self.gym.add_lines(
                self.viewer,
                env_ptr,
                2,
                head_verts,
                np.repeat(color, 2, axis=0),
            )
        return

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


class HumanoidImPainV1(HumanoidImPain):
    """PHC-Pain-v1 task with body-part pain observation.

    v1 intentionally changes the observation contract, so it lives in a
    distinct task class instead of modifying the v0 checkpoint-compatible task.
    Phase 1 only exposes the body-map observation contract; later phases own
    the unilateral knee pain drive and reward mechanism.
    """

    DEFAULT_PAIN_CHANNELS = (
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "back",
        "left_foot", "right_foot",
    )

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        pc = cfg["env"].get("pain", {})
        obs_cfg = pc.get("obs", {})
        self._pain_obs_enabled = bool(obs_cfg.get("enabled", pc.get("append_to_obs", False)))
        self._pain_obs_include_memory = bool(obs_cfg.get("include_memory", True))
        self._pain_body_channels = tuple(obs_cfg.get("channels", self.DEFAULT_PAIN_CHANNELS))
        self._pain_obs_values_per_channel = 1 + int(self._pain_obs_include_memory)
        self._pain_obs_dim = (
            len(self._pain_body_channels) * self._pain_obs_values_per_channel
            if self._pain_obs_enabled else 0
        )
        self._active_knee_side = str(pc.get("active_knee_side", "right"))
        assert self._active_knee_side in ("left", "right", "none"), (
            "env.pain.active_knee_side must be one of {'left', 'right', 'none'}; "
            f"got {self._active_knee_side!r}"
        )
        knee_cfg = pc.get("knee_mechanism", {})
        self._knee_sensitivity = {
            "left": float(knee_cfg.get("left_sensitivity", 1.0)),
            "right": float(knee_cfg.get("right_sensitivity", 1.0)),
        }
        self._knee_threshold = {
            "left": float(knee_cfg.get("left_threshold", pc.get("pain_threshold", 0.10))),
            "right": float(knee_cfg.get("right_threshold", pc.get("pain_threshold", 0.10))),
        }
        self._knee_w_torque = float(knee_cfg.get("w_torque", 0.80))
        self._knee_w_flex = float(knee_cfg.get("w_flex", 0.10))
        self._knee_w_rom = float(knee_cfg.get("w_rom", 0.05))
        self._knee_w_work = float(knee_cfg.get("w_work", 0.05))
        self._knee_torque_ref = float(knee_cfg.get("torque_ref", 100.0))
        self._knee_memory_alpha = float(knee_cfg.get("memory_alpha", 0.05))
        self._knee_proxy_mode = str(knee_cfg.get("proxy_mode", "torque_v13"))
        self._knee_body_weight_ref = float(knee_cfg.get("body_weight_ref", 700.0))
        self._knee_flex_compression_gain = float(knee_cfg.get("flex_compression_gain", 0.25))
        self._knee_loading_rate_ref = float(knee_cfg.get("loading_rate_ref", 1000.0))
        self._knee_kam_ref = float(knee_cfg.get("kam_ref", 50.0))
        self._knee_kfm_ref = float(knee_cfg.get("kfm_ref", 50.0))
        self._knee_w_contact_load = float(knee_cfg.get("w_contact_load", 0.60))
        self._knee_w_moment_load = float(knee_cfg.get("w_moment_load", 0.40))
        self._knee_w_legacy_torque_load = float(knee_cfg.get("w_legacy_torque_load", 0.0))
        self._knee_w_compression = float(knee_cfg.get("w_compression", 0.70))
        self._knee_w_loaded_flex = float(knee_cfg.get("w_loaded_flex", 0.20))
        self._knee_w_loading_rate = float(knee_cfg.get("w_loading_rate", 0.10))
        self._knee_w_kam = float(knee_cfg.get("w_kam", 0.70))
        self._knee_w_kfm = float(knee_cfg.get("w_kfm", 0.30))
        if self._knee_proxy_mode not in ("torque_v13", "oa_contact_v14", "oa_contact_v15"):
            raise ValueError(
                "env.pain.knee_mechanism.proxy_mode must be one of "
                "{'torque_v13', 'oa_contact_v14', 'oa_contact_v15'}; "
                f"got {self._knee_proxy_mode!r}"
            )

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        self.pain_body_state = torch.zeros(
            (self.num_envs, len(self._pain_body_channels)), device=self.device
        )
        self.pain_body_memory = torch.zeros_like(self.pain_body_state)
        self.pain_body_drive = torch.zeros_like(self.pain_body_state)
        self.knee_load_proxy = {
            "left": torch.zeros((self.num_envs,), device=self.device),
            "right": torch.zeros((self.num_envs,), device=self.device),
        }
        self.knee_drive = {
            "left": torch.zeros((self.num_envs,), device=self.device),
            "right": torch.zeros((self.num_envs,), device=self.device),
        }
        self._pain_channel_to_idx = {
            name: i for i, name in enumerate(self._pain_body_channels)
        }
        self._active_knee_channel = (
            f"{self._active_knee_side}_knee"
            if self._active_knee_side in ("left", "right") else None
        )
        if (
            self._active_knee_channel is not None
            and self._active_knee_channel not in self._pain_channel_to_idx
        ):
            raise ValueError(
                f"active knee channel {self._active_knee_channel!r} is not in "
                f"pain obs channels {self._pain_body_channels!r}"
            )
        self._knee_dof_idx = self._build_knee_dof_idx()
        self._knee_load_body_idx = self._build_knee_load_body_idx()
        self._prev_knee_compression = {
            "left": torch.zeros((self.num_envs,), device=self.device),
            "right": torch.zeros((self.num_envs,), device=self.device),
        }

    def get_obs_size(self):
        return super().get_obs_size() + getattr(self, "_pain_obs_dim", 0)

    def get_pain_obs_size(self):
        return self._pain_obs_dim

    def get_pain_obs_metadata(self):
        return {
            "channels": list(self._pain_body_channels),
            "values_per_channel": self._pain_obs_values_per_channel,
            "include_memory": self._pain_obs_include_memory,
            "active_knee_side": self._active_knee_side,
            "active_knee_channel": self._active_knee_channel,
            "obs_dim": self._pain_obs_dim,
        }

    def _compute_pain_obs(self, env_ids):
        if self._pain_obs_dim == 0:
            return torch.zeros((env_ids.shape[0], 0), device=self.device)
        if not hasattr(self, "pain_body_state"):
            return torch.zeros((env_ids.shape[0], self._pain_obs_dim), device=self.device)

        parts = [self.pain_body_state[env_ids]]
        if self._pain_obs_include_memory:
            parts.append(self.pain_body_memory[env_ids])
        return torch.cat(parts, dim=-1)

    def _build_knee_dof_idx(self):
        return {
            "left": self._dof_names.index("L_Knee") * 3 + 1,
            "right": self._dof_names.index("R_Knee") * 3 + 1,
        }

    def _build_knee_load_body_idx(self):
        return {
            "left": {
                "knee": self._body_names.index("L_Knee"),
                "ankle": self._body_names.index("L_Ankle"),
                "toe": self._body_names.index("L_Toe"),
            },
            "right": {
                "knee": self._body_names.index("R_Knee"),
                "ankle": self._body_names.index("R_Ankle"),
                "toe": self._body_names.index("R_Toe"),
            },
        }

    def _log_lower_limb_flexion_summaries(self):
        joints = {
            "left_hip": "L_Hip",
            "right_hip": "R_Hip",
            "left_knee": "L_Knee",
            "right_knee": "R_Knee",
            "left_ankle": "L_Ankle",
            "right_ankle": "R_Ankle",
        }
        for label, dof_name in joints.items():
            idx = self._dof_names.index(dof_name) * 3 + 1
            q = self._dof_pos[:, idx]
            self.extras[f"lower_limb_{label}_flex_mean_rad"] = float(q.mean().item())
            self.extras[f"lower_limb_{label}_flex_abs_rad"] = float(q.abs().mean().item())

    def _compute_knee_contact_proxy(self, side):
        ids = self._knee_load_body_idx[side]
        knee_pos = self._rigid_body_pos[:, ids["knee"], :]
        ankle_pos = self._rigid_body_pos[:, ids["ankle"], :]
        toe_pos = self._rigid_body_pos[:, ids["toe"], :]
        foot_pos = 0.5 * (ankle_pos + toe_pos)
        foot_force = (
            self._contact_forces[:, ids["ankle"], :]
            + self._contact_forces[:, ids["toe"], :]
        )
        knee_flex = self._dof_pos[:, self._knee_dof_idx[side]]

        force_mag = torch.norm(foot_force, dim=-1)
        compression = torch.maximum(force_mag, torch.relu(foot_force[..., 2])) / max(
            self._knee_body_weight_ref, 1.0e-6
        )
        loading_rate = (compression - self._prev_knee_compression[side]) / max(self.dt, 1.0e-6)
        loading_rate[self.progress_buf <= 1] = 0
        self._prev_knee_compression[side][:] = compression.detach()

        return compute_knee_contact_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            knee_flex=knee_flex,
            body_weight_ref=self._knee_body_weight_ref,
            flex_compression_gain=self._knee_flex_compression_gain,
            loading_rate=loading_rate,
            loading_rate_ref=self._knee_loading_rate_ref,
            w_compression=self._knee_w_compression,
            w_loaded_flex=self._knee_w_loaded_flex,
            w_loading_rate=self._knee_w_loading_rate,
        )

    def _compute_knee_moment_proxy(self, side):
        ids = self._knee_load_body_idx[side]
        knee_pos = self._rigid_body_pos[:, ids["knee"], :]
        ankle_pos = self._rigid_body_pos[:, ids["ankle"], :]
        toe_pos = self._rigid_body_pos[:, ids["toe"], :]
        foot_pos = 0.5 * (ankle_pos + toe_pos)
        foot_force = (
            self._contact_forces[:, ids["ankle"], :]
            + self._contact_forces[:, ids["toe"], :]
        )
        return compute_knee_moment_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            kam_ref=self._knee_kam_ref,
            kfm_ref=self._knee_kfm_ref,
            w_kam=self._knee_w_kam,
            w_kfm=self._knee_w_kfm,
        )

    def _compute_knee_proxy(self, side):
        idx = self._knee_dof_idx[side]
        q = self._dof_pos[:, idx]
        dq = self._dof_vel[:, idx]
        tau = self.dof_force_tensor[:, idx]
        q_lo = self.dof_limits_lower[idx:idx + 1]
        q_hi = self.dof_limits_upper[idx:idx + 1]
        rom_proxy = compute_joint_limit_pain(
            q.unsqueeze(-1), q_lo, q_hi, self._p_margin
        ).squeeze(-1)

        torque_load, torque_components = compute_knee_torque_load_proxy(
            tau=tau,
            dq=dq,
            rom_proxy=rom_proxy,
            asset_tau_limit=self.torque_limits[idx],
            torque_ref=self._knee_torque_ref,
            power_ref=self._p_power_ref,
            w_torque=self._knee_w_torque,
            w_flex=self._knee_w_flex,
            w_rom=self._knee_w_rom,
            w_work=self._knee_w_work,
        )
        if self._knee_proxy_mode == "torque_v13":
            return torque_load, torque_components

        contact_load, contact_components = self._compute_knee_contact_proxy(side)
        if self._knee_proxy_mode == "oa_contact_v14":
            components = dict(torque_components)
            components.update(contact_components)
            components["contact_load"] = contact_load
            components["torque_load"] = torque_load
            return contact_load, components

        moment_load, moment_components = self._compute_knee_moment_proxy(side)
        load, combined_components = combine_knee_oa_load_proxy(
            contact_load=contact_load,
            moment_load=moment_load,
            torque_load=torque_load,
            w_contact=self._knee_w_contact_load,
            w_moment=self._knee_w_moment_load,
            w_torque=self._knee_w_legacy_torque_load,
        )
        components = dict(torque_components)
        components.update(contact_components)
        components.update(moment_components)
        components.update(combined_components)
        return load, components

    def _update_pain_buffers(self):
        if not self.pain_enabled or self.pain_mode == "off":
            return

        super()._update_pain_buffers()
        if self._active_knee_side == "none":
            return

        self._log_lower_limb_flexion_summaries()

        for side in ("left", "right"):
            load_proxy, components = self._compute_knee_proxy(side)
            self.knee_load_proxy[side][:] = load_proxy
            drive = self._knee_sensitivity[side] * torch.relu(
                load_proxy - self._knee_threshold[side]
            )
            if side != self._active_knee_side:
                drive = torch.zeros_like(drive)
            self.knee_drive[side][:] = drive

            channel = f"{side}_knee"
            channel_idx = self._pain_channel_to_idx.get(channel)
            if channel_idx is None:
                continue

            self.pain_body_drive[:, channel_idx] = drive
            self.pain_body_state[:, channel_idx] = torch.clamp(
                self.pain_body_state[:, channel_idx] * (1.0 - self._p_decay)
                + self._p_rise * drive,
                0.0,
                self._p_cap,
            )
            self.pain_body_memory[:, channel_idx] = (
                (1.0 - self._knee_memory_alpha) * self.pain_body_memory[:, channel_idx]
                + self._knee_memory_alpha * self.pain_body_state[:, channel_idx]
            )

            prefix = f"pain_v1_{side}_knee"
            self.extras[f"{prefix}_load"] = float(load_proxy.mean().item())
            self.extras[f"{prefix}_drive"] = float(drive.mean().item())
            self.extras[f"{prefix}_state"] = float(self.pain_body_state[:, channel_idx].mean().item())
            for name, value in components.items():
                self.extras[f"{prefix}_{name}"] = float(value.mean().item())
            tau_abs = components["tau_abs"]
            self.extras[f"{prefix}_tau_rms"] = float(
                torch.sqrt(torch.mean(tau_abs * tau_abs)).item()
            )
            self.extras[f"{prefix}_tau_peak"] = float(tau_abs.max().item())

        self.extras["pain_v1_mechanical_proxy_note"] = (
            "synthetic thresholded OA-style knee load proxy; KAM/KFM are "
            "estimated from PHC GRF line-of-action geometry, not true contact force"
        )
        self.extras["pain_v1_proxy_mode"] = self._knee_proxy_mode
        self.extras["pain_v1_obs_enabled"] = self._pain_obs_enabled
        self.extras["pain_v1_reward_enabled"] = self.pain_mode in (
            "reward_only", "guard_and_reward",
        )

    def _compute_reward(self, actions):
        HumanoidIm._compute_reward(self, actions)
        if not self.pain_enabled:
            return

        self._update_pain_buffers()
        if self.pain_mode in ("reward_only", "guard_and_reward"):
            if self._active_knee_channel is not None:
                idx = self._pain_channel_to_idx[self._active_knee_channel]
                affected_pain = self.pain_body_state[:, idx]
            else:
                affected_pain = torch.zeros((self.num_envs,), device=self.device)
            self.rew_buf[:] = self.rew_buf[:] - self._p_lambda * affected_pain
            self.extras["pain_v1_reward_cost_mean"] = float(
                (self._p_lambda * affected_pain).mean().item()
            )

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs

        pain_obs = self._compute_pain_obs(env_ids)
        if pain_obs.shape[-1] > 0:
            obs = torch.cat([obs, pain_obs], dim=-1)

        if self.add_obs_noise and not flags.test:
            obs = obs + torch.randn_like(obs) * 0.1

        if self.obs_v == 4:
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs

        self.extras["pain_obs_dim"] = self._pain_obs_dim
        self.extras["pain_obs_channels"] = list(self._pain_body_channels)
        self.extras["pain_active_knee_side"] = self._active_knee_side

        return obs

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if hasattr(self, "pain_body_state"):
            self.pain_body_state[env_ids] = 0
            self.pain_body_memory[env_ids] = 0
            self.pain_body_drive[env_ids] = 0
            for side in ("left", "right"):
                self.knee_load_proxy[side][env_ids] = 0
                self.knee_drive[side][env_ids] = 0
                self._prev_knee_compression[side][env_ids] = 0

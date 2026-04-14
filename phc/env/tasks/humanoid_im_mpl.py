"""HumanoidImMPL — Motion Plan Layer task class.

Extends HumanoidImVIC with the observations and rewards required by the MPL
(Motion Plan Layer) architecture described in
docs/260410_motion_plan_layer_concept_v09.docx.

Phase 1 of the MPL roadmap: interface-ready environment, single monolithic policy.
Adds:
  - Command observation (v_cmd, yaw_cmd) derived from reference motion root velocity.
  - Command-tracking reward: MSE between actual root velocity/yaw rate and command.
  - Contact consistency reward: penalize foot horizontal slip while in contact.
  - Motion smoothness reward: penalize joint acceleration (jerk proxy).

Reference window for Module G input is already available via
`fut_tracks: True` + `numTrajSamples: K` in the env config (inherited from VIC).

Later phases will split the policy into frozen G + learned R, and introduce
a 5-axis impedance latent decoder. This class is the base environment for
all three stages (A/B/C).
"""

import math
import os
import torch

from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
from phc.utils.stability_utils import compute_stability
from phc.learning.module_g import (
    KP_LATENT_DIM,
    POSE_AA_DIM as G_POSE_DIM,  # 69 (MJCF DoF order)
    load_g_from_checkpoint,
    _smpl_pose_aa_to_mjcf_dof,
)


class HumanoidImMPL(HumanoidImVIC):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Phase 4: these must be set *before* super().__init__ because
        # Humanoid.__init__ calls self.get_action_size() — which our override
        # reads self._mpl_use_g to decide between (69+8) and (69+5).
        self._mpl_use_g = cfg["env"].get("mpl_use_g", False)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        env_cfg = cfg["env"]
        self._mpl_enabled = env_cfg.get("mpl_enabled", True)
        self._mpl_command_obs = env_cfg.get("mpl_command_obs", True)
        self._mpl_cmd_reward_w = env_cfg.get("mpl_cmd_reward_w", 0.3)
        self._mpl_contact_reward_w = env_cfg.get("mpl_contact_reward_w", 0.1)
        self._mpl_smoothness_reward_w = env_cfg.get("mpl_smoothness_reward_w", 0.05)
        self._mpl_contact_threshold = env_cfg.get("mpl_contact_threshold", 10.0)
        self._mpl_cmd_sigma = env_cfg.get("mpl_cmd_sigma", 0.25)  # m/s and rad/s scale

        # Phase 2 — stability margin alpha
        self._mpl_alpha_obs = env_cfg.get("mpl_alpha_obs", True)
        self._mpl_alpha_scale = env_cfg.get("mpl_alpha_scale", 0.2)      # m, XCoM safety buffer
        self._mpl_alpha_polygon_margin = env_cfg.get("mpl_alpha_polygon_margin", 0.05)
        self._mpl_balance_reward_w = env_cfg.get("mpl_balance_reward_w", 0.15)  # alpha-weighted upright reward

        # Phase 4 — Module G + residual policy R (Stage B)
        # self._mpl_use_g was set pre-super() above; read the rest here.
        self._mpl_g_checkpoint = env_cfg.get("mpl_g_checkpoint", "output/module_g.pth")
        self._mpl_residual_scale = env_cfg.get("mpl_residual_scale", 0.2)  # how far R can push q_ref off G
        self._mpl_g_ref_window_K = env_cfg.get("mpl_g_ref_window_K", 5)
        self._mpl_g_obs = env_cfg.get("mpl_g_obs", True)   # append G output to task obs so R can see it
        self._mpl_kp_latent_max = env_cfg.get("mpl_kp_latent_max", 1.5)  # clamp range
        self._mpl_g_hidden = None   # GRU hidden state, [num_layers, num_envs, hidden]

        # Current alpha cache, written in _compute_task_obs and reused in _compute_reward.
        self._alpha_buf = torch.zeros(self.num_envs, device=self.device)

        # Command buffer: [v_x, v_y, yaw_rate, |v|] in root frame
        self._cmd_buf = torch.zeros(self.num_envs, 4, device=self.device)

        # Previous dof_vel cache for smoothness reward (joint acceleration)
        self._prev_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # Previous foot XY position cache for contact-slip reward
        self._prev_foot_xy = torch.zeros(
            self.num_envs, len(self._contact_body_ids), 2, device=self.device
        )

        # Phase 4 — G output cache (written in _compute_task_obs, read in _compute_torques).
        self._g_q_ref_buf = torch.zeros(self.num_envs, G_POSE_DIM, device=self.device)     # [N, 69]
        self._g_kp_latent_buf = torch.zeros(self.num_envs, KP_LATENT_DIM, device=self.device)  # [N, 5]

        # Load frozen Module G and build the fixed 5-axis → 69-DoF impedance decoder.
        if self._mpl_use_g:
            g_path = self._mpl_g_checkpoint
            if not os.path.isabs(g_path):
                # resolve relative to PHC repo root (working directory of run.py).
                g_path = os.path.join(os.getcwd(), g_path)
            print(f"[HumanoidImMPL] loading Module G from {g_path}")
            self._g_model, self._g_cfg = load_g_from_checkpoint(g_path, device=self.device, freeze=True)
            self._mpl_g_ref_window_K = self._g_cfg.ref_window_K
            self._impedance_patterns = self._build_impedance_patterns()  # [5, 69]
            print(
                f"[HumanoidImMPL] G loaded: hidden={self._g_cfg.hidden_dim} "
                f"K={self._g_cfg.ref_window_K} pose_dim={self._g_cfg.pose_dim}; "
                f"residual_scale={self._mpl_residual_scale}"
            )

        print(
            f"[HumanoidImMPL] MPL enabled={self._mpl_enabled} cmd_obs={self._mpl_command_obs} "
            f"cmd_w={self._mpl_cmd_reward_w} contact_w={self._mpl_contact_reward_w} "
            f"smooth_w={self._mpl_smoothness_reward_w} "
            f"alpha_obs={self._mpl_alpha_obs} balance_w={self._mpl_balance_reward_w} "
            f"use_g={self._mpl_use_g}"
        )

    # ------------------------------------------------------------- Phase 4

    def _build_impedance_patterns(self):
        """Fixed 5-axis → 69-DoF decoder used to produce per-joint log-Kp scales.

        `kp_scale_per_dof = patterns.T @ kp_latent` (additive in log-Kp space).
        `kp_final = base_kp * exp(kp_scale_per_dof)`, Kd scaled identically.

        The five axes are semantically fixed:
          0: stance sagittal stiffness  — all leg DoFs (+1.0)
          1: swing compliance           — all leg DoFs (-0.5, opposite direction)
          2: landing damping            — ankle/toe DoFs (+1.0)
          3: lateral stabilization      — hip roll + ankle roll proxies (+1.0)
          4: overall stiffness scale    — all DoFs (+0.5)

        v1 note: stance/swing side gating via contact state is a Phase 5 TODO;
        here axes 0/1 apply symmetrically across both legs and the policy learns
        to modulate via its δKp residual.
        """
        D = 69
        P = torch.zeros(KP_LATENT_DIM, D, device=self.device)
        # MJCF DoF layout (see HumanoidImVIC._build_ccf_group_dof_map):
        #   0-2 L_Hip, 3-5 L_Knee, 6-8 L_Ankle, 9-11 L_Toe,
        #   12-14 R_Hip, 15-17 R_Knee, 18-20 R_Ankle, 21-23 R_Toe,
        #   24-68 upper body
        L_leg = list(range(0, 12))
        R_leg = list(range(12, 24))
        L_ankle_toe = list(range(6, 12))
        R_ankle_toe = list(range(18, 24))
        L_hip = list(range(0, 3))
        R_hip = list(range(12, 15))
        L_ankle = list(range(6, 9))
        R_ankle = list(range(18, 21))
        all_dofs = list(range(D))

        P[0, L_leg + R_leg] = 1.0
        P[1, L_leg + R_leg] = -0.5
        P[2, L_ankle_toe + R_ankle_toe] = 1.0
        P[3, L_hip + R_hip + L_ankle + R_ankle] = 1.0
        P[4, all_dofs] = 0.5
        return P

    def _decode_impedance(self, kp_latent):
        """Map [N, 5] latent → [N, 69] multiplicative Kp scale (`exp` applied)."""
        kp_latent = torch.clamp(kp_latent, -self._mpl_kp_latent_max, self._mpl_kp_latent_max)
        log_scale = kp_latent @ self._impedance_patterns   # [N, 69]
        return torch.exp(log_scale)

    def _run_module_g(self, env_ids, motion_res):
        """Run frozen Module G forward on the selected envs at the current step.

        All inputs come from the *reference* motion (trained distribution):
          pose_aa  — motion's current MJCF-order DoF pose [B, 69]
          root_vel — motion's root linear velocity (world) [B, 3]
          command  — already computed by caller and stored in self._cmd_buf
          ref_win  — next K frames of the motion pose_aa

        Updates self._g_q_ref_buf[env_ids], self._g_kp_latent_buf[env_ids] in place.
        """
        B = env_ids.shape[0]
        K = self._mpl_g_ref_window_K

        # 1) Current pose_aa (MJCF-order 69) from motion library (SMPL canonical 72 -> 69).
        pose_aa_72 = motion_res["motion_aa"]  # [B, 72]
        pose_aa_69 = _smpl_pose_aa_to_mjcf_dof(pose_aa_72)  # [B, 69]
        root_vel = motion_res["root_vel"]     # [B, 3]

        # 2) Reference window: peek K future motion frames via the motion lib.
        # Use the cached motion_times if available, else reconstruct.
        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )
        ref_window = torch.zeros(B, K * G_POSE_DIM, device=self.device)
        for k in range(K):
            tk = motion_times + (k + 1) * self.dt
            motion_ids_k = self._sampled_motion_ids[env_ids]
            # Direct motion_lib call (avoids polluting the ref_motion_cache).
            mres_k = self._motion_lib.get_motion_state(motion_ids_k, tk, offset=self._global_offset[env_ids])
            pa_k_72 = mres_k["motion_aa"]                               # [B, 72]
            pa_k_69 = _smpl_pose_aa_to_mjcf_dof(pa_k_72)               # [B, 69]
            ref_window[:, k * G_POSE_DIM:(k + 1) * G_POSE_DIM] = pa_k_69

        # 3) Command already computed and stored in self._cmd_buf by caller.
        cmd = self._cmd_buf[env_ids]   # [B, 4]

        # 4) Forward pass with T=1 (step-by-step, hidden state maintained via a cache).
        pose_t = pose_aa_69.unsqueeze(1).float()        # [B, 1, 69]
        vel_t = root_vel.unsqueeze(1).float()            # [B, 1, 3]
        cmd_t = cmd.unsqueeze(1).float()                 # [B, 1, 4]
        ref_t = ref_window.unsqueeze(1).float()          # [B, 1, K*69]

        with torch.no_grad():
            q_ref, kp_latent, _ = self._g_model(pose_t, vel_t, cmd_t, ref_t, hidden=None)
            # q_ref, kp_latent shapes: [B, 1, 69], [B, 1, 5]

        self._g_q_ref_buf[env_ids] = q_ref.squeeze(1)
        self._g_kp_latent_buf[env_ids] = kp_latent.squeeze(1)

    # --------------------------------------------------------- action size

    def get_action_size(self):
        """Phase 4: when G is enabled, the action becomes (δq_69, δKp_latent_5) = 74D.

        Otherwise we defer to VIC's action space (69 + n_groups = 77 by default).
        """
        if getattr(self, "_mpl_use_g", False):
            return self._num_actions + KP_LATENT_DIM
        return super().get_action_size()

    # ------------------------------------------------------------------ obs

    def get_task_obs_size(self):
        obs_size = super().get_task_obs_size()
        # Note: this is called from Humanoid.__init__ *before* our MPL attrs are
        # set by __init__, so every attr must use getattr with a default.
        if self._enable_task_obs and getattr(self, "_mpl_command_obs", True):
            obs_size += 4  # (v_x, v_y, yaw_rate, |v|)
        if self._enable_task_obs and getattr(self, "_mpl_alpha_obs", True):
            obs_size += 1  # alpha stability margin
        if self._enable_task_obs and getattr(self, "_mpl_use_g", False) and getattr(self, "_mpl_g_obs", True):
            obs_size += G_POSE_DIM + KP_LATENT_DIM   # G output (69 + 5 = 74)
        return obs_size

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        obs = super()._compute_task_obs(env_ids=env_ids, save_buffer=save_buffer)

        if not self._enable_task_obs:
            return obs

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if not self._mpl_command_obs and not self._mpl_alpha_obs and not getattr(self, "_mpl_use_g", False):
            return obs

        # Command is derived from the *reference* motion's root velocity at current time.
        # This matches the concept-doc prescription: command labels are computed from motion,
        # not hand-annotated.
        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )
        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )
        ref_root_rot = motion_res["root_rot"]        # [B, 4] wxyz? (PHC uses xyzw — see below)
        ref_root_vel = motion_res["root_vel"]        # [B, 3] world frame
        ref_root_ang_vel = motion_res["root_ang_vel"]  # [B, 3] world frame

        # Transform world-frame linear velocity into the reference root's local frame
        # (so the command is heading-invariant — the policy only sees "forward v_x, lateral v_y").
        v_local = _world_to_local_linvel(ref_root_vel, ref_root_rot)
        v_cmd_x = v_local[..., 0:1]
        v_cmd_y = v_local[..., 1:2]
        yaw_rate = ref_root_ang_vel[..., 2:3]  # world-z yaw rate (treadmill = upright, OK)
        speed_mag = torch.linalg.norm(ref_root_vel[..., :2], dim=-1, keepdim=True)

        cmd_obs = torch.cat([v_cmd_x, v_cmd_y, yaw_rate, speed_mag], dim=-1)

        if save_buffer:
            self._cmd_buf[env_ids] = cmd_obs

        if self._mpl_command_obs:
            obs = torch.cat([obs, cmd_obs], dim=-1)

        # ---- Phase 2: stability alpha ----
        if self._mpl_alpha_obs:
            alpha = self._compute_alpha(env_ids)
            if save_buffer:
                self._alpha_buf[env_ids] = alpha
            obs = torch.cat([obs, alpha.unsqueeze(-1)], dim=-1)

        # ---- Phase 4: frozen Module G forward + cache ----
        if getattr(self, "_mpl_use_g", False):
            # Reuse the motion_res we already fetched above for the command obs.
            self._run_module_g(env_ids, motion_res)
            if self._mpl_g_obs:
                g_obs = torch.cat(
                    [self._g_q_ref_buf[env_ids], self._g_kp_latent_buf[env_ids]], dim=-1
                )
                obs = torch.cat([obs, g_obs], dim=-1)

        return obs

    def _compute_alpha(self, env_ids):
        """Compute stability margin alpha for the given env_ids.

        Uses contact forces + foot XY positions + CoM derived from rigid body state.
        CoM is approximated as the root body position/velocity because PHC does not
        expose a dedicated CoM tensor on the fast path; for walking on flat ground
        this is a reasonable proxy (hip height ≈ CoM height + constant offset).
        """
        contact_forces = self._contact_forces[env_ids][:, self._contact_body_ids, :]
        contact_positions = self._rigid_body_pos[env_ids][:, self._contact_body_ids, :]

        root_pos = self._rigid_body_pos[env_ids, 0, :]     # [B, 3]
        root_vel = self._rigid_body_vel[env_ids, 0, :]     # [B, 3]

        result = compute_stability(
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            com_pos=root_pos,
            com_vel=root_vel,
            contact_threshold=self._mpl_contact_threshold,
            polygon_margin=self._mpl_alpha_polygon_margin,
            alpha_scale=self._mpl_alpha_scale,
        )
        return result["alpha"]

    # --------------------------------------------------------------- reward

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        if not self._mpl_enabled:
            return

        # (a) Command tracking — actual root velocity vs command (derived from motion).
        if self._mpl_cmd_reward_w > 0:
            cmd_reward = self._compute_command_reward()
            cmd_reward[self.progress_buf <= 3] = 0
            self.rew_buf[:] += cmd_reward
            self.reward_raw = torch.cat([self.reward_raw, cmd_reward[:, None]], dim=-1)

        # (b) Contact-consistency — penalize foot horizontal slip while in contact.
        if self._mpl_contact_reward_w > 0:
            contact_reward = self._compute_contact_reward()
            contact_reward[self.progress_buf <= 3] = 0
            self.rew_buf[:] += contact_reward
            self.reward_raw = torch.cat([self.reward_raw, contact_reward[:, None]], dim=-1)

        # (c) Motion smoothness — penalize joint acceleration magnitude.
        if self._mpl_smoothness_reward_w > 0:
            smooth_reward = self._compute_smoothness_reward()
            smooth_reward[self.progress_buf <= 3] = 0
            self.rew_buf[:] += smooth_reward
            self.reward_raw = torch.cat([self.reward_raw, smooth_reward[:, None]], dim=-1)

        # (d) Alpha-weighted balance reward — encourages upright posture when
        #     the stability margin is tight. (1 - alpha) * tracking is implicitly
        #     handled because the tracking reward already dominates above; here
        #     we add an explicit alpha-scaled upright term.
        if self._mpl_balance_reward_w > 0:
            balance_reward = self._compute_balance_reward()
            balance_reward[self.progress_buf <= 3] = 0
            self.rew_buf[:] += balance_reward
            self.reward_raw = torch.cat([self.reward_raw, balance_reward[:, None]], dim=-1)

        # Refresh caches for next step.
        self._prev_dof_vel = self._dof_vel.detach().clone()
        foot_xy_now = self._rigid_body_pos[:, self._contact_body_ids, :2].detach().clone()
        self._prev_foot_xy = foot_xy_now

    def _compute_command_reward(self):
        """Reward = exp(-||v_actual - v_cmd||^2 / sigma^2).

        Uses the command that was written into `self._cmd_buf` during `_compute_task_obs`.
        Both actual and target velocities are expressed in the actual root's local frame
        so the reward is heading-invariant.
        """
        actual_root_rot = self._rigid_body_rot[:, 0, :]
        actual_root_vel = self._rigid_body_vel[:, 0, :]
        actual_root_ang_vel = self._rigid_body_ang_vel[:, 0, :]

        v_actual_local = _world_to_local_linvel(actual_root_vel, actual_root_rot)
        yaw_rate_actual = actual_root_ang_vel[..., 2]

        v_cmd_x = self._cmd_buf[..., 0]
        v_cmd_y = self._cmd_buf[..., 1]
        yaw_cmd = self._cmd_buf[..., 2]

        lin_err = (v_actual_local[..., 0] - v_cmd_x) ** 2 + (v_actual_local[..., 1] - v_cmd_y) ** 2
        ang_err = (yaw_rate_actual - yaw_cmd) ** 2
        total_err = lin_err + 0.1 * ang_err

        reward = torch.exp(-total_err / (self._mpl_cmd_sigma ** 2))
        return reward * self._mpl_cmd_reward_w

    def _compute_contact_reward(self):
        """Penalize horizontal foot slip while the foot is in contact with the ground.

        For each contact body:
          * if contact force > threshold → expected zero XY displacement between frames
          * reward = exp(-||delta_xy||^2 / sigma^2) where sigma ~= 2 cm/frame.
          * if not in contact → reward is 1 (don't penalize swing motion).
        """
        contact_forces = self._contact_forces[:, self._contact_body_ids, :]  # [N, C, 3]
        foot_xy_now = self._rigid_body_pos[:, self._contact_body_ids, :2]     # [N, C, 2]

        contact_mag = contact_forces.norm(dim=-1)                              # [N, C]
        in_contact = (contact_mag > self._mpl_contact_threshold).float()       # [N, C]

        delta_xy = foot_xy_now - self._prev_foot_xy                             # [N, C, 2]
        slip_sq = (delta_xy ** 2).sum(dim=-1)                                   # [N, C]
        sigma_sq = (0.02) ** 2                                                  # ~2 cm/frame budget

        per_foot_reward = in_contact * torch.exp(-slip_sq / sigma_sq) + (1.0 - in_contact) * 1.0
        reward = per_foot_reward.mean(dim=-1)  # [N]
        return reward * self._mpl_contact_reward_w

    def _compute_balance_reward(self):
        """Alpha-weighted upright reward.

        When alpha is large (near-fall) we explicitly reward an upright root
        orientation; when alpha is small we don't care. This is the "continuous
        recovery" side of the MPL task-following ↔ recovery blend.
        """
        # Root up-vector alignment: SMPL upright = world +z. Extract R * [0,0,1].
        root_rot = self._rigid_body_rot[:, 0, :]  # [N, 4] xyzw
        qx, qy, qz, qw = root_rot[..., 0], root_rot[..., 1], root_rot[..., 2], root_rot[..., 3]
        # Third column of rotation matrix (world-frame up-direction of body-z axis):
        up_x = 2.0 * (qx * qz + qw * qy)
        up_y = 2.0 * (qy * qz - qw * qx)
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright = up_z.clamp(min=0.0)  # 1 when fully upright, 0 when tipped >=90°
        reward = self._alpha_buf * upright
        return reward * self._mpl_balance_reward_w

    def _compute_smoothness_reward(self):
        """Penalize joint acceleration magnitude (finite-difference of dof_vel).

        Uses exp(-||accel||^2 / sigma^2) so the reward is always in [0, w].
        """
        accel = (self._dof_vel - self._prev_dof_vel) / max(self.dt, 1e-6)  # [N, dof]
        accel_sq = (accel ** 2).mean(dim=-1)
        sigma_sq = 1000.0  # tuned so typical walking accel maps to ~0.6 reward
        reward = torch.exp(-accel_sq / sigma_sq)
        return reward * self._mpl_smoothness_reward_w

    # ------------------------------------------------ Phase 4 action/torque

    def pre_physics_step(self, actions):
        """Copy incoming actions to self.actions with no VIC-curriculum masking.

        For Phase 4 with G enabled, the last 5 dims are δKp_latent (residuals on
        top of G's nominal impedance). We do NOT want VIC's stage-1 masking to
        zero these out, so we skip the parent's masking.
        """
        if getattr(self, "_mpl_use_g", False):
            self.actions = actions.to(self.device).clone()
            return
        return super().pre_physics_step(actions)

    def _compute_torques(self, actions):
        """Phase 4 torque computation: G nominal + R residual + fixed impedance decoder.

        Action layout when `_mpl_use_g=True`:
          * actions[:, :69]    — δq_ref residual on joint position targets
          * actions[:, 69:74]  — δKp_latent residual on G's nominal 5D impedance

        Final q_ref = G_nominal_q + residual_scale * δq  (MJCF DoF order, 69 dims)
        Final kp_latent = G_nominal_kp + δKp_latent
        Per-DoF Kp scale = exp(pattern.T @ kp_latent)  (5-axis → 69 fixed decoder)
        Torque = base_Kp * scale * (q_ref - q) - base_Kd * scale * qdot

        Falls back to VIC's torque computation when G is disabled.
        """
        if not getattr(self, "_mpl_use_g", False):
            return super()._compute_torques(actions)

        # Lazy init of PD gains, matching VIC's behavior.
        if not hasattr(self, "_vic_gains_initialized") or not self._vic_gains_initialized:
            self._init_pd_gains_from_mjcf()
            self._vic_gains_initialized = True

        # Split action.
        delta_q = actions[:, :self._num_actions]                  # [N, 69]
        delta_kp = actions[:, self._num_actions:self._num_actions + KP_LATENT_DIM]  # [N, 5]

        # Combine with G nominal. G output is in absolute radian space
        # (trained on motion-lib dof_pos), so we do NOT go through
        # _action_to_pd_targets — that method scales policy [-1, 1] into
        # radian space, which we don't want here. The residual δq is in the
        # policy's [-1, 1] space, so we scale it by _pd_action_scale (per-DoF
        # joint-range weighting) * residual_scale (global shrink) before adding.
        scaled_residual = self._pd_action_scale * (self._mpl_residual_scale * delta_q)
        pd_tar = self._g_q_ref_buf + scaled_residual                              # [N, 69] absolute radians
        kp_latent = self._g_kp_latent_buf + delta_kp                              # [N, 5]

        # Fixed 5-axis impedance decoder → per-DoF multiplicative Kp/Kd scale.
        kp_scale = self._decode_impedance(kp_latent)                              # [N, 69]
        kp = self.p_gains * kp_scale
        kd = self.d_gains * kp_scale

        torques = kp * (pd_tar - self._dof_pos) - kd * self._dof_vel
        return torch.clamp(torques, -self.torque_limits, self.torque_limits)

    # ---------------------------------------------------------- reset hook

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        # Clear caches so reward at the first post-reset step is not spurious.
        self._prev_dof_vel[env_ids] = 0
        self._prev_foot_xy[env_ids] = (
            self._rigid_body_pos[env_ids][:, self._contact_body_ids, :2].detach().clone()
        )
        self._cmd_buf[env_ids] = 0
        if getattr(self, "_mpl_use_g", False):
            self._g_q_ref_buf[env_ids] = 0
            self._g_kp_latent_buf[env_ids] = 0


# ---------------------------------------------------------------- helpers

def _world_to_local_linvel(v_world, q_root_xyzw):
    """Rotate a world-frame linear velocity into the root body's local frame.

    PHC/IsaacGym quaternions are (x, y, z, w). We only need the yaw component
    of the root rotation for a heading-invariant command, but doing the full
    inverse-quaternion rotation is cheaper and more general.
    """
    # Normalize to guard against any tiny drift.
    q = q_root_xyzw
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # v_local = q^{-1} * v_world * q  (quaternion sandwich using conjugate).
    # Using the standard formula: v' = v + 2 * q_vec × (q_vec × v + q_w * v)
    # but with conjugate quaternion (negate vector part).
    cx, cy, cz, cw = -qx, -qy, -qz, qw
    vx, vy, vz = v_world[..., 0], v_world[..., 1], v_world[..., 2]

    # t = 2 * cross(c_vec, v)
    tx = 2.0 * (cy * vz - cz * vy)
    ty = 2.0 * (cz * vx - cx * vz)
    tz = 2.0 * (cx * vy - cy * vx)

    # v_local = v + cw * t + cross(c_vec, t)
    lx = vx + cw * tx + (cy * tz - cz * ty)
    ly = vy + cw * ty + (cz * tx - cx * tz)
    lz = vz + cw * tz + (cx * ty - cy * tx)

    return torch.stack([lx, ly, lz], dim=-1)

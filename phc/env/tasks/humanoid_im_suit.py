"""WalkOn Suit-native imitation learning task.

Inherits from `Humanoid` (the base IsaacGym env class) and replaces the SMPL
imitation pipeline with a 12-DoF Suit pipeline:
  - Reference motion comes from `SuitMotionLib` (loads L1 IK retargeted .pkl).
  - Observations: base self-obs (rigid body state, written by base class)
    + target features (12 target q + 12 target qvel + 3 dx_base + 1 dyaw).
  - Reward: joint-angle, joint-vel, base-height, base-linear-vel, alive bonus.
  - Reset: Reference State Initialization (RSI) — random clip + start time.
  - Termination: base-link below 0.4 m, or end-of-clip.

We do NOT inherit from HumanoidIm / HumanoidAMPTask since their SMPL-specific
paths conflict with the Suit URDF (24-joint axis-angle pose, AMP discriminator,
etc.). PHC's PPO backbone in `rl_games` only needs the standard
(obs, action, reward, reset) gym interface that BaseTask provides.
"""
from __future__ import annotations

import torch

from phc.env.tasks.humanoid import Humanoid
from phc.utils.motion_lib_suit import SuitMotionLib


class HumanoidImSuit(Humanoid):
    """WalkOn Suit imitation learning task."""

    # 12 + 12 + 3 + 1
    _NUM_TARGET_OBS = 28

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._motion_file = cfg["env"]["motion_file"]
        self._motion_single_clip_idx = int(cfg["env"].get("single_clip_idx", -1))
        self._motion_base_z_offset = float(cfg["env"].get("motion_base_z_offset", 0.0))
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._motion_lib = SuitMotionLib(
            self._motion_file, device=self.device,
            base_z_offset=self._motion_base_z_offset,
            single_clip_idx=self._motion_single_clip_idx,
        )
        self._motion_ids = self._motion_lib.sample_motions(self.num_envs)
        self._motion_start_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_elapsed = torch.zeros(self.num_envs, device=self.device)

    # ----- obs sizing -----
    # Base class calls get_obs_size() BEFORE super().__init__ sets up obs_buf,
    # so include target features here.
    def get_obs_size(self):
        return self.get_self_obs_size() + self._NUM_TARGET_OBS

    # VecTaskPythonWrapper probes these even for non-AMP tasks.
    def get_num_amp_obs(self):
        return 0

    def get_num_enc_amp_obs(self):
        return 0

    # ----- observation -----
    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self_obs = self._compute_humanoid_obs(env_ids)
        target_obs = self._compute_target_obs(env_ids)
        self.obs_buf[env_ids] = torch.cat([self_obs, target_obs], dim=-1)

    def _compute_target_obs(self, env_ids):
        ids = self._motion_ids[env_ids]
        times = self._motion_elapsed[env_ids] + self._motion_start_times[env_ids]
        state = self._motion_lib.get_motion_state(ids, times)

        cur_base_pos = self._rigid_body_pos[env_ids, 0]
        dx_base = state["base_xyz"] - cur_base_pos

        # Current yaw from base quat (xyzw)
        qx, qy, qz, qw = self._rigid_body_rot[env_ids, 0].unbind(-1)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        cur_yaw = torch.atan2(siny_cosp, cosy_cosp)
        dyaw = (state["base_yaw"] - cur_yaw).unsqueeze(-1)

        return torch.cat([state["suit_q"], state["suit_qvel"], dx_base, dyaw], dim=-1)

    # ----- reward -----
    def _compute_reward(self, actions):
        state = self._motion_lib.get_motion_state(
            self._motion_ids,
            self._motion_elapsed + self._motion_start_times,
        )

        # Joints retargeted with content (LH_EXT, LK_EXT, LA_PLA, RH_EXT, RK_EXT, RA_PLA)
        active_idx = torch.tensor([2, 3, 5, 8, 9, 11], device=self.device)
        # Joints with near-zero reference (ABD/ROT/INV both sides)
        passive_idx = torch.tensor([0, 1, 4, 6, 7, 10], device=self.device)

        q_err_active = (self._dof_pos[:, active_idx] - state["suit_q"][:, active_idx]).pow(2).sum(-1)
        q_err_passive = (self._dof_pos[:, passive_idx] - state["suit_q"][:, passive_idx]).pow(2).sum(-1)
        r_q = torch.exp(-2.0 * q_err_active) * 0.5 + torch.exp(-2.0 * q_err_passive) * 0.1

        qv_err = (self._dof_vel - state["suit_qvel"]).pow(2).sum(-1)
        r_qv = torch.exp(-0.1 * qv_err) * 0.15

        cur_base = self._rigid_body_pos[:, 0]
        h_err = (cur_base[:, 2] - state["base_xyz"][:, 2]).pow(2)
        r_h = torch.exp(-50.0 * h_err) * 0.1

        cur_v = self._rigid_body_vel[:, 0]
        v_err = (cur_v - state["base_xyz_vel"]).pow(2).sum(-1)
        r_v = torch.exp(-1.0 * v_err) * 0.1

        r_alive = torch.full_like(r_q, 0.05)

        self.rew_buf[:] = r_q + r_qv + r_h + r_v + r_alive

    # ----- termination -----
    def _compute_reset(self):
        base_z = self._rigid_body_pos[:, 0, 2]
        # Natural standing LINK_BASE z ≈ 1.20 m (after SuitMotionLib base_z_offset).
        # Threshold at 0.7 m flags a clear fall while tolerating moderate squat.
        fell = base_z < 0.7

        clip_durations = self._motion_lib.motion_durations[self._motion_ids]
        end_of_clip = (self._motion_elapsed + self._motion_start_times) >= clip_durations

        timeout = self.progress_buf >= self.max_episode_length - 1

        reset = fell | end_of_clip | timeout
        terminate = fell  # early termination signals "fell", not natural end

        self.reset_buf[:] = reset.long()
        self._terminate_buf[:] = terminate.long()

    # ----- reset (RSI) -----
    def _reset_actors(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return

        self._motion_ids[env_ids] = self._motion_lib.sample_motions(n)
        # Single-clip curriculum: always start at frame 0 of the clip (typically
        # a static standing pose). Avoids spawning in the middle of a stride
        # where the reference has forward base velocity, which the policy
        # cannot exploit without contact.
        if self._motion_single_clip_idx >= 0:
            self._motion_start_times[env_ids] = 0.0
        else:
            self._motion_start_times[env_ids] = self._motion_lib.sample_time(self._motion_ids[env_ids])
        self._motion_elapsed[env_ids] = 0.0

        state = self._motion_lib.get_motion_state(
            self._motion_ids[env_ids], self._motion_start_times[env_ids]
        )

        # DoF
        self._dof_pos[env_ids] = state["suit_q"]
        self._dof_vel[env_ids] = state["suit_qvel"]

        # Root: yaw-only orientation. The reference base_xyz already encodes a
        # per-frame foot-lift correction (v2 pkl) so feet sit ~3 cm above
        # ground at spawn; no extra RSI lift needed.
        yaw = state["base_yaw"]
        cy = torch.cos(yaw / 2)
        sy = torch.sin(yaw / 2)
        zero = torch.zeros_like(cy)
        root_quat = torch.stack([zero, zero, sy, cy], dim=-1)  # xyzw
        root_pos = state["base_xyz"].clone()

        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_quat
        # Force zero base velocity at spawn. Reference base_xyz_vel from
        # walking clips can have ~1 m/s forward momentum, which carries the
        # robot through the air before feet establish ground contact.
        self._humanoid_root_states[env_ids, 7:10] = 0.0
        self._humanoid_root_states[env_ids, 10:13] = 0.0

    # ----- step bookkeeping -----
    def post_physics_step(self):
        self._motion_elapsed += self.dt
        super().post_physics_step()

    # Humanoid._physics_step calls self.render(i=0) but Humanoid.render
    # takes only sync_frame_time. HumanoidIm overrides this same way.
    def render(self, sync_frame_time=False, i=0):
        super().render(sync_frame_time)

import torch
from phc.env.tasks.humanoid_im_vic import HumanoidImVIC


class HumanoidImVICCmd(HumanoidImVIC):
    """VIC with velocity-command conditioning (Substage A - speed only by default).

    Adds per-env velocity command to task observation and reward:
      task_obs += [v_cmd_x, v_cmd_y, w_cmd_yaw]  (3 dims, appended)
      rew       = (1 - w_cmd) * base_rew + w_cmd * cmd_track_rew
        cmd_track_rew = exp(-k_v * ||v_pelvis_xy - v_cmd_xy||^2 - k_w * |w_pelvis_yaw - w_cmd|^2)

    Command resampled at env reset.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._cmd_v_range = env_cfg.get("cmd_v_range", [0.8, 1.3])
        self._cmd_w_range = env_cfg.get("cmd_w_range", [0.0, 0.0])
        self._cmd_tracking_w = env_cfg.get("cmd_tracking_w", 0.3)
        self._cmd_track_kv = env_cfg.get("cmd_track_kv", 2.0)
        self._cmd_track_kw = env_cfg.get("cmd_track_kw", 1.0)

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._current_cmd = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._resample_cmd(torch.arange(self.num_envs, device=self.device))

        print(f"[HumanoidImVICCmd] Command-conditioned VIC enabled. "
              f"v_range={self._cmd_v_range}, w_range={self._cmd_w_range}, "
              f"cmd_tracking_w={self._cmd_tracking_w}, kv={self._cmd_track_kv}, kw={self._cmd_track_kw}")

    def get_task_obs_size(self):
        return super().get_task_obs_size() + 3

    def _resample_cmd(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        n = len(env_ids)
        v_lo, v_hi = self._cmd_v_range
        w_lo, w_hi = self._cmd_w_range
        self._current_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (v_hi - v_lo) + v_lo
        self._current_cmd[env_ids, 1] = 0.0
        self._current_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (w_hi - w_lo) + w_lo

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._resample_cmd(env_ids)

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        base_obs = super()._compute_task_obs(env_ids=env_ids, save_buffer=save_buffer)
        if env_ids is None:
            cmd = self._current_cmd
        else:
            cmd = self._current_cmd[env_ids]
        return torch.cat([base_obs, cmd], dim=-1)

    def _compute_reward(self, actions):
        super()._compute_reward(actions)

        root_vel = self._rigid_body_vel[:, 0, :]
        root_ang_vel_world = self._rigid_body_ang_vel[:, 0, :]
        yaw_rate = root_ang_vel_world[:, 2]

        v_err_sq = (root_vel[:, 0] - self._current_cmd[:, 0]) ** 2 + \
                   (root_vel[:, 1] - self._current_cmd[:, 1]) ** 2
        w_err_sq = (yaw_rate - self._current_cmd[:, 2]) ** 2
        cmd_reward = torch.exp(-self._cmd_track_kv * v_err_sq - self._cmd_track_kw * w_err_sq)
        cmd_reward[self.progress_buf <= 3] = 0

        self.rew_buf[:] = (1.0 - self._cmd_tracking_w) * self.rew_buf + self._cmd_tracking_w * cmd_reward
        self.reward_raw = torch.cat([self.reward_raw, cmd_reward[:, None]], dim=-1)

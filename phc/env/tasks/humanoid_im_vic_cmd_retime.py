import torch
from phc.env.tasks.humanoid_im_vic_cmd import HumanoidImVICCmd

_VEL_KEYS = ("root_vel", "root_ang_vel", "dof_vel", "body_vel", "body_ang_vel")


class HumanoidImVICCmdRetime(HumanoidImVICCmd):
    """VIC with command-conditioned reference retiming (Baseline B, Retime R1).

    Instead of sampling an absolute velocity command independent of the motion,
    this class samples a per-env speed scale `s ~ U(s_lo, s_hi)` and retimes
    the reference motion to play at rate `s`. Reference body/joint velocities
    are multiplied by `s` to match the retimed kinematics (chain rule).

    Observation compatibility: `v_cmd_x = s * v_nat` is written into the same
    3-dim cmd slot used by the parent class, so the policy still sees a
    velocity command in the task obs.

    Retiming mechanism: a per-env scale tensor is applied inside a wrapper
    around `_get_state_from_motionlib_cache`. Callers (`_compute_task_obs`,
    `_compute_reward`) set a context env_ids before calling super(), and the
    wrapper rewrites motion_times to
        retimed = start + s * (motion_times - start)
    and post-scales velocity fields in the returned dict by s.

    AMP demos: the discriminator features include absolute root/body vels
    (humanoid_amp.py build_amp_observations). To keep AMP distribution
    consistent with retimed rollouts, demo velocity fields are also scaled
    by a freshly-sampled s per demo entry when building AMP demos.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._retime_scale_range = env_cfg.get("retime_scale_range", [0.9, 1.1])
        self._retime_v_nat = env_cfg.get("retime_v_nat", 0.85)

        # Keep cmd_v_range consistent with the scale range so any plumbing
        # downstream that relies on cmd_v_range sees a sensible value.
        env_cfg["cmd_v_range"] = [
            self._retime_scale_range[0] * self._retime_v_nat,
            self._retime_scale_range[1] * self._retime_v_nat,
        ]
        env_cfg["cmd_w_range"] = [0.0, 0.0]

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._motion_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self._retime_env_ids_ctx = None
        self._retime_amp_scale_ctx = None
        self._resample_motion_scale(torch.arange(self.num_envs, device=self.device))

        print(f"[HumanoidImVICCmdRetime] Retimed reference enabled. "
              f"s_range={self._retime_scale_range}, v_nat={self._retime_v_nat}, "
              f"derived cmd_v_range={env_cfg['cmd_v_range']}")

    # ------------------------------------------------------------------
    # per-env scale sampling
    # ------------------------------------------------------------------

    def _resample_motion_scale(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        n = len(env_ids)
        lo, hi = self._retime_scale_range
        self._motion_scale[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo
        # Sync the v_cmd observation slot so the policy sees v_cmd_x = s * v_nat.
        self._current_cmd[env_ids, 0] = self._motion_scale[env_ids] * self._retime_v_nat
        self._current_cmd[env_ids, 1] = 0.0
        self._current_cmd[env_ids, 2] = 0.0

    def _resample_cmd(self, env_ids):
        # Disable the parent's random v_cmd sampler — cmd is derived from scale.
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._resample_motion_scale(env_ids)

    # ------------------------------------------------------------------
    # retimed motion time lookup
    # ------------------------------------------------------------------

    def _retimed_motion_times(self, env_ids, motion_times_raw):
        s = self._motion_scale[env_ids]
        start = self._motion_start_times[env_ids] + self._motion_start_times_offset[env_ids]
        m = env_ids.shape[0]
        n = motion_times_raw.shape[0]
        if n == m:
            pass
        elif m > 0 and n % m == 0:
            k = n // m
            s = s.repeat_interleave(k)
            start = start.repeat_interleave(k)
        else:
            return motion_times_raw, None
        elapsed = motion_times_raw - start
        retimed = start + s * elapsed
        return retimed, s

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        # AMP demo path: time not retimed, only velocities scaled by pre-sampled s.
        if self._retime_amp_scale_ctx is not None and self._retime_amp_scale_ctx.shape[0] == motion_times.shape[0]:
            result = super()._get_state_from_motionlib_cache(motion_ids, motion_times, offset)
            s = self._retime_amp_scale_ctx
            for key in _VEL_KEYS:
                if key in result and result[key] is not None:
                    v = result[key]
                    result[key] = v * s.view(-1, *([1] * (v.ndim - 1)))
            return result

        # Training/rollout path: retime motion_times AND scale velocities.
        if self._retime_env_ids_ctx is not None:
            retimed, s = self._retimed_motion_times(self._retime_env_ids_ctx, motion_times)
            if s is None:
                return super()._get_state_from_motionlib_cache(motion_ids, motion_times, offset)
            result = super()._get_state_from_motionlib_cache(motion_ids, retimed, offset)
            for key in _VEL_KEYS:
                if key in result and result[key] is not None:
                    v = result[key]
                    result[key] = v * s.view(-1, *([1] * (v.ndim - 1)))
            return result

        return super()._get_state_from_motionlib_cache(motion_ids, motion_times, offset)

    # ------------------------------------------------------------------
    # wrap observation + reward to install the env_ids retime context
    # ------------------------------------------------------------------

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        if env_ids is None:
            ctx = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            ctx = env_ids
        self._retime_env_ids_ctx = ctx
        try:
            return super()._compute_task_obs(env_ids, save_buffer)
        finally:
            self._retime_env_ids_ctx = None

    def _compute_reward(self, actions):
        self._retime_env_ids_ctx = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        try:
            return super()._compute_reward(actions)
        finally:
            self._retime_env_ids_ctx = None

    # ------------------------------------------------------------------
    # AMP demo velocity scaling
    # ------------------------------------------------------------------

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        lo, hi = self._retime_scale_range
        b = motion_ids.shape[0]
        # `build_amp_obs_demo` in parent expands to (b * num_amp_obs_steps) entries internally
        # before calling `_get_state_from_motionlib_cache`. We therefore pre-sample
        # `num_amp_obs_steps` scaled entries: one scale per (demo, step) pair.
        num_steps = self._num_amp_obs_steps
        s_demo = torch.rand(b, device=self.device) * (hi - lo) + lo
        s_flat = s_demo.unsqueeze(-1).expand(-1, num_steps).flatten().contiguous()
        self._retime_amp_scale_ctx = s_flat
        try:
            return super().build_amp_obs_demo(motion_ids, motion_times0)
        finally:
            self._retime_amp_scale_ctx = None

    def build_amp_obs_demo_steps(self, motion_ids, motion_times0, num_steps):
        lo, hi = self._retime_scale_range
        b = motion_ids.shape[0]
        s_demo = torch.rand(b, device=self.device) * (hi - lo) + lo
        s_flat = s_demo.unsqueeze(-1).expand(-1, num_steps).flatten().contiguous()
        self._retime_amp_scale_ctx = s_flat
        try:
            return super().build_amp_obs_demo_steps(motion_ids, motion_times0, num_steps)
        finally:
            self._retime_amp_scale_ctx = None

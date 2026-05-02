"""V3 task class: residual + AMP + v_cmd on top of frozen phc_3.

V3 redesign rationale (vs V2, see 01_research_docs/260502_personalized_diverse_motion_roadmap.md):

V2 had a "clip-master, v_cmd-slave" reset:
    1. parent picks RANDOM clip via motion_lib.sample_motions()
    2. v_cmd is then OVERWRITTEN to v_natural[clip] ± 0.05 jitter
  → consequence: each episode's v_cmd was constrained to a narrow window
    around its clip's natural speed → policy never learned wide v_cmd response
    within a single episode → mid-episode `_apply_clip_switches` never
    triggered (dead code) → multi-clip exploitation underutilized.

V3 fix — reverse master/slave:
    1. sample v_cmd UNIFORMLY in [V_CMD_MIN, V_CMD_MAX]
    2. pick the clip whose v_natural is CLOSEST to v_cmd
    3. force motion_lib's sample_motions() to return that clip
    4. parent's state init now happens on the v_cmd-selected clip
       → state ↔ reference ↔ v_cmd all consistent
  → consequence: each episode trains a (v_cmd, closest_clip) pair across
    the full v_cmd range → policy learns to exploit each clip's strengths
    in its own v_cmd window. Mid-episode v_cmd schedule (TODO future work)
    will then trigger _apply_clip_switches for real.

Implementation note: motion_lib.sample_motions() is called inside the
parent's _sample_ref_state (HumanoidAMP line 481). We inject our forced
clip ids by monkey-patching the bound method at __init__ to consume a
"pending" buffer that _reset_envs writes BEFORE calling super().

Spec basis: docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md
            (V3 inherits V2's network / agent / AMP setup unchanged)
"""
from __future__ import annotations
import os
import isaacgym  # noqa: F401  # must precede torch
import torch
from phc.env.tasks.humanoid_im import HumanoidIm

# v_cmd config (same as V2)
V_CMD_MIN = 0.76
V_CMD_MAX = 1.23
V_CMD_INIT = 1.0
V_NATURAL = (0.897, 0.975, 1.068)
HYSTERESIS = 0.02

_DIAG_PIN_VCMD = os.environ.get("PHC_PIN_VCMD", "")
_DIAG_PIN_VCMD_VAL = float(_DIAG_PIN_VCMD) if _DIAG_PIN_VCMD else None


class HumanoidImResAMPVCmdV3(HumanoidIm):
    """V3: v_cmd-master clip selection (vs V2's clip-master)."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        init_v = _DIAG_PIN_VCMD_VAL if _DIAG_PIN_VCMD_VAL is not None else V_CMD_INIT
        self._v_cmd_target = torch.full((self.num_envs,), init_v, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), init_v, device=self.device)
        self._v_natural = torch.tensor(list(V_NATURAL), device=self.device)
        self._active_clip = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        if _DIAG_PIN_VCMD_VAL is not None:
            print(f"[V3 diag] PHC_PIN_VCMD={_DIAG_PIN_VCMD_VAL} → v_cmd pinned for all envs")

        # Buffer for v_cmd-driven forced clip ids. _reset_envs writes here
        # BEFORE super(); patched motion_lib.sample_motions consumes it.
        self._pending_forced_clips: torch.Tensor | None = None
        self._install_motion_lib_patch()

    def _install_motion_lib_patch(self):
        """Patch self._motion_lib.sample_motions to read from _pending_forced_clips."""
        orig_sample = self._motion_lib.sample_motions

        def patched_sample_motions(n: int):
            buf = self._pending_forced_clips
            if buf is not None:
                # Consume buffer; default for any extra n beyond buffer length
                # falls back to original behavior.
                self._pending_forced_clips = None
                if buf.shape[0] == n:
                    return buf
                # Length mismatch — log and fall through to original (defensive)
                print(f"[V3] WARN: pending_forced_clips shape {buf.shape} != n={n}, using orig sampler")
            return orig_sample(n)

        self._motion_lib.sample_motions = patched_sample_motions

    def _pick_clip_for_vcmd(self, v_cmd: torch.Tensor) -> torch.Tensor:
        """For each v_cmd, pick clip whose v_natural is closest. Returns long tensor.

        v_cmd shape (n,) → output shape (n,) in {0, 1, 2}.
        """
        diff = (v_cmd.unsqueeze(1) - self._v_natural.unsqueeze(0)).abs()  # (n, 3)
        return diff.argmin(dim=1).long()

    # ------------------------------------------------------------------
    # Observation extension: +1-D v_cmd (normalized to [-1, 1])  — same as V2
    # ------------------------------------------------------------------

    def get_obs_size(self):
        return super().get_obs_size() + 1

    def _v_cmd_norm(self) -> torch.Tensor:
        center = (V_CMD_MIN + V_CMD_MAX) * 0.5
        half = (V_CMD_MAX - V_CMD_MIN) * 0.5
        return (self._v_cmd_ramped - center) / half

    def _compute_observations(self, env_ids=None):
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
    # Multi-clip switching (same hysteresis logic as V2 — used mid-episode
    # if a future v_cmd schedule moves v_cmd across midpoints)
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
        times = torch.zeros_like(self._motion_start_times[ids])
        root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids[ids], times)
        new_root = root_res["root_pos"]
        self._global_offset[ids, :2] = self._humanoid_root_states[ids, :2] - new_root[:, :2]
        if self._global_offset.shape[-1] >= 3:
            self._global_offset[ids, 2] = 0.0
        if hasattr(self, "ref_motion_cache"):
            self.ref_motion_cache.clear()
        self._active_clip = new_clip

    # ------------------------------------------------------------------
    # V3 RESET: v_cmd-master sampling.
    #
    # 1. Sample v_cmd uniformly in [V_CMD_MIN, V_CMD_MAX]
    # 2. Pick closest clip for each env
    # 3. Stage forced clips into _pending_forced_clips buffer
    # 4. Call super() — parent's sample_motions (patched) returns our clips
    #    → state init uses v_cmd-selected clip
    # 5. Confirm v_cmd / active_clip / _sampled_motion_ids
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            super()._reset_envs(env_ids)
            return

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        n = len(env_ids)

        # Step 1: sample v_cmd
        if _DIAG_PIN_VCMD_VAL is not None:
            new_v = torch.full((n,), _DIAG_PIN_VCMD_VAL, device=self.device)
        else:
            new_v = torch.empty(n, device=self.device).uniform_(V_CMD_MIN, V_CMD_MAX)

        # Step 2: pick closest clip per env
        target_clip = self._pick_clip_for_vcmd(new_v)

        # Step 3: stage forced clips for parent's sample_motions consumption.
        # Parent's _sample_ref_state will call self._motion_lib.sample_motions(num_envs)
        # — our patch returns this buffer.
        self._pending_forced_clips = target_clip

        # Step 4: parent reset (state init from v_cmd-selected clip)
        super()._reset_envs(env_ids)

        # Step 5: write v_cmd / active_clip / motion_ids (belt-and-suspenders).
        self._v_cmd_target[env_ids_t] = new_v
        self._v_cmd_ramped[env_ids_t] = new_v
        self._active_clip[env_ids_t] = target_clip
        self._sampled_motion_ids[env_ids_t] = target_clip

    def pre_physics_step(self, actions):
        # Mid-episode clip switching (no-op in V3 unless v_cmd schedule changes
        # v_cmd across midpoints — for now constant-per-episode like V2).
        if _DIAG_PIN_VCMD_VAL is None:
            self._apply_clip_switches()
        # Per-step retime
        v_natural_per_env = self._v_natural[self._active_clip]
        ratio = self._v_cmd_ramped / v_natural_per_env
        self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return super().pre_physics_step(actions)

    # ------------------------------------------------------------------
    # Reward (same as V2: r_track + r_im, no r_survive)
    # ------------------------------------------------------------------

    W_TRACK = 0.5
    W_IM = 0.2
    ALPHA_TRACK = 5.0
    ALPHA_IM = 2.0

    def _compute_reward(self, actions):
        root_vel_xy = self._humanoid_root_states[:, 7:9]
        v_act = torch.linalg.norm(root_vel_xy, dim=-1)
        v_err = v_act - self._v_cmd_ramped
        r_track = torch.exp(-self.ALPHA_TRACK * v_err * v_err)

        ref_pos = getattr(self, "ref_body_pos", None)
        if ref_pos is not None:
            pose_diff = self._rigid_body_pos - ref_pos
            pose_dist = torch.linalg.norm(pose_diff, dim=-1).mean(dim=-1)
            r_im = torch.exp(-self.ALPHA_IM * pose_dist)
        else:
            r_im = torch.zeros_like(r_track)

        self.rew_buf[:] = self.W_TRACK * r_track + self.W_IM * r_im

        if hasattr(self, "reward_raw"):
            self.reward_raw[:] = 0.0
            self.reward_raw[:, 0] = r_track
            self.reward_raw[:, 1] = r_im

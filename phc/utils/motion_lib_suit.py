"""Motion library for L1-retargeted WalkOn Suit reference motion.

Loads suit_reference_lib_*.pkl (built by retarget/build_suit_motion_library.py)
and exposes the API PHC's RL env expects: sample motion ids, query state at
arbitrary times.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import torch


class SuitMotionLib:
    JOINT_ORDER = [
        "JOINT_LH_ABD", "JOINT_LH_ROT", "JOINT_LH_EXT", "JOINT_LK_EXT",
        "JOINT_LA_INV", "JOINT_LA_PLA_foot",
        "JOINT_RH_ABD", "JOINT_RH_ROT", "JOINT_RH_EXT", "JOINT_RK_EXT",
        "JOINT_RA_INV", "JOINT_RA_PLA_foot",
    ]

    def __init__(self, pkl_path: str, device: str = "cuda:0",
                 base_z_offset: float = 0.0, single_clip_idx: int = -1):
        """
        base_z_offset: constant z-shift added to every clip's base_xyz on top
            of whatever is already baked into the pkl. v2 pkl already encodes
            per-frame foot-lift correction, so default is 0.
        single_clip_idx: if >= 0, restrict the library to that single clip.
            Useful for debugging / curriculum (PPO alone struggles with 50
            heterogeneous clips; one clip is much easier).
        """
        self.device = torch.device(device)
        self.base_z_offset = float(base_z_offset)
        self.single_clip_idx = int(single_clip_idx)
        print(f"[SuitMotionLib] loading {pkl_path}  "
              f"(base_z_offset={self.base_z_offset:+.3f} m, single_clip_idx={self.single_clip_idx})")
        d = joblib.load(pkl_path)
        names = list(d.keys())

        if self.single_clip_idx >= 0:
            if self.single_clip_idx >= len(names):
                raise ValueError(f"single_clip_idx={self.single_clip_idx} out of range "
                                  f"(library has {len(names)} clips)")
            chosen = names[self.single_clip_idx]
            print(f"[SuitMotionLib] restricting to clip {self.single_clip_idx}: {chosen}")
            names = [chosen]
            d = {chosen: d[chosen]}
        self.motion_names = names

        self.fps = float(d[names[0]].get("fps", 30.0))
        self.num_motions = len(names)

        suit_q_all = []
        base_xyz_all = []
        base_yaw_all = []
        offsets = [0]
        for n in names:
            c = d[n]
            suit_q_all.append(c["suit_q"])
            base_xyz_all.append(c["base_xyz"])
            base_yaw_all.append(c["base_yaw"])
            offsets.append(offsets[-1] + c["suit_q"].shape[0])

        self.suit_q = torch.from_numpy(np.concatenate(suit_q_all, 0).astype(np.float32)).to(self.device)
        self.base_xyz = torch.from_numpy(np.concatenate(base_xyz_all, 0).astype(np.float32)).to(self.device)
        # Apply the constant z-offset so reference base sits where the suit's
        # LINK_BASE actually stands (rather than at SMPL-pelvis height).
        self.base_xyz[:, 2] += self.base_z_offset
        self.base_yaw = torch.from_numpy(np.concatenate(base_yaw_all, 0).astype(np.float32)).to(self.device)
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        self.motion_lengths = self.offsets[1:] - self.offsets[:-1]
        self.motion_durations = self.motion_lengths.float() / self.fps

        dt = 1.0 / self.fps
        qv = torch.zeros_like(self.suit_q)
        for i in range(self.num_motions):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            if e - s >= 2:
                qv[s + 1:e] = (self.suit_q[s + 1:e] - self.suit_q[s:e - 1]) / dt
        self.suit_qvel = qv

        bv = torch.zeros_like(self.base_xyz)
        for i in range(self.num_motions):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            if e - s >= 2:
                bv[s + 1:e] = (self.base_xyz[s + 1:e] - self.base_xyz[s:e - 1]) / dt
        self.base_xyz_vel = bv

        print(f"[SuitMotionLib] {self.num_motions} clips, "
              f"{self.suit_q.shape[0]} total frames, fps={self.fps}")

    def sample_motions(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.num_motions, (n,), device=self.device, dtype=torch.long)

    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        durs = self.motion_durations[motion_ids]
        return durs * torch.rand_like(durs)

    def _resolve_frame(self, motion_ids, motion_times):
        frame_pos = motion_times * self.fps
        f0 = torch.floor(frame_pos).long().clamp(min=0)
        lengths = self.motion_lengths[motion_ids]
        f0 = torch.minimum(f0, lengths - 2)
        f1 = f0 + 1
        alpha = (frame_pos - f0.float()).clamp(0.0, 1.0)
        starts = self.offsets[motion_ids]
        return starts + f0, starts + f1, alpha

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict:
        i0, i1, a = self._resolve_frame(motion_ids, motion_times)
        a_vec = a.unsqueeze(-1)
        return {
            "suit_q":       (1 - a_vec) * self.suit_q[i0]      + a_vec * self.suit_q[i1],
            "suit_qvel":    (1 - a_vec) * self.suit_qvel[i0]   + a_vec * self.suit_qvel[i1],
            "base_xyz":     (1 - a_vec) * self.base_xyz[i0]    + a_vec * self.base_xyz[i1],
            "base_yaw":     (1 - a) * self.base_yaw[i0]        + a * self.base_yaw[i1],
            "base_xyz_vel": (1 - a_vec) * self.base_xyz_vel[i0] + a_vec * self.base_xyz_vel[i1],
        }

    # --- AMP ---
    AMP_FEATURES_PER_STEP = 12 + 12 + 1 + 3  # dof_q + dof_qvel + base_z + base_lin_vel

    def _amp_features_at_time(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> torch.Tensor:
        """Return (B, AMP_FEATURES_PER_STEP) features for the given (id, time) pairs."""
        s = self.get_motion_state(motion_ids, motion_times)
        # Express base lin vel in heading frame (rotate by -base_yaw around z)
        yaw = s["base_yaw"]
        c, ss = torch.cos(-yaw), torch.sin(-yaw)
        vx = s["base_xyz_vel"][:, 0] * c - s["base_xyz_vel"][:, 1] * ss
        vy = s["base_xyz_vel"][:, 0] * ss + s["base_xyz_vel"][:, 1] * c
        vz = s["base_xyz_vel"][:, 2]
        base_lin_vel_heading = torch.stack([vx, vy, vz], dim=-1)  # (B, 3)
        base_z = s["base_xyz"][:, 2:3]  # (B, 1)
        return torch.cat([s["suit_q"], s["suit_qvel"], base_z, base_lin_vel_heading], dim=-1)

    def get_amp_obs_demo(self, motion_ids: torch.Tensor, motion_times0: torch.Tensor,
                         num_steps: int = 2) -> torch.Tensor:
        """Return (B, num_steps * AMP_FEATURES_PER_STEP) features.

        Step 0 is at `motion_times0`, step k>0 is at `motion_times0 - k*dt`.
        Mirrors HumanoidAMP.build_amp_obs_demo conventions.
        """
        dt = 1.0 / self.fps
        feats = []
        for k in range(num_steps):
            t = (motion_times0 - k * dt).clamp(min=0.0)
            feats.append(self._amp_features_at_time(motion_ids, t))
        return torch.cat(feats, dim=-1)

    def sample_amp_obs_demo(self, n: int, num_steps: int = 2) -> torch.Tensor:
        """Sample (n, num_steps * AMP_FEATURES_PER_STEP) features at random motion times."""
        ids = self.sample_motions(n)
        # Sample uniformly across the clip, but leave room for (num_steps-1) past frames
        dt = 1.0 / self.fps
        max_t = self.motion_durations[ids] - (num_steps - 1) * dt
        max_t = max_t.clamp(min=0.0)
        t0 = max_t * torch.rand_like(max_t) + (num_steps - 1) * dt
        return self.get_amp_obs_demo(ids, t0, num_steps)

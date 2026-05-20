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

    def __init__(self, pkl_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        print(f"[SuitMotionLib] loading {pkl_path}")
        d = joblib.load(pkl_path)
        names = list(d.keys())
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

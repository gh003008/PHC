"""Module G — Nominal Gait Generator for the Motion Plan Layer.

Phase 3 of the MPL roadmap. Module G is a small GRU that maps
    (current self state, task command, future reference window)
  → (nominal joint reference q_ref, nominal 5-axis impedance latent)

This is the "Stage A" offline supervised learning stage from the concept
document (docs/260410_motion_plan_layer_concept_v09.docx §3.3). G is trained
on the motion library with reconstruction + temporal-smoothness loss and
frozen afterwards when Module R is added in Stage B.

Design rationale
----------------
* GRU backbone (hidden 128) — captures short-horizon gait structure without
  the data appetite of a Transformer, and matches the concept doc's guidance
  that "GRU or temporal conv is most realistic for v1".
* Input dim is bounded: 72 (pose_aa) + 3 (root velocity) + 4 (command) +
  K*72 (reference window) = 79 + K*72. With K=5 this is ~439.
* Output: 72 (q_ref) + 5 (Kp_latent) = 77.
* Command labels are computed from the motion itself (concept doc §3.3),
  not hand-annotated. See `_command_from_motion`.
* Kp_latent has 5 axes whose semantic meaning is fixed (stance stiff, swing
  compliance, landing damping, lateral stabilization, overall scale). The
  decoder is not learned here — it lives in the Phase 4 network builder.
  Here we only learn the 5D latent itself with an L2 range penalty.

This file intentionally has no IsaacGym dependency, so it is safe to import
from CPU-only scripts and unit tests.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- constants

# NOTE: G works in the *MJCF DoF* space (69 = 23 non-root joints * 3), NOT the
# 72-dim SMPL canonical pose_aa space. This matches `dof_pos` in the PHC motion
# library so the G output is a drop-in residual for the PD controller.
POSE_AA_DIM = 69            # 23 non-root MJCF joints * 3 axis-angle
ROOT_VEL_DIM = 3            # root linear velocity (world frame)
COMMAND_DIM = 4             # (v_x, v_y, yaw_rate, |v|) in root-local frame
KP_LATENT_DIM = 5           # stance_stiff, swing_compliance, landing_damp,
                            # lateral_stab, overall_scale

# Permutation from SMPL-canonical pose_aa (24 joints, 72 dims, root first) to
# MJCF DoF ordering (23 joints, 69 dims, root dropped).
# SMPL canonical:
#   [Pelvis, L_Hip, R_Hip, Torso, L_Knee, R_Knee, Spine, L_Ankle, R_Ankle,
#    Chest, L_Toe, R_Toe, Neck, L_Thorax, R_Thorax, Head,
#    L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hand, R_Hand]
# MJCF:
#   [L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe,
#    Torso, Spine, Chest, Neck, Head, L_Thorax, L_Shoulder, L_Elbow,
#    L_Wrist, L_Hand, R_Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand]
_SMPL_JOINT_FOR_MJCF = [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]


def _smpl_pose_aa_to_mjcf_dof(pose_aa_72: torch.Tensor) -> torch.Tensor:
    """Convert SMPL-canonical pose_aa [..., 72] to MJCF DoF ordering [..., 69].

    Drops the root (Pelvis) and permutes the remaining 23 joints.
    """
    # Build a flat 69-dim index tensor: for each MJCF joint j in MJCF order,
    # take the 3 axis-angle components from SMPL joint _SMPL_JOINT_FOR_MJCF[j].
    idx = []
    for j_smpl in _SMPL_JOINT_FOR_MJCF:
        idx.extend([j_smpl * 3, j_smpl * 3 + 1, j_smpl * 3 + 2])
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=pose_aa_72.device)
    return pose_aa_72.index_select(-1, idx_t)


@dataclass
class GConfig:
    hidden_dim: int = 128
    encoder_dim: int = 256
    ref_window_K: int = 5
    dropout: float = 0.05
    kp_latent_dim: int = KP_LATENT_DIM
    pose_dim: int = POSE_AA_DIM


# --------------------------------------------------------------- network

class NominalGaitGenerator(nn.Module):
    """Small GRU that outputs (q_ref, Kp_latent) conditioned on state + command + ref."""

    def __init__(self, cfg: GConfig = GConfig()):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.pose_dim + ROOT_VEL_DIM + COMMAND_DIM + cfg.ref_window_K * cfg.pose_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.encoder_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.encoder_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.q_head = nn.Linear(cfg.hidden_dim, cfg.pose_dim)
        self.kp_head = nn.Linear(cfg.hidden_dim, cfg.kp_latent_dim)

    def forward(
        self,
        pose_aa: torch.Tensor,      # [B, T, pose_dim]
        root_vel: torch.Tensor,     # [B, T, 3]
        command: torch.Tensor,      # [B, T, 4]
        ref_window: torch.Tensor,   # [B, T, K*pose_dim]
        hidden: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (q_ref [B,T,pose], kp_latent [B,T,5], next_hidden)."""
        B, T, _ = pose_aa.shape
        x = torch.cat([pose_aa, root_vel, command, ref_window], dim=-1)  # [B, T, in]
        x = self.encoder(x.reshape(B * T, -1)).reshape(B, T, -1)
        out, next_hidden = self.gru(x, hidden)
        q_ref = self.q_head(out)
        kp_latent = self.kp_head(out)
        return q_ref, kp_latent, next_hidden


# --------------------------------------------------------- dataset building

def _command_from_motion(
    root_trans: torch.Tensor,   # [T, 3]
    dt: float,
    sigma_smooth: float = 0.1,
) -> torch.Tensor:
    """Compute command labels (v_x, v_y, yaw_rate, |v|) from motion itself.

    Concept-doc prescription: commands are derived from the motion, not hand-
    annotated. For treadmill data the root yaw is identity, so yaw_rate=0 and
    the command is essentially forward speed.

    Args:
        root_trans: [T, 3] root world-frame trajectory.
        dt: seconds per frame.
        sigma_smooth: unused (reserved for temporal smoothing).
    Returns:
        command: [T, 4] — last value padded from T-1 so shape matches pose.
    """
    T = root_trans.shape[0]
    # Finite-difference velocity, pad last frame.
    vel = torch.zeros_like(root_trans)
    vel[:-1] = (root_trans[1:] - root_trans[:-1]) / dt
    vel[-1] = vel[-2] if T >= 2 else 0

    speed = torch.linalg.norm(vel[..., :2], dim=-1, keepdim=True)
    yaw_rate = torch.zeros(T, 1, device=root_trans.device)
    cmd = torch.cat([vel[..., :2], yaw_rate, speed], dim=-1)  # [T, 4]
    return cmd


def _reference_window(
    pose_aa: torch.Tensor,   # [T, pose_dim]
    K: int,
) -> torch.Tensor:
    """Stack next K frames of pose at each timestep, padding with last frame.

    Returns: [T, K*pose_dim]
    """
    T, P = pose_aa.shape
    # For each t, indices are clamp(t+1, t+2, ..., t+K) to T-1.
    idx = torch.arange(T, device=pose_aa.device).unsqueeze(1) + torch.arange(1, K + 1, device=pose_aa.device).unsqueeze(0)
    idx = idx.clamp(max=T - 1)
    window = pose_aa[idx]                 # [T, K, P]
    return window.reshape(T, K * P)


def build_dataset_from_library(
    library: dict,
    ref_window_K: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Flatten a motion library dict into tensors suitable for G training.

    Args:
        library: {clip_name: {pose_aa, root_trans_offset, fps, ...}}
        ref_window_K: number of future frames in the reference window.

    Returns:
        pose_aa:   [N, T, pose_dim]
        root_vel:  [N, T, 3]     world-frame root linear velocity
        command:   [N, T, 4]     (v_x, v_y, yaw_rate, |v|) root-local
        ref_win:   [N, T, K*P]   next K frames of pose_aa (padded by last frame)
        target_q:  [N, T, pose_dim]  = pose_aa at t+1 (padded by last frame) — G's prediction target
        keys:      list of clip names in order
    """
    pose_list, vel_list, cmd_list, ref_list, tgt_list, key_list = [], [], [], [], [], []
    for key, clip in library.items():
        pose_aa_72 = torch.as_tensor(clip["pose_aa"], dtype=torch.float32)       # [T, 72] SMPL canonical
        # Convert to 69-dim MJCF DoF order once; everything downstream is 69-dim.
        pose_aa = _smpl_pose_aa_to_mjcf_dof(pose_aa_72)                          # [T, 69] MJCF order
        trans = torch.as_tensor(clip["root_trans_offset"], dtype=torch.float32)  # [T, 3]
        fps = clip.get("fps", 30)
        dt = 1.0 / float(fps)

        T = pose_aa.shape[0]
        if T < ref_window_K + 2:
            continue  # skip too-short clips

        # Root linear velocity (world frame), pad last frame.
        vel = torch.zeros_like(trans)
        vel[:-1] = (trans[1:] - trans[:-1]) / dt
        vel[-1] = vel[-2]

        cmd = _command_from_motion(trans, dt)                                    # [T, 4]
        ref = _reference_window(pose_aa, ref_window_K)                           # [T, K*69]

        # Target q_ref = next-frame MJCF DoF pose (supervised reconstruction target).
        target_q = torch.zeros_like(pose_aa)
        target_q[:-1] = pose_aa[1:]
        target_q[-1] = pose_aa[-1]

        pose_list.append(pose_aa)
        vel_list.append(vel)
        cmd_list.append(cmd)
        ref_list.append(ref)
        tgt_list.append(target_q)
        key_list.append(key)

    # Stack: all clips must have the same T. If not, we pad/trunc to the median.
    lengths = [p.shape[0] for p in pose_list]
    if not lengths:
        raise ValueError("Empty library after filtering")
    T_target = int(torch.tensor(lengths).median().item())

    def _fit(x):
        out = torch.zeros(T_target, *x.shape[1:], dtype=x.dtype)
        L = min(x.shape[0], T_target)
        out[:L] = x[:L]
        if L < T_target:
            out[L:] = x[-1:]
        return out

    pose = torch.stack([_fit(p) for p in pose_list])
    vel = torch.stack([_fit(v) for v in vel_list])
    cmd = torch.stack([_fit(c) for c in cmd_list])
    ref = torch.stack([_fit(r) for r in ref_list])
    tgt = torch.stack([_fit(t) for t in tgt_list])
    return pose, vel, cmd, ref, tgt, key_list


# ----------------------------------------------------------------- loss

def g_loss(
    q_pred: torch.Tensor,
    kp_latent: torch.Tensor,
    q_target: torch.Tensor,
    q_prev_pred: torch.Tensor = None,
    q_prev_target: torch.Tensor = None,
    w_recon: float = 1.0,
    w_smooth: float = 0.1,
    w_kp_range: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """Supervised loss for Module G training.

    L = w_recon * MSE(q_pred, q_target)
      + w_smooth * MSE(delta_q_pred, delta_q_target)   if prev supplied
      + w_kp_range * penalty(Kp_latent outside [-2, 2])
    """
    l_recon = F.mse_loss(q_pred, q_target)

    if q_prev_pred is not None and q_prev_target is not None:
        dq_pred = q_pred - q_prev_pred
        dq_tgt = q_target - q_prev_target
        l_smooth = F.mse_loss(dq_pred, dq_tgt)
    else:
        # Intra-sequence smoothness: diff along time.
        dq_pred = q_pred[:, 1:] - q_pred[:, :-1]
        dq_tgt = q_target[:, 1:] - q_target[:, :-1]
        l_smooth = F.mse_loss(dq_pred, dq_tgt)

    # Soft clamp: quadratic penalty outside [-2, 2].
    kp_over = torch.clamp(kp_latent.abs() - 2.0, min=0.0)
    l_kp = (kp_over ** 2).mean()

    total = w_recon * l_recon + w_smooth * l_smooth + w_kp_range * l_kp
    logs = {
        "loss/total": total.item(),
        "loss/recon": l_recon.item(),
        "loss/smooth": l_smooth.item(),
        "loss/kp_range": l_kp.item(),
    }
    return total, logs


# ------------------------------------------------------------- checkpoint io

def load_g_from_checkpoint(
    path: str,
    device: str = "cuda",
    freeze: bool = True,
) -> Tuple[NominalGaitGenerator, GConfig]:
    """Load a trained Module G from a checkpoint saved by `scripts/train_module_g.py`.

    Args:
        path: path to `output/module_g.pth` (or equivalent).
        device: target device for the loaded model.
        freeze: if True, set eval() mode and disable grads (Stage B default).

    Returns:
        (model, cfg) — the loaded GRU and its GConfig.
    """
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    # Only keep keys that GConfig knows about (forward-compat).
    cfg_fields = {f for f in GConfig.__dataclass_fields__}
    cfg = GConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_fields})

    model = NominalGaitGenerator(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, cfg

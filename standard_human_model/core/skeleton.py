"""
skeleton.py — SMPL 휴머노이드 골격 상수 정의

PHC의 SMPL humanoid (smpl_humanoid.xml) 기반.
Body 24개, Joint 23개, DOF 69개 (관절당 3 hinge).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

# =============================================================================
# Body / Joint 정의
# =============================================================================

BODY_NAMES = [
    "Pelvis",       # 0  (root, freejoint)
    "L_Hip",        # 1
    "L_Knee",       # 2
    "L_Ankle",      # 3
    "L_Toe",        # 4
    "R_Hip",        # 5
    "R_Knee",       # 6
    "R_Ankle",      # 7
    "R_Toe",        # 8
    "Torso",        # 9
    "Spine",        # 10
    "Chest",        # 11
    "Neck",         # 12
    "Head",         # 13
    "L_Thorax",     # 14
    "L_Shoulder",   # 15
    "L_Elbow",      # 16
    "L_Wrist",      # 17
    "L_Hand",       # 18
    "R_Thorax",     # 19
    "R_Shoulder",   # 20
    "R_Elbow",      # 21
    "R_Wrist",      # 22
    "R_Hand",       # 23
]

# Joint names = Body names[1:] (Pelvis는 root, 제어 관절 아님)
JOINT_NAMES = BODY_NAMES[1:]  # 23개

NUM_BODIES = 24
NUM_JOINTS = 23
NUM_DOFS = 69  # 23 joints × 3 axes

# =============================================================================
# DOF 매핑
# =============================================================================

def joint_name_to_dof_indices(joint_name: str) -> Tuple[int, int]:
    """관절 이름 → DOF 인덱스 범위 (start, end)."""
    idx = JOINT_NAMES.index(joint_name)
    return (idx * 3, idx * 3 + 3)


# 관절별 DOF 범위 딕셔너리
JOINT_DOF_RANGE: Dict[str, Tuple[int, int]] = {
    name: joint_name_to_dof_indices(name) for name in JOINT_NAMES
}

# 하지 관절만 (보행 분석용)
LOWER_LIMB_JOINTS = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
]

LOWER_LIMB_DOF_INDICES = []
for j in LOWER_LIMB_JOINTS:
    s, e = JOINT_DOF_RANGE[j]
    LOWER_LIMB_DOF_INDICES.extend(range(s, e))

# =============================================================================
# PD Gains (smpl_humanoid.xml의 user attribute에서 추출)
# =============================================================================

# (kp, kd) per joint — MJCF의 user="kp kd 1 kp_max kd_max scale"
JOINT_PD_GAINS: Dict[str, Tuple[float, float]] = {
    "L_Hip":      (250, 2.5),
    "L_Knee":     (250, 2.5),
    "L_Ankle":    (150, 2.5),
    "L_Toe":      (150, 1.0),
    "R_Hip":      (250, 2.5),
    "R_Knee":     (250, 2.5),
    "R_Ankle":    (150, 1.0),
    "R_Toe":      (150, 1.0),
    "Torso":      (500, 5.0),
    "Spine":      (500, 5.0),
    "Chest":      (500, 5.0),
    "Neck":       (150, 1.0),
    "Head":       (150, 1.0),
    "L_Thorax":   (200, 2.0),
    "L_Shoulder": (200, 2.0),
    "L_Elbow":    (150, 1.0),
    "L_Wrist":    (100, 1.0),
    "L_Hand":     (50, 1.0),
    "R_Thorax":   (200, 2.0),
    "R_Shoulder": (200, 2.0),
    "R_Elbow":    (150, 1.0),
    "R_Wrist":    (100, 1.0),
    "R_Hand":     (50, 1.0),
}


def get_kp_kd_tensors(device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """(69,) kp, kd 텐서 반환."""
    kp = torch.zeros(NUM_DOFS, device=device)
    kd = torch.zeros(NUM_DOFS, device=device)
    for joint_name in JOINT_NAMES:
        s, e = JOINT_DOF_RANGE[joint_name]
        _kp, _kd = JOINT_PD_GAINS[joint_name]
        kp[s:e] = _kp
        kd[s:e] = _kd
    return kp, kd


# =============================================================================
# Joint Limits (degrees → radians)
# =============================================================================

# (lower_deg, upper_deg) per DOF axis: joint_x, joint_y, joint_z
_JOINT_LIMITS_DEG: Dict[str, List[Tuple[float, float]]] = {
    "L_Hip":      [(-30, 120), (-45, 45), (-45, 30)],
    "L_Knee":     [(0, 145), (-5.625, 5.625), (-5.625, 5.625)],
    "L_Ankle":    [(-50, 25), (-35, 15), (-20, 20)],
    "L_Toe":      [(-90, 90), (-45, 45), (-45, 45)],
    "R_Hip":      [(-30, 120), (-45, 45), (-30, 45)],
    "R_Knee":     [(0, 145), (-5.625, 5.625), (-5.625, 5.625)],
    "R_Ankle":    [(-50, 25), (-15, 35), (-20, 20)],
    "R_Toe":      [(-90, 90), (-45, 45), (-45, 45)],
    "Torso":      [(-60, 60), (-60, 60), (-60, 60)],
    "Spine":      [(-60, 60), (-60, 60), (-60, 60)],
    "Chest":      [(-60, 60), (-60, 60), (-60, 60)],
    "Neck":       [(-180, 180), (-180, 180), (-180, 180)],
    "Head":       [(-90, 90), (-90, 90), (-90, 90)],
    "L_Thorax":   [(-180, 180), (-180, 180), (-180, 180)],
    "L_Shoulder": [(-180, 180), (-180, 180), (-180, 180)],
    "L_Elbow":    [(-720, 720), (-720, 720), (-720, 720)],
    "L_Wrist":    [(-180, 180), (-180, 180), (-180, 180)],
    "L_Hand":     [(-180, 180), (-180, 180), (-180, 180)],
    "R_Thorax":   [(-180, 180), (-180, 180), (-180, 180)],
    "R_Shoulder": [(-180, 180), (-180, 180), (-180, 180)],
    "R_Elbow":    [(-720, 720), (-720, 720), (-720, 720)],
    "R_Wrist":    [(-180, 180), (-180, 180), (-180, 180)],
    "R_Hand":     [(-180, 180), (-180, 180), (-180, 180)],
}


def get_joint_limits_tensors(device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """(69,) lower, upper 관절 한계 텐서 반환 (radians)."""
    lower = torch.zeros(NUM_DOFS, device=device)
    upper = torch.zeros(NUM_DOFS, device=device)
    deg2rad = np.pi / 180.0
    for joint_name in JOINT_NAMES:
        s, _ = JOINT_DOF_RANGE[joint_name]
        for axis_idx, (lo, hi) in enumerate(_JOINT_LIMITS_DEG[joint_name]):
            lower[s + axis_idx] = lo * deg2rad
            upper[s + axis_idx] = hi * deg2rad
    return lower, upper


# =============================================================================
# Contact Bodies (보행 시 발바닥 접촉 감지용)
# =============================================================================

CONTACT_BODIES = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
CONTACT_BODY_INDICES = [BODY_NAMES.index(b) for b in CONTACT_BODIES]

L_FOOT_BODIES = ["L_Ankle", "L_Toe"]
R_FOOT_BODIES = ["R_Ankle", "R_Toe"]
L_FOOT_INDICES = [BODY_NAMES.index(b) for b in L_FOOT_BODIES]
R_FOOT_INDICES = [BODY_NAMES.index(b) for b in R_FOOT_BODIES]

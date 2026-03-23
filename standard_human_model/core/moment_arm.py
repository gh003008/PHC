"""
moment_arm.py — Moment Arm Matrix R(q)

근육군 → 관절 토크 변환의 핵심.
R(q)는 (num_muscles × num_dofs) 행렬로, bi-articular 근육이면
해당 행에 두 관절 모두 nonzero moment arm이 있다.

τ_joint = R(q)^T @ F_muscle

이 변환을 통해 bi-articular coupling이 자연스럽게 발생한다.

Moment arm 데이터: OpenSim Rajagopal 2016 모델 기반 근사값.

BUG-02 수정: 주요 sagittal plane moment arm에 polynomial fitting 적용.
r_i(q_j) = a0 + a1*q_j + a2*q_j^2
계수는 OpenSim gait2392에서 추출한 근사값.
"""

import torch
import yaml
import os
from typing import Dict, List, Tuple, Optional

from standard_human_model.core.skeleton import (
    JOINT_NAMES, JOINT_DOF_RANGE, NUM_DOFS,
)


# =========================================================================
# Polynomial moment arm coefficients (BUG-02)
# r(q) = a0 + a1*q + a2*q^2  (q in radians, r in meters)
# 출처: OpenSim gait2392 모델에서 추출한 근사값
# key: (muscle_name, joint_name, axis_idx)
# value: (a0, a1, a2)
# =========================================================================
_POLY_COEFFS: Dict[Tuple[str, str, int], Tuple[float, float, float]] = {
    # --- Hamstrings: hip moment arm (신전), q=hip flexion angle ---
    # 무릎 0°→90° 굴곡 시 r: 3.5cm → 5.5cm (대표적 각도 의존성)
    ("hamstrings_L", "L_Hip", 0): (-0.060, -0.008, 0.002),
    ("hamstrings_R", "R_Hip", 0): (-0.060, -0.008, 0.002),
    # --- Hamstrings: knee moment arm (굴곡) ---
    ("hamstrings_L", "L_Knee", 0): (0.030, 0.015, -0.005),
    ("hamstrings_R", "R_Knee", 0): (0.030, 0.015, -0.005),

    # --- Quadriceps/Vastii: knee moment arm (신전) ---
    # 무릎 신전 시 patella가 전방 이동 → moment arm 증가
    ("quadriceps_L", "L_Knee", 0): (-0.040, -0.012, 0.004),
    ("quadriceps_R", "R_Knee", 0): (-0.040, -0.012, 0.004),

    # --- Rectus femoris: knee moment arm ---
    ("rectus_femoris_L", "L_Knee", 0): (-0.040, -0.012, 0.004),
    ("rectus_femoris_R", "R_Knee", 0): (-0.040, -0.012, 0.004),

    # --- Gastrocnemius: knee moment arm ---
    ("gastrocnemius_L", "L_Knee", 0): (0.020, 0.008, -0.003),
    ("gastrocnemius_R", "R_Knee", 0): (0.020, 0.008, -0.003),

    # --- Gastrocnemius: ankle moment arm (족저굴) ---
    # 발목 배굴 시 moment arm 약간 감소
    ("gastrocnemius_L", "L_Ankle", 0): (-0.050, 0.005, 0.002),
    ("gastrocnemius_R", "R_Ankle", 0): (-0.050, 0.005, 0.002),

    # --- Soleus: ankle moment arm ---
    ("soleus_L", "L_Ankle", 0): (-0.050, 0.005, 0.002),
    ("soleus_R", "R_Ankle", 0): (-0.050, 0.005, 0.002),

    # --- Tibialis anterior: ankle moment arm (배굴) ---
    ("tibialis_ant_L", "L_Ankle", 0): (0.030, -0.003, -0.001),
    ("tibialis_ant_R", "R_Ankle", 0): (0.030, -0.003, -0.001),
}


class MomentArmMatrix:
    """Moment arm matrix R(q) 관리 및 토크 변환.

    polynomial fitting으로 R(q)의 각도 의존성 구현 (BUG-02 수정).
    polynomial 계수가 없는 moment arm은 상수값 유지.

    사용법:
        R = MomentArmMatrix(muscle_defs, device="cuda:0")

        # 근육 길이 계산
        l_muscle = R.compute_muscle_length(dof_pos)

        # 근육 속도 계산
        v_muscle = R.compute_muscle_velocity(dof_pos, dof_vel)

        # 근육 힘 → 관절 토크
        tau = R.forces_to_torques(F_muscle, dof_pos)
    """

    def __init__(self, muscle_definitions: List[Dict], device: str = "cpu"):
        """
        Args:
            muscle_definitions: 근육군 정의 리스트. 각 항목:
                {
                    "name": "hamstrings_R",
                    "moment_arms": {
                        "R_Hip": [-0.06, 0, 0],     # [x, y, z] meters
                        "R_Knee": [0.03, 0, 0],
                    },
                    "l_slack": 0.35,  # MTU 이완 시 전체 길이 (m)
                }
        """
        self.device = device
        self.num_muscles = len(muscle_definitions)
        self.muscle_names = [m["name"] for m in muscle_definitions]

        # R_const: 상수 moment arm (num_muscles, num_dofs)
        self.R_const = torch.zeros(self.num_muscles, NUM_DOFS, device=device)

        # l_slack: (num_muscles,) MTU 이완 길이
        self.l_slack = torch.zeros(self.num_muscles, device=device)

        # Polynomial coefficients 저장
        # poly_entries: list of (muscle_idx, dof_idx, a0, a1, a2)
        self._poly_entries = []

        for i, mdef in enumerate(muscle_definitions):
            self.l_slack[i] = mdef.get("l_slack", 0.3)
            muscle_name = mdef["name"]

            for joint_name, arms in mdef.get("moment_arms", {}).items():
                if joint_name not in JOINT_DOF_RANGE:
                    continue
                start, end = JOINT_DOF_RANGE[joint_name]
                for axis_idx, arm_val in enumerate(arms):
                    if axis_idx < (end - start):
                        dof_idx = start + axis_idx
                        self.R_const[i, dof_idx] = arm_val

                        # polynomial 계수가 있으면 등록
                        key = (muscle_name, joint_name, axis_idx)
                        if key in _POLY_COEFFS:
                            a0, a1, a2 = _POLY_COEFFS[key]
                            self._poly_entries.append((i, dof_idx, a0, a1, a2))

        # polynomial 텐서로 변환 (벡터 연산용)
        if self._poly_entries:
            self._poly_muscle_idx = torch.tensor(
                [e[0] for e in self._poly_entries], dtype=torch.long, device=device
            )
            self._poly_dof_idx = torch.tensor(
                [e[1] for e in self._poly_entries], dtype=torch.long, device=device
            )
            self._poly_a0 = torch.tensor(
                [e[2] for e in self._poly_entries], dtype=torch.float32, device=device
            )
            self._poly_a1 = torch.tensor(
                [e[3] for e in self._poly_entries], dtype=torch.float32, device=device
            )
            self._poly_a2 = torch.tensor(
                [e[4] for e in self._poly_entries], dtype=torch.float32, device=device
            )
            self._has_poly = True
        else:
            self._has_poly = False

    def _compute_R(self, dof_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """현재 관절 각도에 따른 R matrix 계산.

        Args:
            dof_pos: (num_envs, num_dofs) 또는 None (상수 R 반환)
        Returns:
            R: (num_envs, num_muscles, num_dofs) 또는 상수일 때 (num_muscles, num_dofs)
        """
        if not self._has_poly or dof_pos is None:
            return self.R_const

        num_envs = dof_pos.shape[0]
        # 상수 R을 환경 수만큼 복제
        R = self.R_const.unsqueeze(0).expand(num_envs, -1, -1).clone()

        # polynomial moment arm 계산
        # q: 해당 DOF의 관절 각도 (num_envs, num_poly_entries)
        q = dof_pos[:, self._poly_dof_idx]  # (num_envs, num_poly)
        r_poly = self._poly_a0 + self._poly_a1 * q + self._poly_a2 * q ** 2

        # R matrix 업데이트
        R[:, self._poly_muscle_idx, self._poly_dof_idx] = r_poly

        return R

    @classmethod
    def from_yaml(cls, yaml_path: str, device: str = "cpu") -> "MomentArmMatrix":
        """YAML 파일에서 근육 정의 로드."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data["muscles"], device=device)

    def compute_muscle_length(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """관절 각도 → 근건 단위(MTU) 길이 계산.

        l_mtu = l_slack - R(q) @ q

        Args:
            dof_pos: (num_envs, num_dofs)
        Returns:
            (num_envs, num_muscles) MTU 길이 (m)
        """
        R = self._compute_R(dof_pos)

        if R.dim() == 3:
            # R: (num_envs, num_muscles, num_dofs), dof_pos: (num_envs, num_dofs)
            delta_l = torch.bmm(
                dof_pos.unsqueeze(1),  # (num_envs, 1, num_dofs)
                R.transpose(1, 2)     # (num_envs, num_dofs, num_muscles)
            ).squeeze(1)               # (num_envs, num_muscles)
        else:
            # 상수 R: (num_muscles, num_dofs)
            delta_l = dof_pos @ R.T

        return self.l_slack - delta_l

    def compute_muscle_velocity(
        self, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> torch.Tensor:
        """관절 속도 → 근건 단위 속도.

        v_mtu = -R(q) @ dq
        (dR/dq 항은 2차 효과로 무시, 보행 범위에서 충분히 작음)

        Args:
            dof_pos: (num_envs, num_dofs)
            dof_vel: (num_envs, num_dofs)
        Returns:
            (num_envs, num_muscles) MTU 속도 (m/s, 양수=신장, 음수=수축)
        """
        R = self._compute_R(dof_pos)

        if R.dim() == 3:
            delta_v = torch.bmm(
                dof_vel.unsqueeze(1),
                R.transpose(1, 2)
            ).squeeze(1)
        else:
            delta_v = dof_vel @ R.T

        return -delta_v

    def forces_to_torques(
        self, F_muscle: torch.Tensor, dof_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """근육 힘 → 관절 토크 변환.

        τ = R(q)^T @ F

        Args:
            F_muscle: (num_envs, num_muscles)
            dof_pos: (num_envs, num_dofs) — polynomial R 사용 시 필요
        Returns:
            (num_envs, num_dofs) 관절 토크
        """
        R = self._compute_R(dof_pos)

        if R.dim() == 3:
            # (num_envs, 1, num_muscles) @ (num_envs, num_muscles, num_dofs)
            return torch.bmm(F_muscle.unsqueeze(1), R).squeeze(1)
        else:
            # F: (num_envs, num_muscles) @ R: (num_muscles, num_dofs)
            return F_muscle @ R

    def get_muscle_index(self, muscle_name: str) -> int:
        """근육 이름 → 인덱스."""
        return self.muscle_names.index(muscle_name)

    def get_coupling_info(self) -> Dict[str, List[str]]:
        """각 근육이 어떤 관절들에 걸려있는지 반환 (디버깅용)."""
        info = {}
        for i, name in enumerate(self.muscle_names):
            coupled_joints = []
            for joint_name in JOINT_NAMES:
                s, e = JOINT_DOF_RANGE[joint_name]
                if self.R_const[i, s:e].abs().sum() > 0:
                    coupled_joints.append(joint_name)
            info[name] = coupled_joints
        return info

    def summary(self) -> str:
        """R matrix 요약 출력."""
        lines = [f"MomentArmMatrix: {self.num_muscles} muscles × {NUM_DOFS} DOFs"]
        if self._has_poly:
            lines.append(f"  Polynomial entries: {len(self._poly_entries)}")
        coupling = self.get_coupling_info()
        for name, joints in coupling.items():
            tag = " (bi-articular)" if len(joints) > 1 else ""
            lines.append(f"  {name}: {', '.join(joints)}{tag}")
        return "\n".join(lines)
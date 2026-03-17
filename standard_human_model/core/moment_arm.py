"""
moment_arm.py — Moment Arm Matrix R(q)

근육군 → 관절 토크 변환의 핵심.
R(q)는 (num_muscles × num_dofs) 행렬로, bi-articular 근육이면
해당 행에 두 관절 모두 nonzero moment arm이 있다.

τ_joint = R(q)^T @ F_muscle

이 변환을 통해 bi-articular coupling이 자연스럽게 발생한다.

Moment arm 데이터: OpenSim Rajagopal 2016 모델 기반 근사값.
"""

import torch
import yaml
import os
from typing import Dict, List, Tuple

from standard_human_model.core.skeleton import (
    JOINT_NAMES, JOINT_DOF_RANGE, NUM_DOFS,
)


class MomentArmMatrix:
    """Moment arm matrix R(q) 관리 및 토크 변환.

    현재는 상수 moment arm (q 무관)으로 구현.
    향후 polynomial fitting으로 R(q)의 각도 의존성 추가 가능.

    사용법:
        R = MomentArmMatrix(muscle_defs, device="cuda:0")

        # 근육 길이 계산
        l_muscle = R.compute_muscle_length(dof_pos)

        # 근육 속도 계산
        v_muscle = R.compute_muscle_velocity(dof_pos, dof_vel)

        # 근육 힘 → 관절 토크
        tau = R.forces_to_torques(F_muscle)
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
                    "l_slack": 0.35,  # 이완 시 근육 길이 (m)
                }
        """
        self.device = device
        self.num_muscles = len(muscle_definitions)
        self.muscle_names = [m["name"] for m in muscle_definitions]

        # R matrix: (num_muscles, num_dofs)
        self.R = torch.zeros(self.num_muscles, NUM_DOFS, device=device)

        # l_slack: (num_muscles,) 이완 길이
        self.l_slack = torch.zeros(self.num_muscles, device=device)

        for i, mdef in enumerate(muscle_definitions):
            self.l_slack[i] = mdef.get("l_slack", 0.3)

            for joint_name, arms in mdef.get("moment_arms", {}).items():
                if joint_name not in JOINT_DOF_RANGE:
                    continue
                start, end = JOINT_DOF_RANGE[joint_name]
                for axis_idx, arm_val in enumerate(arms):
                    if axis_idx < (end - start):
                        self.R[i, start + axis_idx] = arm_val

    @classmethod
    def from_yaml(cls, yaml_path: str, device: str = "cpu") -> "MomentArmMatrix":
        """YAML 파일에서 근육 정의 로드."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data["muscles"], device=device)

    def compute_muscle_length(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """관절 각도 → 근육 길이 계산.

        l_muscle = l_slack - R @ q

        음부호 이유: 근육이 수축(짧아짐)하면 관절이 moment arm 방향(+R)으로 이동.
        따라서 관절이 +R 방향으로 갈수록 근육은 짧아진다.
        예: hamstrings R_knee=+0.03 → 무릎 굴곡 시 hamstrings 짧아짐.

        Args:
            dof_pos: (num_envs, num_dofs)
        Returns:
            (num_envs, num_muscles) 근육 길이
        """
        # R: (num_muscles, num_dofs), dof_pos: (num_envs, num_dofs)
        # (num_envs, num_dofs) @ (num_dofs, num_muscles) → (num_envs, num_muscles)
        delta_l = dof_pos @ self.R.T
        return self.l_slack - delta_l

    def compute_muscle_velocity(
        self, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> torch.Tensor:
        """관절 속도 → 근육 수축 속도.

        v_muscle = -R @ dq

        dl/dt = d(l_slack - R@q)/dt = -R @ dq
        양수 = 근육 신장(늘어남), 음수 = 근육 수축(짧아짐).
        예: hamstrings R_knee=+0.03, 무릎 신전(dq<0) → v=+0.03*positive → 신장 ✓

        Args:
            dof_pos: (num_envs, num_dofs) — 현재는 미사용 (상수 R)
            dof_vel: (num_envs, num_dofs)
        Returns:
            (num_envs, num_muscles) 근육 속도 (양수=신장, 음수=수축)
        """
        return -(dof_vel @ self.R.T)

    def forces_to_torques(self, F_muscle: torch.Tensor) -> torch.Tensor:
        """근육 힘 → 관절 토크 변환.

        τ = R^T @ F

        이 연산에서 bi-articular coupling이 자연스럽게 발생한다:
        hamstrings의 힘이 hip과 knee 두 관절에 동시에 토크를 생성.

        Args:
            F_muscle: (num_envs, num_muscles)
        Returns:
            (num_envs, num_dofs) 관절 토크
        """
        # F: (num_envs, num_muscles) @ R: (num_muscles, num_dofs) → (num_envs, num_dofs)
        return F_muscle @ self.R

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
                if self.R[i, s:e].abs().sum() > 0:
                    coupled_joints.append(joint_name)
            info[name] = coupled_joints
        return info

    def summary(self) -> str:
        """R matrix 요약 출력."""
        lines = [f"MomentArmMatrix: {self.num_muscles} muscles × {NUM_DOFS} DOFs"]
        coupling = self.get_coupling_info()
        for name, joints in coupling.items():
            tag = " (bi-articular)" if len(joints) > 1 else ""
            lines.append(f"  {name}: {', '.join(joints)}{tag}")
        return "\n".join(lines)

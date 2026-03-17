"""
ligament_model.py — 인대/관절낭 Soft Limit

URDF/MJCF의 hard joint limit과 별도로,
ROM 경계 근처에서 지수적으로 증가하는 저항 토크를 생성.

구축(contracture) 환자에서는 soft limit이 더 안쪽에서 시작되고
더 가파르게 증가한다.

τ_ligament = -k_lig * exp(alpha * max(0, q - q_soft_upper))
           + k_lig * exp(alpha * max(0, q_soft_lower - q))
"""

import torch
from typing import Optional, Tuple

from standard_human_model.core.skeleton import NUM_DOFS


class LigamentModel:
    """인대/관절낭 soft limit 토크 모델.

    사용법:
        lig = LigamentModel(num_envs=512, device="cuda:0")
        lig.set_limits(soft_lower, soft_upper, k_lig, alpha)

        tau_lig = lig.compute_torque(dof_pos, dof_vel)
    """

    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device

        # Soft limits (num_dofs,)
        self.soft_lower = torch.zeros(NUM_DOFS, device=device)
        self.soft_upper = torch.zeros(NUM_DOFS, device=device)
        self.k_lig = torch.ones(NUM_DOFS, device=device) * 50.0     # 강성 계수
        self.alpha = torch.ones(NUM_DOFS, device=device) * 10.0     # 지수 기울기
        self.damping = torch.ones(NUM_DOFS, device=device) * 5.0    # 경계 근처 감쇠

    def set_limits(
        self,
        soft_lower: torch.Tensor,    # (num_dofs,) radians
        soft_upper: torch.Tensor,    # (num_dofs,) radians
        k_lig: Optional[torch.Tensor] = None,   # (num_dofs,)
        alpha: Optional[torch.Tensor] = None,   # (num_dofs,)
        damping: Optional[torch.Tensor] = None,  # (num_dofs,)
    ):
        """Soft limit 파라미터 설정."""
        self.soft_lower = soft_lower.to(self.device)
        self.soft_upper = soft_upper.to(self.device)
        if k_lig is not None:
            self.k_lig = k_lig.to(self.device)
        if alpha is not None:
            self.alpha = alpha.to(self.device)
        if damping is not None:
            self.damping = damping.to(self.device)

    def set_limits_from_hard_limits(
        self,
        hard_lower: torch.Tensor,
        hard_upper: torch.Tensor,
        margin_ratio: float = 0.85,
    ):
        """Hard limit에서 margin을 두고 soft limit 자동 설정.

        Args:
            margin_ratio: soft limit을 hard limit의 몇 % 지점에 설정할지.
                          0.85 = hard limit의 85% 지점에서 soft limit 시작.
        """
        center = (hard_lower + hard_upper) / 2
        half_range = (hard_upper - hard_lower) / 2
        self.soft_lower = center - half_range * margin_ratio
        self.soft_upper = center + half_range * margin_ratio

    def compute_torque(
        self,
        dof_pos: torch.Tensor,    # (num_envs, num_dofs)
        dof_vel: torch.Tensor,    # (num_envs, num_dofs)
    ) -> torch.Tensor:
        """인대/관절낭 토크 계산.

        Returns:
            tau_ligament: (num_envs, num_dofs)
        """
        # 상한 초과 → 음의 토크 (복원)
        excess_upper = torch.clamp(dof_pos - self.soft_upper, min=0)
        tau_upper = -self.k_lig * (torch.exp(self.alpha * excess_upper) - 1)

        # 하한 초과 → 양의 토크 (복원)
        excess_lower = torch.clamp(self.soft_lower - dof_pos, min=0)
        tau_lower = self.k_lig * (torch.exp(self.alpha * excess_lower) - 1)

        # 경계 근처 감쇠 (진동 방지)
        in_upper_zone = excess_upper > 0
        in_lower_zone = excess_lower > 0
        in_zone = in_upper_zone | in_lower_zone
        tau_damp = torch.where(in_zone, -self.damping * dof_vel, torch.zeros_like(dof_vel))

        return tau_upper + tau_lower + tau_damp

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """상태 리셋 (현재 stateless이므로 no-op)."""
        pass

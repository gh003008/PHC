"""
muscle_model.py — Hill-type 근육 모델

근육군(muscle group)별 힘 생성 모델.
Active force (F-L, F-V) + Passive force + Damping.

참고: Thelen 2003, Millard 2013
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MuscleParams:
    """단일 근육군의 Hill model 파라미터."""
    name: str
    f_max: float          # 최대 등척성 힘 (N)
    l_opt: float          # 최적 섬유 길이 (정규화, 기본 1.0)
    v_max: float          # 최대 수축 속도 (l_opt/s, 보통 10)
    pennation: float      # 깃 각도 (rad, 보통 0~0.5)
    tau_act: float        # 활성화 시정수 (s, 보통 0.01~0.05)
    tau_deact: float      # 비활성화 시정수 (s, 보통 0.04~0.2)
    # Passive
    k_pe: float           # 수동 강성 계수 (보통 4.0~5.0)
    epsilon_0: float      # 수동 변형률 계수 (보통 0.6)
    # Tendon
    l_tendon_slack: float  # 건 이완 길이 (정규화)
    k_tendon: float        # 건 강성 (보통 35)
    # Damping
    damping: float         # 근육 감쇠 계수 (보통 0.01~0.1)


class HillMuscleModel:
    """Hill-type 근육 모델. 텐서 연산으로 다수 환경 병렬 처리.

    사용법:
        model = HillMuscleModel(num_muscles=16, num_envs=512, device="cuda:0")
        model.set_params(muscle_params_list)

        # 매 step
        F_total = model.compute_force(activation, muscle_length, muscle_velocity)
    """

    def __init__(self, num_muscles: int, num_envs: int, device: str = "cpu"):
        self.num_muscles = num_muscles
        self.num_envs = num_envs
        self.device = device

        # 파라미터 텐서 (num_muscles,) — set_params()에서 채워짐
        self.f_max = torch.zeros(num_muscles, device=device)
        self.l_opt = torch.ones(num_muscles, device=device)
        self.v_max = torch.ones(num_muscles, device=device) * 10.0
        self.pennation = torch.zeros(num_muscles, device=device)
        self.k_pe = torch.ones(num_muscles, device=device) * 4.0
        self.epsilon_0 = torch.ones(num_muscles, device=device) * 0.6
        self.l_tendon_slack = torch.ones(num_muscles, device=device)
        self.k_tendon = torch.ones(num_muscles, device=device) * 35.0
        self.damping = torch.ones(num_muscles, device=device) * 0.05

    def set_params(self, params_list: list):
        """MuscleParams 리스트로 파라미터 텐서 설정."""
        for i, p in enumerate(params_list):
            self.f_max[i] = p.f_max
            self.l_opt[i] = p.l_opt
            self.v_max[i] = p.v_max
            self.pennation[i] = p.pennation
            self.k_pe[i] = p.k_pe
            self.epsilon_0[i] = p.epsilon_0
            self.l_tendon_slack[i] = p.l_tendon_slack
            self.k_tendon[i] = p.k_tendon
            self.damping[i] = p.damping

    def force_length_active(self, l_norm: torch.Tensor) -> torch.Tensor:
        """Active Force-Length relationship (가우시안).

        l_norm: 정규화된 근육 길이 (l_muscle / l_opt)
        Returns: [0, 1] 스케일링 팩터
        """
        # 가우시안: 최적 길이(1.0)에서 최대, 양쪽으로 감소
        # width=0.45는 Thelen 2003 기본값
        width = 0.45
        return torch.exp(-((l_norm - 1.0) ** 2) / (2 * width ** 2))

    def force_length_passive(self, l_norm: torch.Tensor) -> torch.Tensor:
        """Passive Force-Length relationship (지수 함수).

        근육이 최적 길이를 넘어 늘어나면 지수적으로 저항 증가.
        """
        # f_PE = k_pe * max(0, l_norm - 1.0)^2 / epsilon_0
        stretch = torch.clamp(l_norm - 1.0, min=0)
        return self.k_pe * (stretch ** 2) / self.epsilon_0

    def force_velocity(self, v_norm: torch.Tensor) -> torch.Tensor:
        """Force-Velocity relationship (Hill curve).

        v_norm: 정규화된 수축 속도 (v_muscle / v_max)
                음수 = 수축 (concentric), 양수 = 신장 (eccentric)
        Returns: 힘 스케일링 팩터
        """
        # Concentric (수축): v_norm < 0
        # f_FV = (1 + v_norm) / (1 - v_norm / 0.25)  (v_norm ∈ [-1, 0])
        #
        # Eccentric (신장): v_norm > 0
        # f_FV = 1.8 - 0.8 * (1 + v_norm) / (1 + v_norm / 0.18)  (simplified)

        fv = torch.ones_like(v_norm)

        # Concentric
        conc_mask = v_norm < 0
        v_c = v_norm[conc_mask]
        # 안전한 수치 처리: v_norm을 -0.99 이상으로 제한
        v_c = torch.clamp(v_c, min=-0.99)
        fv[conc_mask] = (1.0 + v_c) / (1.0 - v_c / 0.25)

        # Eccentric
        ecc_mask = v_norm > 0
        v_e = v_norm[ecc_mask]
        # 최대 1.8배까지 증가
        fv[ecc_mask] = torch.clamp(
            1.0 + 0.8 * v_e / (v_e + 0.18),
            max=1.8
        )

        return torch.clamp(fv, min=0)

    def compute_force(
        self,
        activation: torch.Tensor,     # (num_envs, num_muscles) [0, 1]
        muscle_length: torch.Tensor,   # (num_envs, num_muscles) 정규화
        muscle_velocity: torch.Tensor, # (num_envs, num_muscles) 정규화
    ) -> torch.Tensor:
        """총 근육 힘 계산.

        Returns:
            F_total: (num_envs, num_muscles) 근육 힘 (N)
        """
        # 정규화
        l_norm = muscle_length / self.l_opt    # (num_envs, num_muscles)
        v_norm = muscle_velocity / self.v_max  # (num_envs, num_muscles)

        # Active force: a * F_max * f_FL(l) * f_FV(v) * cos(pennation)
        f_fl = self.force_length_active(l_norm)
        f_fv = self.force_velocity(v_norm)
        cos_penn = torch.cos(self.pennation)
        F_active = activation * self.f_max * f_fl * f_fv * cos_penn

        # Passive force: F_max * f_PE(l)
        f_pe = self.force_length_passive(l_norm)
        F_passive = self.f_max * f_pe

        # Damping force
        F_damping = self.damping * self.f_max * muscle_velocity

        # Total
        F_total = F_active + F_passive + F_damping

        # 근육은 당기기만 가능 (음의 힘 = 밀기 불가)
        F_total = torch.clamp(F_total, min=0)

        return F_total

    def compute_force_components(
        self,
        activation: torch.Tensor,
        muscle_length: torch.Tensor,
        muscle_velocity: torch.Tensor,
    ) -> dict:
        """디버깅용: 각 성분을 분리하여 반환."""
        l_norm = muscle_length / self.l_opt
        v_norm = muscle_velocity / self.v_max

        f_fl = self.force_length_active(l_norm)
        f_fv = self.force_velocity(v_norm)
        cos_penn = torch.cos(self.pennation)

        F_active = activation * self.f_max * f_fl * f_fv * cos_penn
        F_passive = self.f_max * self.force_length_passive(l_norm)
        F_damping = self.damping * self.f_max * muscle_velocity

        return {
            "F_active": F_active,
            "F_passive": F_passive,
            "F_damping": F_damping,
            "f_FL": f_fl,
            "f_FV": f_fv,
            "l_norm": l_norm,
            "v_norm": v_norm,
        }

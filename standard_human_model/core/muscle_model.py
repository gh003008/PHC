"""
muscle_model.py — Hill-type 근육 모델

근육군(muscle group)별 힘 생성 모델.
Active force (F-L, F-V) + Passive force + Damping.
Rigid tendon (기본) + Elastic tendon (soleus, gastrocnemius 옵션).

참고: Thelen 2003, Millard 2013, De Groote 2016
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MuscleParams:
    """단일 근육군의 Hill model 파라미터."""
    name: str
    f_max: float          # 최대 등척성 힘 (N)
    l_opt: float          # 최적 섬유 길이 (m, Rajagopal 2016)
    v_max: float          # 최대 수축 속도 (l_opt/s, 보통 10)
    pennation: float      # 깃 각도 (rad, 보통 0~0.5)
    tau_act: float        # 활성화 시정수 (s, 보통 0.01~0.05)
    tau_deact: float      # 비활성화 시정수 (s, 보통 0.04~0.2)
    # Passive
    k_pe: float           # 수동 강성 계수 (보통 4.0~5.0)
    epsilon_0: float      # 수동 변형률 계수 (보통 0.6)
    # Tendon
    l_tendon_slack: float  # 건 이완 길이 (m, Rajagopal 2016)
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

    단위 체계:
        - muscle_length: m (MTU 전체 길이, moment_arm.py에서 계산)
        - muscle_velocity: m/s (MTU 속도, moment_arm.py에서 계산)
        - l_opt: m (최적 섬유 길이)
        - l_tendon_slack: m (건 이완 길이)
        - v_max: l_opt/s → 절대 속도 = v_max * l_opt (m/s)

    Fiber length 계산 (rigid tendon):
        l_ce = (l_mtu - l_tendon_slack) / cos(pennation)
        l_norm = l_ce / l_opt
    """

    def __init__(self, num_muscles: int, num_envs: int, device: str = "cpu",
                 elastic_tendon_indices: Optional[List[int]] = None):
        """
        Args:
            elastic_tendon_indices: elastic tendon을 적용할 근육 인덱스 리스트.
                                   None이면 전부 rigid tendon.
        """
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

        # Elastic tendon 적용 근육 (BUG-03)
        self._elastic_mask = torch.zeros(num_muscles, dtype=torch.bool, device=device)
        if elastic_tendon_indices is not None:
            for idx in elastic_tendon_indices:
                self._elastic_mask[idx] = True

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

        l_norm: 정규화된 섬유 길이 (l_ce / l_opt)
        Returns: [0, 1] 스케일링 팩터
        """
        width = 0.45  # Thelen 2003
        return torch.exp(-((l_norm - 1.0) ** 2) / (2 * width ** 2))

    def force_length_passive(self, l_norm: torch.Tensor) -> torch.Tensor:
        """Passive Force-Length relationship (지수 함수).

        근육이 최적 길이를 넘어 늘어나면 지수적으로 저항 증가.
        """
        stretch = torch.clamp(l_norm - 1.0, min=0)
        return self.k_pe * (stretch ** 2) / self.epsilon_0

    def force_velocity(self, v_norm: torch.Tensor) -> torch.Tensor:
        """Force-Velocity relationship (Hill curve).

        v_norm: 정규화된 섬유 속도 (v_ce / (v_max * l_opt))
                음수 = 수축 (concentric), 양수 = 신장 (eccentric)
        Returns: 힘 스케일링 팩터
        """
        fv = torch.ones_like(v_norm)

        # Concentric (수축)
        conc_mask = v_norm < 0
        v_c = v_norm[conc_mask]
        v_c = torch.clamp(v_c, min=-0.99)
        fv[conc_mask] = (1.0 + v_c) / (1.0 - v_c / 0.25)

        # Eccentric (신장)
        ecc_mask = v_norm > 0
        v_e = v_norm[ecc_mask]
        fv[ecc_mask] = torch.clamp(
            1.0 + 0.8 * v_e / (v_e + 0.18),
            max=1.8
        )

        return torch.clamp(fv, min=0)

    def tendon_force_normalized(self, l_tendon_norm: torch.Tensor) -> torch.Tensor:
        """Normalized tendon force (De Groote 2016 simplified).

        f_T = k_tendon * max(0, l_tendon_norm - 1.0)^2

        Args:
            l_tendon_norm: l_tendon / l_tendon_slack
        Returns:
            정규화된 건 힘 (f_T / f_max)
        """
        strain = torch.clamp(l_tendon_norm - 1.0, min=0)
        return self.k_tendon * strain ** 2

    def _compute_fiber_state_rigid(
        self,
        muscle_length: torch.Tensor,
        muscle_velocity: torch.Tensor,
    ) -> tuple:
        """Rigid tendon: 섬유 길이/속도 해석 계산.

        l_ce = (l_mtu - l_tendon_slack) / cos(pennation)
        v_ce = v_mtu / cos(pennation)

        Returns:
            l_norm: (num_envs, num_muscles) 정규화된 섬유 길이
            v_norm: (num_envs, num_muscles) 정규화된 섬유 속도
        """
        cos_penn = torch.cos(self.pennation)

        # 섬유 길이: MTU에서 건 길이를 빼고 pennation으로 보정
        l_ce = (muscle_length - self.l_tendon_slack) / cos_penn
        l_ce = torch.clamp(l_ce, min=1e-4)  # 음수 방지
        l_norm = l_ce / self.l_opt

        # 섬유 속도: rigid tendon이므로 건 속도=0 → v_ce = v_mtu / cos(penn)
        v_ce = muscle_velocity / cos_penn
        # v_max는 l_opt/s 단위 → 절대 속도 = v_max * l_opt (m/s)
        v_norm = v_ce / (self.v_max * self.l_opt)

        return l_norm, v_norm

    def _compute_fiber_state_elastic(
        self,
        activation: torch.Tensor,
        muscle_length: torch.Tensor,
        muscle_velocity: torch.Tensor,
        max_iter: int = 10,
    ) -> tuple:
        """Elastic tendon: Newton-Raphson으로 l_ce 수렴.

        F_tendon(l_mtu - l_ce * cos(penn)) = F_muscle(l_ce, activation)

        soleus, gastrocnemius처럼 tendon/fiber > 3인 근육에서 중요.

        Returns:
            l_norm: (num_envs, num_muscles) 정규화된 섬유 길이
            v_norm: (num_envs, num_muscles) 정규화된 섬유 속도
        """
        cos_penn = torch.cos(self.pennation)

        # 초기 추정: rigid tendon 해
        l_ce = (muscle_length - self.l_tendon_slack) / cos_penn
        l_ce = torch.clamp(l_ce, min=1e-4)

        for _ in range(max_iter):
            # 현재 l_ce에서 건 길이 계산
            l_tendon = muscle_length - l_ce * cos_penn
            l_tendon_norm = l_tendon / self.l_tendon_slack

            # 건 힘 (정규화)
            f_t = self.tendon_force_normalized(l_tendon_norm)

            # 근육 힘 (정규화): a * f_FL + f_PE
            l_norm_iter = l_ce / self.l_opt
            f_fl = self.force_length_active(l_norm_iter)
            f_pe = self.force_length_passive(l_norm_iter)
            f_m = activation * f_fl * cos_penn + f_pe

            # 잔차: F_tendon - F_muscle = 0
            residual = f_t - f_m

            # 수렴 체크 (절대 오차 < 1e-3)
            if residual.abs().max() < 1e-3:
                break

            # Jacobian (수치 미분 대신 해석적 근사)
            # df_t/dl_ce = -cos_penn / l_ts * 2 * k_tendon * max(0, strain)
            strain = torch.clamp(l_tendon_norm - 1.0, min=0)
            df_t = -cos_penn / self.l_tendon_slack * 2.0 * self.k_tendon * strain

            # df_m/dl_ce = a * df_fl/dl_ce * cos_penn + df_pe/dl_ce
            dl = 1e-5
            f_fl_plus = self.force_length_active(l_norm_iter + dl / self.l_opt)
            f_pe_plus = self.force_length_passive(l_norm_iter + dl / self.l_opt)
            df_m = (activation * (f_fl_plus - f_fl) * cos_penn + (f_pe_plus - f_pe)) / dl

            # Newton step
            denom = df_t - df_m
            denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom) * 1e-8, denom)
            delta = residual / denom
            l_ce = l_ce - delta
            l_ce = torch.clamp(l_ce, min=1e-4)

        l_norm = l_ce / self.l_opt

        # 섬유 속도: elastic tendon에서는 근사적으로 rigid와 동일하게 처리
        # (정확한 계산은 implicit integration 필요, 여기서는 1차 근사)
        v_ce = muscle_velocity / cos_penn
        v_norm = v_ce / (self.v_max * self.l_opt)

        return l_norm, v_norm

    def compute_force(
        self,
        activation: torch.Tensor,     # (num_envs, num_muscles) [0, 1]
        muscle_length: torch.Tensor,   # (num_envs, num_muscles) MTU 길이 (m)
        muscle_velocity: torch.Tensor, # (num_envs, num_muscles) MTU 속도 (m/s)
    ) -> torch.Tensor:
        """총 근육 힘 계산.

        Returns:
            F_total: (num_envs, num_muscles) 근육 힘 (N)
        """
        cos_penn = torch.cos(self.pennation)

        # Elastic tendon이 있으면 해당 근육만 elastic 계산
        if self._elastic_mask.any():
            # 기본: rigid tendon으로 전체 계산
            l_norm, v_norm = self._compute_fiber_state_rigid(
                muscle_length, muscle_velocity
            )
            # Elastic tendon 근육만 덮어쓰기
            l_norm_e, v_norm_e = self._compute_fiber_state_elastic(
                activation, muscle_length, muscle_velocity
            )
            l_norm[:, self._elastic_mask] = l_norm_e[:, self._elastic_mask]
            v_norm[:, self._elastic_mask] = v_norm_e[:, self._elastic_mask]
        else:
            l_norm, v_norm = self._compute_fiber_state_rigid(
                muscle_length, muscle_velocity
            )

        # Active force: a * F_max * f_FL(l) * f_FV(v) * cos(pennation)
        f_fl = self.force_length_active(l_norm)
        f_fv = self.force_velocity(v_norm)
        F_active = activation * self.f_max * f_fl * f_fv * cos_penn

        # Passive force: F_max * f_PE(l)
        f_pe = self.force_length_passive(l_norm)
        F_passive = self.f_max * f_pe

        # Damping force (정규화된 속도 사용)
        F_damping = self.damping * self.f_max * v_norm

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
        cos_penn = torch.cos(self.pennation)

        if self._elastic_mask.any():
            l_norm, v_norm = self._compute_fiber_state_rigid(
                muscle_length, muscle_velocity
            )
            l_norm_e, v_norm_e = self._compute_fiber_state_elastic(
                activation, muscle_length, muscle_velocity
            )
            l_norm[:, self._elastic_mask] = l_norm_e[:, self._elastic_mask]
            v_norm[:, self._elastic_mask] = v_norm_e[:, self._elastic_mask]
        else:
            l_norm, v_norm = self._compute_fiber_state_rigid(
                muscle_length, muscle_velocity
            )

        f_fl = self.force_length_active(l_norm)
        f_fv = self.force_velocity(v_norm)

        F_active = activation * self.f_max * f_fl * f_fv * cos_penn
        F_passive = self.f_max * self.force_length_passive(l_norm)
        F_damping = self.damping * self.f_max * v_norm

        return {
            "F_active": F_active,
            "F_passive": F_passive,
            "F_damping": F_damping,
            "f_FL": f_fl,
            "f_FV": f_fv,
            "l_norm": l_norm,
            "v_norm": v_norm,
        }

"""
PatientDynamics: 환자 모델 기반 토크 계산 엔진

9개 파라미터로 정의된 환자 프로파일을 IsaacGym 환경에서 적용한다.
기존 VIC의 _compute_torques()를 확장하는 형태로 설계.

토크 구성:
  human_torque = τ_voluntary + τ_passive + τ_spasticity + τ_endstop + τ_tremor
  exo_torque   = (학습 대상, VIC/CCF)
  total_torque = human_torque + exo_torque
"""

import torch
import numpy as np
from typing import Dict, Optional

from standard_human_model.core.patient_profile import (
    PatientProfile, JOINT_GROUPS, PARAM_DEFAULTS,
)


# VIC CCF 그룹 → DOF 매핑 (SMPL humanoid 69 DOFs)
GROUP_DOF_RANGES = {
    "L_Hip":       (0, 3),    # 3 DOFs
    "L_Knee":      (3, 6),    # 3 DOFs
    "L_Ankle_Toe": (6, 12),   # 6 DOFs
    "R_Hip":       (12, 15),  # 3 DOFs
    "R_Knee":      (15, 18),  # 3 DOFs
    "R_Ankle_Toe": (18, 24),  # 6 DOFs
    "Upper_L":     (24, 54),  # 30 DOFs
    "Upper_R":     (54, 69),  # 15 DOFs
}

NUM_DOFS = 69


class PatientDynamics:
    """환자 dynamics 토크 계산기.

    초기화 시 프로파일의 파라미터를 DOF 단위 텐서로 확장해두고,
    매 step에서 텐서 연산으로 빠르게 토크를 계산한다.

    사용법:
        profile = PatientProfile.load("sci/sci_t10_complete_flaccid")
        dynamics = PatientDynamics(profile, num_envs=512, device="cuda:0")

        # 매 step
        human_torque = dynamics.compute_torques(
            dof_pos, dof_vel, pd_targets, kp, kd, dt, sim_time
        )
    """

    def __init__(self, profile: PatientProfile, num_envs: int,
                 device: str = "cuda:0"):
        self.profile = profile
        self.num_envs = num_envs
        self.device = device

        # 프로파일 → DOF 단위 텐서 확장 (num_envs × num_dofs)
        self._build_param_tensors()

        # Tremor phase (환경마다 랜덤 초기 위상)
        self._tremor_phase = torch.rand(num_envs, NUM_DOFS, device=device) * 2 * np.pi

    def _build_param_tensors(self):
        """프로파일의 8그룹 파라미터를 69 DOF 텐서로 확장."""
        params = self.profile.get_all_params()

        # 각 파라미터를 DOF 차원으로
        self.tau_active = torch.zeros(NUM_DOFS, device=self.device)
        self.k_passive = torch.zeros(NUM_DOFS, device=self.device)
        self.b_passive = torch.zeros(NUM_DOFS, device=self.device)
        self.spasticity = torch.zeros(NUM_DOFS, device=self.device)
        self.spas_dir = torch.zeros(NUM_DOFS, device=self.device)
        self.rom_scale = torch.ones(NUM_DOFS, device=self.device)
        self.k_endstop = torch.zeros(NUM_DOFS, device=self.device)
        self.tremor_amp = torch.zeros(NUM_DOFS, device=self.device)
        self.tremor_freq = torch.zeros(NUM_DOFS, device=self.device)

        for group_name in JOINT_GROUPS:
            start, end = GROUP_DOF_RANGES[group_name]
            p = params[group_name]

            self.tau_active[start:end] = p["tau_active"]
            self.k_passive[start:end] = p["k_passive"]
            self.b_passive[start:end] = p["b_passive"]
            self.spasticity[start:end] = p["spasticity"]
            self.spas_dir[start:end] = p["spas_dir"]
            self.rom_scale[start:end] = p["rom_scale"]
            self.k_endstop[start:end] = p["k_endstop"]
            self.tremor_amp[start:end] = p["tremor_amp"]
            self.tremor_freq[start:end] = p["tremor_freq"]

        # (num_envs, num_dofs)로 확장
        self.tau_active = self.tau_active.unsqueeze(0).expand(self.num_envs, -1)
        self.k_passive = self.k_passive.unsqueeze(0).expand(self.num_envs, -1)
        self.b_passive = self.b_passive.unsqueeze(0).expand(self.num_envs, -1)
        self.spasticity = self.spasticity.unsqueeze(0).expand(self.num_envs, -1)
        self.spas_dir = self.spas_dir.unsqueeze(0).expand(self.num_envs, -1)
        self.rom_scale = self.rom_scale.unsqueeze(0).expand(self.num_envs, -1)
        self.k_endstop = self.k_endstop.unsqueeze(0).expand(self.num_envs, -1)
        self.tremor_amp = self.tremor_amp.unsqueeze(0).expand(self.num_envs, -1)
        self.tremor_freq = self.tremor_freq.unsqueeze(0).expand(self.num_envs, -1)

    def compute_torques(
        self,
        dof_pos: torch.Tensor,       # (num_envs, num_dofs)
        dof_vel: torch.Tensor,        # (num_envs, num_dofs)
        pd_targets: torch.Tensor,     # (num_envs, num_dofs) - 능동 제어 목표
        kp: torch.Tensor,             # (num_dofs,) or (num_envs, num_dofs)
        kd: torch.Tensor,             # (num_dofs,) or (num_envs, num_dofs)
        sim_time: float,              # 현재 시뮬레이션 시간 (초)
        q_rest: Optional[torch.Tensor] = None,   # (num_dofs,) 안정 자세, 없으면 0
        joint_limits_lower: Optional[torch.Tensor] = None,  # (num_dofs,)
        joint_limits_upper: Optional[torch.Tensor] = None,  # (num_dofs,)
    ) -> torch.Tensor:
        """환자의 인간 토크를 계산한다.

        Returns:
            human_torque: (num_envs, num_dofs) 인간 측 총 토크
        """
        if q_rest is None:
            q_rest = torch.zeros(NUM_DOFS, device=self.device)
        q_rel = dof_pos - q_rest

        # 1. 능동 토크 (환자의 자발적 힘, tau_active로 스케일링)
        tau_vol = self.tau_active * (kp * (pd_targets - dof_pos) - kd * dof_vel)

        # 2. 수동 강성 + 감쇠
        tau_passive = -self.k_passive * q_rel - self.b_passive * dof_vel

        # 3. 경직 (속도의존, 방향 비대칭)
        #    spas_dir > 0: 양방향(신전) 속도에 더 강한 저항
        #    spas_dir < 0: 음방향(굴곡) 속도에 더 강한 저항
        dir_mask = torch.where(
            dof_vel > 0,
            1.0 + self.spas_dir,
            1.0 - self.spas_dir,
        )
        tau_spasticity = -self.spasticity * torch.abs(dof_vel) * torch.sign(dof_vel) * dir_mask

        # 4. 끝범위 비선형 강성 (구축)
        tau_endstop = torch.zeros_like(dof_pos)
        if joint_limits_lower is not None and joint_limits_upper is not None:
            scaled_lower = joint_limits_lower * self.rom_scale
            scaled_upper = joint_limits_upper * self.rom_scale

            # 하한 초과
            dist_lower = torch.clamp(scaled_lower - dof_pos, min=0)
            # 상한 초과
            dist_upper = torch.clamp(dof_pos - scaled_upper, min=0)

            tau_endstop = (self.k_endstop * dist_lower ** 2
                           - self.k_endstop * dist_upper ** 2)

        # 5. 떨림 (주기적 비자발적 토크)
        tau_tremor = torch.zeros_like(dof_pos)
        has_tremor = self.tremor_freq[0] > 0  # (num_dofs,) 마스크
        if has_tremor.any():
            phase = 2 * np.pi * self.tremor_freq * sim_time + self._tremor_phase
            tau_tremor = self.tremor_amp * torch.sin(phase)

        # 합산
        human_torque = tau_vol + tau_passive + tau_spasticity + tau_endstop + tau_tremor

        return human_torque

    def get_effective_rom(
        self,
        joint_limits_lower: torch.Tensor,
        joint_limits_upper: torch.Tensor,
    ) -> tuple:
        """ROM_scale이 적용된 실효 관절 한계 반환."""
        return (
            joint_limits_lower * self.rom_scale[0],  # 첫 env 기준 (모두 동일)
            joint_limits_upper * self.rom_scale[0],
        )

    def reset_tremor_phase(self, env_ids: torch.Tensor):
        """특정 환경들의 떨림 위상 리셋 (에피소드 리셋 시 호출)."""
        self._tremor_phase[env_ids] = (
            torch.rand(len(env_ids), NUM_DOFS, device=self.device) * 2 * np.pi
        )

    def info(self) -> Dict:
        """현재 프로파일의 요약 정보 반환."""
        return {
            "name": self.profile.name,
            "injury_type": self.profile.injury_type,
            "num_envs": self.num_envs,
            "device": self.device,
            "has_tremor": bool((self.tremor_freq[0] > 0).any()),
            "has_spasticity": bool((self.spasticity[0] > 0).any()),
            "min_tau_active": float(self.tau_active[0].min()),
            "max_tau_active": float(self.tau_active[0].max()),
        }

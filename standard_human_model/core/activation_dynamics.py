"""
activation_dynamics.py — 신경 명령 → 근육 활성화 ODE

Neural command u ∈ [0,1] → Muscle activation a ∈ [0,1] 변환.
활성화/비활성화 시정수가 다른 1차 ODE로 모델링.

  if u > a:
      da/dt = (u - a) / tau_act        # 활성화 (빠름, ~10-50ms)
  else:
      da/dt = (u - a) / tau_deact      # 비활성화 (느림, ~40-200ms)

환자별 차이:
- 정상: tau_act=0.015, tau_deact=0.060
- 파킨슨: tau_act=0.050, tau_deact=0.200  (bradykinesia)
- 노인: tau_act=0.030, tau_deact=0.100
"""

import torch
from typing import Optional


class ActivationDynamics:
    """근육 활성화 동역학.

    사용법:
        act_dyn = ActivationDynamics(num_muscles=16, num_envs=512, device="cuda:0")
        act_dyn.set_time_constants(tau_act, tau_deact)

        # 매 step
        activation = act_dyn.step(neural_command, dt)

        # 에피소드 리셋
        act_dyn.reset(env_ids)
    """

    def __init__(self, num_muscles: int, num_envs: int, device: str = "cpu"):
        self.num_muscles = num_muscles
        self.num_envs = num_envs
        self.device = device

        # 현재 활성화 상태
        self.activation = torch.zeros(num_envs, num_muscles, device=device)

        # 시정수 (num_muscles,)
        self.tau_act = torch.ones(num_muscles, device=device) * 0.015    # 15ms
        self.tau_deact = torch.ones(num_muscles, device=device) * 0.060  # 60ms

    def set_time_constants(
        self,
        tau_act: torch.Tensor,    # (num_muscles,)
        tau_deact: torch.Tensor,  # (num_muscles,)
    ):
        """근육별 시정수 설정."""
        self.tau_act = tau_act.to(self.device)
        self.tau_deact = tau_deact.to(self.device)

    def step(
        self,
        neural_command: torch.Tensor,  # (num_envs, num_muscles) [0, 1]
        dt: float,
    ) -> torch.Tensor:
        """1 step 진행. 활성화 상태 업데이트 후 반환.

        Args:
            neural_command: u ∈ [0, 1], reflex controller 또는 상위 제어기 출력
            dt: 시뮬레이션 timestep (초)

        Returns:
            activation: (num_envs, num_muscles) [0, 1]
        """
        u = torch.clamp(neural_command, 0, 1)
        a = self.activation

        # 활성화/비활성화 시정수 선택
        activating = u > a
        tau = torch.where(activating, self.tau_act, self.tau_deact)

        # 1차 ODE Euler 적분
        # da/dt = (u - a) / tau
        da = (u - a) / tau * dt
        self.activation = torch.clamp(a + da, 0, 1)

        return self.activation

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """활성화 상태 리셋.

        Args:
            env_ids: 리셋할 환경 인덱스. None이면 전체 리셋.
        """
        if env_ids is None:
            self.activation.zero_()
        else:
            self.activation[env_ids] = 0

    def get_activation(self) -> torch.Tensor:
        """현재 활성화 상태 반환 (읽기 전용)."""
        return self.activation.clone()

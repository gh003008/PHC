"""
reflex_controller.py — 척수 반사 제어기

상위 제어기(RL, CPG 등)와 독립적으로 동작하는 척수 수준 반사.
상위 명령(descending command)에 반사 출력을 더하여 최종 근활성화 결정.

반사 종류:
1. Stretch Reflex (신장 반사) — 근방추, 속도 의존
   - 정상: 관절 안정화
   - 경직: gain 과도 → velocity-dependent resistance
2. GTO Reflex (골지건 반사) — 힘 피드백, 과부하 방지
3. Reciprocal Inhibition — 주동근 활성 시 길항근 억제
4. Cutaneous / Load Reflex — 발바닥 접촉 → 신전근 활성

환자별 차이는 gain 파라미터 조절로 표현:
- 경직(UMN): stretch_gain↑↑, threshold↓, reciprocal_inhibition↓
- 이완(LMN): stretch_gain=0
- 파킨슨: reciprocal_inhibition↓ (co-contraction)
"""

import torch
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class ReflexParams:
    """근육별 반사 파라미터."""
    # Stretch reflex
    stretch_gain: float = 1.0       # 신장 반사 게인 (경직: 5~15)
    stretch_threshold: float = 0.1  # 반사 발동 속도 역치 (경직: 0.01)
    # GTO reflex
    gto_gain: float = 0.5           # GTO 억제 게인
    gto_threshold: float = 0.8     # 힘 역치 (F/F_max 비율)
    # Reciprocal inhibition
    reciprocal_gain: float = 0.3    # 길항근 억제 게인 (파킨슨: 0.05)


class ReflexController:
    """척수 반사 제어기.

    사용법:
        reflex = ReflexController(num_muscles=16, num_envs=512, device="cuda:0")
        reflex.set_params(params_dict)
        reflex.set_antagonist_pairs(pairs)

        # 매 step
        a_total = reflex.compute(
            descending_cmd, muscle_velocity, muscle_force,
            contact_forces, f_max
        )
    """

    def __init__(self, num_muscles: int, num_envs: int, device: str = "cpu",
                 reflex_delay_steps: int = 1):
        self.num_muscles = num_muscles
        self.num_envs = num_envs
        self.device = device
        self.reflex_delay_steps = reflex_delay_steps

        # 반사 파라미터 (num_muscles,)
        self.stretch_gain = torch.ones(num_muscles, device=device)
        self.stretch_threshold = torch.ones(num_muscles, device=device) * 0.1
        self.gto_gain = torch.ones(num_muscles, device=device) * 0.5
        self.gto_threshold = torch.ones(num_muscles, device=device) * 0.8
        self.reciprocal_gain = torch.ones(num_muscles, device=device) * 0.3

        # 길항근 쌍 매핑: antagonist_map[i] = j (i의 길항근이 j)
        # -1이면 길항근 없음
        self.antagonist_map = torch.full((num_muscles,), -1, dtype=torch.long, device=device)

        # 지연 버퍼 (stretch reflex 시간 지연용)
        self._delay_buffer = torch.zeros(
            reflex_delay_steps, num_envs, num_muscles, device=device
        )
        self._buffer_idx = 0

    def set_params(self, params_per_muscle: Dict[int, ReflexParams]):
        """근육별 반사 파라미터 설정.

        Args:
            params_per_muscle: {muscle_index: ReflexParams}
        """
        for idx, p in params_per_muscle.items():
            self.stretch_gain[idx] = p.stretch_gain
            self.stretch_threshold[idx] = p.stretch_threshold
            self.gto_gain[idx] = p.gto_gain
            self.gto_threshold[idx] = p.gto_threshold
            self.reciprocal_gain[idx] = p.reciprocal_gain

    def set_antagonist_pairs(self, pairs: list):
        """길항근 쌍 설정.

        Args:
            pairs: [(agonist_idx, antagonist_idx), ...]
                   예: [(quadriceps_idx, hamstrings_idx), ...]
        """
        for ag, ant in pairs:
            self.antagonist_map[ag] = ant
            self.antagonist_map[ant] = ag

    def compute(
        self,
        descending_cmd: torch.Tensor,    # (num_envs, num_muscles) [0, 1]
        muscle_velocity: torch.Tensor,   # (num_envs, num_muscles)
        muscle_force: torch.Tensor,      # (num_envs, num_muscles)
        contact_forces: Optional[torch.Tensor] = None,  # (num_envs, 4) L/R foot
        f_max: Optional[torch.Tensor] = None,  # (num_muscles,)
    ) -> torch.Tensor:
        """반사 출력 계산.

        Returns:
            a_total: (num_envs, num_muscles) [0, 1] 최종 근활성화 명령
        """
        # 1. Stretch Reflex (신장 반사)
        #    근육이 빠르게 늘어나면 (velocity > 0) 반사적으로 수축
        stretch_signal = torch.clamp(muscle_velocity - self.stretch_threshold, min=0)
        a_stretch_raw = self.stretch_gain * stretch_signal

        # 시간 지연 적용
        a_stretch = self._apply_delay(a_stretch_raw)

        # 2. GTO Reflex (골지건 반사)
        #    근육 힘이 역치를 넘으면 해당 근육 억제 (과부하 방지)
        a_gto = torch.zeros_like(descending_cmd)
        if f_max is not None:
            force_ratio = muscle_force / (f_max + 1e-8)
            gto_signal = torch.clamp(force_ratio - self.gto_threshold, min=0)
            a_gto = -self.gto_gain * gto_signal  # 음수 = 억제

        # 3. Reciprocal Inhibition (상반 억제)
        #    주동근이 활성화되면 길항근을 억제
        a_reciprocal = torch.zeros_like(descending_cmd)
        has_antagonist = self.antagonist_map >= 0
        if has_antagonist.any():
            valid_indices = torch.where(has_antagonist)[0]
            for idx in valid_indices:
                ant_idx = self.antagonist_map[idx].item()
                # 주동근의 하행 명령이 클수록 길항근 억제
                agonist_activation = descending_cmd[:, idx] + a_stretch[:, idx]
                a_reciprocal[:, ant_idx] -= self.reciprocal_gain[idx] * agonist_activation

        # 4. 합산: 하행 명령 + stretch + GTO + reciprocal
        a_total = descending_cmd + a_stretch + a_gto + a_reciprocal

        return torch.clamp(a_total, 0, 1)

    def _apply_delay(self, signal: torch.Tensor) -> torch.Tensor:
        """반사 신호에 시간 지연 적용 (circular buffer)."""
        if self.reflex_delay_steps <= 0:
            return signal

        # 현재 신호를 버퍼에 저장
        self._delay_buffer[self._buffer_idx] = signal

        # 지연된 신호 읽기
        delayed_idx = (self._buffer_idx + 1) % self.reflex_delay_steps
        delayed_signal = self._delay_buffer[delayed_idx].clone()

        # 인덱스 전진
        self._buffer_idx = (self._buffer_idx + 1) % self.reflex_delay_steps

        return delayed_signal

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """리셋."""
        if env_ids is None:
            self._delay_buffer.zero_()
        else:
            self._delay_buffer[:, env_ids] = 0

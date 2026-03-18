"""
humanoid_im_vic_msk.py — VIC + Musculoskeletal Hybrid Environment

VIC의 PD 토크에 근골격계 수동 역학(passive muscle, ligament, reflex)을 합산.
blend_alpha로 근골격계 비중 조절:
  - alpha=0.0: 기존 VIC와 동일 (regression)
  - alpha=0.1~0.5: 하이브리드 (PD + bio)
  - alpha=1.0: 근골격계 수동 역학 최대 반영

전략 C Phase 1 구현.
"""

import torch
import numpy as np
from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
from phc.utils.flags import flags


class HumanoidImVICMSK(HumanoidImVIC):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # MSK config 미리 저장 (super().__init__에서 device 확정 전)
        self._msk_cfg = cfg["env"].get("msk_config", {})

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # super().__init__ 완료 후 self.device 확정 → HumanBody 초기화
        self._init_human_body()

    def _init_human_body(self):
        """근골격계 모델 초기화."""
        from standard_human_model.core.human_body import HumanBody

        muscle_def = self._msk_cfg.get("muscle_def", "muscle_definitions.yaml")
        patient_profile = self._msk_cfg.get("patient_profile", "healthy_baseline.yaml")
        self._msk_blend_alpha = self._msk_cfg.get("blend_alpha", 0.0)
        self._msk_max_torque_ratio = self._msk_cfg.get("max_torque_ratio", 0.5)

        self._human_body = HumanBody.from_config(
            muscle_def_path=muscle_def,
            param_path=patient_profile,
            num_envs=self.num_envs,
            device=self.device,
        )

        print(f"[VIC-MSK] HumanBody initialized: {self._human_body.num_muscles} muscles, "
              f"blend_alpha={self._msk_blend_alpha}, "
              f"max_torque_ratio={self._msk_max_torque_ratio}, "
              f"profile={patient_profile}")

    def _compute_torques(self, actions):
        """VIC PD 토크 + 근골격계 수동 역학 토크 합산."""
        # 1. 기존 VIC PD 토크 계산
        torques_pd = super()._compute_torques(actions)

        # 2. blend_alpha=0이면 bio 토크 계산 스킵 (성능 최적화)
        if self._msk_blend_alpha == 0.0:
            return torques_pd

        # 3. 근골격계 수동 역학 토크 계산
        #    descending_cmd=0 → 자발적 근수축 없음, 수동 역학만 활성
        #    (passive muscle force + ligament + stretch reflex)
        descending_cmd = torch.zeros(
            self.num_envs, self._human_body.num_muscles,
            device=self.device,
        )

        torques_bio = self._human_body.compute_torques(
            self._dof_pos, self._dof_vel,
            descending_cmd,
            dt=self.dt,
        )

        # 4. Bio 토크 크기 제한 (PD 토크 대비 비율로 클램핑)
        if self._msk_max_torque_ratio > 0:
            pd_mag = torques_pd.abs().mean(dim=0, keepdim=True).clamp(min=1.0)
            bio_limit = pd_mag * self._msk_max_torque_ratio
            torques_bio = torch.clamp(torques_bio, -bio_limit, bio_limit)

        # 5. 하이브리드 합산
        torques = torques_pd + self._msk_blend_alpha * torques_bio

        return torch.clamp(torques, -self.torque_limits, self.torque_limits)

    def _reset_envs(self, env_ids):
        """에피소드 리셋 시 근골격계 내부 상태도 리셋."""
        super()._reset_envs(env_ids)
        if hasattr(self, '_human_body'):
            self._human_body.reset(env_ids)

    def _log_msk_metrics(self):
        """wandb 로깅용 근골격계 메트릭."""
        if not hasattr(self, '_human_body'):
            return {}

        metrics = {
            "msk/blend_alpha": self._msk_blend_alpha,
        }

        # 근활성화 통계
        activation = self._human_body.get_activation()
        if activation is not None and activation.numel() > 0:
            metrics["msk/activation_mean"] = activation.mean().item()
            metrics["msk/activation_max"] = activation.max().item()

        return metrics

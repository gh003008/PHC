"""
human_body.py — 통합 인간 모델 파이프라인

전략 문서(0317_project_strategy01.md)의 pre_physics_step 흐름을 구현.
모든 근골격 모듈을 조합하여 하나의 compute_torques() 호출로 동작.

파이프라인:
  1. 상위 명령 (u) 수신
  2. Reflex layer (stretch, GTO, reciprocal inhibition)
  3. Activation dynamics (1st order ODE)
  4. Muscle kinematics (R(q) → 근육 길이/속도)
  5. Force generation (Hill model: active + passive)
  6. Torque mapping (R(q)^T → 관절 토크, coupling 자동 발생)
  7. Ligament forces (soft limit)
  8. 최종 토크 출력
"""

import os
import yaml
import torch
from typing import Optional, Dict

from standard_human_model.core.skeleton import (
    NUM_DOFS, JOINT_NAMES, JOINT_DOF_RANGE,
    get_kp_kd_tensors, get_joint_limits_tensors,
)
from standard_human_model.core.muscle_model import HillMuscleModel, MuscleParams
from standard_human_model.core.moment_arm import MomentArmMatrix
from standard_human_model.core.activation_dynamics import ActivationDynamics
from standard_human_model.core.reflex_controller import ReflexController, ReflexParams
from standard_human_model.core.ligament_model import LigamentModel

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


class HumanBody:
    """통합 인간 모델.

    YAML 설정 파일 2개로 완전히 정의:
    - muscle_definitions.yaml: 근육군 구조 (moment arm, 길항근 쌍)
    - patient profile yaml: 근육 파라미터 (f_max, 반사 gain 등)

    사용법:
        body = HumanBody.from_config(
            muscle_def_path="config/muscle_definitions.yaml",
            param_path="config/healthy_baseline.yaml",
            num_envs=512, device="cuda:0"
        )

        # 매 step (IsaacGym pre_physics_step에서 호출)
        torques = body.compute_torques(
            dof_pos, dof_vel,
            descending_cmd,  # 상위 제어기 출력 (RL, CPG, EMG replay 등)
            dt=1/30, sim_time=0.5
        )
    """

    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.num_muscles = 0

        # 서브 모듈 (from_config에서 초기화)
        self.muscle_model: Optional[HillMuscleModel] = None
        self.moment_arm: Optional[MomentArmMatrix] = None
        self.activation_dyn: Optional[ActivationDynamics] = None
        self.reflex: Optional[ReflexController] = None
        self.ligament: Optional[LigamentModel] = None

        # 관절 한계 (ligament용)
        self.joint_limits_lower, self.joint_limits_upper = get_joint_limits_tensors(device)

    @classmethod
    def from_config(
        cls,
        muscle_def_path: str,
        param_path: str,
        num_envs: int,
        device: str = "cpu",
    ) -> "HumanBody":
        """설정 파일에서 전체 모델 생성.

        Args:
            muscle_def_path: 근육군 구조 정의 YAML
            param_path: 환자/정상인 파라미터 YAML
            num_envs: 병렬 환경 수
            device: "cpu" 또는 "cuda:0"
        """
        body = cls(num_envs, device)

        # --- 1. 근육 구조 로드 ---
        if not os.path.isabs(muscle_def_path):
            muscle_def_path = os.path.join(CONFIG_DIR, muscle_def_path)
        with open(muscle_def_path, "r", encoding="utf-8") as f:
            muscle_def = yaml.safe_load(f)

        muscle_list = muscle_def["muscles"]
        body.num_muscles = len(muscle_list)
        muscle_names = [m["name"] for m in muscle_list]

        # --- 2. 파라미터 로드 ---
        if not os.path.isabs(param_path):
            param_path = os.path.join(CONFIG_DIR, param_path)
        with open(param_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        muscle_params = params.get("muscle_params", {})
        reflex_cfg = params.get("reflex_params", {})
        ligament_cfg = params.get("ligament_params", {})

        # --- 3. Moment Arm Matrix ---
        body.moment_arm = MomentArmMatrix(muscle_list, device=device)

        # --- 4. Hill Muscle Model ---
        body.muscle_model = HillMuscleModel(body.num_muscles, num_envs, device)

        param_objects = []
        for m in muscle_list:
            name = m["name"]
            mp = muscle_params.get(name, {})
            param_objects.append(MuscleParams(
                name=name,
                f_max=mp.get("f_max", 1000),
                l_opt=mp.get("l_opt", 1.0),
                v_max=mp.get("v_max", 10.0),
                pennation=mp.get("pennation", 0.0),
                tau_act=mp.get("tau_act", 0.015),
                tau_deact=mp.get("tau_deact", 0.060),
                k_pe=mp.get("k_pe", 4.0),
                epsilon_0=mp.get("epsilon_0", 0.6),
                l_tendon_slack=mp.get("l_tendon_slack", 1.0),
                k_tendon=mp.get("k_tendon", 35.0),
                damping=mp.get("damping", 0.05),
            ))
        body.muscle_model.set_params(param_objects)

        # --- 5. Activation Dynamics ---
        body.activation_dyn = ActivationDynamics(body.num_muscles, num_envs, device)
        tau_act_tensor = torch.tensor(
            [p.tau_act for p in param_objects], device=device
        )
        tau_deact_tensor = torch.tensor(
            [p.tau_deact for p in param_objects], device=device
        )
        body.activation_dyn.set_time_constants(tau_act_tensor, tau_deact_tensor)

        # --- 6. Reflex Controller ---
        default_reflex = reflex_cfg.get("default", {})
        body.reflex = ReflexController(
            body.num_muscles, num_envs, device,
            reflex_delay_steps=1,
        )

        # 기본 반사 파라미터 적용
        reflex_params_dict = {}
        for i in range(body.num_muscles):
            name = muscle_names[i]
            rp = reflex_cfg.get(name, default_reflex)
            reflex_params_dict[i] = ReflexParams(
                stretch_gain=rp.get("stretch_gain", 1.0),
                stretch_threshold=rp.get("stretch_threshold", 0.1),
                gto_gain=rp.get("gto_gain", 0.5),
                gto_threshold=rp.get("gto_threshold", 0.8),
                reciprocal_gain=rp.get("reciprocal_gain", 0.3),
            )
        body.reflex.set_params(reflex_params_dict)

        # 길항근 쌍
        antagonist_pairs = muscle_def.get("antagonist_pairs", [])
        ant_idx_pairs = []
        for pair in antagonist_pairs:
            try:
                idx_a = muscle_names.index(pair[0])
                idx_b = muscle_names.index(pair[1])
                ant_idx_pairs.append((idx_a, idx_b))
            except ValueError:
                pass
        body.reflex.set_antagonist_pairs(ant_idx_pairs)

        # --- 7. Ligament Model ---
        body.ligament = LigamentModel(num_envs, device)
        margin = ligament_cfg.get("soft_limit_margin", 0.85)
        body.ligament.set_limits_from_hard_limits(
            body.joint_limits_lower, body.joint_limits_upper, margin
        )
        if "k_lig" in ligament_cfg:
            body.ligament.k_lig = torch.ones(NUM_DOFS, device=device) * ligament_cfg["k_lig"]
        if "alpha" in ligament_cfg:
            body.ligament.alpha = torch.ones(NUM_DOFS, device=device) * ligament_cfg["alpha"]
        if "damping" in ligament_cfg:
            body.ligament.damping = torch.ones(NUM_DOFS, device=device) * ligament_cfg["damping"]

        # f_max 텐서 저장 (reflex에서 사용)
        body._f_max = torch.tensor(
            [p.f_max for p in param_objects], device=device
        )

        return body

    def compute_torques(
        self,
        dof_pos: torch.Tensor,           # (num_envs, 69)
        dof_vel: torch.Tensor,           # (num_envs, 69)
        descending_cmd: torch.Tensor,    # (num_envs, num_muscles) [0, 1]
        dt: float,
        sim_time: float = 0.0,
        contact_forces: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """전체 파이프라인 실행. 관절 토크 반환.

        Args:
            dof_pos: 현재 관절 각도
            dof_vel: 현재 관절 각속도
            descending_cmd: 상위 제어기 명령 (RL, CPG, EMG replay)
                            (num_envs, num_muscles) [0, 1]
            dt: 시뮬레이션 timestep
            sim_time: 현재 시뮬레이션 시간 (초)
            contact_forces: (num_envs, 4) 발바닥 접촉력 (선택)

        Returns:
            tau_total: (num_envs, 69) 관절 토크
        """
        # 1. 근육 kinematics
        muscle_length = self.moment_arm.compute_muscle_length(dof_pos)
        muscle_velocity = self.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)

        # 2. 현재 근육 힘 (이전 step의 activation 기반, reflex용)
        current_activation = self.activation_dyn.get_activation()
        current_force = self.muscle_model.compute_force(
            current_activation, muscle_length, muscle_velocity
        )

        # 3. Reflex layer
        a_cmd = self.reflex.compute(
            descending_cmd, muscle_velocity, current_force,
            contact_forces, self._f_max,
        )

        # 4. Activation dynamics
        activation = self.activation_dyn.step(a_cmd, dt)

        # 5. Force generation (Hill model)
        F_muscle = self.muscle_model.compute_force(
            activation, muscle_length, muscle_velocity
        )

        # 6. Torque mapping: τ = R^T @ F (bi-articular coupling 자동 발생)
        tau_muscle = self.moment_arm.forces_to_torques(F_muscle)

        # 7. Ligament forces
        tau_ligament = self.ligament.compute_torque(dof_pos, dof_vel)

        # 8. 합산
        tau_total = tau_muscle + tau_ligament

        return tau_total

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """에피소드 리셋."""
        self.activation_dyn.reset(env_ids)
        self.reflex.reset(env_ids)
        self.ligament.reset(env_ids)

    def get_activation(self) -> torch.Tensor:
        """현재 근활성화 반환 (관측/로깅용)."""
        return self.activation_dyn.get_activation()

    def get_muscle_forces(
        self, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> torch.Tensor:
        """현재 근육 힘 반환 (관측/로깅용)."""
        activation = self.activation_dyn.get_activation()
        ml = self.moment_arm.compute_muscle_length(dof_pos)
        mv = self.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)
        return self.muscle_model.compute_force(activation, ml, mv)

    def info(self) -> Dict:
        """모델 요약 정보."""
        return {
            "num_muscles": self.num_muscles,
            "num_envs": self.num_envs,
            "device": self.device,
            "coupling_info": self.moment_arm.get_coupling_info(),
        }

    def summary(self) -> str:
        """사람이 읽기 쉬운 요약."""
        lines = [
            f"HumanBody: {self.num_muscles} muscles, {self.num_envs} envs, {self.device}",
            "",
            self.moment_arm.summary(),
        ]
        return "\n".join(lines)

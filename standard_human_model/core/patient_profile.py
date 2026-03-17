"""
PatientProfile: YAML 기반 환자 프로파일 로더

환자 YAML 파일을 로드하여 관절그룹별 9개 dynamics 파라미터를 제공한다.
SMPL의 beta parameter처럼, 프로파일 하나로 환자 특성이 결정된다.
"""

import os
import yaml
import copy
from typing import Dict, Optional


# 8개 관절 그룹 (VIC CCF 그룹과 동일)
JOINT_GROUPS = [
    "L_Hip",        # G0: DOFs 0-2
    "L_Knee",       # G1: DOFs 3-5
    "L_Ankle_Toe",  # G2: DOFs 6-11
    "R_Hip",        # G3: DOFs 12-14
    "R_Knee",       # G4: DOFs 15-17
    "R_Ankle_Toe",  # G5: DOFs 18-23
    "Upper_L",      # G6: DOFs 24-53
    "Upper_R",      # G7: DOFs 54-68
]

# 9개 dynamics 파라미터 정의 및 기본값 (정상 성인)
PARAM_DEFAULTS = {
    "tau_active":    1.0,   # [0, 1]       능동 토크 비율
    "k_passive":     3.0,   # [0, inf)     수동 강성 Nm/rad
    "b_passive":     1.5,   # [0, inf)     수동 감쇠 Nm·s/rad
    "spasticity":    0.0,   # [0, inf)     속도의존 강성
    "spas_dir":      0.0,   # [-1, 1]      경직 방향 비대칭 (0=대칭, +1=신전, -1=굴곡)
    "rom_scale":     1.0,   # [0, 1]       가동범위 비율
    "k_endstop":     0.0,   # [0, inf)     끝범위 비선형 강성 Nm/rad^2
    "tremor_amp":    0.0,   # [0, inf)     떨림 진폭 Nm
    "tremor_freq":   0.0,   # [4, 12] Hz   떨림 주파수 (0이면 비활성)
}

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "profiles")


class PatientProfile:
    """환자 프로파일 로더 및 관리 클래스.

    사용법:
        profile = PatientProfile.load("sci/sci_t10_complete_flaccid")
        params = profile.get_group_params("L_Hip")
        all_params = profile.get_all_params()  # {group_name: {param: value}}
    """

    def __init__(self, name: str, description: str, injury_type: str,
                 joint_params: Dict[str, Dict[str, float]],
                 metadata: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.injury_type = injury_type
        self.metadata = metadata or {}

        # 각 그룹에 대해 기본값으로 채우기
        self._joint_params: Dict[str, Dict[str, float]] = {}
        for group in JOINT_GROUPS:
            base = copy.deepcopy(PARAM_DEFAULTS)
            if group in joint_params:
                base.update(joint_params[group])
            self._joint_params[group] = base

    @classmethod
    def load(cls, profile_path: str) -> "PatientProfile":
        """프로파일 YAML 로드.

        Args:
            profile_path: profiles/ 기준 상대 경로 (확장자 생략 가능)
                예: "sci/sci_t10_complete_flaccid"
                    "healthy/healthy_adult"
        """
        if not profile_path.endswith(".yaml"):
            profile_path += ".yaml"

        full_path = os.path.join(PROFILES_DIR, profile_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Patient profile not found: {full_path}")

        with open(full_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            injury_type=data.get("injury_type", "unknown"),
            joint_params=data.get("joint_params", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "PatientProfile":
        """딕셔너리에서 직접 생성 (코드에서 프로그래밍적으로 생성 시)."""
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            injury_type=data.get("injury_type", "custom"),
            joint_params=data.get("joint_params", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def healthy(cls) -> "PatientProfile":
        """정상 성인 프로파일 (기본값) 빠른 생성."""
        return cls(
            name="healthy_adult",
            description="Normal healthy adult",
            injury_type="none",
            joint_params={},  # 모든 그룹 기본값 사용
        )

    def get_group_params(self, group_name: str) -> Dict[str, float]:
        """특정 관절 그룹의 9개 파라미터 반환."""
        if group_name not in self._joint_params:
            raise KeyError(f"Unknown joint group: {group_name}. "
                           f"Available: {JOINT_GROUPS}")
        return copy.deepcopy(self._joint_params[group_name])

    def get_all_params(self) -> Dict[str, Dict[str, float]]:
        """전체 관절 그룹 파라미터 반환. {group_name: {param: value}}"""
        return copy.deepcopy(self._joint_params)

    def get_param_vector(self) -> list:
        """8 groups × 9 params = 72-dim 벡터로 변환 (텐서 변환용).

        순서: [G0_tau, G0_k_p, ..., G0_tr_freq, G1_tau, ..., G7_tr_freq]
        """
        vec = []
        param_keys = list(PARAM_DEFAULTS.keys())
        for group in JOINT_GROUPS:
            for key in param_keys:
                vec.append(self._joint_params[group][key])
        return vec

    def __repr__(self):
        return (f"PatientProfile(name='{self.name}', "
                f"type='{self.injury_type}', "
                f"groups={len(self._joint_params)})")

    def summary(self) -> str:
        """사람이 읽기 쉬운 요약 출력."""
        lines = [f"[{self.name}] {self.description}", f"  Injury type: {self.injury_type}", ""]
        param_keys = list(PARAM_DEFAULTS.keys())
        # 헤더
        header = f"  {'Group':<14}" + "".join(f"{k:>12}" for k in param_keys)
        lines.append(header)
        lines.append("  " + "-" * (14 + 12 * len(param_keys)))
        for group in JOINT_GROUPS:
            vals = self._joint_params[group]
            row = f"  {group:<14}" + "".join(f"{vals[k]:>12.2f}" for k in param_keys)
            lines.append(row)
        return "\n".join(lines)

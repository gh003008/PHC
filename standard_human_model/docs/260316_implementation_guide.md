# Standard Human Model 구현 가이드

작성일: 2026-03-16
대상: standard_human_model/ 모듈의 코드 구조 및 사용법


## 1. 모듈 구조 및 파일별 역할

```
standard_human_model/
├── __init__.py                  → PatientProfile, PatientDynamics를 외부에 export
├── core/
│   ├── __init__.py              → 동일 export
│   ├── patient_profile.py       → YAML 파서, 프로파일 관리, 벡터 변환
│   └── patient_dynamics.py      → GPU 텐서 기반 토크 계산 엔진
├── profiles/                    → 환자별 YAML 프리셋 (데이터만)
│   ├── healthy/healthy_adult.yaml
│   ├── sci/sci_t10_complete_flaccid.yaml
│   ├── sci/sci_incomplete_spastic.yaml
│   ├── stroke/stroke_r_hemiplegia.yaml
│   ├── parkinson/parkinson_moderate.yaml
│   └── cp/cp_spastic_diplegia.yaml
├── docs/                        → 설계 문서
└── examples/
    └── test_patient_dynamics.py → 동작 확인 스크립트
```


## 2. patient_profile.py 상세

### 2.1 상수 정의

JOINT_GROUPS: 8개 관절 그룹 이름 리스트
```python
["L_Hip", "L_Knee", "L_Ankle_Toe", "R_Hip", "R_Knee", "R_Ankle_Toe", "Upper_L", "Upper_R"]
```

PARAM_DEFAULTS: 정상 성인 기준 기본값 딕셔너리
```python
{
    "tau_active": 1.0, "k_passive": 3.0, "b_passive": 1.5,
    "spasticity": 0.0, "spas_dir": 0.0, "rom_scale": 1.0,
    "k_endstop": 0.0, "tremor_amp": 0.0, "tremor_freq": 0.0,
}
```

PROFILES_DIR: profiles/ 폴더의 절대 경로 (자동 계산)

### 2.2 PatientProfile 클래스

생성 방법 3가지:

```python
# 1. YAML에서 로드 (가장 일반적)
profile = PatientProfile.load("sci/sci_t10_complete_flaccid")

# 2. 정상인 빠른 생성
profile = PatientProfile.healthy()

# 3. 딕셔너리에서 직접 생성
profile = PatientProfile.from_dict({
    "name": "custom_patient",
    "injury_type": "stroke",
    "joint_params": {
        "R_Ankle_Toe": {"tau_active": 0.1, "spasticity": 12.0}
        # 명시하지 않은 그룹/파라미터는 기본값 사용
    }
})
```

조회 메서드:

```python
# 특정 그룹의 9개 파라미터
params = profile.get_group_params("R_Ankle_Toe")
# → {"tau_active": 0.15, "k_passive": 8.0, ...}

# 전체 그룹 파라미터
all_params = profile.get_all_params()
# → {"L_Hip": {...}, "L_Knee": {...}, ..., "Upper_R": {...}}

# 72-dim 벡터 (텐서 변환용)
vec = profile.get_param_vector()
# → [G0_tau, G0_kp, ..., G0_tf, G1_tau, ..., G7_tf]

# 사람이 읽기 쉬운 표 출력
print(profile.summary())
```

YAML 파일 필드 설명:

```yaml
name: "프로파일 이름"           # 필수
description: "설명"             # 선택
injury_type: "SCI"              # 선택 (분류용)
metadata:                       # 선택 (임상 정보)
  injury_level: "T10"
  asia_scale: "A"
joint_params:                   # 선택 (없으면 전부 기본값)
  그룹이름:                     # JOINT_GROUPS 중 하나
    파라미터이름: 값            # PARAM_DEFAULTS 키 중 하나
```

누락된 그룹이나 파라미터는 자동으로 기본값(정상)이 채워진다.
이 덕분에 변경이 필요한 그룹만 명시하면 된다.


## 3. patient_dynamics.py 상세

### 3.1 상수 정의

GROUP_DOF_RANGES: 그룹이름 → (start_dof, end_dof) 매핑
```python
{
    "L_Hip": (0, 3), "L_Knee": (3, 6), "L_Ankle_Toe": (6, 12),
    "R_Hip": (12, 15), "R_Knee": (15, 18), "R_Ankle_Toe": (18, 24),
    "Upper_L": (24, 54), "Upper_R": (54, 69),
}
```

NUM_DOFS = 69 (SMPL humanoid 총 자유도)

### 3.2 PatientDynamics 클래스

초기화:
```python
profile = PatientProfile.load("stroke/stroke_r_hemiplegia")
dynamics = PatientDynamics(profile, num_envs=512, device="cuda:0")
```

초기화 시 _build_param_tensors()가 호출되어:
1. 8그룹 파라미터를 69 DOF 1D 텐서로 확장
   - 예: R_Ankle_Toe.spasticity=9.0 → spasticity[18:24] = 9.0
2. unsqueeze(0).expand(num_envs, -1)로 (512, 69) 배치 텐서 생성
3. expand는 메모리를 복사하지 않음 (broadcast view)

생성되는 텐서 목록 (모두 shape: num_envs x 69):
- self.tau_active
- self.k_passive
- self.b_passive
- self.spasticity
- self.spas_dir
- self.rom_scale
- self.k_endstop
- self.tremor_amp
- self.tremor_freq
- self._tremor_phase (랜덤 초기화)

### 3.3 compute_torques() 입력/출력

```python
human_torque = dynamics.compute_torques(
    dof_pos,              # (num_envs, 69) 현재 관절 각도
    dof_vel,              # (num_envs, 69) 현재 관절 각속도
    pd_targets,           # (num_envs, 69) PD 제어 목표
    kp,                   # (69,) 또는 (num_envs, 69) 비례 게인
    kd,                   # (69,) 또는 (num_envs, 69) 미분 게인
    sim_time,             # float, 시뮬레이션 시간 (초)
    q_rest=None,          # (69,) 안정 자세, None이면 0
    joint_limits_lower=None,  # (69,) 관절 하한
    joint_limits_upper=None,  # (69,) 관절 상한
)
# 반환: (num_envs, 69) 인간 토크
```

### 3.4 compute_torques() 내부 5항 계산 순서

항 1: 능동 토크 (tau_vol)
```python
tau_vol = self.tau_active * (kp * (pd_targets - dof_pos) - kd * dof_vel)
```
- 정상 PD 토크에 tau_active를 곱함
- tau_active=0이면 능동 토크 = 0 (완전 마비)
- tau_active=0.15이면 정상의 15% (뇌졸중 마비측 발목)

항 2: 수동 강성 + 감쇠 (tau_passive)
```python
q_rel = dof_pos - q_rest
tau_passive = -self.k_passive * q_rel - self.b_passive * dof_vel
```
- 스프링(k) + 댐퍼(b) 모델
- q_rest에서 벗어나면 복원력, 속도가 있으면 점성 저항
- 이완성 마비: k=0.3, b=0.2 (매우 작음, 흐느적)
- 강직(파킨슨): k=12, b=5 (매우 큼, 뻣뻣)

항 3: 경직 (tau_spasticity)
```python
dir_mask = torch.where(dof_vel > 0, 1.0 + self.spas_dir, 1.0 - self.spas_dir)
tau_spasticity = -self.spasticity * torch.abs(dof_vel) * torch.sign(dof_vel) * dir_mask
```
- 속도의 절대값에 비례 (빠를수록 저항 증가)
- dir_mask로 방향 비대칭:
  - spas_dir=0.7, dof_vel>0: dir_mask=1.7 (신전 방향 저항 1.7배)
  - spas_dir=0.7, dof_vel<0: dir_mask=0.3 (굴곡 방향 저항 0.3배)
- 항상 움직임을 반대하는 방향 (부호: -spas * |dq| * sign(dq))

항 4: 끝범위 비선형 강성 (tau_endstop)
```python
scaled_lower = joint_limits_lower * self.rom_scale
scaled_upper = joint_limits_upper * self.rom_scale
dist_lower = clamp(scaled_lower - dof_pos, min=0)
dist_upper = clamp(dof_pos - scaled_upper, min=0)
tau_endstop = k_endstop * dist_lower**2 - k_endstop * dist_upper**2
```
- ROM 경계를 넘어가면 제곱으로 저항 급증
- joint_limits가 None이면 이 항은 0
- rom_scale=0.7이면 원래 ROM의 70%까지만 허용
- 예: 원래 [-0.5, 0.5] → [-0.35, 0.35]

항 5: 떨림 (tau_tremor)
```python
phase = 2*pi * self.tremor_freq * sim_time + self._tremor_phase
tau_tremor = self.tremor_amp * sin(phase)
```
- tremor_freq=0인 DOF는 자동으로 건너뜀
- 환경마다 _tremor_phase가 다르므로 동기화되지 않음
- 에피소드 리셋 시 reset_tremor_phase()로 위상 재랜덤화

### 3.5 기타 메서드

```python
# ROM_scale 적용된 관절 한계 조회
eff_lower, eff_upper = dynamics.get_effective_rom(joint_lower, joint_upper)

# 에피소드 리셋 시 떨림 위상 초기화
dynamics.reset_tremor_phase(env_ids)  # env_ids: 리셋할 환경 인덱스 텐서

# 현재 프로파일 정보 요약
info = dynamics.info()
# → {"name": "...", "has_tremor": True, "has_spasticity": True, ...}
```


## 4. YAML 프로파일 작성법

### 4.1 최소 프로파일 (변경 부분만)

```yaml
name: "Mild Stroke Left Hemiplegia"
injury_type: "stroke"

joint_params:
  L_Hip:
    tau_active: 0.5
    spasticity: 3.0
  L_Knee:
    tau_active: 0.4
    spasticity: 4.0
  L_Ankle_Toe:
    tau_active: 0.3
    spasticity: 5.0
  Upper_L:
    tau_active: 0.4
    spasticity: 4.0
```

명시하지 않은 그룹(R_Hip, R_Knee 등)은 전부 정상 기본값.
명시한 그룹 내에서도 누락된 파라미터(k_passive 등)는 기본값.

### 4.2 새 환자군 추가 절차

1. profiles/ 아래 적절한 폴더에 YAML 파일 생성
   - 새 질환이면 폴더 새로 생성 (예: profiles/ms/)
2. 임상 특성을 9개 파라미터에 매핑
3. 코드 수정 없이 바로 사용 가능:
   ```python
   profile = PatientProfile.load("ms/ms_moderate")
   ```

### 4.3 파라미터 튜닝 가이드

tau_active 설정:
- MMT(Manual Muscle Test) 등급 기준
  - Grade 5 (정상): 1.0
  - Grade 4 (중력+저항 극복): 0.7
  - Grade 3 (중력 극복): 0.4~0.5
  - Grade 2 (중력 제거 시 움직임): 0.1~0.2
  - Grade 1 (근수축만): 0.05
  - Grade 0 (수축 없음): 0.0

spasticity 설정:
- Modified Ashworth Scale 기준
  - MAS 0: 0
  - MAS 1: 2~4
  - MAS 1+: 4~6
  - MAS 2: 6~8
  - MAS 3: 8~12
  - MAS 4: 12~15+

rom_scale 설정:
- 관절 가동범위 측정값 / 정상 가동범위로 계산
- 예: 정상 발목 배굴 20도, 환자 14도 → 14/20 = 0.7

tremor 설정:
- 파킨슨 안정시 떨림: freq=4~6Hz, amp=1~5 Nm (상지 > 하지)
- 본태성 떨림: freq=8~12Hz, amp=2~8 Nm
- 소뇌성 떨림: freq=3~5Hz


## 5. IsaacGym 환경 통합 (구현 예정)

### 5.1 humanoid_im_vic.py에서의 사용

```python
# __init__에서
from standard_human_model import PatientProfile, PatientDynamics

class HumanoidImVIC(HumanoidIm):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(...)

        # 환자 모델 초기화
        patient_cfg = self.cfg["env"].get("patient_model", {})
        if patient_cfg.get("enabled", False):
            profile = PatientProfile.load(patient_cfg["profile"])
            self.patient_dynamics = PatientDynamics(
                profile, self.num_envs, self.device
            )
        else:
            self.patient_dynamics = None

    def _compute_torques(self, actions):
        # ... (기존 CCF 처리)

        if self.patient_dynamics is not None:
            # 인간 토크
            human_torque = self.patient_dynamics.compute_torques(
                self._dof_pos, self._dof_vel, pd_tar,
                self._kp, self._kd, self.progress_buf[0] * self.dt
            )
            # Exo 토크
            exo_torque = kp_dynamic * (pd_tar - self._dof_pos) - kd_dynamic * self._dof_vel
            # 합산
            torques = human_torque + exo_torque
        else:
            # 기존 방식 (환자 모델 미사용)
            torques = kp_dynamic * (pd_tar - self._dof_pos) - kd_dynamic * self._dof_vel

        return torques

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self.patient_dynamics is not None:
            self.patient_dynamics.reset_tremor_phase(env_ids)
```

### 5.2 env yaml 설정

```yaml
# env_im_walk_vic.yaml에 추가
patient_model:
  enabled: true
  profile: "stroke/stroke_r_hemiplegia"
```


## 6. 테스트 실행

```bash
conda activate phc
cd /home/gunhee/workspace/PHC
python -m standard_human_model.examples.test_patient_dynamics
```

출력 항목:
- 모든 프로파일 로드 성공 여부
- 환자별 관절그룹 토크 비교 (동일 입력, 다른 프로파일)
- 프로파일 요약 표
- healthy() 편의 메서드 검증

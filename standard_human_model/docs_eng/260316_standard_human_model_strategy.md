# 표준화된 인간 모델(Standard Human Model) 설계 전략 및 구현 계획

작성일: 2026-03-16
상태: 초기 설계 완료, 구현 진행 중


## 1. 목표

하나의 인간 모델 프레임워크로 다양한 환자군(SCI, 뇌졸중, 파킨슨, CP 등)의 관절 dynamics를 표현한다.
SMPL이 beta 파라미터 몇 개로 체형을 표현하듯, 관절당 9개 파라미터로 환자 특성을 정의한다.
환자 프로파일 YAML 하나만 선택하면 joint dynamics(임피던스, passive dynamics, 경직, 떨림 등)가 자동 적용된다.

최종 활용: Exoskeleton 제어기 RL 학습 시 "인간 측 dynamics"로 사용.
Exo(VIC/CCF)는 학습 대상, 인간 모델은 환경의 일부로 고정.


## 2. 대상 환자군 및 임상 특성

### 2.1 SCI (척수손상)

완전손상(ASIA A)과 불완전손상(ASIA B-D)으로 나뉜다.

이완성 마비 (Flaccid, LMN 특성)
- 손상 레벨 아래 능동 토크 = 0
- 근긴장 소실: 수동 강성/감쇠 매우 낮음
- 장기 부동 시 구축 발생 (ROM 제한 + 끝범위 강성 증가)
- 대표: T10 완전손상

경직성 마비 (Spastic, UMN 특성)
- 부분적 능동 토크 잔존 (5~15%)
- 속도의존적 경직: 빠르게 움직일수록 저항 급증
- 방향 비대칭: 발목은 배굴 시, 무릎은 신전 시 경직 심함
- 높은 수동 강성, ROM 심하게 제한
- 대표: T8-T12 불완전손상 ASIA C

### 2.2 뇌졸중 (Stroke)

편마비(hemiplegia)가 핵심. 한쪽만 영향.

- 마비측: 근력 저하(tau_active 15~30%) + 경직 동반
- 비마비측: 정상 또는 거의 정상
- 상지 패턴: 굴곡근 경직 (팔꿈치 굽힘, 손목 굽힘)
- 하지 패턴: 신전근 경직 (무릎 펴짐) + 첨족 (발목 배굴 저항)
- Brunnstrom Stage 3 기준 (경직 우세기)

### 2.3 파킨슨병 (Parkinson's Disease)

경직이 아닌 강직(rigidity)이 핵심. 메커니즘이 다르다.

- 강직: 속도 무관, 전 가동범위에서 일정한 저항 (lead-pipe)
  - 구현: k_passive를 높이는 것으로 모델링 (spasticity=0)
- 안정시 떨림 (rest tremor): 4-6Hz, 상지에서 더 심함
- 운동완만 (bradykinesia): tau_active 감소로 근사
- 보행 동결 (freezing of gait): 현재 모델 범위 밖 (별도 메커니즘 필요)

### 2.4 뇌성마비 (Cerebral Palsy)

경직형 양하지마비(spastic diplegia)가 보행 재활에서 가장 흔한 유형.

- 양쪽 하지 대칭적 경직 (상체는 경미)
- 심한 경직 + 구축: ROM 크게 제한
- 가위걸음(scissor gait): 고관절 내전근 경직
- 첨족: 발목 배굴 시 극심한 저항
- GMFCS Level 3 기준 (보조기구 보행)

### 2.5 기타 커버 가능한 환자군

- 근위축: 전신 tau_active 감소, 나머지 정상
- 본태성 떨림: tau_active 정상, tremor_amp/freq만 설정
- 다발성 경화증: 경직 + 근력 저하 조합 (뇌졸중과 유사하게 설정)
- 근긴장이상(dystonia): k_passive 증가로 근사 (유효 강성 증가)


## 3. 파라미터 체계: 관절당 9개

### 3.1 파라미터 정의

| # | 이름 | 범위 | 단위 | 역할 |
|---|---|---|---|---|
| 1 | tau_active | [0, 1] | 비율 | 최대 능동 토크 비율. 0=완전마비, 1=정상 |
| 2 | k_passive | [0, inf) | Nm/rad | 수동 관절 강성. 스프링처럼 위치에 비례하는 저항 |
| 3 | b_passive | [0, inf) | Nm*s/rad | 수동 관절 감쇠. 댐퍼처럼 속도에 비례하는 저항 |
| 4 | spasticity | [0, inf) | Nm*s/rad | 속도의존 경직 계수. 빠를수록 저항 증가 |
| 5 | spas_dir | [-1, 1] | 비율 | 경직 방향 비대칭. 0=대칭, +1=양방향(신전)만, -1=음방향(굴곡)만 |
| 6 | rom_scale | [0, 1] | 비율 | 가동범위 축소 비율. 1=정상, 0.5=절반 |
| 7 | k_endstop | [0, inf) | Nm/rad^2 | ROM 경계 비선형 강성. 구축 시 끝에서 급격한 저항 |
| 8 | tremor_amp | [0, inf) | Nm | 떨림 진폭 |
| 9 | tremor_freq | [0, 12] | Hz | 떨림 주파수. 0이면 비활성 |

총 파라미터 수: 9개 x 8개 관절그룹 = 72개

### 3.2 관절 그룹 정의

SMPL humanoid 69 DOFs를 8개 그룹으로 묶는다 (기존 VIC CCF 그룹과 동일).

| 그룹 | DOF 범위 | DOF 수 | 포함 관절 |
|---|---|---|---|
| G0: L_Hip | 0-2 | 3 | 좌측 고관절 (굴곡/신전, 내전/외전, 회전) |
| G1: L_Knee | 3-5 | 3 | 좌측 무릎 |
| G2: L_Ankle_Toe | 6-11 | 6 | 좌측 발목 + 발가락 |
| G3: R_Hip | 12-14 | 3 | 우측 고관절 |
| G4: R_Knee | 15-17 | 3 | 우측 무릎 |
| G5: R_Ankle_Toe | 18-23 | 6 | 우측 발목 + 발가락 |
| G6: Upper_L | 24-53 | 30 | 좌측 상체 (몸통, 척추, 목, 머리, 좌어깨~손) |
| G7: Upper_R | 54-68 | 15 | 우측 상체 (우어깨~손) |

같은 그룹 내 DOF들은 동일한 파라미터 값을 공유한다.
필요 시 그룹을 더 세분화할 수 있다 (예: Ankle과 Toe 분리).

### 3.3 각 파라미터가 모델링하는 물리 현상

tau_active (능동 토크 비율)
- 모델링 대상: 근력 저하, 마비
- 물리: 환자의 뇌→근육 신경 경로 손상 정도
- 0이면 해당 관절을 자의로 전혀 움직일 수 없음
- 정상 PD 토크에 곱해지므로, 0.3이면 정상의 30% 힘만 발생

k_passive (수동 강성)
- 모델링 대상: 관절 주변 연조직(인대, 관절낭, 비활성 근육)의 탄성
- 물리: 스프링 상수. 위치가 안정 자세에서 벗어날수록 복원력 발생
- 강직(rigidity, 파킨슨)에서 크게 증가
- 이완성 마비에서 크게 감소

b_passive (수동 감쇠)
- 모델링 대상: 연조직의 점성 저항
- 물리: 댐퍼 상수. 속도에 비례하는 저항
- 강직에서 증가, 이완성에서 감소

spasticity (경직 계수)
- 모델링 대상: UMN 병변에 의한 속도의존적 근긴장 증가
- 물리: 근방추 과반사(hyperreflexia)로 인한 속도 비례 저항
- 강직(rigidity)과 다름: 경직은 속도의존적, 강직은 속도 무관
- 빠르게 스트레칭할수록 저항이 급증 (clasp-knife 현상의 기초)

spas_dir (경직 방향 비대칭)
- 모델링 대상: 특정 방향에서만 경직이 심한 임상 패턴
- 물리: 굴곡근/신전근의 경직 정도 차이
- 발목 예시: 배굴(발 위로) 시 족저굴곡근 경직 → spas_dir = +0.7
  - 양방향(+) 속도에 1.7배, 음방향(-) 속도에 0.3배 경직
- 상지 예시: 팔꿈치 신전 시 굴곡근 경직 → spas_dir = -0.3

rom_scale (가동범위 비율)
- 모델링 대상: 구축(contracture)에 의한 ROM 제한
- 물리: 연조직 단축으로 관절이 물리적으로 더 이상 움직이지 못함
- 원래 joint limit에 곱해짐: [-0.5, 0.5] * 0.7 = [-0.35, 0.35]

k_endstop (끝범위 비선형 강성)
- 모델링 대상: ROM 경계에서의 급격한 저항 증가
- 물리: 단축된 연조직이 최대 길이에 도달할 때 비선형적으로 뻣뻣해짐
- 제곱 관계: 경계에서 약간만 벗어나면 약하지만, 많이 벗어나면 급증
- tau = k_endstop * distance_from_boundary^2

tremor_amp, tremor_freq (떨림)
- 모델링 대상: 비자발적 주기적 진동
- 물리: 기저핵/소뇌 병변에 의한 율동적 근수축
- 파킨슨 안정시 떨림: 4-6Hz, 상지 우세
- 본태성 떨림: 8-12Hz
- 환경마다 랜덤 초기 위상으로 다양성 확보


## 4. 토크 계산 수식

매 시뮬레이션 step에서 관절당 인간 토크를 5개 항의 합으로 계산한다.

### 4.1 전체 수식

```
human_torque = tau_vol + tau_passive + tau_spasticity + tau_endstop + tau_tremor
```

각 항의 정의:

```
tau_vol       = tau_active * (kp * (q_target - q) - kd * dq)
tau_passive   = -k_passive * (q - q_rest) - b_passive * dq
tau_spasticity = -spasticity * |dq| * sign(dq) * dir_mask
tau_endstop   = k_endstop * max(0, q_lower - q)^2 - k_endstop * max(0, q - q_upper)^2
tau_tremor    = tremor_amp * sin(2*pi*tremor_freq*t + phi)
```

여기서:
- q: 현재 관절 각도 (dof_pos)
- dq: 현재 관절 각속도 (dof_vel)
- q_target: 능동 제어 목표 (pd_targets)
- q_rest: 안정 자세 (기본 0)
- kp, kd: 시뮬레이터 PD gain (MJCF에서 로드)
- q_lower, q_upper: ROM_scale 적용된 관절 한계
- dir_mask: 경직 방향 비대칭 마스크
  - dq > 0일 때: 1 + spas_dir
  - dq < 0일 때: 1 - spas_dir
- t: 시뮬레이션 시간
- phi: 환경별 랜덤 초기 위상

### 4.2 Exo 토크와의 결합

```
exo_torque  = kp_exo * 2^ccf * (q_target - q) - kd_exo * 2^ccf * dq
total_torque = human_torque + exo_torque
```

human_torque는 환자 프로파일에 의해 고정.
exo_torque는 VIC/CCF policy가 학습.
RL의 목표: human_torque의 한계를 보상하는 exo_torque를 학습하는 것.

### 4.3 수치 예시: 뇌졸중 R_Ankle_Toe

상황: 보행 중 발목 배굴 (발을 위로 올리는 순간)
- dof_pos = 0.0 rad, dof_vel = +2.0 rad/s, pd_target = 0.3 rad
- kp = 200, kd = 20

우측 발목 (마비측: tau_active=0.15, k_passive=8, b_passive=3, spasticity=9, spas_dir=0.7):

```
tau_vol       = 0.15 * (200*(0.3-0) - 20*2.0) = 0.15 * 20 = +3.0 Nm
tau_passive   = -8.0*(0-0) - 3.0*2.0 = -6.0 Nm
tau_spasticity = -9.0 * 2.0 * 1 * (1+0.7) = -30.6 Nm
tau_endstop   = 0 Nm  (ROM 내)
tau_tremor    = 0 Nm  (뇌졸중은 떨림 없음)
합계: +3.0 + (-6.0) + (-30.6) + 0 + 0 = -33.6 Nm ← 배굴 방향 강하게 저항
```

좌측 발목 (정상: tau_active=1.0, k_passive=3, b_passive=1.5, spasticity=0):

```
tau_vol       = 1.0 * (200*(0.3-0) - 20*2.0) = +20.0 Nm
tau_passive   = -3.0*0 - 1.5*2.0 = -3.0 Nm
tau_spasticity = 0 Nm
합계: +20.0 + (-3.0) = +17.0 Nm ← 배굴 방향으로 충분한 힘
```

좌우 차이 50.6 Nm. 이 차이를 Exo가 보상해야 보행이 가능하다.


## 5. 환자군별 파라미터 프리셋 요약

정상인 (baseline):
- 전 관절: tau=1.0, k_p=3.0, b_p=1.5, spas=0, dir=0, ROM=1.0, k_end=0, tr=0

SCI T10 완전 이완성:
- 상체: 정상
- 하체: tau=0, k_p=0.3~0.8, b_p=0.2~0.3, spas=0, ROM=0.8~0.9, k_end=3~8

SCI 불완전 경직성:
- 상체: 정상
- 하체: tau=0.05~0.1, k_p=6~10, b_p=2.5~4, spas=8~12, dir=0.5~0.7, ROM=0.6~0.7, k_end=15~25

뇌졸중 우측 편마비:
- 좌측: 정상
- 우측 하지: tau=0.15~0.3, k_p=5~8, b_p=2~3, spas=5~9, dir=0.4~0.7, ROM=0.7~0.85, k_end=8~15
- 우측 상지: tau=0.2, k_p=6, spas=7, dir=-0.3 (굴곡근 경직)

파킨슨 중등도:
- 전신: tau=0.65~0.7, k_p=8~12, b_p=3.5~5, spas=0(!), ROM=0.8~0.9, tr_amp=1~3, tr_freq=5Hz

CP 경직형 양하지마비:
- 상체: 경미 (tau=0.85, spas=2)
- 하체: tau=0.3~0.45, k_p=8~12, b_p=3.5~5, spas=10~15, dir=0.3~0.8, ROM=0.5~0.65, k_end=18~30


## 6. 구현 아키텍처

### 6.1 폴더 구조

```
standard_human_model/
├── __init__.py                        # PatientProfile, PatientDynamics export
├── core/
│   ├── __init__.py
│   ├── patient_profile.py             # YAML 로더, 프로파일 클래스
│   └── patient_dynamics.py            # 5항 토크 계산 엔진
├── profiles/                          # 환자 파라미터 (데이터만)
│   ├── healthy/
│   │   └── healthy_adult.yaml
│   ├── sci/
│   │   ├── sci_t10_complete_flaccid.yaml
│   │   └── sci_incomplete_spastic.yaml
│   ├── stroke/
│   │   └── stroke_r_hemiplegia.yaml
│   ├── parkinson/
│   │   └── parkinson_moderate.yaml
│   └── cp/
│       └── cp_spastic_diplegia.yaml
├── docs/                              # 설계 문서
│   └── (이 문서)
└── examples/
    └── test_patient_dynamics.py       # 동작 확인 스크립트
```

### 6.2 설계 원칙

데이터와 코드의 분리:
- profiles/ 에는 YAML 파일만 (데이터)
- core/ 에는 Python 코드만 (로직)
- 새 환자군 추가 = YAML 파일 추가 (코드 수정 불필요)

환경단 중심 구현:
- 모델단(MJCF/URDF)에는 ROM, forcelimit, mass 정도만
- 나머지 9개 파라미터 전부 환경단(Python)에서 처리
- 이유: spasticity는 속도의존이라 매 step 계산 필수, MJCF로 표현 불가
- 이유: 좌우 비대칭(편마비)은 환경에서 관절별 적용 필요
- 이유: runtime에 프로파일 변경 가능 (domain randomization)

기존 VIC와의 통합:
- 현재 humanoid_im_vic.py의 _compute_torques()를 확장
- human_torque + exo_torque 구조로 분리
- VIC/CCF는 exo_torque 쪽에서만 작동
- 환자 모델은 human_torque 쪽, 학습 대상이 아닌 환경 파라미터

### 6.3 핵심 클래스

PatientProfile (patient_profile.py):
- YAML 로드: PatientProfile.load("sci/sci_t10_complete_flaccid")
- 빠른 생성: PatientProfile.healthy() (정상 기본값)
- 코드 생성: PatientProfile.from_dict({...})
- 그룹별 조회: profile.get_group_params("R_Ankle_Toe")
- 벡터 변환: profile.get_param_vector() → 72-dim list
- 요약 출력: profile.summary()

PatientDynamics (patient_dynamics.py):
- 초기화: dynamics = PatientDynamics(profile, num_envs=512, device="cuda:0")
  - 8그룹 파라미터를 69 DOF 텐서로 확장
  - (num_envs, 69) 배치 텐서로 GPU 준비
- 매 step: human_torque = dynamics.compute_torques(dof_pos, dof_vel, ...)
  - 5항 합산, 전부 텐서 연산 (GPU 병렬)
- 리셋: dynamics.reset_tremor_phase(env_ids)
- ROM 조회: dynamics.get_effective_rom(lower, upper)

### 6.4 텐서 확장 과정

```
YAML: R_Ankle_Toe.spasticity = 9.0
       ↓
8그룹 dict: {"R_Ankle_Toe": {"spasticity": 9.0, ...}, ...}
       ↓  _build_param_tensors()
69-DOF 텐서: spasticity[18:24] = 9.0  (R_Ankle_Toe는 DOF 18~23)
       ↓  unsqueeze(0).expand(num_envs, -1)
(512, 69) 배치 텐서: 모든 env에서 동일한 환자 파라미터
       ↓  compute_torques()에서 element-wise 연산
(512, 69) 토크 출력
```


## 7. IsaacGym 환경 통합 계획

### 7.1 humanoid_im_vic.py 수정 방향

현재 _compute_torques():
```python
# 단일 PD 제어
torques = kp * 2^ccf * (pd_tar - dof_pos) - kd * 2^ccf * dof_vel
```

수정 후 _compute_torques():
```python
# 인간 토크 (환자 모델)
human_torque = self.patient_dynamics.compute_torques(
    dof_pos, dof_vel, pd_targets, kp, kd, sim_time
)

# Exo 토크 (VIC/CCF 학습 대상)
impedance_scale = 2 ** ccf_full
kp_exo = p_gains * impedance_scale
kd_exo = d_gains * impedance_scale
exo_torque = kp_exo * (pd_tar - dof_pos) - kd_exo * dof_vel

# 합산
torques = human_torque + exo_torque
```

### 7.2 env yaml 설정 추가

```yaml
# env_im_walk_vic.yaml에 추가
patient_model:
  enabled: true
  profile: "stroke/stroke_r_hemiplegia"   # profiles/ 기준 상대 경로
  # 또는 domain randomization 시:
  # profile_pool: ["stroke/stroke_r_hemiplegia", "sci/sci_incomplete_spastic"]
```

### 7.3 초기화 흐름

```
env yaml 로드
    ↓ patient_model.enabled == true
PatientProfile.load(patient_model.profile)
    ↓
PatientDynamics(profile, num_envs, device)
    ↓ self.patient_dynamics에 저장
매 step _compute_torques()에서 호출
```

### 7.4 에피소드 리셋 시

```python
def _reset_envs(self, env_ids):
    # 기존 리셋 로직 ...
    if self.patient_dynamics is not None:
        self.patient_dynamics.reset_tremor_phase(env_ids)  # 떨림 위상 리셋
```


## 8. 모델단(MJCF) 수정이 필요한 경우

대부분 환경단에서 처리하지만, 아래 3가지는 MJCF 수정이 더 적합하다.

ROM 제한 (joint range):
- PhysX가 constraint로 처리해야 물리적으로 정확
- 하지만 rom_scale을 환경단에서 soft하게 처리하는 것도 가능 (k_endstop)
- 우선은 환경단 soft limit으로 진행, 필요 시 MJCF 수정

Actuator force limit:
- tau_active=0인 관절의 forcelimit을 0으로 설정하면 시뮬레이터가 더 효율적
- 하지만 환경단에서 토크를 0으로 곱해도 동일 효과
- MJCF 수정은 선택사항

Segment mass:
- 근위축이 심한 환자는 사지 질량이 감소
- MJCF body mass 수정이 물리적으로 정확
- 현재는 범위 밖, 향후 확장 시 고려


## 9. 확장 계획

### 9.1 단기 (현재)

- 6개 프리셋 프로파일 완성 (완료)
- core 엔진 구현 (완료)
- 단위 테스트 (완료)

### 9.2 중기

- humanoid_im_vic.py에 patient_dynamics 통합
- env yaml에 patient_model 설정 추가
- 정상인 프로파일로 기존 VIC 결과 재현 확인
- 뇌졸중 프로파일로 Exo 학습 실험

### 9.3 장기

- Domain randomization: 환자 파라미터에 노이즈 추가하여 robust한 Exo 학습
- 환자 프로파일 자동 fitting: 임상 데이터(Modified Ashworth Scale, 근력 등급 등)에서 파라미터 자동 추정
- Isaac Lab 마이그레이션 시 동일 프레임워크 적용
- 실제 Exo 하드웨어 sim-to-real에서 환자 모델 활용


## 10. 현재까지 구현되지 않은 현상 (한계점)

다음은 9개 파라미터로 직접 표현되지 않으며, 필요 시 별도 메커니즘이 필요하다:

- Clonus: 경직의 극단적 형태로 5-7Hz 반복 수축-이완. spasticity + tremor 조합으로 근사 가능
- Co-contraction: 주동근-길항근 동시 수축. k_passive 증가로 유효 강성 증가 효과 근사
- 보행 동결 (Freezing of Gait): 파킨슨 특유 현상. 별도 상태 머신 필요
- 피로 (Fatigue): 시간에 따른 tau_active 감소. compute_torques에 time-decay 추가로 확장 가능
- Clasp-knife 현상: 경직 중 갑자기 저항이 사라지는 현상. spasticity에 threshold 추가로 모델링 가능

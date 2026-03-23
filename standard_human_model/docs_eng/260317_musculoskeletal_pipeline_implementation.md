# 근골격 파이프라인 구현 현황

작성일: 2026-03-17
상태: Phase 1 코어 모듈 구현 완료, 검증 실험 진행 예정

---

## 1. 개요

SMPL 휴머노이드(24 body, 23 joint, 69 DOF)에 근육 수준의 동역학을 관절 토크로 변환하는 파이프라인을 구현했다.
기존 IsaacGym의 단순 PD 제어를 대체하여, 수동 역학(passive F-L, damping, ligament), 반사 제어(stretch, GTO, reciprocal inhibition), 이관절근 커플링(bi-articular coupling)을 하나의 compute_torques() 호출로 처리한다.

접근 방식: "Top-down, observation-aligned" — MyoSuite처럼 개별 근육을 시뮬레이션하지 않고, 근육군(muscle group) 단위로 묶어서(lumping) 관절 토크를 생성한다. Surface EMG envelope과 대응 가능한 수준.

---

## 2. 폴더 구조

```
standard_human_model/
├── __init__.py                    # PatientProfile, PatientDynamics, HumanBody export
├── core/                          # 핵심 모듈
│   ├── skeleton.py                # SMPL 골격 상수 (24 body, 23 joint, 69 DOF)
│   ├── muscle_model.py            # Hill-type 근육 힘 생성
│   ├── moment_arm.py              # R(q) 모멘트 암 행렬, 토크 매핑
│   ├── activation_dynamics.py     # 신경→근활성화 ODE
│   ├── reflex_controller.py       # 척수 반사 제어 (stretch, GTO, reciprocal)
│   ├── ligament_model.py          # 인대/관절낭 소프트 리밋
│   ├── human_body.py              # 통합 파이프라인 (메인 진입점)
│   ├── patient_profile.py         # 환자 프로파일 로더 (관절 수준, 9 파라미터)
│   └── patient_dynamics.py        # 환자별 관절 토크 계산 (관절 수준)
├── config/
│   ├── muscle_definitions.yaml    # 20개 근육군 구조 정의
│   └── healthy_baseline.yaml      # 정상 성인 기본 파라미터
├── profiles/                      # 환자군별 관절 수준 프로파일 (9 파라미터)
│   ├── healthy/healthy_adult.yaml
│   ├── sci/sci_t10_complete_flaccid.yaml
│   ├── sci/sci_incomplete_spastic.yaml
│   ├── stroke/stroke_r_hemiplegia.yaml
│   ├── parkinson/parkinson_moderate.yaml
│   └── cp/cp_spastic_diplegia.yaml
├── examples/
│   ├── test_patient_dynamics.py   # 관절 수준 모델 테스트 (6개 환자군)
│   └── test_pipeline.py           # 근골격 파이프라인 테스트 (6개 테스트)
└── docs/                          # 문서
    ├── 0317_project_strategy01.md
    ├── 260316_standard_human_model_strategy.md
    ├── 260316_implementation_guide.md
    ├── 260316_musculoskeletal_joint_level_architecture.md
    └── 260317_musculoskeletal_pipeline_implementation.md  ← 본 문서
```

---

## 3. 토크 생성 파이프라인 (8단계)

HumanBody.compute_torques()가 실행하는 전체 흐름:

```
입력: dof_pos (69), dof_vel (69), descending_cmd (20 muscles), dt
                          │
  ┌───────────────────────┴───────────────────────────┐
  │ 1. Muscle Kinematics (moment_arm.py)              │
  │    l_muscle = l_slack + R(q) @ q                  │
  │    v_muscle = R(q) @ dq                           │
  ├───────────────────────────────────────────────────┤
  │ 2. Current Force (이전 step activation 기반)       │
  │    → reflex 피드백용                               │
  ├───────────────────────────────────────────────────┤
  │ 3. Reflex Layer (reflex_controller.py)            │
  │    a_cmd = descending + stretch + GTO + reciprocal│
  ├───────────────────────────────────────────────────┤
  │ 4. Activation Dynamics (activation_dynamics.py)   │
  │    da/dt = (a_cmd - a) / tau                      │
  │    tau = tau_act (활성화) or tau_deact (비활성화)    │
  ├───────────────────────────────────────────────────┤
  │ 5. Force Generation — Hill Model (muscle_model.py)│
  │    F = a * f_max * fl(l) * fv(v) * cos(pennation) │
  │      + F_passive(l) + F_damping(v)                │
  ├───────────────────────────────────────────────────┤
  │ 6. Torque Mapping (moment_arm.py)                 │
  │    τ_muscle = F @ R   (bi-articular coupling 자동)│
  ├───────────────────────────────────────────────────┤
  │ 7. Ligament Forces (ligament_model.py)            │
  │    τ_lig = -k * (exp(α * excess) - 1) - c * dq   │
  ├───────────────────────────────────────────────────┤
  │ 8. 합산: τ_total = τ_muscle + τ_ligament          │
  └───────────────────────────────────────────────────┘
                          │
출력: tau_total (69)
```

---

## 4. 핵심 모듈 상세

### 4.1 skeleton.py — 골격 상수

SMPL 휴머노이드의 뼈대 정보를 상수로 정의.

- BODY_NAMES: 24개 바디 (Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, ... R_... Torso, Spine, ...)
- JOINT_NAMES: 23개 관절 (각 3 DOF → 69 DOF)
- JOINT_DOF_RANGE: 관절별 DOF 인덱스 범위 (예: L_Hip → [0,3], L_Knee → [3,6])
- JOINT_PD_GAINS: MJCF user attribute에서 가져온 kp/kd 값
- JOINT_LIMITS_DEG: 관절별 각도 제한 (degree)
- 헬퍼 함수: get_kp_kd_tensors(device), get_joint_limits_tensors(device)

### 4.2 muscle_model.py — Hill-type 근육 모델

개별 근육의 힘 생성을 Hill 모델로 계산.

MuscleParams (dataclass):
- f_max: 최대 등척성 힘 (N). 예: soleus 2800N, tibialis_ant 600N
- l_opt: 최적 근육 길이 (정규화, 1.0)
- v_max: 최대 단축 속도 (l_opt/s)
- pennation: 근섬유 우상각 (rad). soleus 0.44 rad (25도)
- tau_act / tau_deact: 활성화/비활성화 시정수 (s)
- k_pe, epsilon_0: 수동 탄성 파라미터
- l_tendon_slack, k_tendon: 건 파라미터
- damping: 점성 감쇠 계수

힘 계산 구성요소:
- force_length_active(l_norm): Gaussian — exp(-((l-1)/0.45)^2)
- force_length_passive(l_norm): 기하급수 — (exp(k_pe*(l-1)/epsilon_0)-1) / (exp(k_pe)-1)
- force_velocity(v_norm): Hill 곡선 — 단축(concentric) vs 신장(eccentric) 비대칭
- compute_force(): F_total = a * f_max * fl * fv * cos(pennation) + f_max * fp + damping * v

### 4.3 moment_arm.py — R(q) 모멘트 암 행렬

근육 힘 → 관절 토크 매핑의 핵심. YAML에서 근육별 moment arm 값을 읽어 (num_muscles × num_dofs) 행렬 구성.

- R 행렬: (20, 69) 크기, 대부분 0 (sparse)
- compute_muscle_length(dof_pos): l = l_slack + R @ q → (num_envs, num_muscles)
- compute_muscle_velocity(dof_pos, dof_vel): v = R @ dq → (num_envs, num_muscles)
- forces_to_torques(F_muscle): τ = F @ R → (num_envs, num_dofs)
  - 이관절근 커플링이 여기서 자동 발생: hamstrings의 F가 hip DOF와 knee DOF 모두에 토크 생성
- get_coupling_info(): 각 근육이 어떤 관절에 작용하는지 딕셔너리 반환

### 4.4 activation_dynamics.py — 신경-근활성화 변환

1차 ODE로 신경 명령(neural command, 0~1)을 근활성화(activation, 0~1)로 변환.

- da/dt = (u - a) / tau
- tau = tau_act (u > a, 활성화 중) 또는 tau_deact (u < a, 비활성화 중)
- 비대칭: 활성화(tau_act=0.015s)가 비활성화(tau_deact=0.060s)보다 빠름
- Soleus는 더 느림: tau_act=0.020s, tau_deact=0.080s (Type I 우세)

### 4.5 reflex_controller.py — 척수 반사 제어

3가지 반사 메커니즘을 결합:

1. Stretch Reflex (신장 반사):
   - 근육이 빠르게 신장될 때 자동 수축 — 경직(spasticity) 모델링의 핵심
   - a_stretch = gain * max(0, |v_stretch| - threshold)
   - gain↑ → 경직 심화, threshold↓ → 민감도 증가

2. GTO Reflex (골지건 반사):
   - 과도한 힘 발생 시 자동 억제 — 건 보호 메커니즘
   - a_gto = -gain * max(0, F/f_max - threshold)

3. Reciprocal Inhibition (상호 억제):
   - 길항근 활성화 시 주동근 억제
   - 길항근 쌍(antagonist_pairs)으로 정의: hip_flexors ↔ gluteus_max, quadriceps ↔ hamstrings 등

반사 지연: 1 step delay buffer (실제 30~50ms 반사 지연 근사)

### 4.6 ligament_model.py — 인대/관절낭

관절 가동 범위(ROM) 근처에서 기하급수적 저항 토크 생성.

- τ = -k_lig * (exp(α * excess) - 1) — excess = 현재각 - soft_limit
- soft_limit = hard_limit * margin (기본 85%)
- 추가로 경계 근처 감쇠: τ_damp = -damping * dq
- 양방향 적용: 상한/하한 모두

### 4.7 human_body.py — 통합 클래스

모든 모듈을 조합하는 메인 진입점.

- HumanBody.from_config(muscle_def_path, param_path, num_envs, device)
  - YAML 2개만으로 전체 모델 생성 (구조 + 파라미터)
  - 환자 변경 = param_path만 교체
- compute_torques(dof_pos, dof_vel, descending_cmd, dt): 8단계 파이프라인 실행
- reset(env_ids): 에피소드 리셋 (activation, reflex buffer, ligament 초기화)
- get_activation(): 현재 근활성화 반환 (관측/로깅용)
- get_muscle_forces(): 현재 근육 힘 반환

---

## 5. 설정 파일

### 5.1 muscle_definitions.yaml — 근육군 구조

20개 근육군 (좌우 각 10개):
- 단관절(mono-articular) 7쌍: hip_flexors, gluteus_max, hip_abductors, hip_adductors, quadriceps, soleus, tibialis_ant
- 이관절(bi-articular) 3쌍: rectus_femoris (고관절↔무릎), hamstrings (고관절↔무릎), gastrocnemius (무릎↔발목)

각 근육 정의 형식:
```yaml
- name: "hamstrings_L"
  type: "bi-articular"
  l_slack: 0.35
  moment_arms:
    L_Hip: [-0.06, 0, 0]    # 고관절 신전
    L_Knee: [0.03, 0, 0]    # 무릎 굴곡
```

길항근 쌍 10개 정의 (좌우 각 5쌍).

### 5.2 healthy_baseline.yaml — 정상 성인 파라미터

근육별 11개 파라미터 + 반사/인대 전역 파라미터.
주요 근육 특성:
- soleus: f_max=2800N (최강), v_max=6.0 (최느림), pennation=0.44 rad (25도) — Type I 근섬유 우세
- quadriceps: f_max=2500N — 두 번째로 강함
- tibialis_ant: f_max=600N — 상대적으로 약함
- hamstrings: f_max=1800N, 이관절

---

## 6. 테스트 결과 요약

test_pipeline.py 실행 결과 (모두 통과):

1. 모델 로드: 20 muscles, 69 DOFs 정상 로드
2. 수동 역학: 능동 명령 0이어도 passive F-L + damping + ligament 토크 발생
3. 능동 수축: hamstrings_L 100% 활성화 → 5 step 후 activation 0.9842
4. Bi-articular Coupling:
   - hamstrings: L_Hip -37.78 Nm (신전) + L_Knee 302.53 Nm (굴곡) 동시 발생
   - gastrocnemius: L_Knee 132.76 Nm (굴곡) + L_Ankle -346.55 Nm (족저굴) 동시 발생
5. Stretch Reflex: 느린 배굴(0.05 rad/s) → 0.0024 Nm vs 빠른 배굴(2.0 rad/s) → 0.0932 Nm (38배 차이)
6. Activation Dynamics: soleus (tau_act=0.020s)가 tibialis_ant (tau_act=0.015s)보다 느리게 활성화 확인

---

## 7. 환자 모델링 방법

환자 변경 = healthy_baseline.yaml의 파라미터만 수정.

예시 — 뇌졸중 우측 편마비:
- 마비측 f_max 40~60% 감소 (근위축)
- stretch_gain 2~3배 증가 (경직)
- stretch_threshold 0.1 → 0.03 (과민)
- tau_act 증가 (느린 활성화)
- 건측은 정상 유지 → 좌우 비대칭

예시 — SCI 완전손상 이완성:
- 손상 수준 아래 f_max → 0 (완전 마비)
- 모든 반사 gain → 0
- 인대 토크만 남음

---

## 8. 다음 단계: 검증 실험

Phase 1 검증 (RL 없이):

1. Pendulum Test
   - 단일 관절(예: 무릎)을 초기 각도에서 놓고 자유 진동 관찰
   - 검증: passive F-L, damping, ligament가 물리적으로 타당한 감쇠 진동 생성하는지
   - 정상인 vs 경직 환자 비교

2. Passive ROM Test
   - 외력으로 관절을 천천히 이동시키며 저항 토크 측정
   - 검증: 관절 각도에 따른 수동 저항 곡선이 생체역학 문헌과 일치하는지
   - 경직 환자의 velocity-dependent 저항 증가 확인

3. Perturbation Response Test
   - 정립 자세에서 순간 외란(impulse) 인가
   - 검증: stretch reflex에 의한 자동 저항 + 반사 지연 확인
   - reciprocal inhibition 동작 확인

4. Co-contraction Test
   - 길항근 쌍 동시 활성화 → 관절 강성 변화 측정
   - 검증: co-contraction이 관절 임피던스를 증가시키는지

5. Bi-articular Coupling Validation
   - 이관절근 단독 활성화 → 두 관절 토크 비율 확인
   - 검증: moment arm 비율과 일치하는지

---

## 9. IsaacGym 통합 계획

현재 파이프라인은 CPU에서 독립 동작. IsaacGym 통합 시:

```python
# pre_physics_step 내부
def pre_physics_step(self, actions):
    # actions → descending_cmd (RL/CPG/EMG replay)
    descending_cmd = actions[:, :self.body.num_muscles]

    torques = self.body.compute_torques(
        self.dof_pos, self.dof_vel,
        descending_cmd, dt=self.dt
    )

    self.gym.set_dof_actuation_force_tensor(
        self.sim, gymtorch.unwrap_tensor(torques)
    )
```

기존 PD 제어를 완전히 대체. CUDA 텐서로 동작하므로 GPU 환경에서도 추가 전송 없음.

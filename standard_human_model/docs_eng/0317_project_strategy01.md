# Isaac Gym 기반 근육 역학 반영 인간 모델 설계 명세서

## 1. 프로젝트 개요

### 목표
Isaac Gym 환경에서 근육 역학이 반영된 표준화된 인간 모델을 구축한다. 이 모델은 다음을 만족해야 한다:
- 근육의 co-contraction에 의한 joint impedance 변화 반영
- bi-articular / multi-articular 근육 특성에 의한 joint coupling 반영
- 근력약화, 뇌졸중(편마비), 파킨슨 등 다양한 환자군에 파라미터 변경만으로 적용 가능
- Real-to-Sim이 가능한 수준의 파라미터 관측가능성 확보
- 인간-로봇 통합 시뮬레이션, VLA, 로봇 제어로의 확장 가능

### 설계 철학
MyoSuite와 같은 full musculoskeletal simulation(bottom-up)이 아닌, **observation-aligned, top-down 접근**을 채택한다.

- 관절 수준에서 관측 가능한 현상(joint impedance, coupling, torque capacity)을 재현
- 파라미터화 방식에 근육 생리학적 구조를 빌려옴
- 모델 파라미터가 임상 측정과 직접 대응되어 Real-to-Sim이 가능

**포지셔닝:**
```
High abstraction  │ Vanilla Isaac Gym (torque-level, 환자 특성 반영 불가)
                  │
                  │ ★ 본 모델 ★ (muscle-inspired joint dynamics, Real-to-Sim 가능)
                  │
Low abstraction   │ MyoSuite (individual muscle level, Real-to-Sim 사실상 불가)
```

---

## 2. 아키텍처 개요

### 계층 구조

```
┌──────────────────────────────────────┐
│  Upper-Level Controller              │  ← RL, CPG, impedance controller,
│  (cortical / supraspinal level)      │     clinical data replay 등 교체 가능
│  output: motor commands (u)          │
└──────────┬───────────────────────────┘
           │ u
           ▼
┌──────────────────────────────────────┐
│  Spinal / Reflex Layer               │  ← 환경단 (pre_physics_step)
│  - stretch reflex                    │
│  - GTO reflex                        │
│  - reciprocal inhibition             │
│  output: a_cmd                       │
├──────────────────────────────────────┤
│  Activation Dynamics                 │  ← 환경단
│  - 1st order ODE: da/dt=(u-a)/τ     │
│  output: activations (a)             │
├──────────────────────────────────────┤
│  Musculotendon Dynamics              │  ← 환경단
│  - Muscle kinematics via R(q)        │
│  - Hill model (active + passive)     │
│  - Torque mapping R(q)ᵀ             │
│  - Ligament / joint capsule forces   │
│  output: joint torques (τ)           │
└──────────┬───────────────────────────┘
           │ set_dof_actuation_force_tensor(τ)
           ▼
┌──────────────────────────────────────┐
│  Isaac Gym Physics Solver            │  ← URDF 로드
│  output: q, dq, contact forces       │──→ feedback to environment
└──────────────────────────────────────┘
```

### 핵심 원칙
- **URDF**: 골격 하드웨어만 담당 (뼈, 관절 축, mass/inertia, hard ROM, collision)
- **환경단**: 모든 soft tissue dynamics 담당 (근육 active/passive, reflex, 인대)
- **상위 제어기와 완전 분리**: 근육 모델 + reflex는 환경의 일부이므로 어떤 컨트롤러든 독립 작동

---

## 3. URDF 구성

### URDF에 포함하는 것
- **Link (뼈)**: mass, inertia tensor (de Leva 1996 등 인체 데이터), collision geometry, visual mesh
- **Joint**: 자유도 타입 (revolute, spherical 등), 축 방향, hard ROM limit (뼈끼리 부딪히는 한계)
- **Foot contact geometry**: 보행 시뮬레이션용 접촉면

### URDF 설정 주의사항
```xml
<joint name="knee" type="revolute">
  <limit lower="0" upper="2.4" effort="300" velocity="15"/>
  <dynamics damping="0.1" friction="0.05"/>  <!-- 수치 안정성용 최소값만! -->
</joint>
```
- `damping`과 `friction`은 **수치 안정성용 최소값**만 설정
- 실제 관절의 점탄성 특성은 전부 환경단에서 처리
- 근육 기인 passive stiffness/damping, 인대 soft limit, 어떤 형태의 joint coupling도 URDF에 넣지 않음

---

## 4. 환경단 구현 상세

### 4.1 파라미터화 레벨 선택

두 가지 레벨이 가능하며, 상황에 따라 선택:

**Level A (EMG 데이터 활용 시):**
```
policy → muscle group activations → Hill model → τ
파라미터: 근육군별 F_max, l_opt, activation dynamics
EMG로 검증/identify 가능
```

**Level B (EMG 없이, 관절 측정만으로):**
```
policy → joint-level commands → joint muscle model → τ
파라미터: 관절별 torque capacity, impedance curve, coupling coefficients
isokinetic dynamometry, passive ROM test, perturbation test로 identify 가능
```

- 기본 모델은 Level B로 구축 (EMG 불필요)
- EMG 데이터가 있으면 Level A로 refinement
- 두 레벨 병행 가능

### 4.2 Moment Arm Matrix R(q) — Joint Coupling의 핵심

R(q)는 (n_muscles × n_joints) 크기의 moment arm matrix:

```python
# 예시: hamstrings (bi-articular)
#              hip_flex  knee_flex  ankle_flex ...
R_hamstrings = [ -0.06,    0.03,     0.0,  ... ]  # meters
```

bi-articular 근육이면 R의 해당 행에 두 관절 모두 nonzero moment arm이 있으므로, `τ = R(q)ᵀ · F`를 통해 자동으로 inter-joint coupling 발생.

**데이터 소스**: OpenSim 모델 (Rajagopal 2016 등)에서 추출 가능. Polynomial fitting이나 lookup table로 R(q)의 관절각 의존성 반영.

**Level B에서의 대안**: R(q)를 명시적으로 쓰는 대신, joint coupling을 직접 파라미터화:
```python
τ_i = τ_active_i(a_i, q_i, dq_i) + τ_passive_i(q_i, dq_i) + Σ_j C_ij(q) · f(q_j, dq_j)
```
여기서 C_ij가 bi-articular muscle이 만드는 inter-joint coupling을 관절 수준에서 직접 표현.

### 4.3 근육 힘 생성 (Hill-type Model)

```python
# Active force
F_active = a * F_max * f_FL(l_m / l_opt) * f_FV(v_m / v_max)

# Passive force (근육이 최적 길이 넘어 늘어나면 exponential 증가)
f_PE(l_norm) = k_PE * max(0, l_norm - 1.0)^2 / epsilon_0
F_passive = F_max * f_PE(l_m / l_opt)

# Muscle damping
F_damping = d * v_m

# Total muscle force
F_total = F_active + F_passive + F_damping
```

- `f_FL`: force-length 관계 (가우시안 커브)
- `f_FV`: force-velocity 관계 (eccentric에서 더 큰 힘)
- 파라미터 참고문헌: Thelen 2003, Millard 2013

### 4.4 근육 길이/속도 계산

```python
l_muscle = l_slack + R(q) @ q_joints    # 근육 길이
v_muscle = R(q) @ dq_joints             # 근육 수축 속도
```

### 4.5 Activation Dynamics

Neural command → muscle activation 사이의 지연:

```python
# u: neural command (0~1), a: muscle activation (0~1)
if u > a:
    da_dt = (u - a) / tau_act    # activation: tau_act ≈ 10~50ms
else:
    da_dt = (u - a) / tau_deact  # deactivation: tau_deact ≈ 40~200ms

a_new = a + da_dt * dt  # Euler integration
```

### 4.6 Joint Torque 계산

```python
# 근육 토크 (여기서 coupling 발생)
tau_muscle = R(q).T @ F_total

# 인대/관절낭 soft limit
tau_ligament = -k_lig * exp(alpha * (q - q_limit))  # ROM 근처에서 exponential 저항

# 최종 토크
tau_total = tau_muscle + tau_ligament
```

### 4.7 Co-contraction → Variable Joint Impedance

agonist/antagonist 분리 시:
```python
tau_net = R_ag * F_ag - R_ant * F_ant           # 순 토크
K_joint = R_ag * dF_ag/dl + R_ant * dF_ant/dl   # 관절 강성
D_joint = R_ag * dF_ag/dv + R_ant * dF_ant/dv   # 관절 댐핑
```
두 근육이 동시 활성화 → net torque 작아도 stiffness 증가 (co-contraction 효과)

### 4.8 Reflex Layer

환경단에 구현, 상위 제어기와 독립:

```python
def compute_reflex(q, dq, a_descending):
    # Muscle spindle: 근육 길이 변화 감지
    l_muscle = compute_muscle_length(q)
    v_muscle = compute_muscle_velocity(q, dq)

    # Stretch reflex: 빠른 신장에 반응
    spindle_signal = k_velocity * ReLU(v_muscle - threshold)

    # Reflex activation (시간 지연 포함)
    a_reflex = delay_buffer.get(spindle_signal, delay_ms=30)

    # 하행 명령 + 반사 = 최종 activation
    a_total = clamp(a_descending + g_reflex * a_reflex, 0, 1)
    return a_total
```

### 4.9 환경단 전체 pre_physics_step 흐름

```python
def pre_physics_step(self, actions):
    u = actions  # motor commands from controller

    # 1. 현재 상태 읽기
    q = self.get_dof_positions()
    dq = self.get_dof_velocities()

    # 2. Reflex layer
    a_cmd = self.compute_reflex(q, dq, u)

    # 3. Activation dynamics
    self.activations += ((a_cmd - self.activations) / self.tau_act_deact) * self.dt

    # 4. Muscle kinematics
    l_m = self.l_slack + self.R_matrix(q) @ q
    v_m = self.R_matrix(q) @ dq

    # 5. Force generation
    F_active = self.activations * self.F_max * self.f_FL(l_m) * self.f_FV(v_m)
    F_passive = self.F_max * self.f_PE(l_m / self.l_opt)
    F_total = F_active + F_passive

    # 6. Torque mapping (coupling 발생 지점)
    tau_muscle = self.R_matrix(q).T @ F_total

    # 7. Ligament forces
    tau_ligament = self.compute_ligament_forces(q)

    # 8. Apply
    tau_total = tau_muscle + tau_ligament
    self.set_dof_actuation_force_tensor(tau_total)
```

---

## 5. 환자군별 파라미터화

동일 모델 구조에서 파라미터만 변경:

| 환자군 | 파라미터 변경 |
|--------|-------------|
| **건강인 (baseline)** | 정상 파라미터 |
| **근력약화** | F_max 감소 (전체 또는 특정 근육군) |
| **편마비 (뇌졸중)** | 환측 F_max↓, 환측 τ_act↑, spasticity → velocity-dependent resistance↑ |
| **경직 (Spasticity)** | g_reflex↑↑, threshold↓, f_FV 커브 변형, passive resistance↑ |
| **파킨슨** | baseline co-contraction↑ (rigidity), reciprocal inhibition↓, tremor noise 추가 |
| **노인** | F_max↓, Type II fiber 감소 → f_FV fast component↓, τ_act↑ |
| **소뇌 손상** | reflex timing 부정확, gain 변동 |

---

## 6. Real-to-Sim 파라미터 규명

### 측정 가능한 항목 → 모델 파라미터 매핑

| 모델 파라미터 | 임상 측정 방법 |
|---|---|
| 근육군별 F_max | Isokinetic dynamometry |
| Joint coupling (R matrix 또는 C_ij) | Multi-joint kinematics + inverse dynamics |
| τ_act, τ_deact | EMG-to-torque delay 측정 (Level A), step response (Level B) |
| Passive stiffness curve | Passive ROM test + torque 측정 |
| Co-contraction level | Agonist/antagonist EMG ratio (Level A) |
| Spasticity 파라미터 | Modified Ashworth Scale, pendulum test |
| Reflex gain (g_reflex) | Perturbation test, H-reflex |

### Level B (EMG 불필요) 파라미터:
```
관절별 파라미터:
  τ_max(q, dq)      ← isokinetic dynamometry로 직접 측정
  K_passive(q)       ← passive ROM test
  K_active(a, q)     ← perturbation test
  coupling_ij(q)     ← multi-joint movement에서 추출
  τ_act, τ_deact     ← step response에서 추출
```

### System Identification 전략
1. Motion capture + force plate → inverse dynamics → joint torque
2. Passive ROM test → passive stiffness curve fitting
3. Isokinetic test → torque capacity function
4. Perturbation test → impedance 파라미터
5. Gradient-based optimization 또는 Bayesian inference로 파라미터 추정 (파라미터 수가 적어 실현 가능)

---

## 7. 상위 제어기 독립성

근육 모델 + reflex는 **환경의 일부**이므로 상위 제어기 종류에 무관:

- **RL**: muscle activation space에서 학습, movement synergy 창발
- **CPG**: central pattern generator 출력을 motor command로 사용
- **Impedance controller**: 임피던스 제어 출력을 motor command로
- **Clinical EMG replay**: 실제 EMG envelope을 muscle group activation으로 직접 replay
- **Trajectory optimization**: 최적 경로 추적

### 이 분리의 실용적 이점:
- 같은 인체 모델 + 다른 컨트롤러 비교
- 같은 컨트롤러 + 다른 환자 파라미터 테스트
- 보조기기(exoskeleton, AFO) co-simulation

---

## 8. 근육군 Lumping 가이드라인

개별 근육 80개+ → 기능적 근육군 ~15-20개로 lumping:

하지 주요 근육군 예시:
- Quadriceps group (rectus femoris 포함 — bi-articular)
- Hamstrings group (bi-articular)
- Gastrocnemius group (bi-articular)
- Soleus
- Tibialis anterior
- Hip flexors (iliopsoas)
- Hip extensors (gluteus maximus)
- Hip abductors (gluteus medius)
- Hip adductors
- Ankle invertors / evertors

Surface EMG envelope과 직접 대응 가능한 수준.

---

## 9. 구현 체크리스트

### Phase 1: 골격 (URDF)
- [ ] 인체 segment mass/inertia 데이터 적용 (de Leva 1996)
- [ ] 관절 자유도 및 hard ROM 설정
- [ ] Joint damping/friction 최소값 설정
- [ ] Foot contact geometry
- [ ] Isaac Gym에서 URDF 로드 확인

### Phase 2: Passive Dynamics
- [ ] Muscle kinematics: R(q) matrix 구축 (OpenSim 데이터 추출)
- [ ] Passive force-length curve (f_PE) 구현
- [ ] Ligament / joint capsule soft limit 구현
- [ ] 관절 간 passive coupling 검증 (SLR test 시뮬레이션 등)

### Phase 3: Active Dynamics
- [ ] Activation dynamics (1st order ODE)
- [ ] Hill model: f_FL, f_FV 구현
- [ ] Active force 생성 및 torque mapping
- [ ] Co-contraction에 의한 impedance 변화 검증

### Phase 4: Reflex Layer
- [ ] Stretch reflex 구현
- [ ] GTO reflex 구현
- [ ] Reciprocal inhibition
- [ ] 시간 지연 buffer

### Phase 5: 환자군 파라미터화
- [ ] 건강인 baseline 파라미터 세팅
- [ ] 각 환자군별 파라미터 프로파일 정의
- [ ] Pathological gait 재현 검증

### Phase 6: Real-to-Sim Pipeline
- [ ] 임상 측정 → 파라미터 추정 파이프라인
- [ ] 검증: 시뮬레이션 kinematics vs 실제 데이터 비교

---

## 10. 참고 문헌 및 데이터 소스

- **인체 세그먼트 데이터**: de Leva, 1996
- **Moment arm 데이터**: OpenSim Rajagopal 2016 model
- **Hill model 파라미터**: Thelen 2003, Millard 2013
- **보행 데이터**: publicly available gait databases
- **Spasticity 모델링**: Modified Ashworth Scale 기반 파라미터화

# MyoSuite / OpenSim 계산 플로우 및 CALM 비교

작성일시: 2026-03-21 (금)

---

## 1. OpenSim 계산 플로우

OpenSim은 근골격계 생체역학의 표준 도구. Simbody 물리 엔진 + Millard 2013 근육 모델.

```
외부 입력
  ├─ q         (N_dof,)     ← 일반화 좌표 [rad/m]
  ├─ qdot      (N_dof,)     ← 일반화 속도
  ├─ excitation (N_muscles,) ← 근육 흥분 신호 [0,1] (CMC/SO에서 계산)
  └─ external_forces          ← GRF 등 외부 하중
         │
         ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                OpenSim::Model.realizeAcceleration()                   ║
║                                                                       ║
║  Step 1: Musculotendon Geometry    (GeometryPath)                     ║
║  Step 2: Activation Dynamics       (FirstOrderActivation)             ║
║  Step 3: Tendon Equilibrium        (Millard2012EquilibriumMuscle)     ║
║  Step 4: Force Generation          (Hill-type + Elastic Tendon)       ║
║  Step 5: Generalized Forces        (MomentArm via GeometryPath)       ║
║  Step 6: Equations of Motion       (Simbody multibody dynamics)       ║
╚═══════════════════════════════════════════════════════════════════════╝
         │
         ▼
  qddot (N_dof,) → 수치 적분 → 다음 스텝 q, qdot
```

### Step별 상세

```
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 1: MUSCULOTENDON GEOMETRY                                       │
│  GeometryPath (wrapping surface 포함)                                 │
│                                                                       │
│  각 근육의 경로 = [Origin → WrapPoint₁ → ViaPoint → WrapPoint₂ → Ins] │
│                                                                       │
│  l_mtu(q) = Σ |P_{i+1} - P_i|     ← 경로점 간 거리 합               │
│  r(q)     = ∂l_mtu / ∂q           ← moment arm (해석적 편미분!)       │
│                                                                       │
│  💡 핵심: moment arm이 관절각도의 연속 함수                             │
│     → wrapping surface (원통/타원/토러스) 위에서 최단 경로 계산         │
│     → 각도 변화에 따라 moment arm 자동 갱신                            │
│                                                                       │
│  출력: l_mtu (N_muscles,), v_mtu (N_muscles,), r(q) (N_muscles, N_dof) │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 2: ACTIVATION DYNAMICS                                          │
│  FirstOrderActivationDynamicModel                                     │
│                                                                       │
│  da/dt = (u - a) / τ(u, a)                                          │
│                                                                       │
│  τ = τ_act × (0.5 + 1.5×a)        (u > a, 근육 활성화)               │
│  τ = τ_deact / (0.5 + 1.5×a)      (u < a, 근육 비활성화)             │
│                                                                       │
│  기본값: τ_act = 10ms, τ_deact = 40ms                                │
│                                                                       │
│  💡 CALM과 차이: τ가 현재 활성화 수준 a에 의존 (비선형)                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ activation (N_muscles,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 3: TENDON EQUILIBRIUM (핵심 차별점!)                             │
│  Millard2012EquilibriumMuscle                                         │
│                                                                       │
│  l_mtu = l_ce × cos(α) + l_tendon                                   │
│                                                                       │
│  근건 평형 조건: F_ce × cos(α) = F_tendon                            │
│                                                                       │
│  F_tendon = f_max × f_T(l_tendon / l_tendon_slack)                   │
│  f_T(ε) = c1 × exp(k_T × ε) - c2     (건 strain ε > 0 일 때)       │
│                                                                       │
│  l_tendon 알려진 값: l_mtu - l_ce × cos(α)                          │
│  미지수: l_ce (근섬유 길이)                                           │
│                                                                       │
│  풀이: Newton-Raphson 반복 (보통 3~5회 수렴)                          │
│    ε_T = (l_mtu - l_ce × cos(α) - l_ts) / l_ts                      │
│    F_tendon(ε_T) == F_ce(a, l_ce, v_ce) 만족하는 l_ce 구함           │
│                                                                       │
│  💡 이 단계가 계산 비용의 ~60% 차지                                    │
│  💡 rigid tendon 옵션도 제공 (DeGrooteFregly2016Muscle)               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ l_ce, v_ce, F_tendon
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 4: FORCE GENERATION                                             │
│  Hill-type (Millard 2012 곡선)                                        │
│                                                                       │
│  l_ce_norm = l_ce / l_opt                                            │
│  v_ce_norm = v_ce / v_max                                            │
│                                                                       │
│  F_active = a × f_max × f_AL(l_ce_norm) × f_FV(v_ce_norm) × cos(α)  │
│  F_passive = f_max × f_PL(l_ce_norm)                                │
│  F_damping = f_max × β × v_ce_norm              (fiber damping)     │
│                                                                       │
│  F_muscle_along_tendon = (F_active + F_passive + F_damping) × cos(α) │
│                                                                       │
│  💡 곡선 차이 (vs CALM):                                              │
│     f_AL: OpenSim = 5차 스플라인 (비대칭, 좌측 완만 / 우측 급락)      │
│           CALM    = Gaussian exp(-((l-1)/σ)²)  (좌우 대칭)            │
│     f_PL: OpenSim = 지수+선형 하이브리드 (l>1.6 에서 급격 증가)       │
│           CALM    = 순수 지수 (유사하나 파라미터 다름)                  │
│     f_FV: OpenSim = De Groote 해석함수 (매끄러움)                     │
│           CALM    = 조각 함수 (단축/신장 분리, 전환점 불연속 가능성)    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ F_muscle (N_muscles,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 5: GENERALIZED FORCES                                          │
│                                                                       │
│  τ_j = Σ_i  r_ij(q) × F_muscle_i                                    │
│                                                                       │
│  r_ij(q) = 관절 j에 대한 근육 i의 moment arm (Step 1에서 계산)        │
│  → 자동으로 bi-articular 커플링 처리                                  │
│                                                                       │
│  + 인대 (CoordinateLimitForce): 관절 제한 토크                        │
│  + 접촉 (SmoothSphereHalfSpaceContactForce): 발-지면 GRF             │
│  + 외부 하중 (ExternalForce): 실험 데이터 기반                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ τ_generalized (N_dof,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 6: EQUATIONS OF MOTION (Simbody)                                │
│                                                                       │
│  M(q) × qddot + C(q, qdot) + G(q) = τ_generalized + J^T × F_ext    │
│                                                                       │
│  M(q):  질량 행렬 (관성, 강체 질량에서 자동 계산)                     │
│  C:     코리올리/원심력                                                │
│  G:     중력항                                                        │
│  J^T:   접촉 자코비안 (GRF → 일반화 좌표 힘)                         │
│                                                                       │
│  Simbody: 다관절체 동역학 자동 계산 (O(n) 알고리즘)                   │
│  💡 CALM/IsaacGym: PhysX가 이 부분 전부 처리 (GPU 병렬)              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. MyoSuite 계산 플로우

MyoSuite (Vittorio+Vikash, 2022): MuJoCo 기반 근골격 RL 환경. 근육 모델은 MuJoCo 내장.

```
외부 입력
  ├─ obs        (N_obs,)     ← 관절 위치/속도/근육 상태 등
  ├─ ctrl       (N_muscles,) ← RL Policy가 출력하는 근육 흥분 [0,1]
  └─ (선택) reference_motion  ← 모방 대상 궤적
         │
         ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                MyoSuite Env.step(ctrl)                                ║
║                                                                       ║
║  Step 1: Activation Dynamics       (MuJoCo 내장 1차 필터)             ║
║  Step 2: Musculotendon Length      (MuJoCo 내장 tendon wrapping)      ║
║  Step 3: Force Generation          (MuJoCo 내장 Hill-type)            ║
║  Step 4: Moment Arm → Torque       (MuJoCo 내장, 경로 기반)           ║
║  Step 5: Forward Dynamics          (MuJoCo 물리 엔진)                 ║
║  Step 6: Reward & Obs              (MyoSuite Python 레이어)           ║
╚═══════════════════════════════════════════════════════════════════════╝
         │
         ▼
  next_obs, reward, done → RL Policy
```

### Step별 상세

```
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 1: ACTIVATION DYNAMICS (MuJoCo 내장)                            │
│                                                                       │
│  MuJoCo XML:                                                         │
│    <general gainprm="1" biasprm="0 -1 0"                            │
│             dyntype="integrator" dynprm="tau_act" .../>              │
│                                                                       │
│  act(t+dt) = act(t) + dt × (ctrl - act) / tau                       │
│  (단순 1차 LPF, 활성화/비활성화 τ 동일)                               │
│                                                                       │
│  💡 OpenSim: τ가 a에 의존하는 비선형 ODE                              │
│  💡 CALM:   활성화/비활성화 τ 분리 (τ_act ≠ τ_deact)                 │
│  💡 MuJoCo: 가장 단순 — 단일 τ, 선형                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ activation (N_muscles,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 2: MUSCULOTENDON GEOMETRY (MuJoCo 내장)                         │
│                                                                       │
│  MuJoCo XML에서 근육 경로 정의:                                       │
│    <spatial>                                                         │
│      <site site="muscle_origin"/>                                    │
│      <geom geom="wrap_cylinder"/>    ← wrapping surface              │
│      <site site="muscle_insertion"/>                                 │
│    </spatial>                                                        │
│                                                                       │
│  MuJoCo 내부:                                                        │
│    l_mtu(q) = 경로점 + wrapping 거리                                 │
│    v_mtu    = dl_mtu/dt                                              │
│    r(q)     = ∂l_mtu/∂q    (해석적)                                  │
│                                                                       │
│  💡 OpenSim과 동일 원리, MuJoCo가 C로 고속 계산                       │
│  💡 CALM: R 상수 행렬 (wrapping 없음, 각도 무관)                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ l_mtu, v_mtu, r(q)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 3: FORCE GENERATION (MuJoCo 내장 Hill-type)                     │
│                                                                       │
│  MuJoCo XML:                                                         │
│    <muscle name="soleus" force="3549" lmin="0.8" lmax="1.6"         │
│           vmax="10" fpmax="1.3" fvmax="1.2" .../>                    │
│                                                                       │
│  Rigid tendon 가정 (기본):                                            │
│    l_ce = l_mtu - l_tendon_slack     (건은 늘어나지 않음)             │
│                                                                       │
│  F_active  = act × force × f_FL(l_norm) × f_FV(v_norm)              │
│  F_passive = force × f_PL(l_norm)                                   │
│  F_total   = F_active + F_passive                                    │
│                                                                       │
│  MuJoCo 곡선 형태:                                                    │
│    f_FL: 구간 선형 보간 (lmin~1.0 상승, 1.0~lmax 하강)               │
│    f_FV: 구간 선형 보간 (단축: 감소, 신장: fvmax까지 증가)            │
│    f_PL: l > 1 에서 (l-1)/(lmax-1) × fpmax                          │
│                                                                       │
│  💡 OpenSim: 매끄러운 스플라인 곡선                                   │
│  💡 MuJoCo: 구간 선형 → 빠르지만 미분 불연속                          │
│  💡 CALM:   Gaussian (f_FL), 조각 함수 (f_FV)                        │
│                                                                       │
│  ★ MuJoCo 2.3+: Elastic tendon 옵션 추가 가능                       │
│    <muscle ... tendon="true" twidth="0.033"/>                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ F_muscle (N_muscles,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 4: MOMENT ARM → TORQUE (MuJoCo 내장)                           │
│                                                                       │
│  τ_j = Σ_i  r_ij(q) × F_muscle_i                                    │
│                                                                       │
│  MuJoCo가 자동 계산: wrapping 기반 moment arm × 근력                  │
│  → tau 배열에 직접 누적                                               │
│                                                                       │
│  💡 코드 레벨에서 사용자가 개입 불가 (블랙박스)                         │
│  💡 CALM: tau = F @ R 로 명시적 계산 → 디버깅 용이                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ τ_generalized (N_dof,)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 5: FORWARD DYNAMICS (MuJoCo)                                    │
│                                                                       │
│  mj_step(model, data)                                                │
│                                                                       │
│  M(q) × qddot = τ_muscle + τ_passive + J^T × F_contact - C - G     │
│                                                                       │
│  MuJoCo 특징:                                                        │
│  - 접촉: soft contact (convex, primitive 기반)                       │
│  - 적분: semi-implicit Euler (기본) / RK4 (선택)                     │
│  - 관절 제한: constraint force 방식 (인대 모델 아님)                  │
│                                                                       │
│  💡 IsaacGym(PhysX): GPU 병렬 수천 환경                               │
│  💡 MuJoCo: CPU 단일 환경 (mujoco-mpc로 일부 병렬)                   │
│  💡 MuJoCo 3.x + MJX: JAX 기반 GPU 가속 지원                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ next_q, next_qdot
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 6: REWARD & OBS (MyoSuite Python 레이어)                        │
│                                                                       │
│  obs = get_obs(data)                                                 │
│    ├─ qpos, qvel        (관절 상태)                                  │
│    ├─ act               (근육 활성화)                                │
│    ├─ muscle_length     (근육 길이)                                  │
│    ├─ muscle_velocity   (근육 속도)                                  │
│    └─ muscle_force      (근육력)                                     │
│                                                                       │
│  reward = task_reward(obs, target)                                   │
│    ├─ tracking_reward   (모션 추적 또는 과제 수행)                    │
│    ├─ effort_penalty    (-w × Σ ctrl²)                               │
│    └─ alive_bonus       (넘어지지 않으면 +1)                          │
│                                                                       │
│  💡 MyoSuite의 핵심 가치: 이 레이어에서 다양한 Task 정의             │
│     - MyoHand (손 조작), MyoLeg (보행), MyoChallenge (경쟁)          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 텐서 흐름 비교 (한눈에)

```
              OpenSim              MyoSuite (MuJoCo)         CALM (ours)
              ─────────            ─────────────────         ───────────
입력       excitation (N_m,)    ctrl (N_m,)               descending_cmd (B,20)
              │                    │                          │
              ▼                    ▼                          ▼
활성화      비선형 ODE             단순 1차 LPF              1차 ODE (τ_act≠τ_deact)
동역학      τ=f(u,a)             τ 고정                     τ 분리
              │                    │                          │
              ▼                    ▼                          ▼
근건        GeometryPath          MuJoCo spatial             R × dof_pos (상수 R)
기하학      wrapping surface      wrapping surface           wrapping 없음
            r(q) = ∂l/∂q          r(q) = ∂l/∂q             r = 상수
              │                    │                          │
              ▼                    ▼                          ▼
건 평형     Elastic Tendon        Rigid (기본)               Rigid (고정)
            Newton-Raphson        Elastic (옵션)             미구현
              │                    │                          │
              ▼                    ▼                          ▼
Hill 곡선   스플라인 (매끄러움)    구간 선형 (빠름)            Gaussian + 조각함수
              │                    │                          │
              ▼                    ▼                          ▼
토크 변환   r(q) × F              r(q) × F                  F @ R (상수)
              │                    │                          │
              ▼                    ▼                          ▼
동역학      Simbody (CPU)         MuJoCo (CPU/GPU*)         PhysX (GPU, 수천 환경)
엔진                              *MJX로 GPU 가능
              │                    │                          │
              ▼                    ▼                          ▼
반사 제어   없음 (별도 구현)       없음 (RL이 대체)           내장 (stretch/GTO/reciprocal)
인대 모델   CoordinateLimitForce  joint constraint           지수함수 soft limit
```

---

## 4. 핵심 차이점 정리

### 4.1 Moment Arm (가장 큰 차이)

| 항목 | OpenSim | MyoSuite | CALM |
|------|---------|----------|------|
| 계산 방식 | wrapping surface + 해석적 편미분 | wrapping surface + 해석적 편미분 | 상수 R 행렬 |
| 각도 의존성 | O (연속 함수) | O (연속 함수) | X (BUG-02) |
| bi-articular | 자동 (경로 기반) | 자동 (경로 기반) | R 비영값으로 수동 지정 |
| 정확도 | 해부학적 정밀 | 해부학적 정밀 | 평지 보행 ±10%, 극단 ±30% 오차 |

### 4.2 Elastic Tendon

| 항목 | OpenSim | MyoSuite | CALM |
|------|---------|----------|------|
| 기본 모드 | Elastic (Millard2012) | Rigid | Rigid |
| 선택지 | Rigid도 가능 (DeGroote) | Elastic 옵션 있음 | 미구현 (BUG-03) |
| 아킬레스건 에너지 | 반영 | 미반영 (기본) | 미반영 |
| 계산 비용 | 높음 (NR 반복) | 낮음 | 낮음 |

### 4.3 Activation Dynamics

| 항목 | OpenSim | MyoSuite | CALM |
|------|---------|----------|------|
| 모델 | 비선형 ODE (τ=f(a)) | 선형 1차 LPF | 선형, τ_act ≠ τ_deact |
| 생리학적 정확도 | 높음 | 낮음 | 중간 |
| 환자 모델링 | τ 직접 조절 | 조절 가능하나 미약 | tau_act/tau_deact per-patient |

### 4.4 Reflex Controller

| 항목 | OpenSim | MyoSuite | CALM |
|------|---------|----------|------|
| 내장 여부 | 없음 | 없음 | 내장 (3종: stretch/GTO/reciprocal) |
| 대안 | 별도 Controller 플러그인 | RL Policy가 학습 | - |
| 병리 모델링 | 별도 구현 필요 | RL이 학습 | stretch_gain으로 경직 직접 모사 |

### 4.5 물리 엔진 & 병렬화

| 항목 | OpenSim | MyoSuite | CALM |
|------|---------|----------|------|
| 물리 엔진 | Simbody (CPU) | MuJoCo (CPU) | PhysX (GPU) |
| GPU 병렬 | 불가 | MJX로 가능 (실험적) | 기본 지원 (수천 환경) |
| RL 학습 속도 | 느림 (보통 CMC/SO 사용) | 중간 | 빠름 |
| 접촉 모델 | 다양 | soft contact | PhysX contact |

### 4.6 Hill 곡선 형태

```
       f_FL 비교                           f_FV 비교
   F                                    F
1.0 ┤    ╱╲                          1.5 ┤              ────── OpenSim
    │   ╱  ╲  OpenSim (비대칭)            │            ╱       (De Groote)
    │  ╱    ╲                             │       ───╱──────── MuJoCo
0.5 ┤ ╱     ╲╲                       1.0 ┤──────╱             (선형)
    │╱   ···· ╲  CALM (Gaussian)          │    ╱  ···········  CALM
    ╱   ·    ·  ╲                         │  ╱  ·              (조각함수)
────┼───┼────┼───┼── l_norm          ─────┼╱──┼───────── v_norm
  0.5   1.0  1.5                        -1  0        +1
```

---

## 5. 근육-동역학 커플링 방식 (가장 근본적인 아키텍처 차이)

세 프레임워크의 가장 근본적인 차이는 **근육 모델과 물리 엔진이 어떻게 결합되는가**이다.

### 5.1 OpenSim — Monolithic Coupled ODE (완전 연립)

OpenSim의 forward dynamics는 근육 상태를 **시스템 상태 변수의 일부**로 포함한다:

```
시스템 상태 벡터:
  y = [q, qdot, a₁...aₙ, l_ce₁...l_ceₙ]
       ─────────  ──────  ──────────────
       다관절체    활성화     근섬유 길이
       상태       상태       상태 (elastic tendon 시)

ODE: dy/dt = f(y) 를 RK4 등으로 한 번에 풀음
```

RK4 적분기가 중간 스텝(k1→k2→k3→k4)을 계산할 때:

```
k1: qdot(t) → v_muscle → f_FV → F_muscle → τ → qddot
k2: qdot(t + dt/2×k1) → v_muscle 갱신 → f_FV 갱신 → F_muscle 갱신 → τ 갱신 → qddot 갱신
k3: qdot(t + dt/2×k2) → ... (반복)
k4: qdot(t + dt×k3)   → ... (반복)

→ 적분 한 스텝 안에서 근육력 ↔ 운동 상태가 4회 상호 갱신
→ 근육이 "속도 변화에 실시간 반응"하면서 수렴
```

핵심: **근육과 동역학이 하나의 ODE 시스템** → 에너지 보존 양호, 수치적으로 정확.

### 5.2 MuJoCo (MyoSuite) — Semi-coupled (부분 연립)

```
MuJoCo mj_step() 내부:

  ① act(t+dt) = act(t) + dt × (ctrl - act) / τ     ← 활성화 먼저 확정
  ② F_muscle = f(act(t+dt), l_mtu(q(t)), v_mtu(t))  ← 현재 상태로 근력 계산
  ③ qddot = M⁻¹(τ_muscle + τ_passive - C - G)       ← semi-implicit Euler
  ④ qdot(t+dt) = qdot(t) + dt × qddot
  ⑤ q(t+dt) = q(t) + dt × qdot(t+dt)               ← 새 qdot 사용 (semi-implicit)

  semi-implicit: ⑤에서 새 qdot을 사용하므로 일부 안정성 확보
  하지만 ②에서 근력 계산 시 qdot(t+dt)은 아직 모름 → 완전 연립은 아님
```

근력 계산은 현재 스텝 상태 기준이나, semi-implicit 적분이 약간의 커플링을 제공.

### 5.3 CALM (ours) — Staggered Decoupled (분리 엇갈림)

```
시간 t에서:

  ┌─ CALM Pipeline (Python/PyTorch) ─────────────────────────────────┐
  │  q(t), qdot(t) ← PhysX에서 읽어옴                                │
  │  tau(t) = compute_torques(q(t), qdot(t), cmd(t))                 │
  │           ※ 이 시점에서 tau 확정, 이후 변하지 않음                  │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ tau(t) = 상수
                                 ▼
  ┌─ PhysX Engine (C++/GPU) ─────────────────────────────────────────┐
  │  substep 1: qddot = M⁻¹(tau(t) + ...) → qdot 갱신               │
  │  substep 2: qddot = M⁻¹(tau(t) + ...) → qdot 갱신   ← tau 그대로│
  │  ...                                                              │
  │  substep N: → 최종 q(t+dt), qdot(t+dt)                           │
  └──────────────────────────────────────────────────────────────────┘

  💡 PhysX substep 도중 qdot이 변해도 → v_muscle 업데이트 안 됨
     → f_FV 업데이트 안 됨 → tau는 시작 시점 값 고정
```

이것은 **explicit staggered coupling** — 근육 계산과 물리 엔진이 번갈아 실행되며, 한쪽이 실행되는 동안 다른 쪽은 고정.

### 5.4 세 방식 비교

```
         OpenSim                MuJoCo                  CALM
         ══════════             ══════════              ══════════
         ┌────────┐            ┌────────┐             ┌─ CALM ──┐  ┌─ PhysX ─┐
         │ muscle │            │ muscle │             │ muscle  │  │         │
         │   ↕    │ 하나의     │   ↓    │ 부분       │    ↓    │  │  tau    │
         │  EoM   │ ODE       │  EoM   │ 연립       │  tau ───┼→│ (상수)  │
         │   ↕    │ 시스템     │   ↑    │            │         │  │   ↓    │
         │ muscle │            │(semi-  │            │ ← q,qd ─┼←│  EoM   │
         └────────┘            │implicit)│            └─────────┘  └────────┘
                               └────────┘
                                                      ← 1 스텝 지연 →

커플링:    완전 (RK4 내부)       부분 (semi-implicit)     없음 (staggered)
에너지:    보존 양호              대체로 양호              인위적 주입/소실 가능
안정성:    dt 크게 가능           dt 보통                  dt 작아야 안정
정밀도:    최고                   중간                     dt 작으면 수렴
```

### 5.5 실질적 영향

**1) 에너지 비보존**

```
CALM 시나리오:
  t=0: qdot=2 → v_muscle=2 → f_FV 계산 → tau = 100N·m
  PhysX substep 중: tau=100 적용 → qdot이 2→4로 변함
  실제 근육이면: qdot=4 시 f_FV 감소 → tau < 100 이어야 함
  → CALM은 tau=100 유지 → 실제보다 더 많은 일(work) → 에너지 인위적 주입
```

**2) 수치 안정성 (높은 강성에서)**

```
f_max이 큰 근육 (soleus 3549N):
  한 스텝에 큰 tau 적용 → qdot 급변 → 다음 스텝에서 반대 방향 tau
  → 진동 가능 (dt가 크면 발산)

OpenSim/MuJoCo: 적분기 내부에서 실시간 보정 → 안정
CALM: 보정 없음 → dt 줄이거나 damping 키워서 대응
```

**3) Elastic tendon 구현 시 더 심각**

```
건 평형: F_ce(l_ce, v_ce) = F_tendon(l_mtu - l_ce×cos(α))
  → l_ce와 F_tendon이 서로 의존 → 연립으로 풀어야 정확
  → CALM 분리 구조에서는 l_mtu가 PhysX 안에서 변하는 걸 반영 못함
  → Elastic tendon 구현 난이도가 OpenSim보다 높음
```

### 5.6 CALM에서 괜찮은 이유 + 완화 전략

**현재 괜찮은 이유:**

| 조건 | 값 | 판단 |
|------|-----|------|
| 시뮬 dt | 1/120s = 8.3ms | 근육 시상수 (15~80ms) 대비 충분히 작음 |
| 주 동작 범위 | 평지 보행 (저속) | 고속 충격 동작이 아님 |
| 목적 | RL 학습 환경 | GPU 수천 환경 병렬 >> 단일 환경 정밀도 |
| 엔진 수정 | PhysX 블랙박스 | 어차피 내부 접근 불가 → 이 구조가 유일한 선택 |

**향후 완화 전략:**

```
1) PhysX substep 수 늘리기 (현재 2 → 4)
   - dt_effective = 8.3ms / 4 = 2.1ms → 커플링 오차 감소
   - 비용: 시뮬 속도 ~2x 느려짐

2) Implicit muscle damping 추가
   - F_damping = -β × v_muscle 항의 β를 키워서
   - explicit 적분의 진동 방지 (수치 감쇠)

3) 예측 보정 (Predictor-Corrector)
   - tau를 계산할 때 qdot(t+dt) 예측값 사용:
     qdot_pred = qdot(t) + dt × qddot(t-dt)   (1차 외삽)
     v_muscle_pred = -(R @ qdot_pred)
     → 1 스텝 지연 보상

4) Isaac Lab 마이그레이션 시
   - PhysX 5의 articulation API로 더 정밀한 제어 가능
   - 하지만 근본 구조 (외부 토크 주입)는 동일
```

---

## 6. CALM 개선 로드맵 (이 비교로부터 도출)

> 5장의 커플링 차이를 감안하여 우선순위 재조정

| 우선순위 | 개선 항목 | 참고 모델 | 난이도 | 영향 |
|---------|-----------|----------|--------|------|
| 1 | l_opt 실측값 적용 | OpenSim Rajagopal 2016 | 쉬움 | BUG-01 해결 |
| 2 | Elastic Tendon (soleus, gastroc) | OpenSim Millard2012 | 어려움 | Push-off 재현 |
| 3 | R(q) polynomial fitting | OpenSim gait2392 export | 보통 | 토크 정확도 30%↑ |
| 4 | f_FL 비대칭 곡선 적용 | OpenSim/De Groote 2016 | 쉬움 | 근력 정확도 소폭 향상 |
| 5 | Activation ODE 비선형화 | OpenSim FirstOrder | 쉬움 | 동적 정확도 향상 |
| 6 | Stretch reflex delay | 자체 (생리학 문헌) | 쉬움 | BUG-04 해결 |

### CALM만의 장점 (유지해야 할 것)

1. **Reflex Controller 내장**: OpenSim/MyoSuite 모두 없음. 병리 환자 모델링에 핵심적.
2. **GPU 대규모 병렬**: PhysX 기반 수천 환경 동시 실행 → RL 학습 속도.
3. **환자 파라미터 YAML**: stretch_gain, f_max scaling 등 직관적 병리 모사.
4. **명시적 텐서 연산**: F @ R 구조로 디버깅/분석 용이 (MyoSuite는 블랙박스).

---

## 7. 관련 문서

| 문서 | 링크 |
|------|------|
| CALM 계산 플로우 | [260320_CALM_계산플로우_시각화.md](260320_CALM_계산플로우_시각화.md) |
| CALM 버그 목록 | [260321_CALM_버그목록.md](260321_CALM_버그목록.md) |
| 검증 체계 | [validation_docs/260320_검증_ver2.md](validation_docs/260320_검증_ver2.md) |
| 참조 데이터 (Winter) | [../../validation/reference_data/winter2009_normative.yaml](../../validation/reference_data/winter2009_normative.yaml) |
| 참조 데이터 (Perry) | [../../validation/reference_data/perry1992_emg_timing.yaml](../../validation/reference_data/perry1992_emg_timing.yaml) |

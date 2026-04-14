# 260410 Motion Plan Layer — 미해결 과제 및 확장 방향

본 문서는 260410_motion_plan_layer_concept_v09.docx의 2-모듈 구조(Nominal Generator G + Residual/Impedance Policy R)를 기반으로, 실제 구현 시 반드시 다뤄야 할 미해결 과제들과 그에 대한 설계 방향을 정리한다.

핵심 요구사항은 다음과 같다.
- Task-following과 recovery 사이의 연속적(smooth) 전환
- 보행 전용이 아닌 범용 동작 정책
- 모든 학습이 실측 데이터에서 출발
- 개인 동작 스타일 보존
- 모션 라이브러리 간 연속 보간(CoM velocity 등 command 기반 제어)

---

## 1. Task-Following과 Recovery 사이의 연속 전환

### 1.1 문제

v09 문서의 구조에서 Residual Policy R은 정상 보행 중의 작은 보정을 담당하도록 설계되어 있다. 그러나 외란이 커질수록 필요한 보정량이 nominal 대비 과도해지며, "정상 추종"과 "낙상 방지"는 질적으로 다른 제어 목표다. 이를 binary mode switching(normal vs recovery)으로 처리하면 전환 경계에서 불연속이 발생한다.

### 1.2 설계 방향: Stability Margin 기반 연속 블렌딩

핵심 아이디어는 이산적인 mode selector 대신, 연속적인 stability margin signal alpha를 도입하여 정책의 행동을 부드럽게 전환하는 것이다.

**Stability margin alpha 정의**

alpha는 0(완전 안정)~1(임계 불안정) 사이의 스칼라로, 다음 물리량들의 가중합으로 산출한다.

(a) CoP(Center of Pressure) 기반 지표
- CoP가 support polygon 내부에 있을 때: alpha 기여 낮음
- CoP가 support polygon 경계에 접근할수록: alpha 기여 증가
- 구체적으로 CoP에서 support polygon 경계까지의 최소 거리를 정규화한 값

(b) CoM-BoS(Base of Support) 관계
- CoM의 수평 투영이 BoS 내부에 있는지
- CoM 속도 벡터가 BoS 밖으로 향하는지 (extrapolated CoM, XCoM)
- XCoM = CoM_pos + CoM_vel / sqrt(g/h) 이 BoS 밖이면 stepping이 필요한 상태

(c) 각운동량(Angular Momentum) 변화율
- 상체 회전 가속도가 급격하면 낙상 전조

**alpha가 정책 출력에 미치는 영향**

alpha를 정책의 observation에 포함시키되, 추가로 출력 합성에도 직접 활용한다.

```
# Nominal과 recovery 범위의 블렌딩
q_ref_range  = (1 - alpha) * nominal_range  + alpha * extended_range
Kp_bias      = (1 - alpha) * 0              + alpha * Kp_emergency_bias
reward_weight = (1 - alpha) * w_tracking     + alpha * w_balance
```

이렇게 하면:
- alpha가 낮을 때 (안정): R은 작은 delta_q만 출력, tracking reward 우세 — 정상 보행 모방에 집중
- alpha가 높을 때 (불안정): R의 출력 범위가 확대되고, balance reward가 우세해지며, 전반적 강성이 올라감 — recovery 행동 유도
- 전환이 alpha 값에 따라 연속적이므로 급격한 mode switching 없음

**학습 시 alpha의 활용**

학습 과정에서 perturbation curriculum을 적용하되, 각 에피소드 내에서 alpha가 자연스럽게 올라가는 상황(외란 인가)과 다시 내려오는 상황(복구 성공)을 반복 경험시킨다. 이를 통해 정책은 "alpha가 올라갈 때 어떻게 반응하고, 복구 후 어떻게 정상으로 돌아오는지"의 연속 스펙트럼을 학습한다.

### 1.3 Ankle/Hip/Stepping 전략의 연속 스펙트럼

인간의 3단계 균형 전략(ankle → hip → stepping)도 alpha와 자연스럽게 대응된다.

- alpha 낮음 (0.0~0.3): ankle strategy — z3(착지 댐핑), z4(좌우 안정화)의 미세 조절로 충분
- alpha 중간 (0.3~0.6): hip strategy — delta_q에서 hip 관련 보정이 커짐, z5(전반 stiffness) 상승
- alpha 높음 (0.6~1.0): stepping strategy — delta_q가 nominal foot placement를 크게 override, 사실상 R이 새로운 발 위치를 결정

이 분류를 명시적으로 코딩하지 않아도, alpha가 observation에 포함되고 perturbation curriculum이 충분하면 정책이 자연스럽게 이 스펙트럼을 학습할 것으로 기대한다. 다만, stepping strategy까지 학습시키려면 충분한 강도의 외란 경험이 필수적이다.

---

## 2. 보행 전용이 아닌 범용 동작 정책

### 2.1 문제

v09 문서는 하지 12 DoF 보행에 한정하여 설계되어 있다. 그러나 최종 목표인 exoskeleton 보조 학습을 위해서는 보행뿐 아니라 앉기-일어서기, 계단, 방향 전환, 정지, 느린 걸음-빠른 걸음 전환 등 다양한 동작을 하나의 정책이 다룰 수 있어야 한다.

### 2.2 설계 방향: Task Command 확장 + Multi-Skill Motion Library

**Task command 확장**

보행 전용의 (v_cmd, yaw_cmd) 대신, 범용 command 체계를 도입한다.

```
c_t = {
    v_cmd:      목표 CoM 선속도 (3D),
    omega_cmd:  목표 yaw 각속도,
    h_cmd:      목표 CoM 높이 (앉기-서기 전환),
    mode_hint:  동작 유형 힌트 (optional, one-hot 또는 embedding)
}
```

mode_hint는 학습 초기에 motion library 검색을 안내하는 역할이며, 학습이 진행되면 정책이 state만으로 적절한 행동을 선택하도록 fade out할 수 있다.

**Motion Library의 구조화**

단일 "보행 clip 모음"이 아니라, command label이 붙은 multi-skill library로 구성한다.

```
Library = {
    (v=1.2, omega=0, h=0.9, mode=walk)    → clip_001, clip_002, ...
    (v=0.5, omega=0, h=0.9, mode=walk)    → clip_010, clip_011, ...
    (v=0,   omega=0, h=0.5, mode=sit)     → clip_020, clip_021, ...
    (v=0,   omega=0, h=0.9, mode=stand)   → clip_030, ...
    (v=1.0, omega=0.3, h=0.9, mode=turn)  → clip_040, ...
}
```

command label은 v09 문서와 동일하게 clip의 실측 kinematics에서 자동 산출한다 (CoM velocity, pelvis height 등). 사람이 수동으로 붙이지 않는다.

**Nominal Generator G의 확장**

G는 현재 command에 가장 부합하는 clip(들)을 library에서 검색하여 target window q_tar(t:t+K)를 입력받고, nominal trajectory를 생성한다. command가 library에 정확히 매칭되지 않는 경우 — 예를 들어 v=0.8인데 library에 v=0.5와 v=1.2만 있는 경우 — G가 두 clip 사이를 보간하는 역할을 한다 (이에 대해서는 제5절에서 상세히 다룸).

### 2.3 동작 전환(Transition)의 처리

보행→정지, 정지→앉기 등 동작 간 전환은 별도의 transition clip을 library에 포함시키는 방법과, G가 자체적으로 보간하는 방법이 있다.

권장 접근: 주요 전환(walk→stand, stand→sit 등)에 대해서는 실측 transition clip을 수집하여 library에 포함시킨다. G는 command가 변할 때 transition clip을 우선 참조하고, 없는 경우 양쪽 clip 사이를 temporal blending으로 보간한다. R은 전환 중의 불안정을 보정하며, alpha가 전환 구간에서 자연스럽게 상승하므로 recovery 여유도 확보된다.

---

## 3. 실측 데이터 기반 학습 파이프라인

### 3.1 원칙

모든 학습의 출발점은 실제 측정 데이터여야 한다. 합성 데이터나 수작업 궤적이 아닌, 실제 사람의 움직임에서 시작하여 물리 시뮬레이션으로 정제한다.

### 3.2 데이터 수집 → 학습 파이프라인

```
[실측 데이터 수집]
    MoCap (optical / IMU) + Force plate + (optional) EMG
    ↓
[전처리]
    SMPL fitting (MoSh++ 등) → joint angle q, dq 추출
    Pelvis trajectory → CoM velocity, yaw rate 산출
    Force plate → contact timing, CoP 산출
    EMG → co-contraction index (optional, impedance proxy)
    ↓
[Command Label 자동 부여]
    각 시간 구간에 c_t = {v, omega, h, mode} 산출
    mode는 contact pattern + CoM height + velocity로 자동 분류
    ↓
[Motion Library 구축]
    command-indexed clip database
    clip 단위: 2~5초 구간, overlap 허용
    메타데이터: subject ID, trial condition, command label
    ↓
[Stage A: Nominal Generator G 학습 — offline supervised]
    입력: (state, command, q_tar window from library)
    출력: nominal q_ref, nominal Kp
    손실: reconstruction + temporal smoothness + (optional) EMG-derived Kp guidance
    ↓
[Stage B: Physics Fine-tune — Isaac Gym RL]
    G freeze → R 학습 (residual + impedance)
    alpha observation 포함, perturbation curriculum 적용
    보상: imitation + command tracking + balance + contact consistency + smoothness
    ↓
[Stage C: Joint Fine-tune]
    G unfreeze (낮은 lr) + R 계속 학습
    전체 시스템이 물리적으로 안정적이면서 데이터 분포를 유지하도록 수렴
```

### 3.3 데이터가 부족한 동작의 처리

실측 데이터가 없는 command 영역(예: 매우 빠른 걸음, 특이한 전환)에서는:
- G의 출력 신뢰도가 낮아짐 → R의 보정 부담 증가 → alpha 상승
- 이 영역에서는 사실상 R이 주도적으로 제어하게 되며, AMP discriminator가 "사람다움" 제약을 유지
- 데이터가 추가 수집되면 library에 편입하여 G를 재학습 가능

---

## 4. 개인 동작 스타일 보존

### 4.1 접근 방식

개인화는 두 수준에서 이루어진다.

**수준 1: Passive/생리학적 파라미터 (측정 기반)**

근력, 관절 가동범위, 수동 강성, 체중/신장, 반사 지연 등은 임상/실험적으로 측정 가능하다. 이 값들은 하위 근골격 모델(또는 현재 구조에서는 PD gain의 nominal 범위)에 직접 주입한다. 학습 대상이 아니다.

**수준 2: Coordination 스타일 (데이터 기반 학습)**

cadence, stride length, arm swing 크기, 체중 이동 패턴, 임피던스 조절 전략 등은 motion library와 학습을 통해 포착한다.

### 4.2 구체적 메커니즘

**(a) Subject-specific motion library**

대상자 S의 MoCap 데이터로만 library를 구성하면, G는 자연스럽게 S의 운동학적 스타일(cadence, stride, timing)을 학습한다. 같은 command(예: v=1.0)에 대해 사람마다 다른 clip이 library에 들어가므로, G의 출력이 달라진다.

**(b) Style discriminator**

S의 데이터로 학습한 discriminator D_S를 RL 단계(Stage B/C)에서 style reward로 사용한다.

```
r_style = log(D_S(s_t, s_{t+1}))
```

이렇게 하면 library에 없는 새로운 command 조합에서도 S의 동작 분포를 벗어나지 않도록 제약한다.

**(c) 임피던스 스타일의 한계와 대안**

v09 문서에서 지적한 것처럼, 임피던스(Kp, Kd)의 ground truth는 직접 측정이 어렵다. 따라서:

- EMG가 있는 경우: co-contraction index를 Kp의 soft supervision으로 사용. "이 phase에서 이 사람은 이 정도로 co-contract한다"는 정보를 G의 Kp 출력 학습에 반영.
- EMG가 없는 경우: 운동학적 모방이 잘 되고 물리적으로 안정적인 Kp를 R이 찾도록 하되, "이것이 실제 그 사람의 임피던스"라고 주장하지 않는다. 다만, 같은 운동학을 재현하기 위한 최적 임피던스는 사람마다 다를 것이므로(체중, 근력 등이 다르므로), 수준 1의 생리학적 파라미터가 개인화된 상태에서 학습하면 임피던스도 간접적으로 개인화된다.

### 4.3 Population Mode vs Personalization Mode

동일한 구조 위에서 두 모드를 운용한다.

- Population mode: 여러 사람의 데이터를 혼합한 library + 범용 discriminator. "평균적 인간"의 동작을 생성. 초기 연구 및 일반 목적에 사용.
- Personalization mode: 특정 사람의 library + D_S. 소량 데이터로 G와 R을 fine-tune. 개인 맞춤 exoskeleton 제어에 사용.

전환은 library 교체 + discriminator 교체 + fine-tune으로 이루어지며, 네트워크 구조 자체는 변경하지 않는다.

---

## 5. 모션 라이브러리 간 연속 보간 (Command-Conditioned Control)

### 5.1 문제

Motion library는 이산적인 clip의 집합이다. 하지만 실제 제어에서는 연속적인 command(예: CoM velocity를 0.5에서 1.2로 서서히 올리기)를 따라야 한다. library에 정확히 해당하는 clip이 없을 때, 인접 clip 사이를 부드럽게 보간해야 한다.

### 5.2 설계 방향: Command-Conditioned Retrieval + G의 보간 능력

**검색(Retrieval)**

현재 command c_t에 대해 library에서 가장 가까운 K개의 clip을 검색한다.

```
clips = top_k_nearest(library, c_t, k=3)
weights = softmax(-distance(c_t, clip.command) / temperature)
q_tar_blended = sum(w_i * q_tar_i for w_i, q_tar_i in zip(weights, clips))
```

이 blended target이 G의 입력으로 들어간다. temperature를 조절하면 보간의 부드러움을 제어할 수 있다.

**G의 역할**

G는 단순히 blended target을 통과시키는 것이 아니라, 물리적으로 일관된 trajectory로 정제한다. 예를 들어:
- 두 clip의 phase가 다를 때: G가 phase alignment을 수행
- 속도가 다른 clip을 blend할 때: G가 dynamically consistent한 궤적을 생성 (CoM dynamics 고려)
- 전환 구간: G가 temporal smoothing을 적용

이를 위해 Stage A에서 G를 학습할 때, 의도적으로 다양한 command 전환 시나리오를 training data에 포함시킨다.

### 5.3 CoM Velocity Trajectory Following

사용자(또는 상위 planner)가 시간에 따른 CoM velocity trajectory를 제공하면:

```
v_cmd(t) = [0.5, 0.5, 0.8, 1.0, 1.2, 1.2, 1.0, 0.5, 0, 0, ...]
```

매 스텝 c_t의 v_cmd가 변하면서 library 검색 결과도 연속적으로 변하고, G의 nominal output도 부드럽게 전환된다. 정책 전체로 보면:

```
시간 →  느린 걸음 → 가속 → 빠른 걸음 → 감속 → 정지
alpha → 낮음      → 약간↑ → 낮음      → 약간↑ → 낮음
G     → 느린 clip → blend  → 빠른 clip → blend  → 정지 clip
R     → 작은 보정 → 전환 보정 → 작은 보정 → 전환 보정 → 자세 유지
```

이 과정 전체가 연속적이며, 명시적인 mode switching 없이 command 변화만으로 동작이 전환된다.

### 5.4 Library에 없는 영역

command가 library의 convex hull 밖(외삽 영역)에 있을 때:
- 가장 가까운 clip의 weight가 지배적이 됨
- G의 출력 신뢰도 저하 → R의 부담 증가
- 학습 시 이런 상황을 경험시키되, 안전 제약(torque limit, joint limit)을 강하게 적용
- 장기적으로는 해당 영역의 데이터를 추가 수집하여 library 확장

---

## 6. 통합 구조도

```
                    ┌──────────────────────────────────┐
                    │        Task Command c_t           │
                    │  (v_cmd, omega_cmd, h_cmd, mode)  │
                    └──────────┬───────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Motion Library     │
                    │ (subject-specific    │
                    │  command-indexed)    │
                    │                      │
                    │  retrieve + blend    │
                    └──────────┬───────────┘
                               │ q_tar(t:t+K) blended
                               ▼
              ┌────────────────────────────────┐
              │    Nominal Generator G          │
              │                                │
              │  입력: state, c_t, q_tar        │
              │  출력: q_ref_nom, Kp_nom        │
              │  (offline supervised 학습)       │
              └────────────────┬───────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  Stability   │  │  Residual /  │  │  Style           │
    │  Margin      │  │  Impedance   │  │  Discriminator   │
    │  alpha       │  │  Policy R    │  │  D_S             │
    │              │  │              │  │  (style reward)  │
    │ CoP, XCoM,   │  │ 입력: state,  │  │                  │
    │ ang.momentum │  │ q_ref_nom,   │  └──────────────────┘
    │              │  │ Kp_nom, alpha│
    └──────┬───────┘  │              │
           │          │ 출력: δq, δKp │
           │          └──────┬───────┘
           │                 │
           ▼                 ▼
    ┌─────────────────────────────────────────┐
    │         Output Synthesis                 │
    │                                         │
    │  q_ref = q_ref_nom + f(alpha) * δq      │
    │  Kp    = Kp_nom    + f(alpha) * δKp     │
    │  Kd    = 2ζ√(J · Kp)                    │
    │                                         │
    │  f(alpha): alpha에 따라 δ의 허용 범위 조절  │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │     Low-level PD Controller              │
    │     torque = Kp*(q_ref-q) - Kd*dq       │
    │     (+ 하위 근골격 모델, 추후 확장)         │
    └─────────────────────────────────────────┘
```

---

## 7. 구현 우선순위

### Phase 1 (현재 VIC 구조 위에서 검증)
- alpha 산출 로직을 humanoid_im_vic.py에 추가 (CoP, XCoM 계산)
- alpha를 observation에 포함시켜 현재 단일 정책이 alpha 변화에 반응하는지 확인
- perturbation curriculum 추가 (random push force)
- 결과: "alpha가 높을 때 CCF가 달라지는가?" 분석

### Phase 2 (2-모듈 분리)
- G 네트워크 구현 + Stage A offline 학습
- 실측 MoCap 데이터 기반 command-indexed library 구축
- Isaac Gym 환경에서 G freeze + R 학습 (Stage B)
- 보행 + 정지 + 속도 전환 범위로 시작

### Phase 3 (범용 동작 + 개인화)
- library에 sit-to-stand, turning, stair 등 추가
- subject-specific library + D_S 구현
- CoM velocity trajectory following 테스트
- EMG 데이터 연동 (가능한 경우)

### Phase 4 (Exoskeleton 통합)
- Isaac Lab (PhysX 5) 마이그레이션
- 인간 agent(Motion Plan Layer) + Exo agent 공동 시뮬레이션
- Exo 제어기가 인간의 (q_ref, Kp, Kd, alpha) 를 관측하며 보조 전략 학습

---

## 8. 미결 사항

1. alpha 산출에 필요한 CoP, support polygon은 Isaac Gym에서 contact force로부터 계산 가능한지 확인 필요. 현재 humanoid_im_vic.py에서 contact boolean은 사용 중이나 CoP 좌표는 미산출 상태.
2. 5-axis impedance latent의 decoder 구조가 보행 이외 동작(앉기 등)에서도 유효한지 검증 필요. 동작 유형에 따라 축의 의미가 달라질 수 있음.
3. Motion library 검색의 실시간 속도. clip 수가 많아지면 nearest neighbor 검색 비용이 증가. Approximate NN 또는 사전 indexing 필요.
4. G의 backbone 선택 (GRU vs temporal conv vs Transformer). 보행 전용이면 GRU로 충분하나 multi-skill로 확장 시 Transformer가 유리할 수 있음.
5. Perturbation curriculum의 구체적 스케줄 (어느 epoch부터, 어떤 강도로, 어떤 방향으로). 현재 VIC의 CCF curriculum 경험을 참고하여 설계.

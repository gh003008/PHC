# Joint-Level Musculoskeletal 모사 아키텍처

작성일: 2026-03-16
목적: Musculoskeletal simulation의 핵심 특성들을 IsaacGym joint-level에서 재현하는 전체 아키텍처 설계


## 1. 문제 정의

### 1.1 왜 필요한가

OpenSim 같은 musculoskeletal simulator는 개별 근육을 모델링하여 아래 현상들을 자연스럽게 재현한다:
- Co-contraction (주동근+길항근 동시 수축 → 관절 강성 증가)
- Bi-articular coupling (이관절근이 인접 관절에 동시 토크 생성)
- Force-Length relationship (근육 길이에 따른 힘 생산 변화)
- Force-Velocity relationship (Hill model, 수축 속도에 따른 힘 변화)
- Activation dynamics (신경 → 근수축 시간 지연)
- Passive muscle mechanics (수동 강성, 경직, 구축)

그러나 musculoskeletal sim은:
- 계산이 무거워 수백 환경 병렬화 어려움
- IsaacGym/Isaac Lab과 통합이 어려움
- RL 학습에 필요한 수만 episode를 돌리기 비현실적

따라서 **joint level에서 이들의 "효과"를 재현**하는 것이 목표다.
근육 하나하나를 시뮬레이션하는 것이 아니라, 근육들이 관절에 미치는 **결과적 토크 특성**을 파라미터화한다.

### 1.2 설계 기준

1. 환자군(SCI, 뇌졸중, 파킨슨, CP)의 임상 특성을 재현할 수 있어야 함
2. Real-to-Sim이 가능해야 함 (임상 측정 장비로 파라미터 측정 가능)
3. GPU 텐서 연산으로 512+ 환경 병렬 실행이 가능해야 함
4. 기존 VIC(Exo 제어기 학습) 프레임워크와 자연스럽게 결합해야 함
5. YAML 프로파일 하나로 환자 특성이 완전히 정의되어야 함


## 2. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL Agent (Policy Network)                     │
│                                                                       │
│  Input: Observation (환자 상태 + 태스크 + 환자 프로파일 임베딩)       │
│  Output: Action (Exo PD targets + CCF + Assistance level)            │
└──────────────────────────┬────────────────────────────────────────────┘
                           │ action
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Environment (IsaacGym Task)                       │
│                                                                       │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐     │
│  │   Exo Controller    │    │     Human Patient Model          │     │
│  │                     │    │                                  │     │
│  │  PD targets → torque│    │  Reference motion → desired q    │     │
│  │  CCF → impedance    │    │  ┌──────────────────────────┐   │     │
│  │  Assist → scaling   │    │  │  Musculoskeletal Pipeline │   │     │
│  │                     │    │  │                          │   │     │
│  │  exo_torque         │    │  │  1. Force-Length scaling  │   │     │
│  │      │              │    │  │  2. Force-Velocity (Hill) │   │     │
│  │      │              │    │  │  3. tau_active scaling    │   │     │
│  │      │              │    │  │  4. Activation dynamics   │   │     │
│  │      │              │    │  │  5. Co-contraction        │   │     │
│  │      │              │    │  │  6. Passive dynamics      │   │     │
│  │      │              │    │  │  7. Spasticity            │   │     │
│  │      │              │    │  │  8. Endstop / ROM         │   │     │
│  │      │              │    │  │  9. Tremor                │   │     │
│  │      │              │    │  │  10. Bi-articular coupling│   │     │
│  │      │              │    │  │          │                │   │     │
│  │      │              │    │  │   human_torque            │   │     │
│  │      │              │    │  └──────────┼───────────────┘   │     │
│  │      │              │    │             │                    │     │
│  └──────┼──────────────┘    └─────────────┼────────────────────┘     │
│         │                                 │                           │
│         └──────────┬──────────────────────┘                           │
│                    ▼                                                   │
│           total_torque = human_torque + exo_torque                    │
│                    │                                                   │
│                    ▼                                                   │
│             IsaacGym Physics (PhysX)                                  │
│                    │                                                   │
│                    ▼                                                   │
│           next_state (dof_pos, dof_vel, body_pos, contacts, ...)     │
│                    │                                                   │
│                    ▼                                                   │
│        Observation + Reward 계산                                      │
└─────────────────────────────────────────────────────────────────────┘
```


## 3. 인간 모델: Musculoskeletal Pipeline 상세

### 3.1 파라미터 체계 (관절당 15개 + 커플링)

기본 9개 (수동 역학 + 병리):

| # | 파라미터 | 범위 | 역할 |
|---|---|---|---|
| 1 | tau_active | [0, 1] | 최대 능동 토크 비율 (마비 정도) |
| 2 | k_passive | [0, inf) Nm/rad | 수동 관절 강성 |
| 3 | b_passive | [0, inf) Nm*s/rad | 수동 관절 감쇠 |
| 4 | spasticity | [0, inf) Nm*s/rad | 속도의존 경직 계수 |
| 5 | spas_dir | [-1, 1] | 경직 방향 비대칭 |
| 6 | rom_scale | [0, 1] | 가동범위 축소 비율 |
| 7 | k_endstop | [0, inf) Nm/rad^2 | ROM 경계 비선형 강성 |
| 8 | tremor_amp | [0, inf) Nm | 떨림 진폭 |
| 9 | tremor_freq | [0, 12] Hz | 떨림 주파수 |

Musculoskeletal 확장 6개:

| # | 파라미터 | 범위 | 역할 |
|---|---|---|---|
| 10 | q_optimal | rad | Force-Length 최적 각도 |
| 11 | fl_sigma | (0, inf) rad | Force-Length 가우시안 폭 |
| 12 | v_max | (0, inf) rad/s | 최대 수축 속도 (Hill F-V) |
| 13 | eccentric_ratio | [1.0, 2.0] | 신장 시 힘 증가 비율 |
| 14 | tau_act | (0, inf) s | 활성화 시정수 |
| 15 | cocontraction | [0, inf) Nm/rad | 동시수축 추가 강성 |

관절간 커플링 (별도 정의):

| 항목 | 설명 |
|---|---|
| source | 원천 관절 그룹 |
| target | 대상 관절 그룹 |
| coefficient | 토크 전달 비율 [-1, 1] |
| muscle | 해당 이관절근 이름 (참조용) |

총 파라미터: 15개 x 8그룹 = 120개 + 커플링 계수 N개

### 3.2 토크 계산 파이프라인 (10단계)

매 시뮬레이션 step마다 아래 순서로 계산한다.
순서가 중요하다: F-L, F-V는 능동 토크 "생산 능력"을 조절하고,
그 결과에 activation dynamics가 적용되고,
마지막에 수동 역학과 병리 토크가 더해진다.

```
입력: dof_pos (q), dof_vel (dq), pd_target, sim_time
      환자 프로파일 파라미터 (15개 x 8그룹)

단계 1: 기본 PD 토크 계산 (능동 의도)
────────────────────────────────────
tau_desired = kp * (pd_target - q) - kd * dq

  의미: 환자의 신경계가 "이 위치로 가고 싶다"는 명령.
        아직 근육 특성이 반영되지 않은 순수한 제어 신호.

단계 2: Force-Length 스케일링
────────────────────────────────────
fl_scale = exp( -(q - q_optimal)^2 / (2 * fl_sigma^2) )

  의미: 현재 관절 각도가 최적 각도에서 벗어날수록 힘 생산 능력 감소.
        가우시안 커브로 근사.

  예시:
    정상인: q_optimal=0, fl_sigma=0.8 → 넓은 범위에서 힘 생산
    CP구축: q_optimal=-0.2, fl_sigma=0.4 → 좁은 범위, 굴곡 위치에서만 힘 생산

  환자 영향:
    - 구축 환자: 근육 단축 → q_optimal이 단축 방향으로 이동
    - 같은 tau_active여도 신전 위치에서 힘이 급격히 감소
    - 이것이 "ROM 내에서도 특정 각도에서 약해지는" 현상을 설명

단계 3: Force-Velocity 스케일링 (Hill model)
────────────────────────────────────
v_norm = dq / v_max  (정규화된 속도)

수축 (concentric: 토크 방향과 속도 방향 같음):
  fv_scale = (1 - v_norm) / (1 + v_norm / 0.25)

  의미: 빠르게 수축할수록 힘 감소.
        v_norm=0 → fv=1.0 (등척성)
        v_norm=0.5 → fv=0.33
        v_norm=1.0 → fv=0 (최대 속도에서 힘 0)

신장 (eccentric: 토크 방향과 속도 방향 반대):
  fv_scale = min(eccentric_ratio, 1.0 + (eccentric_ratio - 1.0) * v_norm * 2)

  의미: 근육이 늘어나면서 힘을 내면 오히려 더 큰 힘 가능.
        최대 eccentric_ratio배 (정상: 1.8배)

  환자 영향:
    - 파킨슨: v_max 감소 → 같은 속도에서도 v_norm이 커져 힘 급감
    - CP: v_max, eccentric_ratio 모두 감소 → 전체적으로 힘 생산 저하

단계 4: 능동 토크 스케일링
────────────────────────────────────
tau_active_raw = tau_active * fl_scale * fv_scale * tau_desired

  의미: 환자가 실제로 근육에서 "생산할 수 있는" 능동 토크.
        3중 스케일링: 마비 정도 x 각도 의존 x 속도 의존

단계 5: Activation Dynamics (시간 지연 필터)
────────────────────────────────────
tau_active_filtered(t) = tau_active_filtered(t-1)
                       + (dt / tau_act) * (tau_active_raw - tau_active_filtered(t-1))

  의미: 신경신호가 근수축으로 이어지기까지의 시간 지연.
        1차 저역통과 필터로 모델링. tau_act가 클수록 느림.

  환자 영향:
    - 정상: tau_act=0.05s (거의 즉시)
    - 파킨슨: tau_act=0.25s (bradykinesia, 현저한 지연)
    - 뇌졸중: tau_act=0.12s (중등도 지연)

  구현 주의: 에피소드 리셋 시 필터 상태도 리셋해야 함.
  이 단계를 거치면 tau_active_filtered는 부드럽게 변화하는 능동 토크가 됨.

단계 6: Co-contraction 토크
────────────────────────────────────
tau_cocontract = -cocontraction * (q - q_rest) - (cocontraction * 0.1) * dq

  의미: 주동근-길항근이 동시에 수축하면 순토크는 작지만 관절이 뻣뻣해짐.
        능동 토크와 무관하게 추가되는 "강성 항".
        스프링(위치 비례) + 약한 댐퍼(속도 비례)로 모델링.

  환자 영향:
    - CP: cocontraction=10 (매우 높음, 관절이 "잠기는" 느낌)
    - 뇌졸중 마비측: cocontraction=8
    - 정상: cocontraction=0

  기존 k_passive와의 차이:
    k_passive = 연조직의 기계적 수동 강성 (항상 존재)
    cocontraction = 신경성 동시수축에 의한 추가 강성 (병리에서 증가)
    두 효과는 물리적으로 합산됨

단계 7: 수동 역학 (강성 + 감쇠)
────────────────────────────────────
tau_passive = -k_passive * (q - q_rest) - b_passive * dq

  의미: 능동 수축과 무관한 조직의 기계적 저항.
        스프링(인대, 관절낭) + 댐퍼(점성).

단계 8: 경직 (Spasticity)
────────────────────────────────────
dir_mask = (dq > 0) ? (1 + spas_dir) : (1 - spas_dir)
tau_spasticity = -spasticity * |dq| * sign(dq) * dir_mask

  의미: UMN 병변에 의한 속도의존적 과반사 저항.
        방향에 따라 비대칭 가능 (spas_dir).

  spasticity와 Hill F-V의 차이:
    Hill F-V = 능동 토크 "생산 능력"의 속도 의존성 (정상인에게도 있음)
    spasticity = 비자발적 "저항"의 속도 의존성 (병리에서만 발생)

단계 9: 끝범위 비선형 강성 + 떨림
────────────────────────────────────
scaled_limits = joint_limits * rom_scale
dist = max(0, |q| - |scaled_limits|)
tau_endstop = -k_endstop * dist^2 * sign(q)

tau_tremor = tremor_amp * sin(2*pi*tremor_freq*t + phi_random)

단계 10: Bi-articular Coupling (관절간 토크 전달)
────────────────────────────────────
for each coupling (source → target, coefficient):
    tau_coupling[target] += coefficient * tau_total[source]

  의미: 이관절근에 의한 인접 관절 토크 전달.

  주요 이관절근과 그 효과:

  햄스트링 (Hamstrings):
    - 관절: 고관절 ↔ 무릎
    - 작용: 고관절 신전 토크 → 무릎 굴곡 토크 유발
    - coefficient: -0.3 (정상), -0.5 (CP 경직)
    - 환자 영향: 경직된 햄스트링 → 고관절 신전 시 무릎이 펴지지 않음 (crouch gait)

  대퇴직근 (Rectus Femoris):
    - 관절: 고관절 ↔ 무릎
    - 작용: 고관절 굴곡 토크 → 무릎 신전 토크 유발
    - coefficient: +0.2 (정상), +0.4 (CP 경직)
    - 환자 영향: 경직 시 swing phase에서 무릎 굴곡 제한 (stiff-knee gait)

  비복근 (Gastrocnemius):
    - 관절: 무릎 ↔ 발목
    - 작용: 무릎 굴곡 토크 → 발목 족저굴 토크 유발
    - coefficient: +0.25 (정상), +0.4 (CP/뇌졸중)
    - 환자 영향: 경직 시 무릎 굴곡할 때 발목도 족저굴 (첨족 악화)

  장딴지근 (Soleus)은 단관절근이므로 coupling 없음 (발목만).

  구현 주의:
    - coupling은 단계 1~9의 합산 후 적용해야 함
    - 양방향 커플링이 아니라 단방향 (근육의 기시-정지 방향)
    - 같은 관절 쌍에 여러 근육이 반대 방향으로 작용 가능
      (햄스트링: hip→knee 음, 대퇴직근: hip→knee 양)

최종 합산:
────────────────────────────────────
human_torque = tau_active_filtered     # 능동 (F-L, F-V, activation 적용됨)
             + tau_cocontract          # 동시수축 강성
             + tau_passive             # 수동 강성/감쇠
             + tau_spasticity          # 경직
             + tau_endstop             # ROM 경계
             + tau_tremor              # 떨림
             + tau_coupling            # 이관절근 커플링
```

### 3.3 각 단계의 상호작용 관계도

```
pd_target ─────────────────────────────────────────┐
                                                    ▼
                                            [1] tau_desired
                                                    │
            q ──→ [2] Force-Length ──→ fl_scale ────┤
                                                    │ (곱)
           dq ──→ [3] Force-Velocity ──→ fv_scale ──┤
                                                    │ (곱)
  tau_active ──→ [4] Active Scaling ──→ tau_active_raw
                                                    │
                                            [5] Activation Filter
                                                    │
                                            tau_active_filtered ──────┐
                                                                       │
            q ──→ [6] Co-contraction ──→ tau_cocontract ──────────────┤
                                                                       │
         q,dq ──→ [7] Passive ──→ tau_passive ────────────────────────┤
                                                                       │ (합)
           dq ──→ [8] Spasticity ──→ tau_spasticity ──────────────────┤
                                                                       │
            q ──→ [9] Endstop + Tremor ──→ tau_endstop + tau_tremor ──┤
                                                                       │
                                                              human_torque (pre-coupling)
                                                                       │
                                                              [10] Bi-articular Coupling
                                                                       │
                                                              human_torque (final)
                                                                       │
                                                                       +
                                                                       │
                                                              exo_torque (from RL)
                                                                       │
                                                                       ▼
                                                              total_torque → PhysX
```


## 4. Exo Controller: Action Space 설계

### 4.1 현재 VIC Action Space

```
action = [pd_targets(69), ccf(8)] = 77 dims
```

pd_targets: 각 DOF의 PD 목표 각도
ccf: 8개 관절 그룹의 임피던스 배율 (2^ccf)

### 4.2 확장 Action Space

환자 모델이 도입되면, Exo 제어기가 학습해야 하는 것이 바뀐다.

```
action = [exo_pd_targets(N), exo_ccf(8), assistance_level(8)]

N = Exo가 구동하는 관절 수 (Exo 설계에 따라 결정)
    예: WalkON F1 = 고관절+무릎+발목 양측 = ~12 DOFs
```

각 요소의 역할:

exo_pd_targets (N dims):
- Exo 액추에이터의 목표 관절 각도
- 현재 VIC의 pd_targets와 동일한 역할이지만, Exo가 구동하는 관절만 해당
- 상체처럼 Exo가 없는 관절은 exo_torque = 0

exo_ccf (8 dims):
- Exo의 관절별 임피던스 조절 (기존 VIC CCF와 동일)
- 2^ccf로 kp, kd 스케일링
- 환자 상태에 따라 CCF를 adaptive하게 조절하는 것을 학습

assistance_level (8 dims):
- 새로 추가. Exo가 환자의 능동 토크를 얼마나 "보조"할지 결정
- assist=0: Exo 자체 제어만 (환자 토크 보조 안 함)
- assist=1: 환자의 부족한 토크를 완전히 보상
- 범위: [0, 1]

```python
# assistance 적용 방식
tau_deficit = (1 - tau_active) * tau_desired  # 환자가 못 내는 만큼의 토크
tau_assist = assistance_level * tau_deficit    # 그 중 얼마를 Exo가 보조할지
exo_torque = exo_pd_torque + tau_assist       # Exo의 자체 PD + 보조
```

assistance_level을 action으로 두는 이유:
- 항상 100% 보조하면 환자의 잔존 근력이 퇴화 (learned non-use)
- 필요한 만큼만 보조하는 "assist-as-needed" 전략을 RL로 학습
- swing phase에서는 보조 높이고, stance에서는 줄이는 등 시간적 조절

### 4.3 Action Space 요약

| 구성 요소 | 차원 | 범위 | 역할 |
|---|---|---|---|
| exo_pd_targets | N (Exo DOFs) | [-1, 1] 정규화 | Exo 관절 목표 각도 |
| exo_ccf | 8 | [-1, 1] | 관절별 임피던스 배율 |
| assistance_level | 8 | [0, 1] | Assist-as-needed 보조 비율 |
| **총** | **N + 16** | | |

WalkON F1 (하지 12 DOFs) 기준: 12 + 16 = **28 dims**
전신 Exo 기준: 69 + 16 = **85 dims**


## 5. Observation Space 설계

### 5.1 관측 구성

Policy가 적절한 action을 내려면, 환자의 현재 상태를 충분히 관측해야 한다.

```
observation = [self_obs, task_obs, phase_obs, patient_obs]
```

self_obs (~166 dims, 기존과 동일):
- dof_pos, dof_vel: 관절 위치/속도
- root_state: 루트 위치/회전/선속도/각속도
- body_rotations: 각 body의 회전

task_obs (288 dims, 기존과 동일):
- 추적 대상 body들의 미래 궤적 (reference motion 기반)

phase_obs (2 dims, 기존과 동일):
- sin(2*pi*gait_phase), cos(2*pi*gait_phase)

patient_obs (새로 추가):
- 이 관측이 핵심. Policy가 "이 환자가 어떤 특성인지" 알아야 적절한 보조 전략을 학습.

### 5.2 patient_obs 설계 방안

방안 A: 파라미터 직접 입력 (120+ dims)

```
patient_obs = [모든 15개 파라미터 x 8그룹 = 120 dims + coupling 계수]
```

장점: 완전한 정보 제공
단점: 차원이 높고, Real-to-Sim 시 모든 파라미터를 정확히 알아야 함

방안 B: 저차원 임베딩 (8~16 dims) ← 권장

```
patient_obs = patient_embedding(profile_vector)
```

120-dim 프로파일 벡터를 작은 MLP로 8~16 dims로 압축.
이 임베딩 네트워크는 policy와 함께 학습.
SMPL의 beta 파라미터처럼 "환자 공간"의 저차원 표현이 됨.

장점: 차원 절약, 일반화 가능
단점: 임베딩 학습이 추가로 필요

방안 C: 효과 기반 관측 (16~32 dims) ← 현실적

환자 파라미터 자체가 아니라, 그 "효과"를 관측으로 제공:

```
patient_obs = [
    last_human_torque_per_group(8),     # 각 그룹의 최근 인간 토크 평균
    tau_active_per_group(8),            # 각 그룹의 능동 토크 비율
    effective_stiffness_per_group(8),   # k_passive + cocontraction + spasticity 효과
    activation_delay_per_group(8),      # tau_act 값
] = 32 dims
```

장점: 실제 물리적 효과를 직접 관측, 차원 적당
단점: 정보 손실 가능

권장: 초기에는 **방안 C (효과 기반)**로 시작하고,
      domain randomization 시 **방안 B (임베딩)**으로 확장.

### 5.3 총 Observation 차원

```
self_obs (166) + task_obs (288) + phase_obs (2) + patient_obs (32)
= 488 dims (기존 456 + 32 추가)
```


## 6. Reward 설계

### 6.1 기존 VIC Reward

```
reward = w_task * imitation_reward + w_disc * amp_discriminator_reward
       + power_penalty + bio_ccf_reward
```

### 6.2 환자 모델 적용 시 Reward 수정

Exo의 목표가 바뀜: "보행 모방"에서 "환자의 보행을 보조"로.

```
reward = w_gait * gait_quality           # 보행 품질 (기존 imitation과 유사)
       + w_assist * assistance_efficiency # 보조 효율성 (새로 추가)
       + w_comfort * patient_comfort      # 환자 편안함 (새로 추가)
       + w_power * power_penalty          # 에너지 효율
       + w_disc * amp_reward              # 자연스러운 보행 (AMP)
```

각 항의 의미:

gait_quality:
- 기존 imitation reward와 유사
- reference motion과의 자세/속도 차이
- 단, 환자 특성상 완벽한 모방은 불가능하므로 가중치/기준 조정 필요

assistance_efficiency (새로 추가):
```python
# Exo가 환자의 잔존 능력을 최대한 활용하도록 유도
# assist-as-needed: 필요 이상으로 보조하면 페널티
tau_patient_contribution = human_torque  # 환자가 실제로 낸 토크
tau_total_needed = total_torque          # 보행에 필요한 총 토크
patient_utilization = |tau_patient_contribution| / (|tau_total_needed| + eps)
r_assist = patient_utilization  # 환자 기여도가 높을수록 좋음
```

patient_comfort (새로 추가):
```python
# 경직이 있는 관절을 급격히 움직이면 불쾌/통증
# Exo 토크 변화율이 낮을수록 편안
r_comfort = -sum(|exo_torque(t) - exo_torque(t-1)|)
# 또는 경직 관절에서의 속도 제한
r_comfort -= sum(spasticity_mask * max(0, |dq| - dq_comfort_threshold))
```


## 7. 환경 클래스 구조

### 7.1 클래스 계층 (수정안)

```
HumanoidIm (기존 base)
    └→ HumanoidImVIC (기존 VIC, CCF 학습)
         └→ HumanoidImExo (새로 추가)
              │
              │ 조합:
              ├── PatientDynamics (인간 모델)
              ├── ExoController (Exo 토크 계산)
              └── RewardCalculator (보행+보조+편안함)
```

### 7.2 HumanoidImExo 핵심 메서드

```python
class HumanoidImExo(HumanoidImVIC):

    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)

        # 환자 모델 초기화
        profile = PatientProfile.load(cfg["env"]["patient_model"]["profile"])
        self.patient_dynamics = PatientDynamics(profile, self.num_envs, self.device)

        # Activation dynamics 필터 상태
        self._tau_active_filtered = torch.zeros(
            self.num_envs, NUM_DOFS, device=self.device
        )

    def _compute_torques(self, actions):
        """핵심: human + exo 토크 합산."""

        # Action 분리
        exo_pd_targets = actions[:, :self.exo_num_dofs]
        exo_ccf = actions[:, self.exo_num_dofs:self.exo_num_dofs+8]
        assist_level = torch.sigmoid(actions[:, -8:])  # [0, 1]

        # 인간 토크 계산 (10단계 파이프라인)
        human_torque = self.patient_dynamics.compute_torques(
            self._dof_pos, self._dof_vel, self._pd_targets_human,
            self._kp, self._kd, self.progress_buf * self.dt,
            q_rest=self._q_rest,
            joint_limits_lower=self._dof_limits_lower,
            joint_limits_upper=self._dof_limits_upper,
        )

        # Exo 토크 계산 (VIC 스타일)
        ccf_expanded = self._expand_ccf_to_dofs(exo_ccf)
        impedance = 2.0 ** ccf_expanded
        exo_pd_torque = (self._kp_exo * impedance * (exo_pd_targets_expanded - self._dof_pos)
                       - self._kd_exo * impedance * self._dof_vel)

        # Assistance 토크
        tau_desired = self._kp * (self._pd_targets_human - self._dof_pos) - self._kd * self._dof_vel
        tau_deficit = (1 - self.patient_dynamics.tau_active) * tau_desired
        assist_expanded = self._expand_assist_to_dofs(assist_level)
        tau_assist = assist_expanded * tau_deficit

        # Exo가 없는 관절은 exo_torque = 0
        exo_torque = (exo_pd_torque + tau_assist) * self._exo_joint_mask

        # 합산
        total_torque = human_torque + exo_torque
        return torch.clamp(total_torque, -self._torque_limits, self._torque_limits)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self.patient_dynamics.reset_tremor_phase(env_ids)
        self._tau_active_filtered[env_ids] = 0  # activation filter 리셋
```

### 7.3 Config YAML 구조

```yaml
# env_im_exo_patient.yaml

env:
  task: HumanoidImExo

  # --- 환자 모델 ---
  patient_model:
    enabled: true
    profile: "stroke/stroke_r_hemiplegia"
    # domain randomization (선택)
    randomize: false
    randomize_params:
      tau_active_noise: 0.1    # 균일 노이즈 범위
      spasticity_noise: 2.0

  # --- Exo 설정 ---
  exo:
    type: "lower_limb"         # lower_limb, full_body
    actuated_groups: ["L_Hip", "L_Knee", "L_Ankle_Toe",
                      "R_Hip", "R_Knee", "R_Ankle_Toe"]
    kp_exo: 100.0
    kd_exo: 10.0

  # --- Action Space ---
  action_space:
    exo_pd_targets: true       # Exo PD 목표
    exo_ccf: true              # Exo 임피던스 조절
    assistance_level: true     # Assist-as-needed

  # --- Reward ---
  reward:
    gait_quality_w: 0.4
    assistance_efficiency_w: 0.3
    patient_comfort_w: 0.2
    power_penalty_w: 0.1
    amp_disc_w: 0.0            # 초기에는 비활성, 안정되면 활성

  # --- 기존 VIC 설정 (하위 호환) ---
  vic_enabled: true
  vic_ccf_num_groups: 8
```


## 8. 학습 전략

### 8.1 커리큘럼 (3단계)

Stage 1: 정상인 + Exo 기본 제어
- 환자 프로파일: healthy_adult
- 목표: Exo가 정상 보행을 방해하지 않는 것을 학습
- human_torque가 충분하므로 Exo는 minimal intervention
- 이 단계에서 기본 보행 dynamics와 Exo 제어 기초를 학습

Stage 2: 경증 환자 + Exo 보조
- 환자 프로파일: 경증 (예: tau_active=0.5~0.7)
- 목표: 부족한 토크를 보조하는 방법 학습
- assistance_level을 적절히 조절하는 것을 학습
- reward에서 assistance_efficiency 비중 높임

Stage 3: 중증 환자 + 전체 보조
- 환자 프로파일: 타겟 환자군 (예: stroke_r_hemiplegia)
- 목표: 경직, 커플링 등 복잡한 dynamics 하에서 보행 보조
- 좌우 비대칭 보조 전략 학습
- patient_comfort 비중 높임

### 8.2 Domain Randomization

학습 중 환자 파라미터에 노이즈를 추가하여 robust한 policy 학습:

```python
# 매 에피소드 리셋 시
tau_active_noisy = tau_active + uniform(-0.1, 0.1)
spasticity_noisy = spasticity + uniform(-2, 2)
# ... 등
```

이렇게 하면 하나의 policy가 다양한 환자에게 적용 가능.
배포 시에는 실제 환자의 파라미터를 측정하여 고정.

### 8.3 Multi-Patient 학습 (장기 목표)

```python
# 512 환경 중 각각 다른 환자 프로파일 할당
profiles = ["sci/...", "stroke/...", "parkinson/...", ...]
for env_id in range(num_envs):
    profile = random.choice(profiles)
    # 환경별로 다른 patient_dynamics 파라미터 설정
```

이때 patient_obs가 중요: policy가 "지금 어떤 환자인지" 알아야
환자에 맞는 보조 전략을 선택할 수 있음.


## 9. 구현 우선순위

### Phase 1 (핵심 인프라)

1. PatientDynamics 확장: 기존 9개 → 15개 파라미터 + bi-articular coupling
2. YAML 프로파일에 musculoskeletal 파라미터 추가
3. compute_torques() 10단계 파이프라인 구현
4. activation dynamics 필터 상태 관리

### Phase 2 (환경 통합)

5. HumanoidImExo 클래스 생성
6. _compute_torques() human + exo 분리
7. action space (pd + ccf + assistance) 구현
8. observation에 patient_obs 추가
9. env yaml 설정 체계

### Phase 3 (학습 검증)

10. 정상인 프로파일로 기존 결과 재현 확인
11. 단일 환자 프로파일로 학습 실험
12. reward 튜닝 (gait + assist + comfort)
13. 커리큘럼 학습 3단계

### Phase 4 (확장)

14. Domain randomization
15. Multi-patient 학습
16. Isaac Lab 마이그레이션
17. Real Exo sim-to-real


## 10. YAML 프로파일 전체 예시 (Musculoskeletal 확장 포함)

```yaml
name: "Stroke Right Hemiplegia - Full Musculoskeletal"
description: "뇌졸중 우측 편마비. Musculoskeletal 특성 포함."
injury_type: "stroke"

metadata:
  affected_side: "right"
  brunnstrom_stage: 3

joint_params:
  # 좌측 (비마비측): 정상
  L_Hip:
    tau_active: 1.0
    k_passive: 3.0
    b_passive: 1.5
    spasticity: 0.0
    spas_dir: 0.0
    rom_scale: 1.0
    k_endstop: 0.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    # musculoskeletal 확장
    q_optimal: 0.0
    fl_sigma: 0.8
    v_max: 8.0
    eccentric_ratio: 1.8
    tau_act: 0.05
    cocontraction: 0.0
  L_Knee:
    tau_active: 1.0
    k_passive: 3.0
    b_passive: 1.5
    spasticity: 0.0
    spas_dir: 0.0
    rom_scale: 1.0
    k_endstop: 0.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: 0.0
    fl_sigma: 0.8
    v_max: 8.0
    eccentric_ratio: 1.8
    tau_act: 0.05
    cocontraction: 0.0
  L_Ankle_Toe:
    tau_active: 1.0
    k_passive: 3.0
    b_passive: 1.5
    spasticity: 0.0
    spas_dir: 0.0
    rom_scale: 1.0
    k_endstop: 0.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: 0.0
    fl_sigma: 0.8
    v_max: 8.0
    eccentric_ratio: 1.8
    tau_act: 0.05
    cocontraction: 0.0

  # 우측 (마비측): 근력 약화 + 경직 + musculoskeletal 변화
  R_Hip:
    tau_active: 0.3
    k_passive: 5.0
    b_passive: 2.0
    spasticity: 5.0
    spas_dir: 0.4
    rom_scale: 0.85
    k_endstop: 8.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: -0.05       # 약간 굴곡 위치에서 최적
    fl_sigma: 0.6          # 정상(0.8)보다 좁음 (근육 단축)
    v_max: 4.0             # 정상(8.0)의 절반 (근력 저하)
    eccentric_ratio: 1.4   # 정상(1.8)보다 낮음
    tau_act: 0.12          # 정상(0.05)보다 느림
    cocontraction: 6.0     # 높은 동시수축
  R_Knee:
    tau_active: 0.25
    k_passive: 6.0
    b_passive: 2.5
    spasticity: 6.0
    spas_dir: 0.5
    rom_scale: 0.8
    k_endstop: 10.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: -0.1        # 굴곡 위치에서 최적 (구축)
    fl_sigma: 0.5
    v_max: 4.0
    eccentric_ratio: 1.4
    tau_act: 0.12
    cocontraction: 8.0     # 무릎 동시수축 심함
  R_Ankle_Toe:
    tau_active: 0.15
    k_passive: 8.0
    b_passive: 3.0
    spasticity: 9.0
    spas_dir: 0.7
    rom_scale: 0.7
    k_endstop: 15.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: -0.15       # 족저굴 위치에서 최적 (첨족 구축)
    fl_sigma: 0.4          # 매우 좁음 (심한 단축)
    v_max: 3.0
    eccentric_ratio: 1.3
    tau_act: 0.15          # 가장 느림
    cocontraction: 10.0    # 동시수축 가장 심함

  Upper_L:
    tau_active: 1.0
    k_passive: 3.0
    b_passive: 1.5
    spasticity: 0.0
    spas_dir: 0.0
    rom_scale: 1.0
    k_endstop: 0.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: 0.0
    fl_sigma: 0.8
    v_max: 8.0
    eccentric_ratio: 1.8
    tau_act: 0.05
    cocontraction: 0.0
  Upper_R:
    tau_active: 0.2
    k_passive: 6.0
    b_passive: 2.5
    spasticity: 7.0
    spas_dir: -0.3
    rom_scale: 0.75
    k_endstop: 10.0
    tremor_amp: 0.0
    tremor_freq: 0.0
    q_optimal: 0.3         # 굴곡 위치 최적 (팔꿈치 굽힘 패턴)
    fl_sigma: 0.5
    v_max: 3.0
    eccentric_ratio: 1.3
    tau_act: 0.15
    cocontraction: 8.0

# Bi-articular coupling (이관절근)
biarticular_coupling:
  # 정상측 (좌측)
  - source: "L_Hip"
    target: "L_Knee"
    coefficient: -0.3
    muscle: "hamstrings_L"
  - source: "L_Hip"
    target: "L_Knee"
    coefficient: 0.2
    muscle: "rectus_femoris_L"
  - source: "L_Knee"
    target: "L_Ankle_Toe"
    coefficient: 0.25
    muscle: "gastrocnemius_L"

  # 마비측 (우측) - 경직으로 인해 커플링 강화
  - source: "R_Hip"
    target: "R_Knee"
    coefficient: -0.5        # 정상(-0.3)보다 강함: 경직된 햄스트링
    muscle: "hamstrings_R"
  - source: "R_Hip"
    target: "R_Knee"
    coefficient: 0.35        # 정상(0.2)보다 강함: 경직된 대퇴직근
    muscle: "rectus_femoris_R"
  - source: "R_Knee"
    target: "R_Ankle_Toe"
    coefficient: 0.4         # 정상(0.25)보다 강함: 경직된 비복근
    muscle: "gastrocnemius_R"
```

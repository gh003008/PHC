# VIC_PHASE_04 결과 분석 (260315)

## 1. 실험 개요

VIC_PHASE_03에서 bio CCF reward의 효과를 확인한 후, 세 가지 개선을 동시에 적용:
1. Bio CCF reward 대상을 ankle → **ankle + knee**로 확대
2. Bio CCF reward weight를 0.05 → **0.15** (3배 강화)
3. Power coefficient를 0.000005 → **0.000002** (에너지 페널티 완화, 종종걸음 해소 목적)
4. 모션을 VIC_PHASE(01)과 동일한 짧은 모션으로 복귀 (4.4s, 4걸음, cycle_motion=False)

## 2. MDP 구성

### 2-1. State (Observation)
| 구성 요소 | Dims | 설명 |
|---|---|---|
| Body positions/rotations | ~166 | dof_pos, vel, root state 등 (self_obs) |
| Target tracking (task_obs) | 288 | 4 bodies x 3 samples x 24 (obs_v=6) |
| Phase obs (sin/cos) | 2 | Gait cycle phase 인코딩 |
| **총 obs dims** | **456** | |

### 2-2. Action
| 구성 요소 | Dims | 설명 |
|---|---|---|
| PD target (joint angles) | 69 | 각 DOF의 목표 관절 각도 |
| CCF (Compliance Control Factor) | 8 | 8 그룹별 임피던스 배율 (2^ccf) |
| **총 action dims** | **77** | |

CCF 그룹 매핑:
- G0: L_Hip (3 DOFs), G1: L_Knee (3), G2: L_Ankle+Toe (6)
- G3: R_Hip (3), G4: R_Knee (3), G5: R_Ankle+Toe (6)
- G6: Upper-L (30), G7: Upper-R (15)

### 2-3. Reward
| Term | 수식/설명 | 가중치 | PHASE_03 대비 변경 |
|---|---|---|---|
| Point goal | distance 기반 위치 보상 | 1.0 | - |
| Imitation | pos/rot/vel/ang_vel 매칭 (k_pos=200, k_rot=10, k_vel=1.0, k_ang_vel=0.1) | 0.5 | - |
| AMP discriminator | 적대적 모션 자연스러움 판별 | curriculum | - |
| Power penalty | -power_coeff * sum(abs(torque * dof_vel)) | 1.0 | **0.000005→0.000002** |
| Bio CCF | stance: clamp(ccf, min=0), swing: clamp(-ccf, min=0) | **0.15** | **0.05→0.15, ankle+knee** |

Reward curriculum: epoch < 10000 → task:0.7/disc:0.3, epoch ≥ 10000 → task:0.3/disc:0.7

### 2-4. Termination
| 조건 | 값 | 설명 |
|---|---|---|
| enableEarlyTermination | True | 넘어지면 종료 |
| terminationHeight | 0.15m | 루트 높이 기준 |
| terminationDistance | 0.25m | reference 대비 위치 오차 |
| episode_length | 300 steps (10s) | 최대 에피소드 길이 |
| 모션 종료 | ~131 steps (4.4s) | cycle_motion=False → 모션 끝나면 종료 |

## 3. 학습 설정

| 항목 | 값 |
|---|---|
| 실험명 | VIC_PHASE_04 |
| 기반 | VIC_PHASE_03 + knee bio CCF + weight 3x + power 0.4x |
| motion_file | amass_isaac_walking_forward_single.pkl (4.4s, 4걸음) |
| cycle_motion | False |
| vic_curriculum_stage | 2 |
| vic_ccf_num_groups | 8 |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_phase_obs | True (+2 dims) |
| vic_bio_ccf_reward_w | 0.15 |
| power_coefficient | 0.000002 |
| 학습 epochs | ~12,500 (자동 종료) |
| num_envs | 512 |

## 4. 학습 결과

### Headless 평가 결과 (greedy policy, epoch 12500)
| 메트릭 | 값 |
|---|---|
| 에피소드 reward | 545.17 (일정) |
| 평균 reward (av reward) | 545.17 |
| 평균 steps (av steps) | 162.0 |
| Phase-CCF log | 6,480 steps |

모든 에피소드에서 162 steps까지 도달. cycle_motion=False이므로 모션 끝(~131 steps=4.4s)까지 완주 후 추가 생존.

시각화 평가: 성큼성큼 걸으며 추종 성능 양호, 보행 품질 개선 (종종걸음 해소).

## 5. 실험 비교

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON2 | 939.51 | 297.39 | CCF sigma=-1.0 분리 |
| VIC_PHASE | 932.36 | 299.1 | Phase obs +2 dims, cycle_motion=True |
| VIC_PHASE_02 | 873.32 | 264.7 | 긴 모션 + cycle_motion=False |
| VIC_PHASE_03 | 873.59 | 262.0 | Bio CCF reward (ankle, w=0.05) |
| **VIC_PHASE_04** | **545.17** | **162.0** | **Bio CCF (ankle+knee, w=0.15), power 0.4x** |

주의: VIC_PHASE_04의 reward/steps가 낮은 이유:
- 짧은 모션(4.4s) + cycle_motion=False → 에피소드 상한이 ~162 steps
- VIC_PHASE(01)과 동일 모션이나, PHASE(01)은 cycle_motion=True로 300 steps까지 생존 가능했음
- 절대 reward 비교는 에피소드 길이 차이로 의미 제한적

## 6. Phase-CCF 분석 (수정된 Gait Cycle Phase 기준)

### 중요: Phase Logging 버그 수정 경위

초기 분석에서 knee의 극적인 phase-dependent 변화(0.73→1.61x, 2.2배)를 발견했으나, 이는 **phase logging 버그**로 인한 오해였다.

**버그 내용**: `phase_ccf_log`에 기록되는 phase가 **gait cycle phase가 아닌 clip phase**였다.
- Clip phase: 전체 모션 클립(4.4s) 내 진행률 (0→1, 한 번만)
- Gait cycle phase: 한 보행 주기(~0.87s) 내 진행률 (0→1, 4~5회 반복)

clip phase로 보면 "초반→후반"의 변화가 보이지만, 이는 gait cycle 4~5번을 하나로 합쳐 평균낸 것이므로 **실제 gait cycle 내 stance/swing 분화와 무관**하다. 만약 진짜 gait cycle 내 분화가 있었다면 clip phase 플롯에서도 주기적 오르락내리락이 보여야 하는데, 그것도 없었다.

**수정**: `humanoid_im_vic.py`의 phase logging을 `_gait_phase_table` 기반 gait cycle phase로 변경 후 재평가.

### 6-1. Group별 Impedance Scale — 수정된 Gait Cycle Phase 기준

전체 실험을 수정된 코드로 재평가한 결과:

| Group | VIC_PHASE | VIC_PHASE_03 | VIC_PHASE_04 | PHASE_03→04 변화 |
|---|---|---|---|---|
| L_Hip | 1.111x | 0.982x | 1.428x | +45% |
| L_Knee | 0.739x | 0.647x | **1.005x** | **+55% (compliant→neutral)** |
| L_Ankle+Toe | 1.513x | 1.750x | 1.644x | -6% |
| R_Hip | 1.019x | 1.156x | 1.131x | -2% |
| R_Knee | 0.723x | 0.659x | **0.995x** | **+51% (compliant→neutral)** |
| R_Ankle+Toe | 1.483x | 1.796x | 1.738x | -3% |
| Upper-L | 0.646x | 0.540x | 0.646x | +20% |
| Upper-R | 0.679x | 0.629x | 0.696x | +11% |

Bio CCF reward를 knee에 확대한 결과 knee 전체 평균 임피던스가 0.65x(compliant)에서 1.0x(neutral)로 크게 상승. 이는 bio CCF reward의 **전체 수준 조정 효과**가 확인된 것이다.

### 6-2. Phase 구간별 Impedance Scale — 핵심 결과 (수정 후)

**VIC_PHASE_04 (gait cycle phase 기준):**

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 | 변화폭 |
|---|---|---|---|---|---|
| L_Hip | 1.390 | 1.424 | 1.469 | 1.428 | ±3% |
| L_Knee | 0.975 | 0.965 | 1.067 | 1.005 | **±5%** |
| L_Ankle+Toe | 1.612 | 1.622 | 1.701 | 1.644 | ±3% |
| R_Hip | 1.077 | 1.135 | 1.179 | 1.131 | ±5% |
| R_Knee | 0.976 | 0.963 | 1.038 | 0.995 | **±4%** |
| R_Ankle+Toe | 1.719 | 1.719 | 1.777 | 1.738 | ±2% |
| Upper-L | 0.632 | 0.651 | 0.654 | 0.646 | ±2% |
| Upper-R | 0.674 | 0.702 | 0.711 | 0.696 | ±3% |

**이전 실험 비교 (gait cycle phase 기준 재평가):**

VIC_PHASE (bio CCF reward 없음):

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 | 변화폭 |
|---|---|---|---|---|---|
| L_Knee | 0.741 | 0.725 | 0.756 | 0.739 | ±2% |
| R_Knee | 0.724 | 0.711 | 0.737 | 0.723 | ±2% |
| L_Ankle | 1.574 | 1.422 | 1.585 | 1.513 | ±5% |
| R_Ankle | 1.483 | 1.495 | 1.432 | 1.483 | ±2% |

VIC_PHASE_03 (bio CCF reward ankle only, w=0.05):

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 | 변화폭 |
|---|---|---|---|---|---|
| L_Knee | 0.647 | 0.660 | 0.639 | 0.647 | ±2% |
| R_Knee | 0.660 | 0.675 | 0.645 | 0.659 | ±2% |
| L_Ankle | 1.770 | 1.712 | 1.773 | 1.750 | ±2% |
| R_Ankle | 1.806 | 1.780 | 1.804 | 1.796 | ±1% |

### 6-3. 핵심 발견: Phase-dependent 분화는 전혀 없다

세 실험 모두에서 **gait cycle phase에 따른 CCF 변화가 사실상 없다**:
- 모든 관절 그룹이 phase에 걸쳐 **±2~5% 범위의 거의 수평선**
- Stance/swing 구간 간 체계적 차이 없음
- L/R antiphase 패턴 없음

Bio CCF reward의 효과:
- **전체 수준 변화는 성공**: knee 0.65x → 1.0x, ankle 1.51x → 1.75x (bio reward 방향으로 전체 평균 이동)
- **Phase-dependent 분화는 실패**: Stance에서 stiff, swing에서 compliant하게 변하는 gait-cycle-level 패턴은 미출현

## 7. Phase-dependent 분화 실패 원인 분석

### 7-1. Bio CCF Reward 구조의 근본적 한계 — "항상 stiff해도 손해가 적다"

현재 bio CCF reward 설계:
```
stance: clamp(ccf, min=0)    → ccf>0이면 보상, ccf<0이면 보상 0 (벌칙 없음)
swing:  clamp(-ccf, min=0)   → ccf<0이면 보상, ccf>0이면 보상 0 (벌칙 없음)
```

Policy가 **상수 ccf=+0.5 (항상 stiff)**를 출력하는 경우:
- Stance(~60% 시간): clamp(0.5, min=0) = 0.5 보상 받음
- Swing(~40% 시간): clamp(-0.5, min=0) = 0 보상 (벌칙 없음!)
- 기대 bio reward = 0.6 × 0.5 × 0.25 × 0.15 = **0.01125/step**

반면 **phase-dependent ccf** (stance=+0.5, swing=-0.5):
- Stance: 0.5 보상
- Swing: 0.5 보상
- 기대 bio reward = 1.0 × 0.5 × 0.25 × 0.15 = **0.01875/step**

**차이 = 0.0075/step**. 전체 step reward(~5/step)의 **0.15%**에 불과하다. Policy가 이 미세한 차이를 위해 phase-dependent 전략을 학습할 gradient가 극히 약하다.

### 7-2. Phase Obs 신호의 구조적 약점

- Phase obs = 2 dims / 전체 456 dims = **0.4%** 비중
- MLP(1024-1024-512-512)에서 456차원 입력 중 2차원 phase 신호를 추출하여 8차원 CCF를 gait phase에 따라 조절해야 함
- CCF는 action의 일부(8/77 = 10%)로, phase→CCF 경로의 gradient 전파가 매우 간접적
- 다른 reward 항목(imitation, AMP, power)이 CCF에 미치는 gradient가 bio CCF reward보다 훨씬 강력

### 7-3. 벌칙 부재 — Wrong-phase CCF에 대한 페널티가 없다

"Soft guidance" 설계(올바른 방향만 보상, 반대 방향은 0) 자체가 **"항상 한쪽 방향(stiff)"이 안전한 전략**이 되는 근본 원인:
- Swing에서 stiff(ccf>0)해도 벌칙 = 0
- 따라서 "항상 stiff" = risk-free 전략
- Phase-dependent 전환은 복잡도만 증가시키고 marginal gain이 미미

### 7-4. CCF 변화의 기능적 필요성(Functional Pressure) 부재

현재 환경에서 CCF가 상수여도 보행이 충분히 잘 된다:
- Imitation + AMP reward로 보행 모방이 잘 학습됨
- CCF를 phase에 따라 바꾸면 오히려 보행이 불안정해질 위험
- 외란(perturbation)이 없어 CCF 변화가 필요한 상황이 발생하지 않음
- 정리: **CCF를 바꿔야 할 물리적 이유가 환경에 없다**

### 7-5. 요약: 세 가지 동시 부재

Phase-dependent CCF 분화가 일어나려면 아래 세 가지가 모두 필요하나, 현재는 모두 부재:

| 조건 | 현재 상태 | 필요한 상태 |
|---|---|---|
| 1. 충분히 강한 보상 신호 | bio reward가 전체의 0.15% | 10% 이상의 비중 |
| 2. Wrong-phase 벌칙 | 벌칙 없음 (soft guidance) | 양방향 보상/벌칙 |
| 3. 기능적 필요성 | 외란 없음, 상수 CCF로 충분 | 외란으로 가변 CCF 필요 |

## 8. 결론

VIC_PHASE_04의 **bio CCF reward**는:
- **전체 임피던스 수준 조정에는 성공**: knee 0.65x→1.0x (compliant→neutral), ankle 유지
- **Gait cycle 내 phase-dependent 분화에는 실패**: 모든 관절이 phase에 걸쳐 거의 수평선

이 결과는 VIC_PHASE, VIC_PHASE_03에서도 동일하게 재확인되었다 (gait cycle phase 기준 재평가). 이전 분석에서 관찰된 "phase-dependent 변화"는 clip phase vs gait cycle phase 혼동에 의한 착시였다.

**핵심 교훈**: Soft guidance(벌칙 없는 방향성 보상)는 전체 수준을 이동시키는 데는 효과적이나, phase-dependent 분화를 유도하기에는 신호가 너무 약하다.

## 9. 후속 방향

| 우선순위 | 방향 | 설명 | 기대 효과 |
|---|---|---|---|
| 1 | **양방향 보상 (벌칙 추가)** | swing에서 ccf>0이면 벌칙 부여. 상수 stiff 전략이 손해가 되도록 | "항상 stiff" 전략 차단 |
| 2 | **Bio CCF weight 대폭 강화** | w=0.15 → 0.5~1.0으로 전체 reward에서 유의미한 비중 확보 | Gradient 증폭 |
| 3 | **외란(perturbation) 추가** | 랜덤 외력으로 CCF 변화의 functional 필요성 생성 | 상수 CCF로는 대응 불가 |
| 4 | **Phase-CCF 직접 보상** | phase 구간별 target CCF를 명시적으로 지정 | 가장 직접적 (자율 학습 의미 약화) |
| 5 | **CCF 아키텍처 분리** | Phase obs → CCF 전용 네트워크 분리 | Phase-CCF 경로 gradient 강화 |

가장 효과적인 조합: **1번 (양방향 보상) + 2번 (weight 강화)**. 현재 reward가 "soft guidance"로 너무 약한 것이 핵심 문제이다.

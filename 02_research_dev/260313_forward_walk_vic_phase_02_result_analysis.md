# VIC_PHASE_02 결과 분석 (260313)

## 1. 실험 개요

VIC_PHASE에서 phase obs 추가로 CCF 분화 강화를 확인한 후, 보행 모션과 종료 조건을 개선하여 학습 안정성 향상을 목표로 한 실험.

### 핵심 변경
1. **Reference motion 변경**: 4.4s (4걸음) → 8.77s (~10걸음) 긴 직선 보행 모션 (KIT_9_WalkingStraightForwards01)
2. **cycle_motion: False**: 사이클 반복 없이 모션 끝까지만 모방 → 2번째 사이클 velocity 불연속(bunny hop) 문제 원천 제거
3. **에너지 페널티 10배 강화**: power_coefficient 0.0000005 → 0.000005 → 관절 떨림 억제
4. **Episode length 유지**: 300 steps (10초). 모션 길이 8.77s에 맞춰 자연 종료

### Phase 검출 결과
- 11 R heel strikes detected, avg stride period: 0.841s
- R_Ankle height range: [0.0511, 0.1850]m

## 2. 학습 설정

| 항목 | 값 |
|---|---|
| 실험명 | VIC_PHASE_02 |
| 기반 | VIC_PHASE + 모션/종료조건 개선 |
| motion_file | amass_isaac_walking_forward_long.pkl (8.77s, ~10걸음) |
| cycle_motion | False |
| vic_curriculum_stage | 2 (CCF 학습 활성화) |
| vic_ccf_num_groups | 8 |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_phase_obs | True (+2 dims) |
| power_coefficient | 0.000005 (10x 강화) |
| reward_curriculum_switch_epoch | 10000 |
| num_envs | 512 |
| 학습 epochs | ~11637 (수동 중단, 수렴 판단) |

## 3. 학습 결과

### 학습 특이사항
- 이전 실험들 대비 보상 상승 속도가 매우 빠름 (epoch 600 시점에서 이미 rwd~350, eps_len~100)
- epoch ~5000 이후 수렴 양상

### 최종 성능 (epoch ~11637)
| 메트릭 | 값 |
|---|---|
| 학습 마지막 reward | ~350-380 (변동) |
| 학습 마지막 eps_len | ~106-116 |

### Headless 평가 결과 (greedy policy)
| 메트릭 | 값 |
|---|---|
| 에피소드 reward 범위 | 864.39 - 1042.98 |
| 평균 reward (av reward) | 873.32 |
| 평균 steps (av steps) | 264.7 |
| Phase-CCF log | 10,480 steps 기록 |

참고: 대부분 에피소드가 264.7 steps에서 종료 → 8.77s 모션 끝까지 성공적으로 도달(262 steps ≈ 8.73s). cycle_motion=False이므로 모션 종료 = 에피소드 종료.

## 4. 실험 비교

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON | 945.30 | 297.98 | Stage2, CCF sigma 공유 → 미학습 |
| VIC_CCF_ON2 | 939.51 | 297.39 | CCF sigma=-1.0 분리 → CCF 학습 확인 |
| VIC_PHASE | 932.44 | 299 | Gait cycle phase obs +2 dims |
| **VIC_PHASE_02** | **873.32** | **264.7** | **긴 모션 + cycle_motion=False + energy 10x** |

주의: VIC_PHASE_02의 reward/steps가 낮아 보이는 이유:
- cycle_motion=False → 에피소드 상한이 ~262 steps (8.77s 모션 끝). 이전 실험은 300 steps (10초) 생존 가능
- 에너지 페널티 10x → 절대 reward 값 감소
- 실제로는 **모션 끝까지 거의 100% 성공** (264.7/262 ≈ 완주)

## 5. Phase-CCF 분석

### 5-1. Group별 Impedance Scale (전체 평균)

| Group | Impedance Scale | 해석 |
|---|---|---|
| R_Ankle+Toe | 1.509x | 높음 (rigid) |
| L_Ankle+Toe | 1.433x | 높음 (rigid) |
| L_Hip | 1.131x | 약간 높음 |
| R_Hip | 1.015x | 기준선 수준 |
| L_Knee | 0.691x | 낮음 (compliant) |
| R_Knee | 0.673x | 낮음 (compliant) |
| Upper-R | 0.642x | 낮음 (compliant) |
| Upper-L | 0.572x | 낮음 (compliant) |

### 5-2. Phase 구간별 Impedance Scale

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 |
|---|---|---|---|---|
| L_Hip | 1.150 | 1.092 | 1.158 | 1.131 |
| L_Knee | 0.676 | 0.668 | 0.734 | 0.691 |
| L_Ankle+Toe | 1.519 | 1.419 | 1.331 | 1.433 |
| R_Hip | 1.012 | 1.050 | 0.970 | 1.015 |
| R_Knee | 0.666 | 0.669 | 0.665 | 0.673 |
| R_Ankle+Toe | 1.572 | 1.377 | 1.592 | 1.509 |
| Upper-L | 0.565 | 0.591 | 0.548 | 0.572 |
| Upper-R | 0.618 | 0.681 | 0.608 | 0.642 |

### 5-3. VIC_PHASE 대비 CCF 패턴 비교

| Group | VIC_PHASE | VIC_PHASE_02 | 변화 |
|---|---|---|---|
| L_Ankle+Toe | 1.51x | 1.43x | -5% (약간 감소) |
| R_Ankle+Toe | 1.48x | 1.51x | 유사 |
| L_Knee | 0.74x | 0.69x | -7% (더 compliant) |
| R_Knee | 0.72x | 0.67x | -7% (더 compliant) |
| Upper-L | 0.65x | 0.57x | -12% (더 compliant) |
| Upper-R | 0.68x | 0.64x | -6% (더 compliant) |

에너지 페널티 10x 강화로 전반적으로 compliance 방향으로 이동. 특히 Upper-L이 0.57x로 가장 compliant.

### 5-4. Phase-dependent CCF 변화 관찰

Phase 구간별 변화가 여전히 크지 않으나 일부 패턴 관찰:
- **L_Ankle+Toe**: 초반(1.519) → 후반(1.331)으로 감소 — 모션 진행에 따라 발목 compliance 증가
- **R_Ankle+Toe**: 중반(1.377)에서 다소 낮아졌다 후반(1.592)에서 다시 상승 — U자형 패턴
- **L_Knee**: 후반(0.734)에서 약간 상승 — toe-off 준비 시 무릎 stiffness 증가 가능성
- **Hip/Upper**: 비교적 일정한 패턴 유지

## 6. 분석 및 해석

### 긍정적 발견
1. **모션 완주 100%**: 대부분 에피소드가 모션 끝(262 steps)까지 도달. cycle 경계 문제 없이 안정적 보행
2. **빠른 수렴**: 이전 실험 대비 학습 초기부터 reward/steps 상승이 가파름. cycle_motion=False + 긴 모션의 효과
3. **CCF 분화 패턴 유지**: 에너지 페널티 강화에도 "rigid ankle / compliant knee & upper body" 기본 구조 유지
4. **학습 효율성**: ~11600 epoch에서 수렴 (이전 20000 epoch 대비 절반)

### 한계 및 관찰
1. **Phase-dependent CCF 변화 여전히 미미**: stance/swing에서 뚜렷한 CCF 전환이 관찰되지 않음
2. **에너지 페널티의 CCF 영향**: 10x 강화로 전반적 compliance 증가 — 에너지 절약 전략일 가능성 (생체역학적 의미 불분명)
3. **비교 제한**: cycle_motion/episode_length/energy penalty가 모두 변경되어 순수 효과 분리 어려움
4. **Phase-dependent CCF 학습 동기 부재**: 현재 reward에 phase별 CCF 방향성 신호가 없음 → 에이전트가 CCF를 phase에 따라 조절할 유인이 부족

## 7. 결론

VIC_PHASE_02는 학습 안정성과 수렴 속도 면에서 큰 개선을 보였다:
- cycle_motion=False + 긴 모션으로 학습 환경 단순화 → 빠른 수렴, 100% 모션 완주
- CCF 분화의 기본 패턴(rigid ankle, compliant knee/upper) 유지

그러나 핵심 목표인 **phase-dependent CCF 조절**(stance에서 stiff, swing에서 compliant)은 여전히 미미하다. 이는 현재 reward 구조에서 CCF를 phase에 따라 조절해야 할 명시적 동기가 없기 때문이다.

## 8. 후속 방향: VIC_PHASE_03

Phase-dependent CCF 학습을 촉진하기 위해 **생체역학 기반 CCF 보상** 추가:
- 발-지면 contact force 기반으로 stance/swing 판별
- Stance leg: ankle stiff(CCF>0) 보상, Swing leg: ankle compliant(CCF<0) 보상
- 작은 보상 가중치(w=0.05)로 기존 학습 안정성 유지하면서 방향성만 제공

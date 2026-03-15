# VIC_PHASE_03 결과 분석 (260314)

## 1. 실험 개요

VIC_PHASE_02에서 phase-dependent CCF 변화가 미미했던 원인이 reward에 CCF 방향성 신호가 없기 때문이라고 분석. 생체역학 기반 CCF 보상(bio CCF reward)을 추가하여 stance/swing에서 적절한 ankle 임피던스 학습을 유도하는 실험.

### 핵심 변경
- **Bio CCF Reward 추가**: 발-지면 contact force 기반으로 stance/swing 판별 후, ankle CCF에 방향성 보상
  - Stance leg (접촉력 > 10N): ankle CCF > 0 (stiff) 보상
  - Swing leg (접촉력 ≤ 10N): ankle CCF < 0 (compliant) 보상
  - 가중치: w = 0.05 (기존 imitation reward 대비 매우 작음)

## 2. MDP 구성

### 2-1. State (Observation)
| 구성 요소 | Dims | 설명 |
|---|---|---|
| Body positions/rotations | ~400 | 전신 관절 위치, 회전, 속도, 각속도 (local root frame) |
| DOF positions/velocities | 69+69 | 관절 각도 및 각속도 |
| Target tracking | ~20 | 미래 trajectory 샘플 (key body targets) |
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
| Term | 수식/설명 | 가중치 |
|---|---|---|
| Point goal | distance 기반 위치 보상 | 1.0 |
| Imitation | pos/rot/vel/ang_vel 매칭 (k_pos=200, k_rot=10, k_vel=1.0, k_ang_vel=0.1) | 0.5 (far 아닌 경우) |
| AMP discriminator | 적대적 모션 자연스러움 판별 | curriculum (0.3→0.7) |
| Power penalty | -0.000005 * sum(abs(torque * dof_vel)) | 1.0 |
| **Bio CCF (신규)** | **stance: clamp(ankle_ccf, min=0), swing: clamp(-ankle_ccf, min=0)** | **0.05** |

Reward curriculum: epoch < 10000 → task:0.7/disc:0.3, epoch ≥ 10000 → task:0.3/disc:0.7

### 2-4. Termination
| 조건 | 값 | 설명 |
|---|---|---|
| enableEarlyTermination | True | 넘어지면 종료 |
| terminationHeight | 0.15m | 루트 높이 기준 |
| terminationDistance | 0.25m | reference 대비 위치 오차 |
| episode_length | 300 steps (10s) | 최대 에피소드 길이 |
| 모션 종료 | ~262 steps (8.77s) | cycle_motion=False → 모션 끝나면 종료 |

## 3. 학습 설정

| 항목 | 값 |
|---|---|
| 실험명 | VIC_PHASE_03 |
| 기반 | VIC_PHASE_02 + bio CCF reward |
| motion_file | amass_isaac_walking_forward_long.pkl (8.77s, ~10걸음) |
| cycle_motion | False |
| vic_curriculum_stage | 2 |
| vic_ccf_num_groups | 8 |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_phase_obs | True (+2 dims) |
| vic_bio_ccf_reward_w | 0.05 (신규) |
| power_coefficient | 0.000005 |
| max_epochs | 20000 |
| num_envs | 512 |
| wandb run | gallant-terrain-53 |

## 4. 학습 결과

### Headless 평가 결과 (greedy policy, epoch 20000)
| 메트릭 | 값 |
|---|---|
| 에피소드 reward | 878.89 (거의 일정) |
| 평균 reward (av reward) | 878.89 |
| 평균 steps (av steps) | 262.0 |
| Phase-CCF log | 10,480 steps |

모든 에피소드에서 262 steps(모션 끝)까지 100% 완주. Reward 편차가 거의 없음 (878.886~878.890).

## 5. 실험 비교

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON2 | 939.51 | 297.39 | CCF sigma=-1.0 분리 |
| VIC_PHASE | 932.44 | 299 | Phase obs +2 dims |
| VIC_PHASE_02 | 873.32 | 264.7 | 긴 모션 + cycle_motion=False |
| **VIC_PHASE_03** | **878.89** | **262.0** | **Bio CCF reward (w=0.05)** |

VIC_PHASE_02 대비 reward 미세 증가(+5.57), steps 일정(262). Bio CCF reward 추가가 기존 성능에 악영향을 주지 않음.

## 6. Phase-CCF 분석

### 6-1. Group별 Impedance Scale (전체 평균)

| Group | VIC_PHASE_02 | VIC_PHASE_03 | 변화 |
|---|---|---|---|
| L_Hip | 1.131x | 0.982x | -13% (더 compliant) |
| L_Knee | 0.691x | 0.647x | -6% (더 compliant) |
| L_Ankle+Toe | 1.433x | **1.750x** | **+22% (더 rigid!)** |
| R_Hip | 1.015x | **1.155x** | +14% (더 rigid) |
| R_Knee | 0.673x | 0.660x | 유사 |
| R_Ankle+Toe | 1.509x | **1.795x** | **+19% (더 rigid!)** |
| Upper-L | 0.572x | 0.540x | -6% |
| Upper-R | 0.642x | 0.630x | -2% |

**핵심 변화: Ankle 임피던스가 크게 증가!** Bio CCF reward가 stance에서 ankle CCF>0을 보상한 효과가 명확함.
- L_Ankle: 1.43x → 1.75x (+22%)
- R_Ankle: 1.51x → 1.80x (+19%)

### 6-2. Phase 구간별 Impedance Scale

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 |
|---|---|---|---|---|
| L_Hip | 0.889 | 1.090 | 0.946 | 0.982 |
| L_Knee | 0.650 | 0.653 | 0.627 | 0.647 |
| L_Ankle+Toe | 1.828 | 1.653 | 1.798 | 1.750 |
| R_Hip | 1.255 | 0.987 | 1.315 | 1.155 |
| R_Knee | 0.662 | 0.653 | 0.656 | 0.660 |
| R_Ankle+Toe | 1.838 | 1.743 | 1.825 | 1.795 |
| Upper-L | 0.536 | 0.544 | 0.539 | 0.540 |
| Upper-R | 0.622 | 0.642 | 0.615 | 0.630 |

### 6-3. L/R Comparison Plot 분석

Hip 플롯에서 흥미로운 패턴 관찰:
- **R_Hip이 L_Hip보다 전반적으로 stiff** (R: 1.15x vs L: 0.98x)
- **Phase 40% 부근에서 L_Hip이 상승, R_Hip이 하강** → 좌우 교차 패턴 (antiphase) 일부 관찰!
- 이는 보행 시 stance leg의 hip이 stiff해지는 것과 일치

Ankle 플롯:
- L/R 모두 전반적으로 매우 stiff (1.7~1.8x)
- **Phase 55-65% 구간에서 L_Ankle이 dip** (1.83→1.52 정도) — swing phase 진입 시 compliance 증가 가능성
- R_Ankle은 중반에서 약간 하강(1.74x) 후 회복

Knee 플롯:
- L/R 모두 일정하게 compliant (~0.65x), phase 변화 미미

## 7. 분석 및 해석

### 긍정적 발견
1. **Bio CCF reward 효과 확인**: Ankle 임피던스가 VIC_PHASE_02 대비 20% 이상 증가. Contact force 기반 보상이 stance ankle stiffness를 강화함
2. **Hip antiphase 패턴 출현**: L/R Hip CCF가 phase에 따라 교차하는 패턴이 일부 관찰 — stance leg hip stiff, swing leg hip compliant의 초기 징후
3. **L_Ankle phase-dependent dip**: Phase 55-65%에서 L_Ankle compliance가 증가 — swing 진입 시 ankle이 풀리는 생체역학적 패턴과 일치 가능성
4. **성능 유지**: Bio CCF reward 추가에도 100% 모션 완주, reward 유지

### 한계 및 관찰
1. **Swing phase에서 ankle compliance 유도 효과 부족**: Bio reward가 swing에서 CCF<0을 보상하지만, 전체적으로 ankle이 매우 stiff(1.75~1.80x). Stance 보상이 dominant하고 swing 보상은 상대적으로 약한 듯
2. **Hip antiphase가 아직 미약**: 패턴이 관찰되지만 변화 폭이 작음 (L_Hip: 0.89~1.09, R_Hip: 0.99~1.32)
3. **Knee CCF 변화 없음**: Bio reward가 ankle만 대상으로 하므로 knee에는 영향 없음
4. **Bio reward weight 0.05가 적절한지 검토 필요**: 더 높이면 phase-dependent 변화가 강해질 수 있으나, 학습 안정성 저하 위험

## 8. 결론

VIC_PHASE_03는 **생체역학 기반 CCF 보상의 효과를 처음으로 확인**한 실험이다:
- Contact force 기반 stance/swing 판별 → ankle stiffness 보상이 ankle CCF를 크게 증가시킴
- Hip에서 L/R antiphase 패턴이 출현 (부수적 효과)
- L_Ankle에서 swing 진입 시 compliance dip 관찰

Bio CCF reward가 "방향성"을 제공하는 효과는 확인되었으나, stance dominant → swing compliance 유도가 부족. 보상 구조 조정이나 knee 포함 확대가 다음 과제.

## 9. 후속 방향

1. **Bio CCF reward weight 튜닝**: w=0.05 → w=0.1~0.2로 강화 실험
2. **Knee 포함 확대**: Hip, Knee에도 contact-aware CCF 보상 적용
3. **Swing compliance 강화**: swing 보상 가중치를 stance보다 높여서 balance 조정
4. **외란(Perturbation) 추가**: 랜덤 push로 임피던스 분화 압력 제공
5. **Gait cycle 단위 분석**: heel strike 기준 정규화된 gait phase별 CCF 비교 (현재는 motion phase 기반)

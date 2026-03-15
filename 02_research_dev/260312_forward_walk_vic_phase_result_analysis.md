# VIC_PHASE 결과 분석 (260312)

## 1. 실험 개요

VIC_CCF_ON2에서 CCF 학습 성공 확인 후, 에이전트에게 **보행 한 주기(gait cycle) 내 현재 위치** 정보를 추가로 제공하여 phase-dependent CCF 학습을 강화하는 실험.

### 핵심 변경: Gait Cycle Phase Observation
- Reference motion의 R_Ankle 높이에서 heel strike(local minima) 검출
- 연속 heel strike 사이를 0→1로 선형 보간 → gait phase lookup table 구축
- sin/cos 인코딩으로 obs에 +2 dims 추가 (총 456 dims)

### 검출 결과
- 6 R heel strikes detected, avg stride period: 0.873s
- R_Ankle height range: [0.0523, 0.2231]m

## 2. 학습 설정

| 항목 | 값 |
|---|---|
| 실험명 | VIC_PHASE |
| 기반 | VIC_CCF_ON2 동일 설정 + vic_phase_obs: True |
| vic_curriculum_stage | 2 (CCF 학습 활성화) |
| vic_ccf_num_groups | 8 |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_phase_obs | True (+2 dims) |
| reward_curriculum_switch_epoch | 10000 |
| max_epochs | 20000 |
| num_envs | 512 |
| wandb run | pious-feather-51 |

## 3. 학습 결과

### 최종 성능 (epoch 20000)
| 메트릭 | 값 |
|---|---|
| 학습 마지막 reward | ~590-660 (변동 있음) |
| 학습 마지막 eps_len | ~195-220 |

### Headless 평가 결과 (greedy policy)
| 메트릭 | 값 |
|---|---|
| 첫 에피소드 reward | 932.44 |
| 첫 에피소드 steps | 299 |
| 평균 reward | 1815.65 |
| 평균 steps | 565.75 |
| Phase-CCF log | 11960 steps 기록 |

참고: 평균이 높은 것은 일부 에피소드가 여러 cycle을 성공적으로 생존했기 때문.

## 4. 실험 비교

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON | 945.30 | 297.98 | Stage2, CCF sigma 공유 → 미학습 |
| VIC_CCF_ON2 | 939.51 | 297.39 | CCF sigma=-1.0 분리 → CCF 학습 확인 |
| **VIC_PHASE** | **932.44** | **299** | **Gait cycle phase obs +2 dims** |

Per-episode 성능은 VIC_CCF_ON2와 거의 동일 (reward ~932 vs ~939, steps ~299 vs ~297). Phase obs 추가가 성능을 저하시키지 않았으나, 단순 reward/steps 지표에서 뚜렷한 개선은 관찰되지 않음.

## 5. Phase-CCF 분석

### 5-1. Group별 Impedance Scale (전체 평균)

| Group | Impedance Scale | 해석 |
|---|---|---|
| L_Ankle+Toe | 1.512x | 높음 (rigid) |
| R_Ankle+Toe | 1.476x | 높음 (rigid) |
| L_Hip | 1.102x | 약간 높음 |
| R_Hip | 1.007x | 기준선 수준 |
| L_Knee | 0.741x | 낮음 (compliant) |
| R_Knee | 0.723x | 낮음 (compliant) |
| Upper-R | 0.680x | 낮음 (compliant) |
| Upper-L | 0.645x | 낮음 (compliant) |

### 5-2. Phase 구간별 Impedance Scale

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) |
|---|---|---|---|
| L_Hip | 0.965 | 1.199 | 1.045 |
| L_Knee | 0.701 | 0.757 | 0.750 |
| L_Ankle+Toe | 1.500 | 1.452 | 1.581 |
| R_Hip | 0.864 | 1.067 | 1.002 |
| R_Knee | 0.690 | 0.739 | 0.722 |
| R_Ankle+Toe | 1.492 | 1.427 | 1.499 |
| Upper-L | 0.603 | 0.675 | 0.633 |
| Upper-R | 0.674 | 0.687 | 0.676 |

### 5-3. VIC_CCF_ON2 대비 CCF 패턴 비교

| Group | VIC_CCF_ON2 | VIC_PHASE | 변화 |
|---|---|---|---|
| L_Ankle+Toe | 1.28x | 1.51x | +18% (더 rigid) |
| R_Ankle+Toe | 1.49x | 1.48x | 유사 |
| L_Knee | 0.91x | 0.74x | -19% (더 compliant) |
| R_Knee | 0.94x | 0.72x | -23% (더 compliant) |
| Upper-L | 0.80x | 0.65x | -19% (더 compliant) |
| Upper-R | 0.86x | 0.68x | -21% (더 compliant) |

**주요 변화**: VIC_PHASE에서 CCF 분화가 더 뚜렷해짐!
- 발목: VIC_CCF_ON2 대비 더 rigid (특히 L_Ankle 1.28→1.51)
- 무릎: 더 compliant (0.91~0.94 → 0.72~0.74)
- 상체: 더 compliant (0.80~0.86 → 0.65~0.68)
- 전반적으로 "rigid ankle / compliant knee & upper body" 패턴이 강화됨

### 5-4. Phase-dependent CCF 변화 관찰

Phase 구간별 변화 폭이 크지 않으나 일부 트렌드 관찰:
- **Hip**: 초반 낮고 중반에 상승 (L_Hip: 0.965→1.199, R_Hip: 0.864→1.067)
  - Heel strike 직후 compliant → mid-stance에서 rigid 전환
- **Ankle**: 전반적으로 높으나 후반에 약간 증가 (L_Ankle: 1.500→1.581)
  - Toe-off 준비 시 stiffness 증가 가능성
- **Knee/Upper**: 비교적 일정한 compliant 패턴 유지

## 6. 분석 및 해석

### 긍정적 발견
1. **CCF 분화 강화**: Phase obs 추가로 에이전트가 gait cycle 위치를 인지하면서 CCF를 더 적극적으로 분화시킴
2. **생체역학적 일관성**: "rigid ankle + compliant knee/upper body" 패턴이 문헌과 더 강하게 일치
3. **성능 유지**: reward/steps 지표가 VIC_CCF_ON2와 거의 동일 — phase obs가 학습에 해를 끼치지 않음
4. **다중 사이클 생존**: 평균 steps=565.75로, 일부 에피소드가 여러 gait cycle을 안정적으로 통과

### 한계 및 관찰
1. **Phase-dependent CCF 변화 미미**: phase 구간별 impedance 변화 폭이 기대보다 작음 (Hip에서 일부 변화만 관찰)
2. **Reward 개선 없음**: 단순 reward/steps 지표에서 VIC_CCF_ON2 대비 뚜렷한 개선 없음
3. **Bunny hop 완화 미확인**: 시각화를 통한 정성적 평가 필요

## 7. 결론

VIC_PHASE 실험은 gait cycle phase observation이 CCF 분화를 **강화**하는 효과를 확인했다:
- 발목 stiffness 증가, 무릎/상체 compliance 증가 → 생체역학적 패턴 강화
- 전체 성능(reward/steps)은 유지

다만, 기대했던 phase-dependent CCF 조절(stance vs swing에서 뚜렷한 차이)은 제한적이었다. 이는 현재 단일 모션 + 짧은 gait cycle (~0.87s)에서 CCF 변화의 시간 해상도가 충분하지 않을 수 있음을 시사한다.

## 8. 후속 방향

1. **Gait cycle 내 CCF 시간 해상도 분석**: phase를 10구간으로 세분화하여 stance/swing별 임피던스 비교
2. **Contact force 기반 CCF 분석**: 실제 발-지면 접촉력과 CCF의 상관관계 분석
3. **주기적 보행 모션 교체**: walk-stop → cyclic walking loop로 근본적 사이클 경계 문제 해결
4. **Phase obs + 추가 보상**: phase-dependent CCF 조절에 대한 명시적 보상 추가 검토

# VIC_PHASE_02 구현 상세 (260313)

## 1. 실험 목적

VIC_PHASE에서 phase obs 추가로 CCF 분화 강화를 확인했으나, phase-dependent CCF 변화는 여전히 미미했다. VIC_PHASE_02에서는 학습 환경 자체를 개선하여 더 안정적이고 긴 보행을 학습시키는 것이 목적이다.

핵심 문제:
- VIC_PHASE의 모션이 4.4초(4걸음)로 짧아 다양한 gait cycle 경험 부족
- cycle_motion=True 시 2번째 사이클 경계에서 velocity 불연속(bunny hop) 발생
- 에너지 페널티가 약해(0.0000005) 관절 떨림이 있었음

## 2. 핵심 변경 사항 (VIC_PHASE 대비)

### 2-1. Reference Motion 변경
기존 4.4초(4걸음) 짧은 모션에서 8.77초(~10걸음) 긴 직선 보행 모션으로 교체.

| 항목 | VIC_PHASE | VIC_PHASE_02 |
|---|---|---|
| motion_file | amass_isaac_walking_forward_single.pkl | amass_isaac_walking_forward_long.pkl |
| 모션 길이 | ~4.4초 (4걸음) | ~8.77초 (~10걸음) |
| 출처 | KIT_WalkingStraightForwards05 | KIT_9_WalkingStraightForwards01 |

### 2-2. cycle_motion: False
사이클 반복 없이 모션 끝까지만 모방 학습. 2번째 사이클 진입 시 velocity 불연속(bunny hop) 문제를 원천 제거.

```yaml
cycle_motion: False  # (기존: True)
```

모션 길이 8.77초 ≈ 262 steps이므로, 에피소드는 모션 끝에서 자연 종료된다.

### 2-3. 에너지 페널티 10배 강화
관절 떨림 억제를 위해 power_coefficient를 10배 증가.

```yaml
power_coefficient: 0.000005  # (기존: 0.0000005)
```

Power reward 수식: `-power_coefficient * sum(abs(torque * dof_vel))`

### 2-4. Phase 검출 결과 변화
긴 모션으로 heel strike 검출 수 증가:

| 항목 | VIC_PHASE | VIC_PHASE_02 |
|---|---|---|
| R heel strikes | 6개 | 11개 |
| Avg stride period | 0.873s | 0.841s |
| R_Ankle height range | 0.0523~0.2231m | 0.0511~0.1850m |

## 3. 코드 수정 내용

### 수정 파일 목록
코드 자체 변경은 없다. VIC_PHASE와 동일한 `humanoid_im_vic.py`를 사용한다. 변경은 설정(yaml) 파일에서만 이루어졌다.

### env yaml 변경 (`env_im_walk_vic.yaml`)
```yaml
# 변경 1: Reference motion
motion_file: "sample_data/amass_isaac_walking_forward_long.pkl"  # (기존: single.pkl)

# 변경 2: Cycle motion 비활성화
cycle_motion: False  # (기존: True)

# 변경 3: 에너지 페널티 강화
power_coefficient: 0.000005  # (기존: 0.0000005)
```

### learning yaml 변경 (`im_walk_vic.yaml`)
```yaml
name: VIC_PHASE_02  # (기존: VIC_PHASE)
```

나머지 설정은 VIC_PHASE와 동일:
- vic_curriculum_stage: 2
- vic_ccf_num_groups: 8
- vic_ccf_sigma_init: -1.0
- vic_phase_obs: True
- reward_curriculum_switch_epoch: 10000

## 4. Obs/Action 구조

VIC_PHASE와 동일. 코드 변경 없음.

| 구분 | 크기 | 설명 |
|---|---|---|
| self_obs | 166 | dof_pos, vel, root state 등 |
| task_obs (base) | 288 | 4 bodies x 3 samples x 24 |
| gait_phase_obs | 2 | sin/cos(2π × gait_cycle_phase) |
| 합계 | 456 | |

action_space: 69 (PD targets) + 8 (CCF groups) = 77

## 5. 실험 설정 백업

`exp_config/forward_walking/260313_VIC_PHASE_02/`에 스냅샷:
- env_im_walk_vic.yaml
- im_walk_vic.yaml
- humanoid_im_vic.py

## 6. 설계 의도

1. **긴 모션**: 10걸음의 직선 보행으로 다양한 gait cycle을 경험, 더 일반화된 보행 학습 유도
2. **cycle_motion=False**: 사이클 경계 문제를 회피하여 학습 안정성 확보. 모션 끝에서 자연 종료로 "모션 모방 성공"을 명확히 측정 가능
3. **에너지 페널티 강화**: 관절 떨림 억제, 에너지 효율적인 보행 유도. 부수적으로 CCF를 더 compliant 방향으로 유도할 수 있음

## 7. 기대 효과 및 실제 결과

| 기대 | 실제 |
|---|---|
| 학습 안정성 향상 | 확인: 수렴 속도 매우 빠름, ~11600 epoch에서 수렴 |
| 모션 완주율 향상 | 확인: 거의 100% 모션 끝(262 steps)까지 도달 |
| Bunny hop 제거 | 확인: cycle_motion=False로 원천 제거 |
| Phase-dependent CCF 강화 | 미달: 여전히 미미, reward에 CCF 방향성 신호 부재 |

## 8. 후속 방향

Phase-dependent CCF가 여전히 미미한 원인은 reward 구조에 CCF를 phase에 따라 조절해야 할 명시적 동기가 없기 때문이다. 이를 해결하기 위해 VIC_PHASE_03에서 생체역학 기반 CCF 보상(bio CCF reward) 추가를 계획.

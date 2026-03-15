# VIC_PHASE 구현 상세 (260312)

## 1. 실험 목적

VIC_CCF_ON2에서 CCF 학습 성공을 확인한 후, 에이전트에게 **보행 한 주기(gait cycle) 내 현재 위치** 정보를 추가로 제공하여:
- Stance/Swing 구분을 에이전트가 인지할 수 있게 함
- Phase-dependent CCF 조절을 더 명확하게 학습 (발목은 stance에서 rigid, swing에서 compliant 등)
- 사이클 경계(cycle_motion=True) 전환 시 사전 준비 가능 (bunny hop 완화)

## 2. Phase 정의: Gait Cycle Phase (Foot Contact 기반)

### 핵심 개념
**Gait cycle phase = 보행 한 주기(right heel strike → next right heel strike) 내 위치 (0→1)**

기존 모션 클립 phase(전체 5.4초 대비 현재 시점)가 아닌, 실제 **보행 한 걸음 주기** 기준의 phase다.

### 구현 방식: Reference Motion Foot Height 기반 사전 계산
1. 초기화 시 reference motion(KIT_WalkingStraightForwards05)을 1000개 시점에서 샘플링
2. R_Ankle 높이 궤적에서 **local minima**(발이 가장 낮은 시점 = heel strike) 검출
3. 연속 heel strike 사이를 0→1로 선형 보간 → gait phase lookup table 구축
4. 런타임: 현재 motion time으로 lookup table에서 gait phase 조회

### 검출 결과 (실제 학습 로그)
```
6 R heel strikes detected, avg stride period: 0.873s
R_Ankle height range: [0.0523, 0.2231]m
```
- Stance 시 발목 높이: ~5.2cm (지면 바로 위)
- Swing peak 시 발목 높이: ~22.3cm
- 평균 보폭 주기: 0.873초 (정상 보행 범위)

### 인코딩: sin/cos
```python
phase_obs = [sin(2π × gait_phase), cos(2π × gait_phase)]  # [N, 2]
```
phase=0(heel strike)과 phase=1(next heel strike) 사이가 연속적으로 매핑됨.

### 이전 버전(clip-level phase)과의 차이

| | Clip-level Phase | Gait Cycle Phase (현재) |
|---|---|---|
| 의미 | 전체 모션 클립 대비 위치 | 보행 한 주기 내 위치 |
| 1 cycle | ~5.4초 (전체 클립) | ~0.87초 (한 걸음) |
| Stance/Swing 구분 | 불가 (한 클립에 4걸음 혼재) | 가능 (phase 0~0.6≈stance, 0.6~1.0≈swing) |
| 검출 방법 | curr_time / motion_length | Reference motion foot height local minima |

## 3. 코드 수정 내용

### 수정 파일: `phc/env/tasks/humanoid_im_vic.py` (단 하나)

#### 3-1. import 추가
```python
import math
```

#### 3-2. __init__(): 플래그 추가
```python
self._vic_phase_obs = cfg["env"].get("vic_phase_obs", False)
```

#### 3-3. get_task_obs_size(): +2 dims
```python
if self._enable_task_obs and self._vic_phase_obs:
    obs_size += 2
```

#### 3-4. _precompute_gait_phase(): 핵심 메서드 (신규)
Reference motion에서 gait cycle phase를 사전 계산하는 메서드.

동작 원리:
1. `self._motion_lib.get_motion_state()`로 reference motion을 1000개 시점에서 샘플링
2. R_Ankle body의 z-높이 궤적 추출
3. Local minima 검출 (양쪽 이웃보다 낮은 점) + 최소 0.5초 간격 필터
4. 연속 heel strike 사이를 선형 보간하여 phase 0→1 할당
5. 첫 heel strike 이전: 첫 stride period로 역방향 외삽
6. 마지막 heel strike 이후: 마지막 stride period로 순방향 외삽 (clamp 1.0)
7. `self._gait_phase_table` [1000]에 저장

Lazy init: `_compute_task_obs()` 첫 호출 시 자동 실행.

#### 3-5. _compute_task_obs(): gait phase lookup
```python
if self._vic_phase_obs:
    if not hasattr(self, '_gait_phase_table'):
        self._precompute_gait_phase()

    # 현재 motion time → clip 내 시간 (cyclic 처리)
    time_in_clip = motion_times % motion_len

    # Lookup table에서 gait phase 조회
    frame_idx = (time_in_clip / motion_length * num_samples).long().clamp(0, num_samples-1)
    gait_phase = self._gait_phase_table[frame_idx]

    # sin/cos 인코딩 후 obs에 concat
    phase_obs = [sin(2π×gait_phase), cos(2π×gait_phase)]
    obs = cat([obs, phase_obs], dim=-1)
```

## 4. 설정

### env yaml (`env_im_walk_vic.yaml`)
```yaml
vic_phase_obs: True
```

### learning yaml (`im_walk_vic.yaml`)
```yaml
name: VIC_PHASE
```

나머지 VIC_CCF_ON2와 동일:
- vic_curriculum_stage: 2 (CCF 학습 활성화)
- vic_ccf_num_groups: 8
- vic_ccf_sigma_init: -1.0
- reward_curriculum_switch_epoch: 10000
- max_epochs: 20000

## 5. Obs/Action 구조

| 구분 | 크기 | 설명 |
|---|---|---|
| self_obs | 166 | dof_pos, vel, root state 등 |
| task_obs (base) | 288 | 4 bodies x 3 samples x 24 (obs_v=6) |
| gait_phase_obs | 2 | sin/cos(2π×gait_cycle_phase) |
| 합계 | 456 | 기존 454 + 2 |

action_space: 69 (PD targets) + 8 (CCF groups) = 77 (변경 없음)

## 6. 실험 설정 백업

`exp_config/forward_walking/260312_VIC_PHASE/`에 스냅샷:
- env_im_walk_vic.yaml
- im_walk_vic.yaml
- humanoid_im_vic.py

## 7. 기대 효과

1. 에이전트가 stance/swing 위치를 직접 인지 → phase-dependent CCF 학습 강화
2. Stance에서 발목 stiffness 증가, swing에서 감소 등 생체역학적 패턴 더 명확히 분화
3. Phase-resolved CCF 분석에서 gait cycle 내 임피던스 변화 직접 관찰 가능
4. 사이클 경계 전환 시 부드러운 속도 조절 기대

## 8. 후속 방향

- Gait cycle phase + CCF 상관 분석 (stance vs swing별 임피던스 비교)
- 필요 시 양발 contact 기반 sub-phase 추가 (L heel strike 정보도 포함)
- 주기적 보행 모션 교체: walk-stop → cyclic walking loop (근본적 해결)

# Phase State 구현 계획 (260309)

## 1. 문제 정의

현재 VIC11/VIC_CCF_ON의 핵심 문제:
- `cycle_motion: True` + 단일 모션(4걸음 walk-stop)이 반복
- 두 번째 사이클 시작 시 **reference가 정지 자세로 초기화**되지만 에이전트의 **물리 속도/관성은 유지**
- 결과: 사이클 경계에서 상태 불일치 → bunny hop, 부자연스러운 전환

지금은 에이전트가 obs에서 "지금이 모션의 어느 위치인지" 전혀 알 수 없다. 사이클이 언제 끝나는지 모르기 때문에 사전에 준비(속도 줄이기, 자세 정렬)가 불가능하다.

## 2. Phase State 개요

**Phase = 현재 모션 사이클에서의 위치 (0.0 ~ 1.0)**

```
phase = clamp(curr_time / motion_length, 0, 1)
curr_time = progress_buf * dt + _motion_start_times + _motion_start_times_offset
```

Phase를 obs에 추가하면 에이전트가:
- "phase → 0에 근접" = 사이클 시작 근처 (reference는 정지 자세)
- "phase → 1에 근접" = 사이클 종료 임박 → 속도 줄이고 정지 준비 가능

## 3. 인코딩 방법: sin/cos

Phase 스칼라를 직접 쓰면 1→0 점프 불연속이 있다. sin/cos로 인코딩하면 연속:

```python
phase_sin = sin(2π * phase)   # [N,]
phase_cos = cos(2π * phase)   # [N,]
# phase_obs = [phase_sin, phase_cos]  # [N, 2]
```

obs 추가 차원: +2 (task obs에 concat)

## 4. 구현 방법

### 4-1. 수정 파일

단 하나: `phc/env/tasks/humanoid_im_vic.py`

### 4-2. get_task_obs_size() 오버라이드

```python
def get_task_obs_size(self):
    base_size = super().get_task_obs_size()
    if self._enable_task_obs and self._vic_phase_obs:
        return base_size + 2  # + (sin_phase, cos_phase)
    return base_size
```

`_vic_phase_obs` 플래그는 env yaml에서 `vic_phase_obs: True`로 켜고 끌 수 있게 한다.

### 4-3. _compute_task_obs() 오버라이드

```python
def _compute_task_obs(self, env_ids=None, save_buffer=True):
    task_obs = super()._compute_task_obs(env_ids, save_buffer)

    if not self._vic_phase_obs:
        return task_obs

    if env_ids is None:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    # 현재 모션에서의 시간
    curr_time = (self.progress_buf[env_ids] * self.dt
                 + self._motion_start_times[env_ids]
                 + self._motion_start_times_offset[env_ids])
    motion_len = self._motion_lib._motion_lengths[self._sampled_motion_ids[env_ids]]

    phase = torch.clamp(curr_time / motion_len, 0.0, 1.0)  # [N,]
    phase_obs = torch.stack([
        torch.sin(2 * math.pi * phase),
        torch.cos(2 * math.pi * phase),
    ], dim=-1)  # [N, 2]

    return torch.cat([task_obs, phase_obs], dim=-1)
```

### 4-4. __init__() 파라미터 추가

```python
self._vic_phase_obs = cfg["env"].get("vic_phase_obs", False)
```

### 4-5. env yaml 추가

```yaml
vic_phase_obs: True
```

## 5. 현재 obs 구조 (obs_v=6 기준)

| 구분 | 크기 | 설명 |
| :--- | :--- | :--- |
| self_obs | humanoid 자체 상태 | dof_pos, vel, root 등 |
| task_obs (base) | 4 bodies × 3 samples × 24 = 288 | target bodies의 ref 상태 |
| **phase_obs (추가)** | **2** | **sin/cos(2π×phase)** |

Phase 추가 후 총 obs = 기존 + 2 (처음부터 학습 시 문제 없음. 기존 체크포인트 재사용은 불가.)

## 6. 기대 효과 및 한계

기대 효과:
- 에이전트가 사이클 종료를 사전에 인지하여 정지 준비 가능
- 두 번째 사이클 초반 bunny hop 감소 기대
- 사이클 경계 전환의 부드러움 향상

한계:
- 근본 원인(velocity 불연속)은 해결 안 됨. Phase obs는 에이전트에게 "언제" 문제가 오는지 알려주는 것이지 물리 불연속 자체를 없애지는 않는다.
- 효과의 크기는 에이전트가 phase 정보를 충분히 활용하는지에 달림

## 7. 보완 방향 (Phase State 이후)

Phase State만으로 부족하면 추가 고려:

(A) **Lookahead obs**: 현재 + 미래 N 프레임 reference 상태를 obs에 포함 (이미 `_num_traj_samples=3`으로 일부 구현됨)

(B) **모션 파일 교체**: walk-stop 단발 모션 대신 진짜 주기적 보행 모션(in-place walking loop)으로 교체하면 사이클 경계 velocity 불연속 자체가 없어짐. 근본 해결책이나 데이터 의존적.

(C) **Speed-matching termination**: 사이클 경계 직전 속도가 일정 이하일 때만 pass, 아니면 리셋. Curriculum으로 점차 조건 강화.

## 8. 실험 명칭 및 순서

VIC_CCF_ON 결과 확인 후:
- 사이클 경계 문제가 여전히 뚜렷하면 → VIC_PHASE (vic_phase_obs: True)
- 이미 개선됐으면 → 관찰 후 판단

VIC_PHASE 설정 (예상):
- vic_curriculum_stage: 2 (CCF 유지)
- vic_phase_obs: True
- reward_curriculum_switch_epoch: 10000
- max_epochs: 20000

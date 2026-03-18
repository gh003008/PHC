# VIC 추종 성능 분석: 모션 리타게팅과 체형 불일치 (260315)

---

## 1. 질문

레퍼런스 모션을 완벽히 추종하지 못하는 원인이 에이전트와 모션 데이터의 신체 비율(체형) 차이 때문인가?

---

## 2. 결론: 현재 실험에서는 체형 불일치가 원인이 아니다

현재 사용 중인 모션 파일 확인 결과:

```
파일: sample_data/amass_isaac_walking_forward_single.pkl
beta: [0, 0, 0, ..., 0]  (16차원, 전부 0)
gender: 'neutral'
```

에이전트도 기본적으로 neutral beta (zeros)를 사용하므로, 모션 소스와 에이전트의 체형이 동일하다.
따라서 **체형 불일치는 추종 오차의 원인이 아니다.**

---

## 3. PHC의 모션 리타게팅 구조

### 3-1. 표준 모방학습 (HumanoidIm, HumanoidImVIC)

자동 리타게팅을 수행하지 않는다.

- 모션 데이터의 beta가 포함되어 있지만, forward kinematics 시 에이전트의 skeleton tree를 사용
- 유일한 적응: `fix_trans_height()` — 발이 바닥에 안 박히도록 높이 보정
- 보상 계산에서 `ref_body_pos - body_pos`를 직접 유클리드 거리로 비교
- 팔다리 길이가 다르면 모든 관절 위치를 동시에 정확히 맞추는 것이 원리적으로 불가능

### 3-2. MCP 변형 (HumanoidImMCPDemo)

명시적 limb length scaling을 수행한다.

```python
# humanoid_im_mcp_demo.py:265-272
limb_lengths = []
for i in range(6):
    parent = self.skeleton_trees[0].parent_indices[i]
    if parent != -1:
        limb_lengths.append(np.linalg.norm(ref_rb_pos[:, parent] - ref_rb_pos[:, i], axis=-1))
limb_lengths = np.array(limb_lengths).transpose(1, 0)
scale = (limb_lengths / self.mean_limb_lengths).mean(axis=-1)
ref_rb_pos /= scale[:, None, None]
```

parent-child 간 limb length 비율을 계산해서 reference body position을 스케일링한다.

### 3-3. Shape Variation 모드

`has_shape_variation: True` 설정 시:
- 에이전트의 beta가 랜덤 샘플링됨 (shape_resampling_interval=500 마다)
- 다양한 체형에서 보행을 학습
- 그러나 모션 데이터의 beta와 에이전트의 beta를 매칭하지는 않음

---

## 4. 그러면 추종 오차의 실제 원인은?

현재 beta가 일치하는데도 완벽히 추종하지 못하는 진짜 원인들:

### 4-1. PD 제어의 물리적 한계
- `torque = kp * 2^ccf * (target - current) - kd * 2^ccf * vel`
- PD 제어는 one-step delay가 있고, 관성/중력/접촉 등 외란을 즉시 보상하지 못한다
- action_scale=0.5로 target position 범위가 제한됨

### 4-2. 보상 함수 trade-off
- task_reward (모션 추종) vs disc_reward (AMP 자연스러움) 사이의 균형
- 현재 task:disc = 0.5:0.5 또는 curriculum으로 변동
- AMP discriminator가 "자연스러운 보행"을 요구하면, 레퍼런스와 약간 달라도 자연스러운 방향으로 학습될 수 있다

### 4-3. Power penalty
- `power_coefficient: 0.000002` — 에너지 소모에 패널티
- 큰 토크가 필요한 구간에서 절약하려는 방향으로 학습 → 추종 정확도 저하

### 4-4. Random State Initialization
- `stateInit: "Random"` — 에피소드 시작 시 모션의 랜덤 시점에서 초기화
- 초기 상태와 레퍼런스 간 불일치에서 회복하는 과정에서 추종 오차 발생

### 4-5. 시뮬레이션 해상도
- controlFrequencyInv=2, substeps=2 → 30Hz 제어, 60Hz 물리
- 빠른 동작(발목 충돌, 무릎 굴신)에서 시간 해상도 부족 가능

---

## 5. VIC에 MCP 스타일 limb scaling 구현이 필요한가?

### 현재: 불필요

현재 모션 데이터(neutral beta)와 에이전트(neutral beta)가 동일 체형이므로 스케일링 효과 없음.

### 미래: 조건부 필요

다음 경우에 필요해진다:

1. **다양한 체형의 모션 데이터 사용 시** (AMASS 전체 데이터셋 등) — 각 모션마다 다른 beta
2. **Shape Variation 학습 시** — 에이전트 beta가 랜덤으로 바뀜
3. **WalkON + SMPL 통합 시** — 엑소에 맞춰 SMPL beta를 조정하면 모션과 체형 불일치 발생

### 구현 방향

만약 필요해지면 MCP 방식보다는 더 정확한 방법이 좋다:
- MCP는 6개 body만으로 전체 스케일 1개를 계산 (거친 근사)
- 관절별 limb length ratio로 per-joint scaling이 더 정확
- 또는 모션 데이터의 beta로 FK → 에이전트 beta로 재FK하는 retargeting pipeline 구축

---

## 6. 요약

| 항목 | 상태 |
|---|---|
| 모션 데이터 beta | [0, ..., 0] (neutral) |
| 에이전트 beta | [0, ..., 0] (neutral) |
| 체형 불일치 여부 | 없음 |
| 리타게팅 구현 | 표준 모방학습에는 없음, MCP에만 있음 |
| 추종 오차 원인 | PD 제어 한계, 보상 trade-off, power penalty, random init |
| VIC에 scaling 추가 필요 | 현재 불필요, 다른 체형 모션 사용 시 필요 |

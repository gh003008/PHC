# VIC07 구현 계획: Action Rate Penalty + 상체 ROM 제한 (260302)

## 1. 목표

VIC06 (CCF gradient noise 제거)에 추가로 두 가지 구조적 수정을 적용한다.
(A) Action rate penalty: 지터링 해결, 저주파 액션 유도
(B) 상체 ROM 제한: 불필요한 상체 자유도 억제, 하지 학습에 집중

## 2. 현재 문제 요약

VIC03~05에서 reward weight를 다양하게 조정했으나 결과가 거의 동일 (av reward ~220, av steps ~70).
시각화 관찰: 일부 에이전트는 걷지만, 다수가 서서 떨림(jittering). 몸 전체가 고주파로 진동.
reward tuning으로는 해결 불가능한 구조적 문제.

## 3. 수정 (A): Action Rate Penalty

### 3-1. 개념

연속된 타임스텝의 액션 차이를 패널티로 부여한다.
penalty = -coefficient * sum((a_t - a_{t-1})^2)

이렇게 하면 네트워크가 부드러운 저주파 액션을 출력하도록 유도된다.
기존 power_reward(= |torque x velocity|)는 "힘을 적게 쓰라"는 의미이고,
action rate penalty는 "급격하게 변하지 마라"는 의미로, 성격이 다르다.

### 3-2. 참고 코드

humanoid_teleop.py에 이미 동일한 구현이 존재한다:

```python
# humanoid_teleop.py:40
self.last_actions = torch.zeros_like(self.actions)

# humanoid_teleop.py:159 (매 스텝 업데이트)
self.last_actions[:] = self.actions[:]

# humanoid_teleop.py:272-274
def _reward_action_rate(self):
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

### 3-3. 구현 방안

humanoid_im_vic.py에 추가할 내용:

(1) __init__에서 초기화:
    self.last_actions = None  # 첫 스텝 처리용
    self._action_rate_penalty_w = cfg["env"].get("action_rate_penalty_w", 0.0)

(2) pre_physics_step에서 last_actions 업데이트:
    if self.last_actions is None:
        self.last_actions = torch.zeros_like(self.actions)
    (기존 actions 처리 후)
    -> 스텝 끝에 self.last_actions[:] = self.actions[:]

(3) _compute_reward에서 penalty 적용:
    if self._action_rate_penalty_w > 0:
        action_diff = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        action_rate_penalty = -self._action_rate_penalty_w * action_diff
        self.rew_buf[:] += action_rate_penalty

(4) env_im_walk_vic.yaml에 파라미터 추가:
    action_rate_penalty_w: 0.005  (초기값, 튜닝 필요)

주의: last_actions 업데이트 타이밍이 중요하다.
  - _compute_reward에서 penalty를 계산한 후에 last_actions를 업데이트해야 한다
  - pre_physics_step에서 먼저 업데이트하면 a_t - a_t = 0이 되어 의미 없음
  - 순서: pre_physics_step(actions 저장) -> physics_step -> post_physics_step(_compute_reward에서 penalty 계산) -> last_actions 업데이트

올바른 구현 위치:
  - last_actions 업데이트는 _compute_reward 마지막 또는 post_physics_step 마지막에서 수행

### 3-4. coefficient 선정 기준

action_rate_penalty_w 값은 다른 reward 항목과의 스케일을 고려해야 한다.
- 138차원 액션, 각 차원의 변화량이 ~0.1 정도라면: sum(0.1^2 * 138) = 1.38
- penalty = -0.005 * 1.38 = -0.0069 (매우 작음)
- imitation reward는 0~1 범위이므로, 0.005~0.01 정도에서 시작하여 효과를 확인

## 4. 수정 (B): 상체 ROM 제한

### 4-1. 개념

보행 과제에서 상체(팔, 손, 목, 머리)는 자연스럽게 흔들리기만 하면 된다.
현재 상체 관절이 넓은 ROM을 가지고 있어, 네트워크가 비정상적인 상체 자세를 취하면서 균형을 잡으려 할 수 있다.
상체 ROM을 실제 보행 시 범위로 제한하면 학습 탐색 공간이 줄어들고, 하지 제어에 집중할 수 있다.

### 4-2. SMPL body 구조

전체 24개 body (Pelvis 포함), 23개 body에 각 3 DOF = 69 DOF.
_dof_names = body_names[1:] (Pelvis 제외):

하지 (보행 핵심, 8개 body x 3 DOF = 24 DOF):
  L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe

몸통 (3개 body x 3 DOF = 9 DOF):
  Torso, Spine, Chest

상체 (12개 body x 3 DOF = 36 DOF):
  Neck, Head, L_Thorax, L_Shoulder, L_Elbow, L_Wrist, L_Hand,
  R_Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand

### 4-3. 구현 방안

ROM 제한은 IsaacGym의 dof_prop에서 lower/upper limit을 수정하거나,
_action_to_pd_targets 출력을 클램핑하는 방식으로 구현할 수 있다.

방법 1: MJCF XML에서 상체 관절의 range를 제한 (가장 깔끔하지만 원본 수정)
방법 2: _compute_torques에서 pd_tar을 클램핑 (VIC 코드에서만 수정, 규칙 준수)

방법 2 구현:
  - 상체 관절 인덱스 목록을 정의 (Neck, Head, Thorax, Shoulder, Elbow, Wrist, Hand)
  - _compute_torques에서 pd_tar의 상체 부분을 좁은 범위로 클램핑
  - 또는 더 간단하게: 상체의 action_scale을 줄여서 상체 동작 범위 자체를 축소

방법 3: bounds_loss_coef 활용
  - 이미 learning config에 bounds_loss_coef: 10이 있음
  - 이는 네트워크 출력(mu)이 [-1, 1] 범위를 넘으면 L2 패널티를 주는 것
  - 상체에 특화된 것은 아니지만, 전체적인 액션 범위 제한에 기여

추천: 방법 2 (pd_tar 클램핑). VIC 규칙(기존 PHC 코드 미수정)을 준수하면서 구현 가능.

### 4-4. 상체 ROM 값 결정

보행 시 상체 관절의 자연스러운 범위를 정해야 한다.
- Neck/Head: 거의 안 움직임 (+-10도 정도)
- Shoulder: 팔 흔들기 범위 (전후 +-30도, 좌우 +-10도)
- Elbow: 살짝 굽힘 (0~30도)
- Wrist/Hand: 거의 안 움직임 (+-5도)
- Thorax: 약간의 회전 (+-10도)

구체적인 값은 reference motion 데이터에서 상체 관절 범위를 추출하여 결정하는 것이 가장 정확하다.
또는 보수적으로 현재 dof_limits의 50% 범위로 제한하는 것도 방법이다.

## 5. VIC07 전체 변경 요약

VIC06 대비 변경:
(1) CCF gradient noise 제거 (VIC06에서 이미 적용)
(2) Action rate penalty 추가 (지터링 해결)
(3) 상체 ROM 제한 (학습 공간 축소)

Config 변경 (env_im_walk_vic.yaml):
  action_rate_penalty_w: 0.005 (신규)
  upper_body_rom_scale: 0.5 (신규, 또는 개별 관절별 설정)

코드 변경 (humanoid_im_vic.py):
  - __init__: last_actions, action_rate_penalty_w, 상체 인덱스 초기화
  - pre_physics_step: CCF zeroing 유지
  - _compute_reward: action rate penalty 추가, last_actions 업데이트
  - _compute_torques: 상체 pd_tar 클램핑

## 6. 구현 순서

1. VIC06 결과 확인 (CCF fix 효과 검증)
2. action rate penalty 구현 및 적용
3. 상체 ROM 제한 구현 및 적용
4. VIC07 학습 및 평가

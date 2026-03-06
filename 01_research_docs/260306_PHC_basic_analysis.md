# PHC 기본 스크립트 분석 (260306)

원본 PHC 코드베이스의 State / Action / Reward / Termination / 모델 / AMP / 트릭들을 정리.
기준 파일: humanoid_amp.py, humanoid_im.py, amp_agent.py, amp_network_builder.py, im_amp.py

---

## 1. State (관측 공간)

### 1-1. Self Observation (body state)

`_compute_humanoid_obs()`에서 계산. SMPL 휴머노이드 기준.

| 항목 | 차원 | 설명 |
|:--|:--:|:--|
| Root height (z) | 1 | 루트 절대 높이 |
| Root rotation | 6 | tan-norm 6D 표현 (quaternion → 6D rotation) |
| Root linear velocity | 3 | 로컬 프레임 (heading 방향 기준) |
| Root angular velocity | 3 | 로컬 프레임 |
| Body positions | 3 × N_body | 루트 기준 상대 위치 (global - root_pos) |
| Body rotations | 6 × N_body | tan-norm 6D 표현 |
| Body velocities | 3 × N_body | 각 body의 선속도 |
| Body angular velocities | 3 × N_body | 각 body의 각속도 |
| DOF positions | 69 | 23개 관절 × 3 DOF |
| DOF velocities | 69 | 23개 관절 × 3 DOF |
| (Optional) SMPL betas | 10 | 체형 파라미터 (_has_shape_obs=True 시) |
| (Optional) Limb weights | 10 | 팔다리 질량 파라미터 (_has_limb_weight_obs=True 시) |

SMPL body 수: 24개 (Pelvis 포함), 관절 수: 23개 (Pelvis 제외).
`local_root_obs: True` → root vel/ang_vel을 world 대신 local heading 프레임으로 표현.
`root_height_obs: True` → root height 포함.

### 1-2. Task Observation (imitation target 추종)

`obs_v: 6` 기준. track bodies에 대해 현재 상태와 reference motion 간의 차이를 observation에 포함.

| 항목 | 차원 | 설명 |
|:--|:--:|:--|
| 위치 차이 | 3 × 4 | ref_pos - curr_pos (track bodies 4개) |
| 속도 차이 | 3 × 4 | ref_vel - curr_vel |
| 회전 차이 | 6 × 4 | 상대 rotation (tan-norm 6D) |

`trackBodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]` → 4개 body, 총 48 dims.
`fut_tracks: False` → 미래 reference 샘플 없이 현재 시점 1개만 사용.

### 1-3. AMP Observation (discriminator 입력)

`numAMPObsSteps: 10` → 10 타임스텝 concatenate.

per-step AMP obs 구성 (SMPL 기준):
| 항목 | 차원 |
|:--|:--:|
| Root height | 1 |
| Root rotation (quat) | 4 |
| Root linear velocity | 3 |
| Root angular velocity | 3 |
| DOF positions | 69 |
| DOF velocities | 69 |
| Key body velocities (4개 × 3) | 12 |
| **per-step 합계** | **161** |

총 AMP obs = 161 × 10 = **1610 dims**.

### 1-4. Observation Normalization

- policy input과 AMP obs 모두 running mean/std로 normalize.
- `normalize_input: True`, `normalize_amp_input: True`

---

## 2. Action (행동 공간)

### 2-1. 기본 설정

- **차원**: 69 (SMPL 23관절 × 3 DOF)
- **제어 방식**: PD control (`control_mode: "pd"`)
- **제어 주파수**: 30 Hz (`controlFrequencyInv: 2` → 60Hz 시뮬레이션 ÷ 2)
- **action_scale**: 0.5

### 2-2. PD Target 변환

```
pd_tar = pd_action_offset + pd_action_scale * (action * action_scale)
```

- `pd_action_offset` = 0.5 × (dof_upper_limit + dof_lower_limit) → 관절 범위의 중심
- `pd_action_scale`  = 0.5 × (dof_upper_limit - dof_lower_limit) → 관절 범위의 절반

action ∈ [-1, 1] → pd_tar은 [lower_limit, upper_limit] 내에서 움직임.
`action_scale=0.5`가 추가로 곱해지므로 실제 탐색 범위는 관절 범위의 ±50%.

### 2-3. 특이사항

- **무릎 관절 (Knee) 스케일 부스트**: knee joints의 pd_action_scale을 5배 적용 (코드 내 하드코딩).
  → 무릎은 굽힘이 중요한 관절이라 action 영향력을 강제로 키움.
- **SMPL MJCF의 PD gains**: MJCF XML의 `user` 어트리뷰트에 kp/kd가 저장됨.
  IsaacGym의 dof_prop stiffness/damping은 0으로 설정되어 있어 직접 파싱해서 사용해야 함.
  (VIC 코드에서 발견한 버그로, PHC 기본 코드는 motorized actuator를 통해 처리)

### 2-4. Bounds Loss (PPO 정규화)

`bounds_loss_coef: 10` → 네트워크 출력 mu가 [-1, 1]을 넘으면 L2 패널티.
policy가 극단적인 action 출력을 하지 않도록 부드럽게 제약.

---

## 3. Reward

### 3-1. Imitation Reward

`humanoid_im.py`의 `_compute_reward()`에서 계산. track bodies 기준.

```
r_pos     = exp(-k_pos     × mean(||ref_pos - curr_pos||²))
r_rot     = exp(-k_rot     × mean(||Δangle||²))        # 쿼터니언 각도 차이
r_vel     = exp(-k_vel     × mean(||ref_vel - curr_vel||²))
r_ang_vel = exp(-k_ang_vel × mean(||ref_ang_vel - curr_ang_vel||²))

imitation_reward = w_pos × r_pos + w_rot × r_rot + w_vel × r_vel + w_ang_vel × r_ang_vel
```

현재 설정값:
| 파라미터 | 값 | 역할 |
|:--|:--|:--|
| k_pos | 200 | 위치 차이에 민감 (작은 차이도 큰 패널티) |
| k_rot | 10 | 회전 차이 |
| k_vel | 1.0 | 속도 차이 |
| k_ang_vel | 0.1 | 각속도 차이 |
| w_pos | 0.4 | 위치 가중치 |
| w_rot | 0.3 | 회전 가중치 |
| w_vel | 0.2 | 속도 가중치 |
| w_ang_vel | 0.1 | 각속도 가중치 |

**Range**: 각 r_* ∈ [0, 1] → imitation_reward ∈ [0, 1].

### 3-2. Discriminator (AMP) Reward

`amp_agent.py`에서 계산.

```
disc_logit = discriminator.forward(amp_obs)
prob = sigmoid(disc_logit)                      # 얼마나 "real"처럼 보이는가
disc_reward = -log(max(1 - prob, 0.0001))       # CrossEntropy 형태
disc_reward *= disc_reward_scale                # scale = 2.0
```

agent가 "real"하게 보일수록 prob → 1, disc_reward → ∞.
실제로는 agent가 reference motion처럼 자연스럽게 움직이면 높은 reward.

### 3-3. Power (Energy) Reward

```
power = sum(|dof_force × dof_vel|)             # 전체 관절 소비 에너지
power_reward = -power_coefficient × power
power_coefficient = 0.000005                   # VIC09 기준 (기본: 0.00005)

power_reward[progress_buf ≤ 3] = 0            # 처음 3프레임은 제외
```

에너지를 많이 쓸수록 패널티. 떨림(jittering) 억제 역할.

### 3-4. Total Reward 조합

```
total_reward = task_reward_w × imitation_reward + disc_reward_w × disc_reward + power_reward
            = 0.5 × imitation_reward + 0.5 × disc_reward + power_reward
```

`task_reward_w: 0.5`, `disc_reward_w: 0.5`.

### 3-5. zero_out_far: 거리 기반 reward zeroing

`zero_out_far: True` 시, agent가 reference position에서 너무 멀면 imitation reward를 0으로:

```
if ||root_pos - ref_root_pos|| > terminationDistance (0.25m):
    imitation_reward = 0
    → direction-based goal reward로 대체 (거리 감소분 × scale)
```

학습 초반 agent가 넘어지더라도 reference 방향으로 이동하면 보상을 받도록 설계.

---

## 4. Termination (종료 조건)

### 4-1. Tracking Distance Termination (주요)

```python
terminationDistance: 0.25    # meters
# track bodies의 평균 또는 최대 거리가 threshold 초과 시 reset
has_fallen = mean(||body_pos - ref_body_pos||) > 0.25
```

### 4-2. Height-based Termination

```python
terminationHeight: 0.15      # 머리 높이가 이 이하로 떨어지면 종료
```

fall_contact (비접촉 body가 지면 접촉) AND fall_height (높이 미달) 동시 조건.

### 4-3. Episode Length Termination

```python
episodeLength: 300            # 30Hz에서 10초
reset: progress_buf >= episode_length - 1
```

### 4-4. Reset 전략

`stateInit: "Random"`, `hybridInitProb: 0.5`

- 50% 확률: 에피소드 시작 시 motion의 랜덤 시간점에서 초기화 (Reference State Initialization, RSI)
- 50% 확률: 일반 초기 자세에서 시작
- RSI는 에이전트가 reference motion의 다양한 위상을 경험하게 하여 학습 안정성 향상.

`recoveryEpisodeProb: 0.5`, `fallInitProb: 0.3`
- 50% 확률로 recovery 에피소드 (넘어진 상태에서 시작)
- fallInitProb: 30% 확률로 임의 낙하 위치에서 시작

---

## 5. 모델 구조

### 5-1. Policy Network (Actor)

```
Input: self_obs + task_obs
MLP: [1024, 1024, 512, 512] units, ReLU activation
Output: Gaussian distribution (mu, fixed sigma)
  sigma = exp(-2.9) ≈ 0.055 (fixed_sigma: True, learn_sigma: False)
```

`separate: True` → actor와 critic을 독립된 네트워크로 분리.

### 5-2. Value Network (Critic)

Actor와 동일한 MLP 구조 `[1024, 1024, 512, 512]`.

### 5-3. Discriminator

```
Input: AMP obs (1610 dims = 161 per step × 10 steps)
MLP: [1024, 512] units, ReLU activation
Output: 1 scalar logit (real=positive / fake=negative)
```

### 5-4. 초기화

`sigma_init: const_initializer, val: -2.9`
→ 초기 action std가 매우 작아 (≈0.055), 학습 초반 탐색이 제한적.
→ 하지만 reference motion에서 크게 벗어나지 않아 초반 학습 안정성 확보.

---

## 6. AMP (Adversarial Motion Prior) 설정

### 6-1. Discriminator Loss

```python
# Binary Cross Entropy
loss_agent = BCE(disc_logit_agent, label=0)    # agent는 "fake" (0)
loss_demo  = BCE(disc_logit_demo,  label=1)    # demo는 "real" (1)
disc_loss  = 0.5 × (loss_agent + loss_demo)

# Regularization 항목들:
+ disc_logit_reg    × sum(disc_logit_weights²)  # = 0.01 × weight_reg
+ disc_grad_penalty × mean(||∇disc(demo_obs)||²) # = 5 × grad_penalty
+ disc_weight_decay × sum(disc_weights²)         # = 0.0001 × L2
```

총 discriminator loss = disc_coef × disc_loss + 정규화
`disc_coef: 5`.

### 6-2. Buffer 구성

```yaml
amp_obs_demo_buffer_size: 200000     # reference motion에서 샘플한 real obs 저장
amp_replay_buffer_size: 200000       # agent가 생성한 fake obs 저장
amp_replay_keep_prob: 0.01           # 각 스텝에서 replay buffer에 저장할 확률

amp_batch_size: 512                  # disc 학습용 배치 (real + fake)
amp_minibatch_size: 2048             # PPO와 함께 처리되는 미니배치
```

`amp_replay_keep_prob: 0.01`은 매 스텝을 1% 확률로만 저장 → 다양한 시간대의 agent 경험 유지.

### 6-3. Motion Demo Sampling

- demo obs는 reference pkl에서 random time sampling으로 수집.
- `numAMPObsSteps: 10` → 연속 10스텝의 obs를 함께 저장.
- 단일 모션 사용 시 (VIC09): disc가 한 가지 스타일만 학습하면 되므로 수렴이 빠를 것으로 기대.

---

## 7. 특별한 트릭들

### 7-1. zero_out_far (거리 기반 reward 전환)

```python
zero_out_far: True
zero_out_far_train: False  # 학습 중에는 비활성화 (eval에서만)
terminationDistance: 0.25
```

agent가 reference에서 멀어지면 imitation reward → 0 → direction goal reward.
학습 초반 "어디로든 가려는 시도"를 유지시키는 역할.

### 7-2. hard_negative Mining

```python
hard_negative: False  # 현재 비활성화
```

활성화 시: termination이 자주 발생하는 모션을 더 자주 샘플링.
어려운 모션에 집중적으로 학습 기회를 부여하는 전략.

### 7-3. cycle_motion

```python
cycle_motion: True
```

에피소드가 motion 길이보다 짧을 때, motion을 반복(cyclic)으로 제공.
보행처럼 반복 동작에 적합. 에이전트가 reference와 동기화를 유지하도록 도움.

### 7-4. Shape Resampling

```python
shape_resampling_interval: 500  # 에폭 단위
```

주기적으로 SMPL body shape (beta 파라미터)를 재샘플링.
다양한 체형 (키, 체중, 팔다리 비율)에서 학습하여 범용성 확보.

### 7-5. Hybrid State Initialization (RSI)

```python
stateInit: "Random"
hybridInitProb: 0.5
```

Reference State Initialization (RSI): motion의 임의 시간점에서 시작.
→ 에이전트가 "어느 위상에서도 시작해서 계속 따라갈 수 있도록" 학습.
→ 에피소드가 항상 처음 자세에서 시작하면, 나중 phase를 학습하기 어렵다는 문제 해결.

### 7-6. Getup Schedule

```python
getup_schedule: False
getup_udpate_epoch: 1000
recoverySteps: 90
```

넘어진 후 일어나는 동작을 스케줄에 따라 학습.
활성화 시: 초반에는 recovery episode 없이 시작, 에폭이 지남에 따라 recovery episode 도입.

### 7-7. normalize_advantage

```python
normalize_advantage: True
```

PPO의 advantage를 배치 내에서 정규화. 학습 안정성 향상.

### 7-8. Observation Normalization (Running Stats)

```python
normalize_input: True
normalize_amp_input: True
normalize_value: True
```

모든 입력과 value function output을 running mean/std로 정규화.
reward scale이 달라도 안정적으로 학습 가능.

### 7-9. auto_pmcp (Progressive Motion Clip Pruning)

```python
auto_pmcp: False
auto_pmcp_soft: True
```

학습 중 에이전트가 잘 따라가지 못하는 모션 클립을 점진적으로 제거.
soft 버전: 완전히 제거하지 않고 sampling 확률만 낮춤.

### 7-10. Gradient Clipping

```python
truncate_grads: True
grad_norm: 50.0
```

gradient norm이 50을 초과하면 clip. 폭발적 gradient 방지.

---

## 8. 학습 설정 (PPO + AMP)

```yaml
algo: im_amp                 # AMP + Imitation 복합 알고리즘
gamma: 0.99                  # discount factor
tau: 0.95                    # GAE lambda (advantage estimation)
learning_rate: 5e-5          # 매우 작은 학습률 (안정성 우선)
lr_schedule: constant        # 학습률 고정

horizon_length: 32           # rollout 길이 (매우 짧음, 512 env × 32 = 16384 samples/update)
minibatch_size: 2048         # PPO minibatch
mini_epochs: 6               # PPO epoch 수

e_clip: 0.2                  # PPO clipping parameter
critic_coef: 5               # value loss 가중치
entropy_coef: 0.0            # entropy bonus 없음

max_epochs: 15000            # 총 학습 에폭
save_frequency: 100          # 체크포인트 저장 주기
save_best_after: 50          # 50 에폭 이후부터 best 체크포인트 저장
```

---

## 9. 파일 구조 요약

| 파일 | 역할 |
|:--|:--|
| `phc/env/tasks/humanoid_amp.py` | AMP obs 계산, disc obs 구성, AMP reward |
| `phc/env/tasks/humanoid_amp_task.py` | Task obs 추가, task reward 계산 |
| `phc/env/tasks/humanoid_im.py` | Imitation reward, reset 로직, motion sampling |
| `phc/env/tasks/humanoid_im_vic.py` | VIC 전용: CCF 처리, PD gains 파싱 |
| `phc/learning/im_amp.py` | 학습 entry point, AMP + PPO 루프 |
| `phc/learning/amp_agent.py` | Discriminator loss, replay buffer 관리 |
| `phc/learning/amp_network_builder.py` | Actor/Critic/Disc 네트워크 구조 |
| `phc/utils/motion_lib_smpl.py` | AMASS 모션 데이터 로딩 및 sampling |

---

## 10. VIC vs 기본 PHC 차이점

| 항목 | 기본 PHC (V4) | VIC (VIC08/09) |
|:--|:--|:--|
| Action 차원 | 69 | 138 (q_target 69 + CCF 69) |
| CCF 사용 | 없음 | Stage 1: 0 고정, Stage 2: 학습 |
| PD gains | IsaacGym dof_prop 직접 | MJCF XML 파싱 (버그 수정) |
| 성능 (15k 에폭) | av reward ~461, steps ~143 | av reward ~200~254, steps ~65~80 |
| 학습 난이도 | 상대적으로 쉬움 | CCF로 인한 학습 공간 확장으로 어려움 |

# PHC-Pain v1.5 Training and Reward Design Notes

작성일: 2026-04-30

이 문서는 PHC-Pain v1.5에서 수행한 주요 학습 trial들이 각각 어떤 설정으로 학습되었고, reward와 pain cost를 어떻게 구성했는지 정리한다. 정량 요약은 `trial_metric_summary.csv`, 시각 비교 영상은 `videos/` 폴더를 함께 본다.

## 공통 학습 프레임워크

모든 v1.5 trial은 PHC의 `im_pnn` 학습 프레임워크를 사용했다.

- 학습 task: `env=env_im_pain_v1`
- policy: PHC PNN imitation policy
- primitive 수: `env.num_prim=4`
- humanoid: `robot=smpl_humanoid_shape`
- motion set: `sample_data/amass_isaac_walking_forward_subset23.pkl`
- 병렬 환경 수: `env.num_envs=256`
- rollout horizon: `learning.params.config.horizon_length=32`
- minibatch: `learning.params.config.minibatch_size=8192`
- PPO mini epochs: `learning.params.config.mini_epochs=4`
- GPU: Slurm `idx3`, `--mem=15G`
- pain mode: `env.pain.mode=reward_only`
- active impaired side: `env.pain.active_knee_side=right`
- pain observation: `env.pain.obs.enabled=True`

기본 reward는 기존 PHC imitation/task reward를 유지하고, 여기에 right-knee pain cost를 subtract하는 구조다.

```text
reward = PHC_imitation_reward - pain_reward_cost - optional_contact_regularizer
```

여기서 `learning.params.config.disc_reward_w`와 `learning.params.config.task_reward_w`는 PHC 원래 reward 쪽의 imitation/style vs task 계열 tradeoff를 조절한다. Pain trial에서는 이 값을 낮추거나 유지하면서, pain term이 policy 행동을 바꿀 수 있을 만큼 영향력을 주는지를 탐색했다.

## Pain State and Reward Definition

v1.5의 핵심은 오른쪽 무릎 통증을 단순 raw torque가 아니라 OA-style knee load proxy로 정의한 점이다.

### Knee Load Proxy

각 side의 knee load는 다음 성분들로 구성된다.

```text
knee_load =
    w_contact_load * contact_load
  + w_moment_load  * moment_load
  + w_legacy_torque_load * torque_load
```

현재 v1.5 주요 trial에서는 `legacy_torque_load`의 비중은 0에 가깝거나 사용하지 않고, contact/moment 계열을 중심으로 보았다.

구성 요소:

- `contact_load`: 발/발목 contact force, compression, loaded flexion, loading-rate 기반 proxy
- `moment_load`: GRF line-of-action과 knee 위치로부터 추정한 KAM/KFM 계열 proxy
- `KAM`: frontal-plane knee adduction moment surrogate
- `KFM`: sagittal-plane knee flexion moment surrogate
- `torque_load`: 이전 torque/work 기반 legacy proxy

주의: KAM/KFM은 실제 force plate + anatomical joint contact force가 아니라, IsaacGym contact force와 PHC rigid-body geometry에서 만든 synthetic surrogate다. 따라서 임상적 절대값이 아니라, 같은 시뮬레이션 조건 내 상대 비교 지표로 사용해야 한다.

### Pain Drive and Pain State

right knee load가 threshold를 넘으면 pain drive가 생긴다.

```text
drive_right = right_sensitivity * relu(right_knee_load - right_threshold)
drive_left  = 0   # active side가 right인 trial 기준
```

이 drive는 body-part pain state로 누적된다.

```text
pain_state_t = clamp(
    pain_state_{t-1} * (1 - decay) + rise * drive_t,
    0,
    pain_cap
)
```

별도로 `pain_body_memory`도 exponential moving average로 저장된다. Policy observation에는 body-part pain state/memory가 붙기 때문에, policy가 pain을 단순 penalty가 아니라 상태로 볼 수 있다.

### Pain Reward Cost

기본 pain reward cost는 다음과 같다.

```text
pain_reward_cost = lambda_p * right_knee_pain_state * pain_gate
reward -= pain_reward_cost
```

초기 trial에서는 `pain_gate = 1`이었다. `walkgate` 이후부터는 reference motion의 root speed를 보고 standing 구간에서는 pain penalty를 거의 꺼서, 시작/끝 standing pose에서 발을 들어버리는 꼼수를 줄였다.

```text
walk_gate = clamp((ref_speed - min_ref_speed) / transition_width, 0, 1)
pain_gate = standing_scale + (1 - standing_scale) * walk_gate
```

사용한 gated 설정:

- `min_ref_speed=0.35`
- `transition_width=0.20`
- `standing_scale=0.03`

### Contact Regularizer

Aggressive pain에서는 오른발을 거의 안 딛는 hopping/non-contact cheat가 나왔다. 이를 막기 위해 `walkgate` 이후에는 reference right-stance phase에서 최소 contact force를 요구하는 regularizer를 추가했다.

```text
contact_deficit = relu(min_force - right_foot_force) / min_force
contact_penalty = weight * stance_gate * clamp(contact_deficit, 0, max_penalty)
reward -= contact_penalty
```

`stance_gate`는 reference right foot이 낮고 느릴 때 켜진다.

- `ref_stance_height=0.08`
- `ref_stance_speed=0.35`

핵심 의도는 “오른발을 절대 들지 말라”가 아니라, “reference stance 중 오른발을 완전히 버리는 cheat는 막되, 무릎 moment를 줄이는 보상 전략은 허용한다”였다.

## Trial 0: PHC Pretrained Baseline

### Experiment

- exp: `phc_shape_pnn_iccv`
- checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth`
- pain: 없음
- 역할: 모든 trial의 visual/metric 기준

### Purpose

PHC pretrained가 walking motion을 정상적으로 따라가는지 확인하고, 이후 pain fine-tuning 결과가 단순 tracking collapse인지, 실제 보상 전략인지 비교하기 위한 기준으로 사용했다.

### Result

Viewer에서 walking을 잘 따라가는 것을 확인했다. Overlay 영상에서는 파란색 모델로 사용한다.

## Trial 1: v1.5 OA Sanity / Long

### Experiment

- exp: `phc_shape_pnn_iccv_pain_v15_oa_sanity_idx3`
- scripts:
  - `sbatch_pain_v15_sanity_idx3.sh`
  - `sbatch_pain_v15_long_idx3.sh`
- seed checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth`
- first sanity time limit: `00:30:00`
- long time limit: `12:00:00`

### Reward / Pain Parameters

```text
env.pain.lambda_p=0.05
env.pain.mode=reward_only
env.pain.obs.enabled=True
env.pain.active_knee_side=right
```

No explicit override was given for:

```text
right_sensitivity
right_threshold
w_contact_load
w_moment_load
w_kam
w_kfm
reward_gating
contact_regularizer
```

So this was the first weak-pain OA proxy run using the v1.5 default knee mechanism.

### Training Behavior

This run verified that the OA knee load proxy and pain logging were wired correctly. It also showed that even a weak pain reward can reduce the logged proxy without visibly changing the gait much.

Key scalar trend:

```text
reward:                 60.77 -> 65.77
episode length:         90.12 -> 91.86
pain reward cost:        0.052 -> 0.037
right knee load:         0.338 -> 0.290
right knee moment load:  0.320 -> 0.229
right KAM:               0.319 -> 0.226
right KFM:               0.321 -> 0.235
right knee tau RMS:     16.20  -> 15.15
```

### Interpretation

This was not a failure. It taught the key lesson:

```text
OA knee load proxy is trainable, but lambda_p=0.05 is too weak to create visible compensation.
```

The model found small internal reductions while preserving the original PHC gait. That motivated stronger pain weights and more explicit tradeoff experiments.

## Trial 2: Aggressive Pain

### Experiment

- exp: `phc_shape_pnn_iccv_pain_v15_oa_aggressive_idx3`
- script: `sbatch_pain_v15_aggressive_idx3.sh`
- seed checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth`

### Reward / Pain Parameters

```text
env.pain.lambda_p=1.0
env.pain.knee_mechanism.right_sensitivity=6.0
env.pain.knee_mechanism.right_threshold=0.08
env.pain.knee_mechanism.w_contact_load=0.30
env.pain.knee_mechanism.w_moment_load=0.70
env.pain.knee_mechanism.w_kam=0.80
env.pain.knee_mechanism.w_kfm=0.20
env.pain.knee_mechanism.memory_alpha=0.12
learning.params.config.disc_reward_w=0.15
learning.params.config.task_reward_w=0.85
```

No reward gating or contact regularizer was used.

### Rationale

The previous weak-pain run moved metrics but not visible behavior. This trial intentionally made pain strong enough to see if an obvious compensatory gait would emerge.

### Training Behavior

Key scalar trend:

```text
reward:                -183.16 -> 21.06
episode length:          90.87 -> 63.09
pain reward cost:         2.740 -> 0.092
right knee load:          0.334 -> 0.003
right knee moment load:   0.326 -> 0.003
right KAM:                0.326 -> 0.003
right KFM:                0.327 -> 0.003
right knee tau RMS:      15.36  -> 18.78
```

### Visual Result

The model reduced right knee load almost to zero, but it did so by avoiding right-leg contact. Viewer inspection showed hopping/right-leg non-contact behavior rather than a clinically useful limping-like gait.

### Interpretation

This trial established a hard constraint for later reward design:

```text
Pain pressure alone is not enough. It creates a non-contact cheat.
```

The next trials needed a mechanism that allows partial compensation but prevents the model from simply lifting or abandoning the painful leg.

## Trial 3: Middle Pain

### Experiment

- exp: `phc_shape_pnn_iccv_pain_v15_oa_middle_idx3`
- script: `sbatch_pain_v15_middle_idx3.sh`
- seed checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth`

### Reward / Pain Parameters

```text
env.pain.lambda_p=0.25
env.pain.knee_mechanism.right_sensitivity=3.0
env.pain.knee_mechanism.right_threshold=0.15
env.pain.knee_mechanism.w_contact_load=0.45
env.pain.knee_mechanism.w_moment_load=0.55
env.pain.knee_mechanism.w_kam=0.75
env.pain.knee_mechanism.w_kfm=0.25
env.pain.knee_mechanism.memory_alpha=0.08
learning.params.config.disc_reward_w=0.30
learning.params.config.task_reward_w=0.70
```

No reward gating or contact regularizer was used.

### Rationale

This trial was the midpoint between weak v1.5 OA sanity and overly aggressive pain. It reduced pain pressure enough to avoid immediate hopping collapse while keeping pain strong enough to affect the policy.

### Training Behavior

Key scalar trend:

```text
reward:                  9.49 -> 24.53
episode length:         90.23 -> 90.93
pain reward cost:        0.619 -> 0.405
right knee load:         0.330 -> 0.138
right knee moment load:  0.315 -> 0.118
right KAM:               0.315 -> 0.116
right KFM:               0.315 -> 0.122
right knee tau RMS:     16.01  -> 16.52
```

### Visual Result

Walking looked close to pretrained. However, start/end standing segments still showed right-foot lifting behavior. The model reduced knee load metrics, but the compensation was not reliably expressed as visible limping during walking.

### Interpretation

This trial taught two things:

```text
Moderate pain can lower knee proxy while keeping rollout stable.
Standing/start/end phases create an easy foot-lift loophole.
```

That motivated reward gating: penalize pain mainly during walking, not during low-speed standing portions of the reference motion.

## Trial 4: Walk-Gated + Contact-Regularized

### Experiment

- exp: `phc_shape_pnn_iccv_pain_v15_oa_walkgate_idx3`
- script: `sbatch_pain_v15_walkgate_idx3.sh`
- seed checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv_pain_v15_oa_middle_idx3/Humanoid.pth`

### Reward / Pain Parameters

Base pain parameters were inherited from the middle setting:

```text
env.pain.lambda_p=0.25
env.pain.knee_mechanism.right_sensitivity=3.0
env.pain.knee_mechanism.right_threshold=0.15
env.pain.knee_mechanism.w_contact_load=0.45
env.pain.knee_mechanism.w_moment_load=0.55
env.pain.knee_mechanism.w_kam=0.75
env.pain.knee_mechanism.w_kfm=0.25
env.pain.knee_mechanism.memory_alpha=0.08
learning.params.config.disc_reward_w=0.30
learning.params.config.task_reward_w=0.70
```

Added walking-speed reward gate:

```text
env.pain.reward_gating.enabled=True
env.pain.reward_gating.min_ref_speed=0.35
env.pain.reward_gating.transition_width=0.20
env.pain.reward_gating.standing_scale=0.03
```

Added right-stance contact regularizer:

```text
env.pain.contact_regularizer.enabled=True
env.pain.contact_regularizer.side=right
env.pain.contact_regularizer.weight=0.50
env.pain.contact_regularizer.min_force=120.0
env.pain.contact_regularizer.ref_stance_height=0.08
env.pain.contact_regularizer.ref_stance_speed=0.35
env.pain.contact_regularizer.max_penalty=1.0
```

### Rationale

The model needed to stop exploiting standing-phase pain and right-foot lifting. The reward gate downweighted pain during reference standing/slow phases, while the contact regularizer prevented full right-leg abandonment during right stance.

### Training Behavior

Key scalar trend:

```text
reward:                 37.60 -> 41.71
episode length:         90.01 -> 91.07
pain reward cost:        0.235 -> 0.251
right contact force:   169.68 -> 247.88
right knee load:         0.146 -> 0.202
right knee moment load:  0.125 -> 0.162
right KAM:               0.125 -> 0.161
right KFM:               0.128 -> 0.167
right knee tau RMS:     16.20  -> 16.95
```

### Visual Result

Right-foot contact recovered. However, because the contact floor was strong (`min_force=120N`, `weight=0.50`), the model was pulled back toward normal right-leg loading rather than toward pain-avoiding compensation.

### Interpretation

This trial fixed the non-contact cheat but overcorrected:

```text
Contact regularization must prevent foot abandonment, not force normal right-leg loading.
```

That led to the final moment-biased setting: lower contact floor, weaker contact regularizer, stronger moment/KAM/KFM pain.

## Trial 5: Moment-Biased Pain

### Experiment

- exp: `phc_shape_pnn_iccv_pain_v15_oa_momentbias_idx3`
- script: `sbatch_pain_v15_momentbias_idx3.sh`
- seed checkpoint: `output/HumanoidIm/phc_shape_pnn_iccv_pain_v15_oa_walkgate_idx3/Humanoid.pth`

### Reward / Pain Parameters

Pain pressure was increased, but contact regularization was softened.

```text
env.pain.lambda_p=0.45
env.pain.knee_mechanism.right_sensitivity=4.0
env.pain.knee_mechanism.right_threshold=0.13
env.pain.knee_mechanism.w_contact_load=0.20
env.pain.knee_mechanism.w_moment_load=0.80
env.pain.knee_mechanism.w_kam=1.20
env.pain.knee_mechanism.w_kfm=0.80
env.pain.knee_mechanism.memory_alpha=0.08
```

Reward gating was kept:

```text
env.pain.reward_gating.enabled=True
env.pain.reward_gating.min_ref_speed=0.35
env.pain.reward_gating.transition_width=0.20
env.pain.reward_gating.standing_scale=0.03
```

Contact regularizer was softened:

```text
env.pain.contact_regularizer.enabled=True
env.pain.contact_regularizer.side=right
env.pain.contact_regularizer.weight=0.20
env.pain.contact_regularizer.min_force=70.0
env.pain.contact_regularizer.ref_stance_height=0.08
env.pain.contact_regularizer.ref_stance_speed=0.35
env.pain.contact_regularizer.max_penalty=0.7
```

PHC imitation/task weights were slightly loosened:

```text
learning.params.config.disc_reward_w=0.25
learning.params.config.task_reward_w=0.65
```

### Rationale

This was the first setting explicitly designed around the desired middle behavior:

```text
Keep right-foot contact, but do not force normal right-leg loading.
Make knee moment/KAM/KFM avoidance more important than raw contact-load avoidance.
```

### Training Behavior

Key scalar trend:

```text
reward:                 18.86 -> 20.16
episode length:         89.72 -> 91.09
pain reward cost:        0.507 -> 0.482
right contact force:   248.89 -> 248.74
right knee load:         0.318 -> 0.301
right knee moment load:  0.334 -> 0.314
right KAM:               0.166 -> 0.155
right KFM:               0.169 -> 0.160
right knee tau RMS:     16.84  -> 16.72
```

### Visual Result

Overlay videos show visible deviation from pretrained while keeping right-foot contact. This is currently the best candidate for pain-aware compensation gait.

### Interpretation

This setting achieved the best tradeoff so far:

```text
No full hopping cheat.
Right foot remains in contact.
Moment/KAM/KFM proxies decrease.
Visual behavior starts to differ from pretrained.
```

The key remaining question is whether this visible difference is clinically interpretable as limping-like gait or just a mild style shift. The generated sagittal-right overlay videos are the current primary artifact for that decision.

## Summary Table

| Trial | Pain strength | Contact control | Main outcome |
|---|---:|---:|---|
| v1.5 OA sanity/long | weak, `lambda_p=0.05` | none | proxy decreased, visual gait mostly unchanged |
| aggressive | very strong, `lambda_p=1.0`, sensitivity 6.0 | none | right knee load vanished via hopping/non-contact cheat |
| middle | moderate, `lambda_p=0.25`, sensitivity 3.0 | none | stable, proxy decreased, but standing foot-lift remained |
| walkgate | moderate + walking gate | strong floor, 120N / 0.50 | foot contact recovered, but knee load increased |
| momentbias | stronger moment pain, `lambda_p=0.45` | soft floor, 70N / 0.20 | best current tradeoff: contact maintained, moment proxies reduced |

## Current Recommendation

Use `phc_shape_pnn_iccv_pain_v15_oa_momentbias_idx3` as the current best checkpoint for visual/report artifacts. The previous trials should be presented as ablations:

1. Weak pain proves the OA proxy is trainable but visually insufficient.
2. Aggressive pain proves pain-only optimization cheats through non-contact.
3. Middle pain finds a stable but visually weak regime.
4. Walkgate proves contact regularization is necessary but can overconstrain.
5. Momentbias is the current best compromise.

For presentations, show `videos/05_momentbias_vs_pretrained_sagittal_right.mp4` together with `trial_metric_summary.csv`. For method discussion, emphasize that PHC-Pain v1.5 is not simply copying a pathological gait; it changes the objective so that a different movement strategy can emerge under an OA-style knee pain/load proxy.

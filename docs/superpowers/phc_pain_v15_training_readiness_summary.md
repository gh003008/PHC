# PHC-Pain v1.5 Training Readiness Summary

Date: 2026-04-30
Scope: sanity training before the next PHC-Pain v1.5 fine-tune

## 1. Current Conclusion

현재 코드는 짧은 sanity training을 시작할 수 있는 상태다. 다만 이 상태가 의미하는 것은 "OA-style knee load proxy가 학습 루프에 연결되어 있고, metric이 finite하게 나온다"는 것이지, 아직 "병적 보행이 생성되었다"는 뜻은 아니다.

이번 단계의 핵심 변화는 pain을 단순 actuator torque penalty로 보지 않고, 무릎 OA에 더 가까운 synthetic load proxy로 재정의한 것이다. 즉 오른무릎 pain state는 이제 다음 흐름으로 만들어진다.

```text
IsaacGym contact force + body geometry + knee flexion
  -> contact/compression proxy
  -> KAM/KFM-style moment-arm proxy
  -> OA knee load proxy
  -> threshold/sensitivity
  -> right-knee pain drive
  -> leaky right-knee pain state
  -> reward penalty and observation channel
```

따라서 다음 학습에서 봐야 할 질문은 "보행이 바로 절뚝거리느냐"가 아니라, 먼저 "pretrained walking competence를 유지한 채 right-knee OA load/state가 줄어드는 방향으로 policy update가 일어나는가"이다.

## 2. Base PHC Training Framework

이 repo의 기본 학습 구조는 PHC의 imitation + AMP/PPO 구조를 그대로 따른다.

주요 구성은 다음과 같다.

- Environment task: `HumanoidIm` 계열 task
- Policy/training config: `learning=im_pnn`
- Network: `amp_pnn`
- Optimizer loop: rl_games PPO/AMP 기반
- Main task reward: reference motion imitation reward
- Additional regularization: power reward
- Discriminator reward: AMP-style motion prior reward
- Pretrained checkpoint: PHC walking/motion prior로 사용

기본 imitation reward는 `HumanoidIm._compute_reward()` 안에서 현재 body pose/rotation/velocity와 reference motion을 비교해서 계산된다. 이 reward가 PHC의 "잘 걷는 motor competence prior"를 만든다. `im_pnn.yaml`에서는 PPO와 AMP가 같이 설정되어 있고, `task_reward_w: 0.5`, `disc_reward_w: 0.5`로 task imitation reward와 discriminator reward를 함께 사용한다.

이 말은 PHC-Pain이 처음부터 병적 보행을 새로 배우는 구조가 아니라는 뜻이다. 기존 PHC가 이미 보행을 잘 따라갈 수 있으므로, pain fine-tuning은 이 motor prior 위에서 reward landscape를 살짝 바꾸는 방식이다.

## 3. Why We Kept PHC as the Motor Prior

처음 목표는 "환자 보행을 imitation으로 복사"하는 것이 아니었다. 현재 목표는 synthetic mechanism proof다.

```text
특정 무릎에 pain/load sensitivity를 주었을 때,
policy가 reference walking을 유지하려고 하면서도
그 무릎의 pain/load를 줄이는 방향으로 행동을 바꾸는가?
```

그래서 PHC는 병적 보행 자체의 정답이 아니라, 정상 locomotion competence를 제공하는 prior 역할을 한다. pain term은 이 prior 위에서 추가된 cost다.

좋은 결과는 다음 조건을 동시에 만족해야 한다.

- reference walking을 완전히 포기하지 않는다.
- episode length와 forward motion이 무너지지 않는다.
- affected side, 현재는 right knee, 의 OA load proxy가 줄어든다.
- 감소가 단순히 멈춤, collapse, bilateral shutdown 때문에 생긴 것이 아니다.
- left/right asymmetry나 knee/hip/ankle compensation이 해석 가능하게 나타난다.

## 4. Pain Task Structure

Pain task는 `phc/env/tasks/humanoid_im_pain.py`에 있다.

### v0: `HumanoidImPain`

`HumanoidImPain`은 기존 `HumanoidIm`에 environment-side pain estimator를 얹은 첫 버전이다. 이 버전은 기존 observation/checkpoint shape을 유지하는 방향이었다.

v0 pain은 대략 다음 항을 조합했다.

- joint-limit pain
- actuator torque pain
- power/work pain
- contact pain
- optional action guard
- optional reward penalty

여기서 action guard는 policy가 pain을 이해하게 만드는 주 메커니즘이 아니라, action을 외부에서 잘라서 위험한 행동을 제한하는 safety/debug 장치에 가깝다. 현재 v1 mechanism claim에서는 action guard가 핵심이 아니다.

### v1: `HumanoidImPainV1`

`HumanoidImPainV1`은 observation contract를 바꾼 별도 task다. 즉 기존 PHC observation 뒤에 body-part pain observation을 추가한다.

현재 pain channels는 다음과 같은 body map 구조다.

```text
left_hip, right_hip,
left_knee, right_knee,
left_ankle, right_ankle,
back,
left_foot, right_foot
```

각 channel은 현재 pain state를 가진다. 설정에 따라 memory도 같이 observation에 붙는다. 현재 config는 `include_memory: True`이므로 policy는 "지금 아픈가"뿐 아니라 "최근에 지속적으로 아팠는가"도 볼 수 있다.

현재 active side는 config에서 `active_knee_side: "right"`로 설정되어 있다. 즉 right knee만 reward-facing pain channel로 활성화된다. left knee metric도 계산/로그할 수 있지만, 기본 sensitivity는 left 0, right 1이다.

## 5. Checkpoint Adaptation

v1은 pain observation을 추가하므로 observation dimension이 기존 PHC checkpoint와 달라진다. 그래서 pretrained PHC checkpoint를 그대로 load하면 첫 layer input shape mismatch가 난다.

이를 해결하기 위해 `phc/learning/amp_agent.py`에 checkpoint adaptation 경로가 들어가 있다.

동작은 다음과 같다.

```text
saved policy first-layer weights:
  [original PHC obs columns]

target v1 policy first-layer weights:
  [original PHC obs columns | new pain obs columns]

adaptation:
  original columns: pretrained weight copy
  new pain columns: zero initialization
```

이 설계의 의미는 중요하다. fine-tune 시작 시점에서 policy는 기존 PHC와 거의 같은 행동을 한다. pain observation column은 0으로 시작하므로 처음부터 행동을 망가뜨리지 않는다. 학습이 진행되면서 PPO update가 pain obs column과 reward penalty를 이용해 행동을 바꾸게 된다.

## 6. Reward Structure

현재 reward는 기존 PHC reward에 pain penalty를 빼는 구조다.

```text
reward_total
  = PHC imitation / AMP reward
  + power regularization
  - lambda_p * affected_knee_pain_state
```

`HumanoidImPainV1._compute_reward()`는 의도적으로 `HumanoidIm._compute_reward()`를 직접 호출한 뒤, pain buffer를 업데이트하고, `reward_only` 또는 `guard_and_reward` 모드일 때 affected knee pain state만 penalty로 뺀다.

현재 config의 핵심 값은 다음과 같다.

```yaml
pain:
  enabled: True
  mode: "reward_only"
  append_to_obs: True
  active_knee_side: "right"
  lambda_p: 0.05
```

즉 현재 main mechanism은 guard가 아니라 reward-only pain pressure다. 짧은 sanity training에서는 `lambda_p=0.05`가 너무 약한지, 혹은 너무 강해서 imitation이 깨지는지 확인해야 한다.

## 7. v1.5 OA Knee Load Proxy

가장 큰 변화는 `pain_state`를 무엇으로 drive하느냐다.

이전 torque-centric proxy는 다음 한계가 있었다.

- actuator torque가 0 또는 작게 보일 수 있다.
- torque magnitude가 줄어도 GRF나 joint loading이 줄었다고 말하기 어렵다.
- knee OA pain을 설명하려면 "무릎에 걸리는 compressive/moment load"가 더 자연스럽다.

그래서 v1.5에서는 torque를 reward-facing load에서 빼고, OA-style load proxy를 기본으로 쓴다.

현재 default:

```yaml
knee_mechanism:
  proxy_mode: "oa_contact_v15"
  w_contact_load: 0.60
  w_moment_load: 0.40
  w_legacy_torque_load: 0.00
```

### 7.1 Contact/Compression Proxy

`compute_knee_contact_load_proxy()`는 ankle/toe contact force를 합쳐 stance foot loading을 추정한다.

주요 component:

- `compression`: foot contact force magnitude 또는 vertical force를 body-weight reference로 normalize
- `loaded_flex`: compression이 걸린 상태에서 knee flexion이 커질 때 증가
- `loading_rate`: compression이 빠르게 증가할 때 증가

현재 config:

```yaml
body_weight_ref: 700.0
flex_compression_gain: 0.25
loading_rate_ref: 1000.0
w_compression: 0.70
w_loaded_flex: 0.20
w_loading_rate: 0.10
```

즉 stance force가 크고, 그 상태에서 knee flexion이 크고, loading이 급격히 올라가면 pain-driving load가 커진다.

### 7.2 KAM/KFM Moment Proxy

`compute_knee_moment_load_proxy()`는 knee position과 foot center position 사이의 lever arm, 그리고 vertical GRF를 사용해 KAM/KFM-style proxy를 만든다.

현재 구현:

- `kam`: frontal-plane lever component times vertical GRF
- `kfm`: sagittal-plane lever component times vertical GRF

현재 config:

```yaml
kam_ref: 50.0
kfm_ref: 50.0
w_kam: 0.70
w_kfm: 0.30
```

주의할 점은 이 값이 true medial tibiofemoral contact force가 아니라는 것이다. PHC rigid body geometry와 contact tensor로부터 만든 synthetic surrogate다. 따라서 논문/보고서에서는 "estimated OA-style load proxy", "KAM/KFM-style surrogate"라고 써야 한다.

### 7.3 Final OA Load

최종 right-knee load는 다음처럼 조합된다.

```text
oa_load
  = 0.60 * contact_load
  + 0.40 * moment_load
  + 0.00 * legacy_torque_load
```

legacy torque는 완전히 없앤 것이 아니라 metric으로 계속 남긴다. 그래서 학습 후에도 다음을 비교할 수 있다.

- pain_v1_right_knee_contact_load
- pain_v1_right_knee_moment_load
- pain_v1_right_knee_torque_load
- pain_v1_right_knee_tau_rms
- pain_v1_right_knee_tau_peak

이렇게 둔 이유는, pain은 OA proxy로 학습하되 "실제로 actuator torque도 같이 줄었는가/아닌가"를 별도 diagnostic으로 보기 위해서다.

## 8. Pain State Dynamics

raw load가 바로 reward penalty로 들어가는 것은 아니다. v1에서는 load가 threshold와 sensitivity를 거쳐 drive가 되고, drive가 leaky state를 만든다.

현재 right knee:

```text
load_proxy
  -> relu(load_proxy - right_threshold)
  -> right_sensitivity * above_threshold_load
  -> right_knee_drive
  -> leaky update
  -> right_knee_pain_state
```

현재 config:

```yaml
right_sensitivity: 1.0
right_threshold: 0.30
memory_alpha: 0.05
```

이 구조는 "부하가 항상 pain이다"가 아니라, threshold를 넘는 load가 반복/누적될 때 pain state가 올라가는 형태다. sanity training에서는 `right_knee_load`가 threshold 근처에서 어떻게 분포하는지 확인해야 한다. 만약 대부분 threshold 아래라면 pain reward가 거의 작동하지 않는다. 반대로 처음부터 매우 높고 계속 saturated라면 training이 불안정해질 수 있다.

## 9. Observation Path

`HumanoidImPainV1._compute_observations()`는 기존 PHC observation을 만든 뒤 pain observation을 뒤에 붙인다.

구조:

```text
self_obs
  + task_obs/reference motion obs
  + pain_obs
```

pain_obs는 `pain_body_state`와 `pain_body_memory`를 concat한 것이다. 따라서 policy는 현재 관절/몸 상태, reference motion, 그리고 pain body map을 동시에 본다.

이 설계가 중요한 이유는 reward-only penalty만 있고 observation에 pain이 없으면, policy가 어떤 상태가 아픈 상태인지 직접 구분하기 어렵기 때문이다. 현재 v1은 pain을 observation에도 넣고 reward에도 넣는다. 그래서 policy가 "이 stance/loading pattern으로 가면 right knee pain state가 올라간다"는 것을 학습할 수 있다.

## 10. Logging and Metrics

학습 중 `pain_v1_*` scalar는 training log로 올라가도록 되어 있다.

`phc/learning/common_agent.py`는 env `infos`에서 key가 `pain_v1_`로 시작하는 scalar를 모아서 logging accumulator에 넣는다. viewer/headless probe에서는 `PHC_PAIN_PROBE_JSON` 환경변수를 주면 player가 `pain_v1_`, `lower_limb_`, `pain_mean`, `pain_max` 값을 JSON으로 저장한다.

중요 metric:

```text
pain_v1_right_knee_load
pain_v1_right_knee_state
pain_v1_right_knee_drive
pain_v1_right_knee_contact_load
pain_v1_right_knee_moment_load
pain_v1_right_knee_compression
pain_v1_right_knee_loaded_flex
pain_v1_right_knee_loading_rate
pain_v1_right_knee_kam
pain_v1_right_knee_kfm
pain_v1_right_knee_torque_load
pain_v1_right_knee_tau_rms
pain_v1_right_knee_tau_peak
```

해석 우선순위:

1. `right_knee_state`와 `right_knee_load`가 줄어드는가.
2. 그 감소가 `contact_load` 때문인지 `moment_load` 때문인지 본다.
3. `tau_*`는 reward target이 아니라 diagnostic으로 본다.
4. episode length, average reward, imitation quality가 같이 유지되는지 본다.
5. left knee나 전체 locomotion이 같이 망가지는지 본다.

## 11. Completed Verification

Phase 9 구현 후 다음 검증은 통과했다.

```bash
conda run --no-capture-output -n phc python -m unittest tests.test_pain_knee_proxy -v
```

결과: 5 tests OK.

```bash
conda run --no-capture-output -n phc python -m py_compile \
  phc/env/util/pain_baseline.py \
  phc/env/tasks/humanoid_im_pain.py \
  phc/learning/im_amp_players.py
```

결과: passed.

```bash
conda run --no-capture-output -n phc python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("phc/data/cfg/env/env_im_pain_v1.yaml").read_text())
assert cfg["pain"]["knee_mechanism"]["proxy_mode"] == "oa_contact_v15"
assert cfg["pain"]["knee_mechanism"]["w_legacy_torque_load"] == 0.0
print("env_im_pain_v1.yaml OK")
PY
```

결과: passed.

Headless pretrained probe도 실행했고, 다음 OA proxy metrics가 finite하게 나왔다.

```text
pain_v1_right_knee_compression: 0.18638019636273384
pain_v1_right_knee_loaded_flex: 0.01365843357052654
pain_v1_right_knee_loading_rate: 0.0023503638803958893
pain_v1_right_knee_kam: 0.05676730442792177
pain_v1_right_knee_kfm: 0.058967757504433393
pain_v1_right_knee_contact_load: 0.13343286886811256
pain_v1_right_knee_moment_load: 0.057427442632615566
pain_v1_right_knee_torque_load: 0.28062527626752853
pain_v1_right_knee_load: 0.10303070209920406
pain_v1_right_knee_state: 0.0011279801838099957
```

주의: 이 probe는 metric wiring 확인용이다. reward가 낮고 step도 짧은 설정이었으므로 gait-quality evidence로 쓰면 안 된다.

## 12. What the Short Sanity Training Should Test

짧은 sanity training의 목적은 성능 확인이 아니다. 다음 네 가지를 확인하면 된다.

1. PPO training loop가 `env_im_pain_v1` + `oa_contact_v15`에서 crash 없이 돈다.
2. TensorBoard 또는 event log에 `pain_v1_*` scalar가 찍힌다.
3. episode length가 즉시 collapse하지 않는다.
4. `right_knee_load/state/contact_load/moment_load`가 관측 가능한 scale로 나온다.

짧은 sanity training에서 병적 보행이 눈에 보이지 않아도 괜찮다. 이 단계는 "학습이 가능한가"와 "metric이 살아 있는가"를 보는 gate다.

## 13. Recommended Sanity Training Setup

기본 방향:

- pretrained PHC checkpoint에서 시작한다.
- walking motion subset으로 시작한다.
- 너무 긴 epoch을 돌리지 않는다.
- log 주기는 너무 촘촘하게 하지 않는다.
- `games_num`이나 viewer 설정은 training에는 쓰지 않는다.

권장 override 방향:

```text
learning=im_pnn
env=env_im_pain_v1
exp_name=<new sanity run name>
epoch=-1
test=False
env.motion_file=sample_data/amass_isaac_walking_forward_subset23.pkl
robot.has_shape_obs_disc=True
```

VRAM이 15GB 수준이면 `env.num_envs`는 보수적으로 낮춰 시작하는 것이 낫다. 기존 reduced-scale pain-v1 run이 안정적이었으므로, 처음부터 원본 PHC 대규모 config를 그대로 쓰기보다 sanity run에서는 작은 env 수로 crash/logging을 먼저 확인하는 편이 안전하다.

## 14. Success and Failure Criteria

Short sanity pass:

- training process가 정상 시작된다.
- checkpoint adaptation message가 나오거나, expanded observation checkpoint가 정상 load된다.
- `pain_v1_right_knee_*` metrics가 event log에 기록된다.
- episode length가 1-2 epoch 안에 완전히 붕괴하지 않는다.

Short sanity fail:

- checkpoint shape mismatch가 난다.
- `pain_v1_*` scalar가 전혀 없다.
- `right_knee_load/state`가 전부 0 또는 NaN이다.
- 시작 직후 모든 env가 collapse한다.
- reward가 통째로 NaN이 된다.

## 15. After Sanity Training

sanity가 통과하면 그 다음은 긴 fine-tune이다. 이때 봐야 할 것은 visual보다 먼저 metric이다.

긴 run에서 기대하는 1차 evidence:

- pretrained 대비 `right_knee_load` 감소
- pretrained 대비 `right_knee_state` 감소
- contact/moment 중 어떤 channel이 줄었는지 분해 가능
- episode length와 imitation reward 유지
- left/right side specificity 유지

그 다음 viewer에서 볼 것:

- 오른무릎 stance loading이 줄어드는지
- 오른 stance time이나 GRF vector가 바뀌는지
- trunk/pelvis/hip/ankle compensation이 생기는지
- 단순히 멈추거나 무너지는 방식으로 pain을 줄이지 않았는지

## 16. Current Risk

가장 큰 risk는 `lambda_p=0.05`가 너무 약해서 보행이 거의 변하지 않는 것이다. 지금 `right_knee_load` probe 값은 threshold 0.30보다 낮게 나온 짧은 rollout이 있었기 때문에, 실제 walking rollout에서 load distribution이 threshold를 충분히 넘는지 확인해야 한다.

가능한 조정 방향:

- threshold를 낮춘다.
- right sensitivity를 올린다.
- `lambda_p`를 올린다.
- `w_contact_load`, `w_moment_load`, `w_kam`/`w_kfm` 비율을 조정한다.
- imitation reward pressure를 줄이는 별도 experiment를 둔다.

하지만 첫 sanity training 전에는 이 값을 크게 바꾸지 않는 것이 좋다. 먼저 현재 default로 log scale과 stability를 확인하고, 그 다음 조정하는 것이 원인 분리가 쉽다.

## 17. Bottom Line

지금까지의 작업은 "PHC walking controller에 OA-style right-knee pain/load state를 reward와 observation으로 연결하는 코드 준비"까지 완료한 것이다.

다음 한 줄 요약:

```text
이제 할 일은 짧은 sanity training으로 oa_contact_v15가 PPO fine-tune loop에서 안정적으로 돌고,
pain_v1_right_knee_* metric이 기록되는지 확인하는 것이다.
```


# 중간 공유 보고서

Exo Lab 인간-로봇 통합 시뮬레이션

0.

담당: Jinsu / PHC-Pain 진행

날짜: 26.04.28

관련 모듈 / 영역: PHC / Humanoid imitation / pain estimation / PPO fine-tuning / evaluation gates

## 1. Objective (이번 사이클 목표)

- 기존 PHC humanoid imitation controller 위에 pain-aware simulation layer를 붙여, 통증/부하가 행동을 바꾸는지 검증할 수 있는 실험 기반을 만든다.
- v0 목표: 기존 observation contract를 깨지 않고 pain estimator, action guard, reward penalty를 붙인 뒤 안정성 및 fine-tune 가능성을 확인한다.
- v1 목표: body-part pain map을 observation에 포함하고, synthetic unilateral knee pain reward를 통해 "통증 회피 행동이 policy learning으로 생기는지"를 검증할 수 있는 구조를 만든다.
- 현재 회의에서의 핵심 목적은 구현 완료 여부보다 "어디까지 claim 가능하고, 다음 GPU 실험에서 무엇을 봐야 하는지"를 명확히 하는 것이다.

## 2. Current Progress

- PHC-Pain-v0는 구현 및 재실험까지 완료했다. `HumanoidImPain`, pure-torch pain helpers, `env_im_pain.yaml`, `env_im_pain_finetune.yaml`이 추가되었다.
- v0는 A5000 서버에서 짧은 fine-tune을 수행했고, guard가 걸려도 controller가 무너지지 않는 상태까지 확인했다.
- PHC-Pain-v1은 별도 task `HumanoidImPainV1`로 구현했다. v0 checkpoint-compatible task는 유지하고, v1에서만 observation dimension을 확장한다.
- v1은 9개 body-part pain channel과 memory를 observation에 붙이고, 오른쪽/왼쪽 knee pain drive를 reward-only cost로 쓰는 구조까지 구현했다.
- expanded observation checkpoint adaptation을 추가했다. 기존 PHC input column은 pretrained weight를 복사하고, 새 pain-observation column은 zero init한다.
- 평가 패키지와 ablation command matrix를 만들었다. main / no_obs / no_reward 조건을 비교해 pain reduction, locomotion competence, side-specificity를 판정한다.
- 초기 main run에서 pain scalar가 TensorBoard에 안 남는 문제가 있었고, 26.04.28에 `pain_v1_*` scalar logging instrumentation과 서버 probe까지 완료했다.

## 3. Key Results / Findings

- 기존 PHC baseline은 standing motion에서 reward 941.05, 999 steps로 정상 동작했다.
- v0 default guard는 너무 강했다. `guard_only`에서 38 steps / reward 27.15로 collapse했다.
- v0 retuned guard는 collapse를 해결했다. pretrained checkpoint 기준 `guard_only`에서 999 steps / reward 912.75를 기록했다.
- A5000 fine-tune 이후 v0 `guard_only`는 999 steps / reward 975.69로 PASS했다. 즉 "guard를 견디는 controller"는 확보했다.
- 반면 v0 `log_only` pain_max는 0.522로 baseline과 동일했다. 즉 "통증을 내재적으로 줄이는 behavior learning"은 v0에서 아직 증명되지 않았다.
- v1 eval smoke는 expanded obs shape 963, checkpoint restore 성공, 999 steps / reward 936.74로 통과했다.
- v1 main training run은 reward가 15.38에서 350.27까지 증가하고 episode length가 19에서 461까지 올라가 training stability signal은 있다.
- 단, main run 당시 required pain metrics가 없어서 mechanism claim은 보류했다. 이후 scalar probe에서 `pain_v1_right_knee_load`, `drive`, `state`, `reward_cost_mean`이 TensorBoard에 찍히는 것을 확인했다.

## 4. Issues / Blockers

- v0의 핵심 한계: action guard survival은 됐지만 pain avoidance는 안 됐다. pain_max 0.522가 줄지 않아 "learns to avoid pain" claim은 불가하다.
- v1의 핵심 blocker였던 pain metric logging은 26.04.28 probe로 해소했다.
- 아직 v1 ablation matrix는 실행 전이다. main / no_obs / no_reward 비교가 없으므로 pain observation의 효과와 reward-only mechanism의 효과를 분리해 말할 수 없다.
- synthetic knee load proxy는 medial tibiofemoral contact force나 KAM 직접값이 아니다. 현재는 torque, flexion torque, ROM near-limit, positive work 기반 proxy로 제한된다.
- 8 GB GPU에서는 PHC fine-tuning이 현실적으로 어렵다. Isaac Gym fixed GPU overhead 때문에 24 GB급 서버 GPU를 계속 사용해야 한다.

## 5. Discussion Points

- v1.0 claim boundary를 synthetic mechanism proof로 유지할지 확인이 필요하다. 현재 단계에서는 patient-specific gait reproduction이나 clinical pain model claim은 과하다.
- ablation 우선순위: main, no_obs, no_reward를 먼저 돌리고 left-knee side-specific run은 그 다음에 돌리는 것이 비용 대비 명확하다.
- pain penalty strength를 현재 값으로 유지할지, main run의 pain scalar 분포를 본 뒤 `lambda_p` 또는 threshold를 조정할지 결정해야 한다.
- locomotion competence gate를 episode length만으로 볼지, forward progress / no-fall / reward component까지 같이 볼지 논의가 필요하다.
- 향후 환자 데이터가 들어오면 imitation target으로 바로 쓰기보다 parameter identification / validation signal로 쓰는 방향이 더 안전하다.

## 6. Decision / Help Needed

- 오늘 결정할 것: v1.0의 공식 목표를 "synthetic unilateral knee pain mechanism proof"로 고정할지.
- 필요 리소스: A5000/4090/A100급 GPU에서 Phase 8 ablation matrix를 실행할 시간.
- 필요한 피드백: 현재 knee load proxy가 회의에서 설명 가능한 수준인지, 아니면 KAM/GRF 계열 proxy를 더 보강해야 하는지.
- 필요한 결정: ablation 결과가 pain reduction은 보이지만 reward/episode length가 떨어질 경우, 어느 정도 trade-off까지 허용할지.

## 7. Next Plan

- Step 1: scalar logging patch가 들어간 상태로 main condition 짧은 재실행을 돌려 pain metric distribution을 확인한다.
- Step 2: main / no_obs / no_reward minimal ablation을 같은 checkpoint 및 같은 budget으로 실행한다.
- Step 3: `scripts/phc_pain_v1_evaluate.py`에 aggregate JSON을 넣어 PASS / FAIL / BLOCKED를 판정한다.
- Step 4: side-specificity 확인을 위해 right-knee와 left-knee impairment를 비교한다.
- Step 5: 결과가 약하면 `lambda_p`, knee threshold, active side sensitivity를 조정하되, locomotion collapse를 막는 gate를 먼저 둔다.

## 8. Timeline / Due

- 26.04.28: 회의 공유, claim boundary 및 ablation 우선순위 확정.
- 26.04.28-26.04.29: scalar logging 적용 상태로 main/no_obs/no_reward 짧은 ablation 실행.
- 26.04.29 이후: 결과가 유효하면 evaluation template 채우고 v1.0 mechanism proof 판정. 결과가 약하면 penalty/threshold sweep으로 전환.

## 9. Misc / Notes

- v0 final disposition은 partial success다. "survive the guard"는 성공했지만 "learn to avoid pain"은 실패했다.
- v1은 이 한계를 직접 해결하기 위해 pain을 observation에 넣고, action guard가 아니라 reward-only learning으로 행동 변화를 유도하는 방향이다.
- 보고서의 수치는 git commit log와 `.planning/phases/*/SUMMARY.md`, `docs/superpowers/specs/*_impl_log.md` 기준이다.

## 10. Appendix

### A. 주요 커밋 흐름

- `6b6ac45`: PHC-Pain-v0 구현. pain helper, task subclass, config, parse_task 등록.
- `312f27d`: v0 retuned fine-tune config와 body-group pain logging 추가.
- `0d2f7e0`: A5000 서버 재실행 결과 기록. v0 V5.3/V6/V7 결과 정리.
- `17fd119`: PHC-Pain-v1 task observation contract 추가.
- `16008de`: synthetic knee pain reward mechanism 추가.
- `cea80db`: expanded observation checkpoint adaptation 추가.
- `739db3e`: evaluation protocol, evidence template, evaluator script 추가.
- `58731a0`, `5273526`: player/checkpoint restore path에서 expanded obs shape 대응.
- `c2a6874`, `77a76a6`: `pain_v1_*` scalar logging instrumentation 및 서버 probe evidence 기록.

### B. 핵심 수치

| 항목 | 결과 | 해석 |
|---|---:|---|
| PHC baseline | 941.05 reward / 999 steps | 기존 checkpoint 정상 |
| v0 default guard_only | 27.15 reward / 38 steps | guard가 너무 강해 collapse |
| v0 retuned pretrained guard_only | 912.75 reward / 999 steps | retune만으로 생존 회복 |
| v0 A5000 fine-tuned guard_only | 975.69 reward / 999 steps | guard survival PASS |
| v0 A5000 log_only pain_max | 0.522 | pain avoidance FAIL |
| v1 eval smoke | 936.74 reward / 999 steps | expanded obs checkpoint restore 정상 |
| v1 main run signal | reward 15.38 -> 350.27, ep_len 19 -> 461 | training stability signal |
| scalar probe right_knee_state | 0.0631 | logging path 확인 |
| scalar probe reward_cost_mean | 0.00315 | reward cost scalar 확인 |

### C. 현재 claim 가능 / 불가능

- 가능: PHC-Pain-v0는 pain guard를 걸어도 retuned/fine-tuned controller가 episode를 끝까지 버틸 수 있다.
- 가능: PHC-Pain-v1은 body-map pain obs, unilateral knee pain drive, checkpoint adaptation, evaluation gate, scalar logging infrastructure가 준비되어 있다.
- 불가능: v0가 pain을 줄이는 행동을 학습했다고 말할 수 없다.
- 아직 불가능: v1이 pain-reducing compensatory gait를 학습했다고 말할 수 없다. ablation metric이 필요하다.

### D. 코드 근거

- `HumanoidImPainV1`: `phc/env/tasks/humanoid_im_pain.py`
- checkpoint adaptation: `phc/learning/amp_agent.py`
- scalar logging: `phc/learning/common_agent.py`, `phc/learning/amp_agent.py`
- evaluator: `scripts/phc_pain_v1_evaluate.py`
- evaluation docs: `docs/superpowers/phc_pain_v1_evaluation_protocol.md`, `docs/superpowers/phc_pain_v1_evidence_template.md`

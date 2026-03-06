# VIC08 결과 분석 (260305)

## 1. 실험 설정

VIC07에서 도입한 두 가지 수정(action rate penalty + upper body ROM limit)이 모두 성능을 저하시켰다.
VIC08은 이 두 변경을 제거하고, VIC06에서 적용한 CCF gradient noise fix 하나만 유지한 채로
15,000 에폭을 처음부터 온전히 학습한다.

VIC06은 OOM으로 7642 에폭에서 중단되어 연속성이 깨진 상태였으므로,
VIC08이 사실상 CCF noise fix 단독 효과를 검증하는 실험이다.

변경 사항 (VIC07 대비):
- humanoid_im_vic.py: action rate penalty 코드 전체 제거 (_action_rate_penalty_w, _last_actions, _upper_body_dof_start, _upper_body_rom_limit), _compute_torques에서 상체 스케일링 제거, _reset_envs 단순화
- env_im_walk_vic.yaml: action_rate_penalty_w, upper_body_rom_limit 항목 제거

유지 사항:
- CCF gradient noise fix (pre_physics_step에서 Stage 1 시 CCF 차원을 0으로 덮어씀)
- reward_specs: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1
- task_reward_w=0.5, disc_reward_w=0.5
- power_coefficient: 0.00005 (VIC09 이전 원래값)
- motion_file: amass_isaac_walking_forward.pkl (30개 모션)

## 2. 학습 경과

15,000 에폭 정상 완료.

학습 곡선 형태:
- VIC07의 S자형과 달리, VIC03~06과 유사한 위로 볼록한 단조 증가 후 수렴 형태로 돌아옴.
- action rate penalty 제거로 초반 탐색 억제가 없어진 결과로 해석 가능.

## 3. 정량적 평가

평가 환경: num_envs=16, test 모드, 최종 체크포인트 (15000 에폭)

| 지표 | VIC03 | VIC04 | VIC05 | VIC07 | VIC08 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 평가 Avg Reward | 254.0 | 218.5 | 224.3 | 207.8 | 223.8 |
| 평가 Avg Steps | 79.1 (2.6초) | 67.8 (2.3초) | 69.3 (2.3초) | 65.9 (2.2초) | 69.8 (2.3초) |

VIC08은 VIC03~05 범위(av reward 218~254)에 속하며, VIC07보다는 개선되었다.
그러나 V4 수준(av reward ~461, av steps ~143)에는 전혀 도달하지 못했다.

## 4. 분석

(1) CCF noise fix 단독으로는 부족함이 확인됨

VIC03~08 모두 유사한 수준(av reward 207~254)에서 수렴했다. CCF gradient noise fix를
도입한 VIC06/VIC08이 VIC03~05보다 크게 개선되지 않았으므로, CCF noise가 학습 부진의
주요 원인이 아니었음을 알 수 있다.

V4(PHC 기본 스크립트)와 VIC(VIC 스크립트)의 성능 차이의 원인이 CCF noise 외의
다른 요소에 있을 가능성이 높아졌다.

(2) VIC07이 VIC03보다 낮았던 이유 재확인

VIC08이 VIC07보다 높은 결과를 보임으로써, VIC07의 action rate penalty와
upper body ROM limit이 실제로 성능을 저하시켰음이 다시 확인된다.

(3) VIC03~08 공통 원인 탐색 필요

이 범위에서 벗어나지 못하는 근본 원인이 남아 있다. 가능한 후보:
- energy penalty: power_coefficient=0.00005가 보폭이 작은 보행 전략을 유도할 수 있음
- 30개 다양한 모션: AMP discriminator가 다양한 스타일을 동시에 학습해야 하므로 어려울 수 있음
- Stage 1 CCF=0 고정: 에이전트가 순수 PD 제어만으로 작동하는 한계가 있을 수 있음

## 5. 결론 및 다음 실험 방향

CCF noise fix는 필요조건이지만 충분조건이 아니다. VIC03~08이 모두 비슷한
수준에서 멈추는 근본 원인을 해결하기 위해 VIC09를 계획한다.

VIC09 방향:
- energy penalty 1/10 완화 (0.00005 → 0.000005): 작은 보폭 전략 억제
- 단일 모션 사용 (30개 → 1개): AMP discriminator 및 policy 학습 단순화

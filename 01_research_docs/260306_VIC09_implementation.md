# VIC09 구현 계획: 단일 모션 + Energy Penalty 완화 (260306)

## 1. 목표

VIC03~08에서 일관되게 av reward ~200~254, av steps ~65~79 수준에서 벗어나지 못하고 있다.
VIC09는 학습 난이도 자체를 낮추는 두 가지 변화를 적용한다.

(A) Reference motion을 30개에서 1개로 축소 -> 목표 분포 단순화
(B) Energy penalty 가중치를 1/10로 완화 -> 작은 보폭으로 버티는 전략 억제

## 2. 변경 사항 (VIC08 대비)

### 2-1. 단일 모션 사용

변경:
- motion_file: "sample_data/amass_isaac_walking_forward.pkl" (30개)
             -> "sample_data/amass_isaac_walking_forward_single.pkl" (1개)

선택 모션: 0-KIT_11_WalkingStraightForwards05_poses
- 30개 중 인덱스 0번, 이름 그대로 직진 보행 모션
- sample_data/amass_isaac_walking_forward_single.pkl로 추출 저장

의도:
- 30개의 다양한 보행 스타일을 동시에 학습하면, discriminator도 다양한 스타일을 학습해야 하고
  policy도 모든 스타일을 모방해야 하므로 학습 공간이 넓다.
- 1개 모션만 사용하면 policy가 해당 모션의 step cycle에만 집중할 수 있어 학습이 단순해진다.
- AMP discriminator도 단일 스타일만 구분하면 되므로 훈련이 쉬워진다.

### 2-2. Energy Penalty 완화

변경:
- power_coefficient: 0.00005 -> 0.000005 (1/10)

의도:
- 현재 에이전트들이 "서서 작은 동작으로 버티기" 전략을 취하는 경향이 있다.
- power_reward = -coefficient * |torque × velocity| 는 움직임에 패널티를 준다.
- 보행은 필연적으로 토크와 속도가 커야 하므로, 패널티가 강하면 오히려 보폭을 줄이는 것이 유리해진다.
- 1/10으로 낮추면 보행에 따른 에너지 패널티가 현저히 줄어들어 더 적극적인 보행이 가능해진다.
- 단, 너무 낮추면 떨림(jittering) 억제 효과도 줄어들 수 있으나, 이 문제는 나중에 조정.

## 3. 유지 사항

- CCF gradient noise fix (VIC06에서 도입): pre_physics_step에서 Stage 1 시 CCF = 0 유지
- VIC07에서 제거한 action rate penalty, upper body ROM limit: 유지하지 않음 (VIC08과 동일)
- reward_specs: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1 (VIC08과 동일)
- task_reward_w=0.5, disc_reward_w=0.5

수정 파일:
- phc/data/cfg/env/env_im_walk_vic.yaml: motion_file, power_coefficient 변경
- phc/data/cfg/learning/im_walk_vic.yaml: name VIC09로 변경
- humanoid_im_vic.py: 변경 없음 (VIC08과 동일)

## 4. 기대 효과

- 단일 모션으로 목표가 단순해져서 더 높은 imitation reward 달성 가능
- energy penalty 완화로 에이전트가 더 적극적으로 보행을 시도할 것으로 기대
- VIC03~08 대비 av reward 및 av steps 개선 여부로 두 변화의 효과를 확인

## 5. 다음 단계 (VIC09 결과에 따라)

VIC09가 개선되지 않는다면, 더 근본적인 구조 개편 검토:
- State에 gait phase 추가: 보행 주기(0~1)를 observation에 포함하여 발 움직임을 명시적으로 유도
- CCF curriculum learning: Stage 1에서 CCF를 0으로 고정하는 대신, reward penalty로 CCF를 점진적으로 활성화

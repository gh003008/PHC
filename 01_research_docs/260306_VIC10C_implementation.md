# VIC10C 구현 계획: 단일 모션 + Energy Penalty 추가 완화 (260306)

## 1. 목표

VIC09에서 단일 모션 사용과 energy penalty 1/10 완화를 동시에 적용하여
av reward 272.2, av steps 84.4로 지금까지의 최고 성능을 달성했다.

VIC10에서는 원인 분리 ablation과 추가 개선 방향을 분리하여 세 가지 후보를 정의했다.

VIC10A (ablation - 단일 모션 효과 분리):
- motion_file: 30개 복원 (amass_isaac_walking_forward.pkl)
- power_coefficient: 0.000005 유지
- 목적: 단일 모션만 제거했을 때 성능 변화로 단일 모션의 기여도 측정

VIC10B (ablation - energy penalty 효과 분리):
- motion_file: 단일 모션 유지 (amass_isaac_walking_forward_single.pkl)
- power_coefficient: 0.00005 복원 (원래 값, VIC09의 10배)
- 목적: energy penalty만 원래대로 복원했을 때 성능 변화로 에너지 완화의 기여도 측정

VIC10C (계속 개선 - energy penalty 추가 완화):
- motion_file: 단일 모션 유지
- power_coefficient: 0.0000005 (VIC09의 1/10, 총 VIC08 대비 1/100)
- 목적: energy penalty를 더 낮췄을 때 추가 개선 가능 여부 확인

현재 진행: VIC10C. 에너지 완화가 VIC09 개선의 주요 원인이라는 가정 하에
ablation 없이 방향성을 계속 밀어붙여 빠르게 가능성을 탐색한다.
VIC10A, B는 VIC10C 결과에 따라 필요 시 후속 실험으로 진행할 수 있다.

## 2. 변경 사항 (VIC09 대비)

### 2-1. Energy Penalty 추가 완화

변경:
- power_coefficient: 0.000005 → 0.0000005 (1/10)

의도:
- VIC09에서 0.00005 → 0.000005 (1/10)으로 낮춰 성능이 개선됐으므로,
  같은 방향으로 한 번 더 낮춰서 추가 개선 여부를 확인한다.
- 지나치게 낮추면 jittering 억제 효과가 완전히 사라질 수 있으나,
  VIC09에서 시각화 상 jittering은 문제가 되지 않았으므로 여유가 있다.
- power_coefficient=0.0000005는 VIC08(0.00005) 대비 1/100이다.

### 2-2. 단일 모션 유지

유지:
- motion_file: "sample_data/amass_isaac_walking_forward_single.pkl" (1개)

VIC09와 동일하게 단일 모션을 유지한다.
energy penalty 변화만 가한다.

## 3. 유지 사항

- CCF gradient noise fix (VIC06에서 도입): pre_physics_step에서 Stage 1 시 CCF = 0 유지
- reward_specs: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1
- task_reward_w=0.5, disc_reward_w=0.5
- humanoid_im_vic.py: VIC09와 동일 (추가 수정 없음)

수정 파일:
- phc/data/cfg/env/env_im_walk_vic.yaml: power_coefficient만 변경
- phc/data/cfg/learning/im_walk_vic.yaml: name VIC10C로 변경
- humanoid_im_vic.py: 변경 없음

## 4. 기대 효과

에너지 완화가 VIC09 개선의 주요 원인이었다면:
- power_coefficient를 추가로 낮추면 더 적극적인 보행이 허용되어 추가 성능 개선 가능
- av reward > 272.2, av steps > 84.4 달성 여부로 에너지 완화 방향의 한계 탐색

에너지 완화가 주요 원인이 아니었다면:
- VIC10C 결과가 VIC09와 유사하거나 낮을 수 있음
- 이 경우 단일 모션(VIC10A 비교)이 주요 원인임이 사후 추론 가능

## 5. 다음 단계 (VIC10C 결과에 따라)

VIC10C가 개선되면:
- 에너지 완화 방향이 유효함을 확인, energy penalty를 더 낮추거나 완전히 제거 실험 가능
- 또는 Stage 2 (Learnable CCF)로 전환하여 VIC 본래 목표 진행

VIC10C가 개선되지 않으면:
- 단일 모션이 주요 원인일 가능성 -> VIC10A (30개 모션으로 복원, 에너지 낮게 유지) 비교 가능
- 또는 더 근본적인 구조 개편 (state에 gait phase 추가, CCF curriculum with reward penalty)

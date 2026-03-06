# VIC09 결과 분석 (260306)

## 1. 실험 설정

VIC03~08이 av reward 207~254 범위에서 벗어나지 못하는 원인으로
energy penalty와 motion 다양성을 지목하고 두 가지를 동시에 변경한 실험.

변경 사항 (VIC08 대비):
- motion_file: amass_isaac_walking_forward.pkl (30개) → amass_isaac_walking_forward_single.pkl (1개)
  선택 모션: 0-KIT_11_WalkingStraightForwards05_poses (직진 보행)
- power_coefficient: 0.00005 → 0.000005 (1/10)

유지 사항:
- CCF gradient noise fix (VIC06에서 도입)
- reward_specs: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1
- task_reward_w=0.5, disc_reward_w=0.5
- humanoid_im_vic.py: VIC08과 동일 (추가 수정 없음)

## 2. 학습 경과

15,000 에폭 정상 완료.

학습 곡선:
- VIC03~08 대비 전반적으로 높은 reward 수준으로 수렴
- 학습 초반부터 reward가 빠르게 상승하여 이전 실험들보다 높은 최종값 도달

## 3. 정량적 평가

평가 환경: num_envs=16, test 모드, 최종 체크포인트 (15000 에폭)

| 지표 | VIC03 | VIC07 | VIC08 | VIC09 |
| :--- | :--- | :--- | :--- | :--- |
| 평가 Avg Reward | 254.0 | 207.8 | 223.8 | 272.2 |
| 평가 Avg Steps | 79.1 (2.6초) | 65.9 (2.2초) | 69.8 (2.3초) | 84.4 (2.8초) |

VIC09가 지금까지 VIC 실험 중 최고 성능. VIC03 대비 av reward +18.2, av steps +5.3.
V4(av reward ~461, av steps ~143)에는 여전히 미치지 못하나, 처음으로 VIC03을 넘어섰다.

## 4. 시각화 관찰

- 발을 시원시원하게 들고 스윙하는 보행 패턴이 관찰됨
- 걷는 개체 수가 이전 실험들 대비 확연히 증가
- 이전 실험들의 "작은 보폭으로 버티기" 경향이 줄어든 것으로 관찰됨
- VIC03~08 대비 더 자연스럽고 적극적인 보행 스타일

## 5. 분석: 무엇이 성능을 올렸나

단일 모션과 energy penalty 완화 두 가지를 동시에 변경했으므로 원인을 분리할 수 없다.
각 변화의 가능한 기여:

(1) Energy penalty 완화 (power_coefficient 1/10)
- 기존 0.00005는 보행 시 발생하는 토크와 속도에 패널티를 부과하여 "작은 동작으로 버티기"가 유리했음
- 0.000005로 낮추면 보행에 따른 에너지 패널티가 크게 줄어 더 큰 보폭의 보행이 허용됨
- 시각화에서 더 적극적인 발 스윙이 관찰된 것과 일치

(2) 단일 모션 사용
- 30개 다양한 스타일의 모션을 동시 학습 시 AMP discriminator와 policy 모두 더 어려운 문제를 풀어야 함
- 단일 모션으로 줄이면 AMP discriminator가 단일 스타일만 구분하면 되고, policy도 하나의 step cycle에 집중 가능
- 목표 분포가 단순해져서 imitation reward 달성이 쉬워짐

현재 어느 쪽이 주된 원인인지 불명확하다.
원인 분리를 위해 VIC10 ablation 실험을 계획한다.

## 6. 다음 실험 방향 (VIC10)

원인 분리 및 추가 개선을 위한 세 가지 후보:

VIC10A (ablation - 단일 모션 효과 분리):
- motion_file: 30개 복원 (amass_isaac_walking_forward.pkl)
- power_coefficient: 0.000005 유지 (VIC09 낮은 에너지 유지)
- VIC09 대비 단일 모션만 30개로 복원 → 단일 모션의 기여도 측정

VIC10B (ablation - energy penalty 효과 분리):
- motion_file: 단일 모션 유지 (amass_isaac_walking_forward_single.pkl)
- power_coefficient: 0.00005 복원 (원래 값)
- VIC09 대비 energy penalty만 원래대로 복원 → energy penalty 완화의 기여도 측정

VIC10C (계속 개선 - energy penalty 추가 완화):
- motion_file: 단일 모션 유지
- power_coefficient: 0.0000005 (VIC09의 1/10, 추가 완화)
- VIC09가 에너지 완화 방향으로 성능이 개선됐다는 가정 하에 더 낮춰서 검증

현재 진행: VIC10C (계속 개선 방향으로 선택). VIC09에서의 개선이 에너지 완화
방향으로 해석하여, 더 낮춘 결과를 먼저 확인한다.
VIC10A, VIC10B는 VIC10C 결과에 따라 필요 시 후속 실험으로 진행한다.

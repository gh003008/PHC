# VIC07 결과 분석 (260304)

## 1. 실험 설정

VIC06 (CCF gradient noise 제거) 위에 두 가지 구조 수정을 추가한 실험.
지터링 해결과 상체 자유도 낭비 억제를 동시에 적용.

변경 사항 (VIC06 대비):
- humanoid_im_vic.py: action rate penalty 구현
  - __init__: _action_rate_penalty_w, _last_actions, _upper_body_dof_start, _upper_body_rom_limit 초기화
  - pre_physics_step: _last_actions 버퍼 초기화 (첫 스텝)
  - _compute_reward: penalty = -w * sum((a_t - a_{t-1})^2) 계산 후 reward에 합산, 이후 _last_actions 업데이트
  - _reset_envs: 에피소드 종료 시 _last_actions[env_ids] = 0 리셋
  - _compute_torques: 상체 q_targets (DOF 33~68, Neck~Hand) *= upper_body_rom_limit (0.5)
- env_im_walk_vic.yaml: action_rate_penalty_w=0.005, upper_body_rom_limit=0.5 추가

수정 파일:
- phc/env/tasks/humanoid_im_vic.py: action rate penalty, 상체 스케일링, _reset_envs 추가
- phc/data/cfg/env/env_im_walk_vic.yaml: action_rate_penalty_w, upper_body_rom_limit 추가
- phc/data/cfg/learning/im_walk_vic.yaml: name VIC07로 변경

구현 상 주의점:
- 상체 ROM 제한을 pd_tar 절댓값 클램핑으로 구현했다가 CUDA illegal memory access 발생.
  reference motion 초기화 시 관절이 클램핑 범위 밖에 있어 PD 토크가 폭발하는 문제였음.
  q_targets (액션 스케일)을 줄이는 방식으로 수정하여 해결.

## 2. 학습 경과

15,000 에폭 완료. 정상 종료.

학습 곡선 형태 (사용자 관찰):
- VIC03~06: 위로 볼록한 형태로 단조 증가 후 수렴 (전형적인 RL 학습 곡선)
- VIC07: S자형 - 초반에 아래로 볼록 (학습 느림), 변곡점 이후 위로 볼록 (급격히 개선), 이후 수렴
- 수렴값 자체는 이전 실험과 비슷한 수준

## 3. 정량적 평가

평가 환경: num_envs=16, test 모드, 최종 체크포인트 (15000 에폭)

| 지표 | VIC03 | VIC04 | VIC05 | VIC07 |
| :--- | :--- | :--- | :--- | :--- |
| 평가 Avg Reward | 254.0 | 218.5 | 224.3 | 207.8 |
| 평가 Avg Steps | 79.1 (2.6초) | 67.8 (2.3초) | 69.3 (2.3초) | 65.9 (2.2초) |

VIC07이 지금까지 시도한 실험 중 가장 낮은 수치. VIC03 대비 av reward -46.2, av steps -13.2.

## 4. 시각화 관찰

- 발을 떼려는 시도 자체는 이전보다 있어 보임
- 하지만 실제로 보행에 성공하는 개체 수가 VIC03~05 대비 감소
- 걷는 개체가 더 줄어든 것으로 관찰됨
- 지터링이 줄었는지는 명확하지 않음

## 5. 분석: 왜 더 안좋아졌는가

(1) Action rate penalty가 보행을 억제했을 가능성
보행은 본질적으로 다리를 번갈아 사용하는 주기적 동작이므로, 액션이 규칙적으로 크게 변한다.
penalty = -0.005 * sum((a_t - a_{t-1})^2) 는 이 변화 자체를 패널티로 처리.
결과적으로 네트워크가 "변화가 적고 작은 액션을 유지"하는 전략을 학습하게 유도되었을 수 있다.
즉, 지터링(고주파 랜덤 변화)뿐만 아니라 보행에 필요한 저주파 대진폭 변화까지 억제했을 가능성이 있다.

(2) S자형 학습 곡선이 이를 지지
초반에 학습이 매우 느렸던 것은, penalty 때문에 탐색 자체가 억제됐기 때문으로 해석 가능.
어느 시점에서 네트워크가 "penalty를 피하면서도 reward를 얻는" 협소한 전략을 찾으면서 급격히 개선.
그러나 그 전략이 지터링 없는 보행이 아니라, 작은 액션으로 버티는 전략이었을 수 있음.

(3) Upper body ROM limit이 보행 균형을 방해했을 가능성
인간 보행에서 팔 흔들기는 각운동량 보상(angular momentum compensation) 역할을 한다.
상체 액션을 0.5배로 줄이면 팔 흔들기 범위가 줄어들어 하체 보행과의 균형 잡기가 어려워진다.
에이전트가 팔을 통한 균형 보조 없이 하체만으로 보행을 유지해야 하므로 더 어려운 문제가 됨.

(4) 두 변화를 동시에 적용해서 원인 분리 불가
action rate penalty와 upper body ROM limit 중 어느 쪽이 성능 저하의 주원인인지 알 수 없다.
두 변화를 분리하여 각각 단독 실험이 필요하다.

(5) action_rate_penalty_w=0.005 계수의 스케일 재검토
초기 설계 시 138차원에서 각 dim 변화량이 ~0.1일 때 penalty ~0.007로 계산했다.
하지만 보행 중 액션 변화량이 이보다 클 경우 (각 dim ~0.3 수준이면 penalty ~0.062) reward에 비해 큰 영향을 줄 수 있다.
적절한 계수를 찾기 위해 더 작은 값(0.001 또는 0.0005)에서 시작하는 것이 필요했을 수 있다.

## 6. 다음 단계 제안

원인 분리를 위해 단일 변인 실험 필요:

VIC08 후보 A: action rate penalty만 제거, upper body ROM limit만 유지
- upper body ROM limit 단독 효과 검증

VIC08 후보 B: upper body ROM limit 제거, action rate penalty만 유지 (계수 축소: 0.001)
- 더 작은 penalty에서 지터링 억제 효과 검증

VIC08 후보 C: 두 변화 모두 제거하고 VIC06 (CCF noise fix만)의 순수 효과를 평가
- VIC06을 실제로 시각화/평가하지 않았으므로, CCF noise fix 단독 효과가 얼마나 되는지 모름

현재 가장 큰 미지수는 VIC06 (CCF noise fix 단독)의 실제 성능이다.
VIC03~05가 동일한 수준에서 멈춘 근본 원인이 CCF noise였다면,
VIC06은 V4 수준(av reward ~461, av steps ~143)에 근접했을 수도 있다.
이를 확인하지 않고 VIC07로 넘어갔기 때문에, CCF noise fix 효과가 가려진 상태.

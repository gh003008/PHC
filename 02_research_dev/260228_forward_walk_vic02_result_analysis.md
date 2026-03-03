# VIC02 결과 분석 (260301)

## 1. 실험 설정

VIC01에서 reward weight를 변경한 실험.
- task_reward_w: 0.3 -> 0.5
- disc_reward_w: 0.7 -> 0.5
- 나머지 설정은 VIC01과 동일 (Stage 1, CCF=0 고정)
- 15,000 에폭 학습

## 2. 결과

| 지표 | VIC01 | VIC02 |
| :--- | :--- | :--- |
| 평가 Avg Reward | (VIC01 수준) | 222.3 |
| 평가 Avg Steps | (VIC01 수준) | 69.6 (약 2.3초) |

wandb 비교:
- returns, mb_rewards: VIC01 대비 소폭 상승
- body_rot: VIC01 대비 소폭 하락
- 나머지 지표: 거의 동일

## 3. 시각화 관찰

- 대부분의 에이전트가 발을 떼지 못하고 서있기만 함
- reference motion이 먼저 출발하고, 에이전트는 따라가지 못해 거리가 벌어지면 종료
- 넘어지는 것은 아니고, 몸 전체가 굳어있는 느낌
- 일부 걷는 에이전트도 reference 대비 보폭이 짧음
- 걷는 것처럼 보이는 에이전트도 실제로는 리셋 시 받은 초기 운동량(momentum)으로 미끄러지듯 이동하는 것

## 4. 치명적 버그 발견

VIC02 분석 과정에서 PD 게인 버그를 발견했다.

원인: VIC의 _compute_torques()에서 IsaacGym의 dof_prop['stiffness']와 dof_prop['damping']을 읽어서 p_gains, d_gains로 사용하고 있었다. 그런데 SMPL MJCF의 motor actuator는 IsaacGym의 dof_prop에 stiffness/damping 값을 채우지 않는다. 결과적으로 p_gains = 0, d_gains = 0이 되어 모든 토크가 0이었다.

즉, VIC01과 VIC02 모두 에이전트가 아무런 근력 없이 학습된 것이다.

V4 (isaac_pd)에서는 IsaacGym 엔진이 MJCF motor actuator를 내부적으로 처리하기 때문에 문제가 없었다. VIC는 DOF_MODE_EFFORT를 사용해 수동으로 토크를 계산하기 때문에, 엔진이 채우지 않는 dof_prop 값에 의존하면 안 된다.

에이전트가 "걷는 것처럼 보인" 이유: 에피소드 리셋 시 reference motion의 위치뿐 아니라 속도(velocity)도 초기값으로 설정된다. 이 초기 운동량으로 관성에 의해 잠시 앞으로 미끄러지다가, 토크가 없으므로 자세를 유지하지 못하고 reference와 멀어져 종료되는 것.

## 5. 결론

VIC01, VIC02의 모든 결과는 토크=0 상태에서의 학습이므로 유효하지 않다. PD 게인을 MJCF XML에서 직접 파싱하는 수정이 필요하며, 이를 VIC03에서 적용한다.

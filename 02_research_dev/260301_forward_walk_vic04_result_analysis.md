# VIC04 결과 분석 (260301)

## 1. 실험 설정

VIC03에서 task_reward_w를 높여 imitation 추종 압력을 강화한 실험.

변경 사항:
- task_reward_w: 0.5 -> 0.7
- disc_reward_w: 0.5 -> 0.3
- 나머지 설정은 VIC03과 동일

의도: "서있기만 하면 손해"가 되도록 imitation reward 비중을 높여, 에이전트가 적극적으로 reference motion을 추종하게 유도.

## 2. 결과

| 지표 | VIC03 (task 0.5) | VIC04 (task 0.7) |
| :--- | :--- | :--- |
| 학습 말미 Avg Reward | ~200 | ~200 |
| 학습 말미 Avg Steps | ~70 | ~70 |
| 평가 Avg Reward | 254.0 | 218.5 |
| 평가 Avg Steps | 79.1 (2.6초) | 67.8 (2.3초) |

VIC03 대비 오히려 소폭 하락했다.

## 3. 시각화 관찰

- VIC03과 시각적으로 거의 차이 없음
- 잘 걷는 에이전트, 서서 떨리는 에이전트 분포가 유사
- task_reward_w 변경만으로는 "서서 떨리기" 문제를 해결하지 못함

## 4. 분석

(1) task_reward_w 증가가 효과 없었던 이유
- task reward 자체가 위치(w_pos=0.5), 자세(w_rot=0.3), 속도(w_vel=0.1), 각속도(w_ang_vel=0.1)의 가중합
- 서있기만 해도 초반 몇 스텝은 reference와 가깝기 때문에 위치/자세 reward가 높음
- 서있을 때 속도 reward(w_vel=0.1)만 낮아지는데, 비중이 0.1로 작아서 패널티 효과가 미미
- 따라서 task_reward_w를 아무리 올려도, 그 안에서 속도 추종 비중이 낮으면 움직임을 유도하기 어려움

(2) disc_reward_w 감소의 부작용
- AMP discriminator는 "인간다운 움직임"을 보상
- 비중을 0.5에서 0.3으로 줄이면, 실제로 걷는 동작의 스타일 보상이 약해짐
- 이미 걷는 에이전트의 보행 품질이 떨어질 수 있음

(3) 핵심 문제는 속도 추종 가중치
- 현재 w_vel=0.1은 전체 imitation reward의 10%에 불과
- 에이전트 입장에서 "가만히 서서 위치/자세만 맞추기" vs "걸어서 속도까지 맞추기" 중 전자가 더 쉬운 전략
- w_vel을 높이면 정지 상태에 대한 불이익이 커져서 움직임이 유도될 수 있음

(4) power_reward의 역효과 가능성
- power = |torque x velocity| -> 계수 0.00005로 패널티
- 서있으면 velocity가 낮아 penalty 작음, 걸으면 velocity 커서 penalty 큼
- 움직이지 않는 것이 에너지적으로 유리한 구조
- 다만 power_reward는 떨림 억제 용도이므로 제거하면 떨림이 악화될 수 있음

## 5. 다음 단계 제안

1순위: reward_specs에서 w_vel 증가 (0.1 -> 0.2)
- 속도 매칭 비중을 높여 정지 상태에 불이익 부여
- w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1 또는 w_pos=0.5, w_rot=0.2, w_vel=0.2, w_ang_vel=0.1
- task_reward_w는 0.5로 복원 (disc의 스타일 보상도 유지)

2순위: power_coefficient 조정 (0.00005 -> 0.00002 또는 더 낮게)
- 떨림 억제 효과는 유지하되, 움직임 억제 효과를 줄임

3순위: VIC01 분석에서 제안된 상지 CCF 제거
- 액션 공간 축소로 학습 효율 개선

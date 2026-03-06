# VIC05 결과 분석 (260301)

## 1. 실험 설정

VIC04에서 속도 추종 비중이 낮아서 정지 전략이 유리하다는 분석을 바탕으로 w_vel을 높인 실험.

변경 사항:
- reward_specs: w_pos 0.4 -> 0.3, w_vel 0.2 -> 0.3 (속도 추종 비중 강화)
- task_reward_w: 0.7 -> 0.5 (VIC04에서 복원, disc 스타일 보상도 유지)
- disc_reward_w: 0.3 -> 0.5 (VIC03 수준으로 복원)
- 나머지 설정은 VIC04와 동일

의도: "서있으면 속도 패널티가 크다"는 신호를 강화하여 에이전트가 이동을 택하도록 유도.

수정 파일:
- phc/data/cfg/env/env_im_walk_vic.yaml: w_pos, w_vel 변경, task_reward_w/disc_reward_w 복원
- phc/data/cfg/learning/im_walk_vic.yaml: name VIC05로 변경

## 2. 결과

| 지표 | VIC03 | VIC04 | VIC05 |
| :--- | :--- | :--- | :--- |
| 평가 Avg Reward | 254.0 | 218.5 | 224.3 |
| 평가 Avg Steps | 79.1 (2.6초) | 67.8 (2.3초) | 69.3 (2.3초) |

VIC04 대비 소폭 개선됐지만 VIC03보다 낮다. VIC03~05가 모두 av reward 218~254, av steps 67~79 범위에서 거의 동일한 수준.

## 3. 시각화 관찰

- VIC03, VIC04와 시각적으로 구분이 어려운 수준
- 잘 걷는 에이전트가 일부 존재하고, 다수는 서서 고주파 떨림을 보임
- w_vel 변경만으로는 "서서 떨리기" 패턴을 해소하지 못함

## 4. 분석

(1) reward weight 튜닝의 한계 확인
- VIC03, VIC04, VIC05 세 실험이 모두 유사한 결과를 보임
- task_reward_w 변경(VIC04), w_vel 변경(VIC05) 모두 효과가 미미함
- reward weight 조정으로는 해결되지 않는 구조적 문제가 있음

(2) w_vel 증가의 실제 효과
- w_vel=0.3으로 높였지만, 전체 imitation reward에서 여전히 30% 비중
- 에이전트가 서있을 때 위치/자세 reward가 높기 때문에 속도 패널티를 상쇄할 수 있음
- 속도 reward 항이 exp(-k_vel * ||v_pred - v_ref||^2) 형태이므로, 서있을 때 v_pred≈0, v_ref≠0이면 k_vel=1.0 기준으로 penalty가 크지 않을 수 있음

(3) V4 (non-VIC) 대비 성능 격차
- V4는 동일 에폭에서 av reward ~461, av steps ~143 수준
- VIC03~05는 일관되게 V4의 약 50~55% 수준에 머묾
- VIC의 액션 공간이 138차원(q_target 69 + CCF 69)으로 V4의 2배인데, CCF가 PPO 버퍼에 랜덤 값으로 저장되면서 gradient noise를 유발하는 것이 유력한 원인 (-> VIC06에서 수정)

## 5. 다음 단계 결정

reward weight 튜닝은 더 이상 시도하지 않기로 결정. 구조적 문제 해결이 선행되어야 함.

확인된 구조적 문제:
- CCF(69차원)가 Stage 1에서 환경에는 0으로 마스킹되지만, PPO 버퍼에는 랜덤 값이 저장됨
- PPO gradient에서 CCF 69차원이 reward와 무관한 noise를 주입 -> q_target 학습 방해
- 수정 방향: pre_physics_step에서 Stage 1 시 self.actions의 CCF 부분도 0으로 덮어씀 (VIC06)

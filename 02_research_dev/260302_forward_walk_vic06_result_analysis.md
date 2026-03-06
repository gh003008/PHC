# VIC06 결과 분석 (260302)

## 1. 실험 설정

VIC03~05의 공통 원인인 CCF gradient noise를 수정한 실험. reward weight는 VIC03 기준으로 복원하여 통제변인을 CCF noise 수정 하나로 한정.

변경 사항:
- humanoid_im_vic.py pre_physics_step: Stage 1에서 self.actions의 CCF 69차원을 0으로 덮어씀 (3줄 추가)
  - 기존: CCF가 마스킹되어 환경에는 0이 적용되지만, PPO 버퍼에는 랜덤 값이 저장됨
  - 수정: PPO 버퍼에도 CCF=0을 저장 -> gradient에서 CCF noise 제거
- reward_specs: w_pos=0.3/w_vel=0.3(VIC05) -> w_pos=0.4/w_vel=0.2(VIC03 기준으로 복원)
- task_reward_w=0.5, disc_reward_w=0.5 유지

수정 파일:
- phc/env/tasks/humanoid_im_vic.py: pre_physics_step에 CCF zeroing 3줄 추가
- phc/data/cfg/env/env_im_walk_vic.yaml: reward_specs w_pos, w_vel 복원
- phc/data/cfg/learning/im_walk_vic.yaml: name VIC06으로 변경

## 2. 학습 경과

총 15,000 에폭 목표로 시작. 에폭 7642에서 GPU OOM 크래시 발생.
이후 60GB 디스크 공간 확보 후 7700 에폭 체크포인트에서 이어서 학습. 최종 15,000 에폭까지 완료.

wandb 학습 곡선은 OOM 직전인 ~7642 에폭까지만 기록됨 (이후 재시작 분도 wandb sync 완료).

## 3. 정량적 평가

평가 없이 VIC07로 넘어감 (학습 곡선 확인 후 수렴 판단, 별도 시각화/평가 미진행).

## 4. 분석

(1) CCF gradient noise 수정의 의미
- Stage 1에서 CCF는 어차피 0이 되므로, PPO 버퍼에도 0을 저장하는 것이 일관된 동작
- 랜덤 CCF 값이 log_prob 계산에 포함되면, PPO가 "어떤 CCF 값이 좋은가"를 학습하게 됨
- CCF는 reward와 무관하므로 이 gradient가 순수 noise로 작용 -> q_target 부분 학습 방해
- 수정 후 q_target 69차원만 실질적인 gradient를 받게 되어 V4와 동등한 학습 조건

(2) V4 수준에 도달했는지 여부
- VIC06 학습 결과를 시각화/평가하지 않아 정량적 확인 불가
- 학습 곡선 상에서 reward 추이만 확인한 수준
- CCF noise 제거만으로 충분하지 않다면, 추가 구조적 문제(jittering, 상체 자유도 낭비)가 남아있음

(3) 한계
- 15,000 에폭 대비 V4 수준 달성 여부를 확인하지 못함
- jittering 여부, 보행 품질을 시각화로 확인하지 않음
- CCF noise 제거 효과를 단독으로 검증하지 못한 채 VIC07로 넘어감

## 5. 다음 단계 결정

VIC07로 넘어가서 두 가지 추가 구조 수정 동시 적용:
- Action rate penalty (w=0.005): 연속 액션 변화량 패널티로 jittering 억제
- Upper body ROM limit (scale=0.5): 상체 관절 액션 스케일을 절반으로 줄여 학습 공간 축소

VIC06 체크포인트는 별도로 보관하지 않고 VIC07을 처음부터 학습.

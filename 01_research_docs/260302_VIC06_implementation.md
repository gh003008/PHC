# VIC06 구현: Stage 1 CCF Gradient Noise 제거 (260302)

## 1. 문제 정의

VIC03~05에서 reward weight를 다양하게 조정했으나 결과가 거의 동일했다 (av reward 218~254, av steps 67~79).
V4 (isaac_pd, CCF 없음)는 동일 에폭에서 av reward 461.8, av steps 143.5를 달성했다.
reward tuning이 아닌 구조적 문제가 있다.

## 2. 원인 분석: PPO 버퍼에 저장되는 랜덤 CCF

현재 코드 흐름 (Stage 1, CCF=0 고정):

(1) 정책 네트워크가 138차원 액션 출력 (q_target 69 + CCF 69)
(2) pre_physics_step에서 self.actions = actions.clone() -> PPO 버퍼에 원본 액션 저장 (랜덤 CCF 포함)
(3) _compute_torques에서 CCF를 0으로 마스킹 -> 환경에는 CCF=0 적용
(4) reward는 CCF와 무관하게 계산됨
(5) PPO가 log_prob을 계산할 때, 버퍼에 저장된 랜덤 CCF 값도 포함됨

문제: PPO의 policy gradient에서 CCF 69차원이 reward와 무관한 noise를 gradient에 주입한다.
네트워크의 hidden layer는 q_target과 CCF가 공유하므로, CCF의 gradient noise가 q_target 학습을 방해한다.

V4는 69차원 액션만 출력하므로 이 문제가 없다.

## 3. 수정 방안

pre_physics_step에서 Stage 1일 때 self.actions의 CCF 부분도 0으로 덮어씌운다.

수정 전 (humanoid_im_vic.py, pre_physics_step):
  self.actions = actions.to(self.device).clone()

수정 후:
  self.actions = actions.to(self.device).clone()
  if self._vic_enabled and self._vic_curriculum_stage == 1:
      self.actions[:, self._num_actions:] = 0  # CCF를 PPO 버퍼에도 0으로 저장

효과:
- PPO 버퍼에 CCF=0이 저장됨
- log_prob 계산에서 CCF 차원이 항상 동일한 값(0)이므로 gradient에 noise를 주입하지 않음
- 네트워크 구조는 138차원 출력 유지 -> Stage 2 전환 시 체크포인트 호환성 유지
- _compute_torques의 기존 마스킹 코드도 유지 (이중 안전장치)

## 4. 기대 효과

- q_target 학습에 CCF noise가 없어져서 V4에 가까운 수렴 가능
- 이 수정으로 V4 수준 성능이 나오면: CCF gradient noise가 근본 원인이었음
- 여전히 안 나오면: manual PD 자체의 문제 (PD 주기 60Hz vs 120Hz 등)를 다음으로 조사

## 5. 추가 변경사항

- reward_specs를 VIC03 (V4와 동일) 설정으로 복원: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1
- task_reward_w: 0.5, disc_reward_w: 0.5 유지 (VIC03 기준)
- 통제변인: CCF gradient noise 제거만 변경, 나머지는 VIC03과 동일

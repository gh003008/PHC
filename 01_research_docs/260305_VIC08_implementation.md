# VIC08 구현 계획: VIC07 부작용 제거, CCF Noise Fix 단독 재검증 (260305)

## 1. 목표

VIC07에서 도입한 두 가지 수정(action rate penalty, upper body ROM limit)이 오히려 성능을 저하시켰다.
VIC08은 이 두 변경을 모두 제거하고, VIC06에서 적용한 CCF gradient noise fix만 남긴 상태로
15,000 에폭을 처음부터 온전히 학습한다.

VIC06은 7642 에폭에서 OOM으로 중단된 후 7700 에폭 체크포인트에서 재시작했기 때문에,
학습 연속성이 깨진 상태였다. VIC08은 동일한 코드로 중단 없이 15,000 에폭을 완주하여
CCF noise fix의 순수 효과를 깨끗하게 측정한다.

## 2. VIC07 문제 요약 및 제거 이유

(1) Action rate penalty (w=0.005) 제거
- 의도: 연속된 액션 변화량 패널티로 jittering 억제
- 부작용: 보행은 다리를 번갈아 크게 움직이는 주기적 동작이므로 액션 변화가 본질적으로 크다.
  penalty가 jittering뿐 아니라 보행에 필요한 저주파 대진폭 변화까지 억제.
  초반에 학습 탐색 자체가 억제되는 S자 학습 곡선으로 나타남.

(2) Upper body ROM limit (scale=0.5) 제거
- 의도: 상체 관절 액션 범위를 절반으로 줄여 학습 공간 축소, 하지 학습 집중
- 부작용: 인간 보행에서 팔 흔들기는 각운동량 보상(angular momentum compensation) 역할.
  상체 스케일을 줄이면 팔을 통한 균형 보조가 약해져서 오히려 보행이 어려워짐.

(3) 두 변화 동시 적용의 문제
- 어느 쪽이 성능 저하의 주원인인지 원인 분리가 불가능했음.
- VIC08에서는 두 변화를 모두 제거하여 VIC06 기반으로 돌아감.

## 3. VIC08 변경 사항 (VIC07 대비)

### 3-1. 코드 변경 (humanoid_im_vic.py)

제거한 것:
- __init__에서 _action_rate_penalty_w, _upper_body_dof_start, _upper_body_rom_limit 초기화 제거
- pre_physics_step에서 _last_actions 초기화 코드 제거
- _compute_reward에서 action rate penalty 계산 블록 제거, _last_actions 업데이트 제거
- _reset_envs 메서드 전체 제거 (부모 클래스 호출만 남김)
- _compute_torques에서 상체 q_targets 스케일링 제거

유지한 것 (VIC06에서 도입):
- pre_physics_step에서 Stage 1 시 self.actions의 CCF 69차원을 0으로 덮어씀
  ```python
  if self._vic_enabled and self._vic_curriculum_stage == 1:
      self.actions[:, self._num_actions:] = 0
  ```

### 3-2. 설정 변경 (env_im_walk_vic.yaml)

제거한 것:
- action_rate_penalty_w: 0.005
- upper_body_rom_limit: 0.5

유지한 것: 모든 나머지 설정 (VIC06과 동일)

### 3-3. 학습 설정 (im_walk_vic.yaml)

- name: VIC07 -> VIC08
- 나머지 hyperparameter는 VIC06/07과 동일

## 4. VIC08과 VIC06의 관계

코드 및 설정 면에서 VIC08 = VIC06과 사실상 동일하다.
차이점:
- VIC06: OOM으로 7642 에폭 중단 -> 7700 에폭 체크포인트에서 재시작 (학습 연속성 손상)
- VIC08: 처음부터 15,000 에폭 완주 (학습 연속성 보장)

"매 학습마다 결과가 조금씩 다를 수 있나?"에 대한 답:
seed=0으로 고정되어 있지만, CUDA 연산의 비결정론적 특성(cuDNN 내부 병렬 연산 순서)으로 인해
완전히 동일한 결과가 보장되지 않는다. 특히 GPU 점유 상태, 다른 프로세스 간섭 등에 따라
약간씩 다를 수 있다. 그러나 동일한 조건이라면 평균적으로 유사한 수렴값을 보인다.

## 5. 기대 효과

- VIC06이 OOM 없이 정상 완주했다면 어떤 결과였을지를 추정할 수 있음
- CCF noise fix 단독의 순수 효과를 정량적으로 측정 (시각화 + 평가 포함)
- 결과가 VIC03~05 (av reward ~220~254) 수준이면: CCF noise fix만으로는 부족 -> 다른 접근 필요
- 결과가 V4 (av reward ~461) 수준에 가깝다면: CCF noise fix가 핵심이었고, VIC07은 역효과였음을 확인

## 6. 다음 실험 방향 (VIC08 결과에 따라)

시나리오 A: VIC08이 V4 수준에 근접할 경우
- CCF noise fix가 핵심 원인이었음 확인
- Stage 2 (Learnable CCF)로 전환하여 VIC 본래 목표로 진행

시나리오 B: VIC08도 VIC03~05와 유사한 수준일 경우
- CCF noise fix만으로는 부족
- PD 주기 문제 (60Hz vs V4의 120Hz), 관측 공간 차이 등 다른 원인 탐색 필요
- 또는 jitter 억제를 다른 방식으로 재시도 (더 작은 penalty w, 주파수 도메인 접근)

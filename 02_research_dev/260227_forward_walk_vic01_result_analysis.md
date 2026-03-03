# VIC01 결과 분석 및 다음 단계 (260227)

## 1. VIC01 현재 문제

Stage 1 (CCF=0 고정) 15,000 에폭 학습 후 평가 결과:
- V4 (isaac_pd, CCF 없음) 대비 보행 성능 현저히 저하
- 몸 전체가 저주파(2-3Hz)로 떨림
- 발을 제대로 떼지 못하고, 아장아장 따라가다 실패
- 서있는 것은 가능하나 전방 보행 추종 불가

## 2. PD 제어 주기 분석 (팩트 체크)

V4와 VIC01의 sim/physics 설정은 동일하다:
- step_dt: 1/60 (60Hz), substeps: 2, controlFrequencyInv: 2
- 정책 주기: 30Hz (둘 다 동일)

차이는 control_mode 뿐이다:
- V4: isaac_pd (env_im_walk.yaml:90)
- VIC01: pd (env_im_walk_vic.yaml:97)

실제 PD 적용 빈도 차이:
- V4 (isaac_pd): set_dof_position_target_tensor로 타겟 설정 후, PhysX 엔진이 매 substep마다 현재 joint state를 읽어 PD 토크를 재계산한다. 실질 PD 주기는 120Hz (60Hz x 2 substeps).
- VIC01 (pd): _physics_step 루프에서 _compute_torques -> set_dof_actuation_force_tensor -> simulate 순으로 실행된다. simulate 내부 2 substeps 동안 토크는 고정값으로 유지된다. 실질 PD 주기는 60Hz.

결론: PD 주기 차이는 120Hz vs 60Hz (2배)이며, 이것이 2-3Hz 저주파 떨림의 주원인은 아닌 것으로 판단한다. PD 주기 차이는 고주파 진동을 유발하지, 저주파 떨림을 만들지 않는다.

## 3. 떨림의 실제 원인 분석

(1) 액션 공간 2배 확장으로 인한 최적화 난이도 증가
- 기존 N차원(Joint Target Pose)에서 2N차원(+ CCF)으로 확장
- Stage 1에서 CCF를 0으로 마스킹하더라도, 네트워크는 여전히 2N 출력을 생성
- 네트워크 용량의 절반이 사실상 무의미한 출력에 소비되고 있음
- 동일 에폭에서의 수렴도가 V4 대비 낮을 수밖에 없음

(2) Energy penalty 부족
- power_coefficient: 0.00005는 V4에서도 사용하는 값
- 액션 공간이 2배로 커지면 불필요한 action 진동을 억제할 별도 텀이 필요
- vic_metabolic_reward_w: 0.01은 코드상 존재하나, Stage 1에서는 CCF=0이므로 실질적 효과 없음

(3) CCF 마스킹에도 gradient가 흐르는 문제
- torch.zeros_like(ccf_raw)로 마스킹해도 네트워크 출력 자체는 생성됨
- Policy gradient에서 CCF 출력에 대한 gradient가 noise로 작용할 가능성

## 4. 시뮬레이션-강화학습 루프 구조

전체 루프는 계층적으로 동작한다:

(계층 1) 강화학습 루프 - play_steps() (amp_agent.py:309)
  horizon_length번 반복하며 experience 수집:
    obs -> policy network -> action -> env.step(action) -> reward, next_obs
  수집된 데이터로 PPO 업데이트 (train_epoch)

(계층 2) 환경 스텝 - step() (base_task.py:216)
  pre_physics_step(actions): 정책 출력을 물리 명령으로 변환
  _physics_step(): 물리 시뮬레이션 실행
  post_physics_step(): obs, reward, done 계산

(계층 3) 물리 스텝 - _physics_step() (humanoid.py:1602)
  control_freq_inv(=2)번 반복:
    pd 모드: _compute_torques -> set_dof_actuation_force_tensor -> simulate
    isaac_pd 모드: simulate (엔진 내부 PD)

(계층 4) PhysX 내부 - simulate() 1회 호출
  substeps(=2)번 반복: 충돌 검출, 구속 조건 풀기, 적분

중요한 점: 여기서 60Hz, 30Hz는 실제 벽시계 시간(wall-clock time)이 아니다.
물리 시뮬레이션은 고정된 가상 시간(simulation time)으로 진행된다.
simulate() 한 번 호출하면 가상 시간이 step_dt(=1/60초)만큼 진행되는 것이며,
이 연산이 실제로 1ms 걸리든 100ms 걸리든 가상 시간은 동일하게 1/60초가 흐른다.

즉, 연산 지연이 발생해도 시뮬레이션 물리 결과에는 영향이 없다.
실시간 시각화에서만 프레임 드랍으로 보일 뿐, 물리 자체는 항상 결정론적(deterministic)이다.

정리하면:
- 정책 1스텝 = 가상 시간 1/30초 (controlFrequencyInv x step_dt = 2 x 1/60)
- 그 안에서 simulate() 2번 호출 = 가상 시간 2/60초
- 각 simulate() 내부에서 substep 2번 = PhysX가 1/120초씩 2번 적분
- 전체: 정책 30Hz -> 환경 60Hz -> PhysX 120Hz (가상 시간 기준)

## 5. 다음 단계 제안

### 5-1. 상지 액션 공간 축소 검토

현재: 전체 관절에 대해 CCF를 출력 (2N 액션)
제안: 하지 관절(Hip, Knee, Ankle, Toe)에만 CCF를 적용하고, 상지는 CCF 없이 기존 PD로 구동

이유:
- 보행 과제에서 상지의 임피던스 조절은 우선순위가 낮다
- 액션 공간을 N + 하지CCF(약 12차원)으로 줄이면 최적화 난이도가 크게 감소
- 하지에서 VIC 효과가 검증되면 상지로 점진적 확장 가능

구현 방법:
- humanoid_im_vic.py의 _compute_torques에서 CCF를 하지 관절 인덱스에만 적용
- 상지 관절의 CCF는 항상 0으로 고정 (impedance_scale = 1.0)
- 네트워크 출력 차원도 N + 하지_DOF로 축소

### 5-2. 공격적 MPJPE 학습

현재 reward_specs에서 위치/회전/속도 가중치가 균형적(0.4/0.3/0.2/0.1)인데,
MPJPE를 낮추는 방향으로 공격적으로 학습하려면:

(A) reward weight 조정:
- w_pos를 0.6~0.8로 높이고, w_vel/w_ang_vel을 줄임
- 속도 추종보다 위치 추종에 집중하면 MPJPE가 더 빠르게 내려감
- 단, 속도 추종이 약해지면 동작의 자연스러움이 줄어들 수 있음

(B) early termination 완화:
- terminationDistance: 0.25 -> 0.5 또는 더 크게
- enableEarlyTermination: False로 임시 비활성화
- 균형을 못 잡더라도 에피소드를 끝까지 진행하면서 위치 추종 학습 기회를 더 줌
- 넘어져도 리셋 안 되면 "넘어지면서도 따라가려는" 행동을 학습할 수 있음

(C) AMP discriminator 비중 조정:
- 현재 im_walk_vic.yaml에서 task_reward_w: 0.3, disc_reward_w: 0.7
- task_reward_w를 0.5~0.7로 올리면 위치 추종에 더 집중
- 대신 동작의 자연스러움(AMP quality)은 양보해야 함

### 5-3. 우선 추천 순서

1순위: 상지 CCF 제거 + CCF 출력 차원 축소 -> 재학습
- 액션 공간 축소만으로도 수렴 속도와 안정성이 개선될 가능성이 높음
- 기존 보행 성능 회복이 최우선

2순위: early termination 완화 (terminationDistance 증가)
- 걷다 넘어지는 경우에도 학습 기회 제공

3순위: reward weight 튜닝 (w_pos 증가)
- 1,2순위 적용 후에도 MPJPE가 높으면 시도

4순위: task_reward_w / disc_reward_w 비율 조정
- 자연스러움을 다소 양보하더라도 추종 성능 확보

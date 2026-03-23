# IsaacGym 통합 계획 — standard_human_model

작성일: 2026-03-17

## 1. 현재 상태

현재 standard_human_model은 CPU에서 독립적으로 동작하는 근골격계 파이프라인이다.
IsaacGym과 연결되어 있지 않으며, 단일 관절 단위 검증만 완료된 상태.

기존 VIC 환경(humanoid_im_vic.py)은 PD 제어 + CCF 임피던스 스케일링으로 토크를 생성한다:

```
torque = kp * 2^ccf * (q_target - q) - kd * 2^ccf * dq
```

통합 목표: 이 PD 토크 계산을 근골격계 파이프라인(compute_torques)으로 대체하거나 보강하는 것.


## 2. 통합 전략 비교

### 전략 A: 완전 대체 (Musculoskeletal-only)

PD 제어를 완전히 제거하고 근골격계 모델만으로 토크 생성.

```
Policy → descending_cmd (20 muscles) → Reflex → Activation → Hill → R^T@F → torque
```

장점: 생체역학적으로 가장 현실적
단점: Action space가 69 DOF → 20 muscles로 완전히 바뀜. 기존 VIC 학습 결과 활용 불가.
      학습 난이도 급격히 상승 (근육 조합으로 보행을 처음부터 학습해야 함).

### 전략 B: 하이브리드 (PD + Musculoskeletal 보강)

기존 VIC PD 제어를 유지하면서, 근골격계 모델의 수동 역학만 추가.

```
torque_pd   = kp * 2^ccf * (q_target - q) - kd * 2^ccf * dq     ← 기존 VIC
torque_bio  = tau_passive(muscle) + tau_ligament + tau_reflex       ← 근골격계
torque_final = torque_pd + alpha * torque_bio
```

장점: 기존 학습 파이프라인 유지. alpha를 점진적으로 올려서 안정적 전환.
      수동 역학(ligament, passive muscle, stretch reflex)만으로도 환자 특성 표현 가능.
단점: 완전한 근육 기반 제어는 아님.

### 전략 C: 2단계 접근 (권장)

1단계: 전략 B(하이브리드)로 시작하여 안정성 확보
2단계: 학습이 안정화되면 전략 A(완전 대체)로 전환

이유: 기존 VIC에서 쌓은 학습 결과를 활용하면서 점진적으로 근골격계 비중을 높일 수 있음.


## 3. 구현 계획 (전략 C 기준)

### Phase 1: 하이브리드 통합

#### 3.1 새 태스크 파일 생성

humanoid_im_vic.py를 복사하여 humanoid_im_vic_msk.py 생성 (MSK = Musculoskeletal).
기존 VIC 코드 수정 금지 원칙 유지.

```
phc/env/tasks/humanoid_im_vic_msk.py    ← 새 파일
```

parse_task.py에 등록:
```python
"HumanoidImVICMSK": humanoid_im_vic_msk.HumanoidImVICMSK
```

#### 3.2 _compute_torques 수정 위치

현재 VIC의 _compute_torques (humanoid_im_vic.py:1380):

```python
def _compute_torques(self, actions):
    # ... CCF 처리 ...
    kp = self.p_gains * impedance_scale
    kd = self.d_gains * impedance_scale
    torques = kp * (pd_tar - self._dof_pos) - kd * self._dof_vel   # ← 여기에 bio torque 추가
    return torch.clamp(torques, -self.torque_limits, self.torque_limits)
```

수정 후:

```python
def _compute_torques(self, actions):
    # ... 기존 CCF/PD 토크 계산 동일 ...
    torques_pd = kp * (pd_tar - self._dof_pos) - kd * self._dof_vel

    # 근골격계 수동 역학 추가
    descending_cmd = self._make_descending_cmd(actions)  # 또는 zeros (수동만)
    torques_bio = self._human_body.compute_torques(
        self._dof_pos, self._dof_vel,
        descending_cmd, dt=self.dt
    )

    # 하이브리드 합산
    alpha = self._msk_blend_alpha  # yaml에서 설정, 0.0~1.0
    torques = torques_pd + alpha * torques_bio

    return torch.clamp(torques, -self.torque_limits, self.torque_limits)
```

#### 3.3 HumanBody 초기화 위치

__init__에서 HumanBody를 생성하되, IsaacGym 환경 생성 이후(super().__init__ 이후)에 호출해야
device가 확정된 상태에서 CUDA 텐서 생성이 가능하다.

```python
class HumanoidImVICMSK(HumanoidImVIC):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # 근골격계 모델 초기화 (super().__init__ 이후 → self.device 확정)
        from standard_human_model.core.human_body import HumanBody
        msk_cfg = cfg["env"].get("msk_config", {})
        self._human_body = HumanBody.from_config(
            muscle_def_path=msk_cfg.get("muscle_def", "muscle_definitions.yaml"),
            param_path=msk_cfg.get("patient_profile", "healthy_baseline.yaml"),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._msk_blend_alpha = msk_cfg.get("blend_alpha", 0.0)  # 초기: bio 토크 비활성
```

#### 3.4 리셋 처리

에피소드 리셋 시 HumanBody의 내부 상태(activation, reflex delay buffer)도 리셋해야 한다.

```python
def _reset_envs(self, env_ids):
    super()._reset_envs(env_ids)
    if hasattr(self, '_human_body'):
        self._human_body.reset(env_ids)
```

현재 VIC의 리셋 위치: _reset_envs() 또는 _reset_env_tensors() 내부.

#### 3.5 Observation 확장

근활성화 상태를 observation에 추가하면 policy가 근육 상태를 인지할 수 있다.

```python
# 추가 obs (선택적)
activation = self._human_body.get_activation()  # (num_envs, 20)
obs = torch.cat([obs_base, activation], dim=-1)
```

obs 차원 증가분: +20 (근육 수)
이는 yaml에서 on/off 설정 가능하게 한다.

#### 3.6 Config 파일

env_im_walk_vic_msk.yaml (새 파일):

```yaml
# 기존 VIC 설정 상속
defaults:
  - env_im_walk_vic

# 근골격계 설정 추가
msk_config:
  muscle_def: "muscle_definitions.yaml"          # standard_human_model/config/ 기준
  patient_profile: "healthy_baseline.yaml"
  blend_alpha: 0.0        # 0.0=PD only, 1.0=MSK fully added
  add_activation_obs: false
  add_muscle_force_obs: false
```


### Phase 2: 완전 대체 (향후)

Phase 1이 안정적으로 동작하면:

1. blend_alpha를 1.0으로 올리고 PD 토크를 제거
2. Action space를 근육 명령 (20 dims) + CCF (8 dims)로 변경
3. descending_cmd를 policy 출력으로 직접 매핑

이 단계에서는 Action space 변경으로 인해 처음부터 학습해야 한다.


## 4. 핵심 기술 이슈

### 4.1 DOF 매핑 문제

standard_human_model의 skeleton.py는 자체 DOF 정의(69 DOFs)를 사용한다.
IsaacGym의 SMPL 모델도 69 DOFs이지만, 순서가 다를 수 있다.

확인 필요 사항:
- standard_human_model의 JOINT_NAMES 순서 vs IsaacGym의 dof_names 순서
- 불일치 시 permutation 인덱스 매핑 테이블 생성

확인 방법:
```python
# IsaacGym 측
dof_names = self.gym.get_actor_dof_names(self.envs[0], self.humanoid_handles[0])

# standard_human_model 측
from standard_human_model.core.skeleton import JOINT_NAMES, JOINT_DOF_RANGE
```

### 4.2 시뮬레이션 주기 (dt)

- IsaacGym dt: 보통 1/60초 (sim_params에서 설정)
- Activation dynamics ODE: dt 의존적 (Euler 적분)
- Reflex delay: step 단위

IsaacGym의 dt를 그대로 activation_dyn.step(dt=self.dt)에 전달하면 된다.
다만, IsaacGym에서 substep을 사용하는 경우 주의 필요:
- substep=2이면 _compute_torques가 2번 호출되지만 activation은 1번만 업데이트해야 할 수 있음
- 현재 PHC는 substep=1 (확인 필요: sim_params.substeps)

### 4.3 Contact Force 전달

stretch reflex의 load reflex (발바닥 접촉 → 신전근 활성)에는 contact force 정보가 필요.
IsaacGym에서 contact force 읽기:

```python
# humanoid_im_vic.py에서 이미 사용 중인 패턴:
contact_forces = self._contact_forces  # (num_envs, num_bodies, 3)

# 발바닥 바디 인덱스 찾기
l_foot_idx = self._body_names.index("L_Ankle")  # 또는 L_Toe
r_foot_idx = self._body_names.index("R_Ankle")
foot_contacts = contact_forces[:, [l_foot_idx, r_foot_idx], :]  # (num_envs, 2, 3)
foot_contact_mag = foot_contacts.norm(dim=-1)  # (num_envs, 2)
```

### 4.4 토크 안정성

근골격계 토크가 PD 토크와 합산될 때 발산할 수 있다.
안전 장치:
- torque_limits 클램핑 (기존 VIC에 이미 있음)
- bio 토크에 별도 스케일링 (blend_alpha 외에 max_bio_torque 설정)
- 초기에 alpha=0에서 시작하여 학습 중 점진적 증가 (커리큘럼)


## 5. 구현 순서 (체크리스트)

```
Phase 1: 하이브리드 통합
[ ] 1. DOF 순서 매핑 확인 (skeleton.py vs IsaacGym dof_names)
[ ] 2. humanoid_im_vic_msk.py 생성 (VIC 상속)
[ ] 3. HumanBody 초기화 추가 (__init__)
[ ] 4. _compute_torques에 bio torque 추가
[ ] 5. _reset_envs에 HumanBody.reset() 추가
[ ] 6. env_im_walk_vic_msk.yaml 생성
[ ] 7. parse_task.py에 HumanoidImVICMSK 등록
[ ] 8. blend_alpha=0으로 기존 VIC 동일 동작 확인 (regression test)
[ ] 9. blend_alpha=0.1~0.3에서 학습 안정성 테스트
[ ] 10. wandb에 bio torque 관련 메트릭 추가 로깅

Phase 2: 완전 대체 (Phase 1 안정화 후)
[ ] 11. Action space 변경 (69 DOF → 20 muscles + 8 CCF)
[ ] 12. descending_cmd를 policy 직접 출력으로 연결
[ ] 13. PD 토크 제거, 근골격계 토크만 사용
[ ] 14. 새로운 학습 시작 (from scratch)
```


## 6. 예상 파일 변경

```
신규 파일:
  phc/env/tasks/humanoid_im_vic_msk.py         ← 메인 태스크
  phc/data/cfg/env/env_im_walk_vic_msk.yaml     ← 환경 설정

수정 파일:
  phc/utils/parse_task.py                       ← 태스크 등록 1줄 추가

기존 파일 수정 없음:
  phc/env/tasks/humanoid_im_vic.py              ← 건드리지 않음
  standard_human_model/                         ← 건드리지 않음
```


## 7. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| DOF 순서 불일치 | 토크가 엉뚱한 관절에 적용 | 매핑 테이블 생성, 검증 테스트 |
| Bio 토크 발산 | 시뮬레이션 crash | torque clamp + alpha 커리큘럼 |
| 학습 속도 저하 | 수렴 시간 증가 | alpha=0에서 warm-up 후 점진적 활성화 |
| Muscle 파라미터 비현실적 | 비정상적 보행 | 검증 실험 결과 기반 튜닝 |
| PhysX 4 한계 | closed chain 불가 | Isaac Lab 전환 시점까지 open chain으로 운용 |

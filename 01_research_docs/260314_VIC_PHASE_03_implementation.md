# VIC_PHASE_03 구현 상세 (260314)

## 1. 실험 목적

VIC_PHASE_02에서 phase-dependent CCF 변화가 미미했던 원인은 reward 구조에 CCF를 phase에 따라 조절해야 할 명시적 동기가 없기 때문이다. VIC_PHASE_03에서는 **생체역학 기반 CCF 보상(bio CCF reward)**을 추가하여 stance/swing에서 적절한 ankle 임피던스 학습을 유도한다.

생체역학적 근거:
- Stance phase (발이 지면 접촉): 체중 지지를 위해 발목 stiffness가 높아야 함
- Swing phase (발이 공중): 다리 스윙을 위해 발목이 compliant해야 함

## 2. 핵심 변경 사항 (VIC_PHASE_02 대비)

### 2-1. Bio CCF Reward 추가
발-지면 contact force를 기반으로 stance/swing을 실시간 판별하고, ankle CCF에 방향성 보상을 제공.

| 조건 | 판별 기준 | CCF 보상 방향 |
|---|---|---|
| Stance leg | foot contact force > 10N | ankle CCF > 0 (stiff) 보상 |
| Swing leg | foot contact force ≤ 10N | ankle CCF < 0 (compliant) 보상 |

보상 가중치: w = 0.05 (기존 imitation reward 대비 매우 작음, 방향성만 제공)

## 3. 코드 수정 내용

### 수정 파일: `phc/env/tasks/humanoid_im_vic.py`

#### 3-1. __init__(): Bio CCF 가중치 읽기
```python
# VIC: Biomechanical CCF reward (contact-force based stance/swing CCF guidance)
self._vic_bio_ccf_reward_w = cfg["env"].get("vic_bio_ccf_reward_w", 0.0)
```

#### 3-2. _compute_torques(): CCF raw 값 저장
bio CCF reward 계산에 필요한 환경별 CCF raw 값을 저장하도록 추가.

```python
# 기존 코드에 1줄 추가
if self._vic_curriculum_stage == 2:
    self._last_ccf_mean = ccf_raw.mean().item()
    self._last_ccf_std = ccf_raw.std().item()
    self._last_ccf_group_mean = ccf_raw.mean(dim=0).detach()
    self._last_ccf_raw = ccf_raw.detach()  # [N, n_groups] 추가됨
```

#### 3-3. _compute_reward(): Bio CCF reward 호출
power reward 계산 이후에 bio CCF reward를 추가로 계산하여 rew_buf에 합산.

```python
# VIC: Biomechanical CCF reward — stance ankle stiff, swing ankle compliant
if self._vic_enabled and self._vic_bio_ccf_reward_w > 0 and self._vic_curriculum_stage == 2:
    bio_ccf_reward = self._compute_bio_ccf_reward()
    bio_ccf_reward[self.progress_buf <= 3] = 0  # 초기 3스텝은 무시 (초기화 노이즈)
    self.rew_buf[:] += bio_ccf_reward
    self.reward_raw = torch.cat([self.reward_raw, bio_ccf_reward[:, None]], dim=-1)
```

#### 3-4. _compute_bio_ccf_reward(): 핵심 신규 메서드

```python
def _compute_bio_ccf_reward(self):
    # 1. Contact force 기반 stance/swing 판별
    contact_forces = self._contact_forces[:, self._contact_body_ids, :]  # [N, 4, 3]
    # contact_bodies 순서: ["R_Ankle"=0, "L_Ankle"=1, "R_Toe"=2, "L_Toe"=3]
    l_foot_force = contact_forces[:, 1, :].norm(dim=-1) + contact_forces[:, 3, :].norm(dim=-1)
    r_foot_force = contact_forces[:, 0, :].norm(dim=-1) + contact_forces[:, 2, :].norm(dim=-1)

    contact_threshold = 10.0  # N
    l_is_stance = (l_foot_force > contact_threshold).float()
    r_is_stance = (r_foot_force > contact_threshold).float()

    # 2. CCF 그룹에서 ankle CCF 추출
    l_ankle_ccf = self._last_ccf_raw[:, 2]  # G2: L_Ankle+Toe
    r_ankle_ccf = self._last_ccf_raw[:, 5]  # G5: R_Ankle+Toe

    # 3. 방향성 보상 계산
    # Stance: CCF > 0 보상 (stiff), Swing: CCF < 0 보상 (compliant)
    l_reward = l_is_stance * clamp(l_ankle_ccf, min=0) + (1-l_is_stance) * clamp(-l_ankle_ccf, min=0)
    r_reward = r_is_stance * clamp(r_ankle_ccf, min=0) + (1-r_is_stance) * clamp(-r_ankle_ccf, min=0)

    bio_reward = (l_reward + r_reward) * 0.5 * self._vic_bio_ccf_reward_w
    return bio_reward
```

보상 특성:
- Stance에서 ankle CCF=1.0이면: clamp(1.0, min=0) = 1.0 → 최대 보상
- Stance에서 ankle CCF=-0.5이면: clamp(-0.5, min=0) = 0.0 → 보상 없음 (벌칙도 없음)
- Swing에서 ankle CCF=-0.5이면: clamp(0.5, min=0) = 0.5 → 보상
- Swing에서 ankle CCF=1.0이면: clamp(-1.0, min=0) = 0.0 → 보상 없음
- 즉, 원하는 방향의 CCF만 보상하고, 반대 방향은 0 (벌칙 없음). Soft guidance 방식.

### 수정 파일: `phc/data/cfg/env/env_im_walk_vic.yaml`
```yaml
# 1줄 추가
vic_bio_ccf_reward_w: 0.05
```

### learning yaml 변경 없음
VIC_PHASE_02와 동일. exp_name만 VIC_PHASE_03으로 변경해야 했으나, 실수로 VIC_PHASE_02로 유지되어 체크포인트가 덮어쓰여짐. 이후 수동으로 VIC_PHASE_03.pth로 복사하여 보존.

## 4. Obs/Action 구조

VIC_PHASE_02와 동일. Bio CCF reward는 기존 state/action을 활용할 뿐 새로운 obs/action을 추가하지 않음.

| 구분 | 크기 | 설명 |
|---|---|---|
| self_obs | 166 | dof_pos, vel, root state 등 |
| task_obs (base) | 288 | 4 bodies x 3 samples x 24 |
| gait_phase_obs | 2 | sin/cos(2π × gait_cycle_phase) |
| 합계 | 456 | |

action_space: 69 (PD targets) + 8 (CCF groups) = 77

## 5. Reward 구조

| Term | 수식/설명 | 가중치 |
|---|---|---|
| Point goal | distance 기반 위치 보상 | 1.0 |
| Imitation | pos/rot/vel/ang_vel 매칭 | 0.5 |
| AMP discriminator | 적대적 모션 자연스러움 판별 | curriculum |
| Power penalty | -0.000005 * sum(abs(torque * dof_vel)) | 1.0 |
| **Bio CCF (신규)** | stance: clamp(ankle_ccf, min=0), swing: clamp(-ankle_ccf, min=0) | **0.05** |

## 6. 평가 파이프라인 개선

VIC_PHASE_03와 함께 평가 파이프라인도 개선되었다.

### analyze_phase_ccf.py 리팩토링
- `analyze(npy_path)` 함수로 모듈화 (CLI와 코드 호출 모두 가능)
- `plot_per_group()`: 기존 8그룹 subplot 플롯
- `plot_lr_comparison()`: L vs R 비교 플롯 (Hip, Knee, Ankle+Toe 3 subplot)

### im_amp_players.py 자동 분석 통합
평가 완료 후 phase_ccf_log.npy 저장 시 자동으로 분석 및 플롯 생성:
```python
try:
    from analyze_phase_ccf import analyze
    analyze(save_path)
except Exception as e:
    print(f"[VIC] Phase-CCF analysis failed: {e}")
```

## 7. 실험 설정 백업

`exp_config/forward_walking/260313_VIC_PHASE_03/`에 스냅샷:
- env_im_walk_vic.yaml
- im_walk_vic.yaml
- humanoid_im_vic.py

## 8. 설계 의도

1. **최소 개입 원칙**: w=0.05로 매우 작은 가중치 사용. 기존 학습 안정성을 유지하면서 CCF에 방향성만 제공
2. **벌칙 없는 Soft guidance**: 원하는 방향의 CCF만 보상하고, 반대 방향은 벌칙 없이 0. Policy가 자유롭게 탐색할 수 있되, 방향성만 인센티브 제공
3. **Contact force 기반 실시간 판별**: Reference motion의 사전 계산이 아닌, 실제 시뮬레이션의 발-지면 접촉력으로 stance/swing 판별. 물리적으로 정확한 기준

## 9. 기대 효과 및 실제 결과

| 기대 | 실제 |
|---|---|
| Ankle CCF 분화 강화 | 확인: L_Ankle 1.43→1.75x (+22%), R_Ankle 1.51→1.80x (+19%) |
| Phase-dependent CCF 변화 | 일부 확인: L_Ankle swing dip (phase 55-65%), Hip antiphase 출현 |
| 성능 유지 | 확인: 100% 모션 완주, reward 878.89 (PHASE_02: 873.32 대비 미세 증가) |
| Swing compliance 유도 | 미달: 전반적으로 ankle이 stiff(1.75~1.80x). Stance 보상이 dominant |

## 10. 후속 방향

1. Bio CCF reward weight 튜닝 (w=0.1~0.2)
2. Knee, Hip 포함 확대
3. Swing 보상 가중치를 stance보다 높이기
4. 외란(Perturbation) 추가로 임피던스 분화 압력 제공
5. 에너지 페널티 조정 (종종걸음 문제 관련)

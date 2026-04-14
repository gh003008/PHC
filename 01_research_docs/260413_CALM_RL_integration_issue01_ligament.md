# CALM-PHC RL 통합 이슈 #01: Ligament Model이 SMPL axis-angle과 호환 안 됨

## 요약

CALM의 근골격 모델을 PHC의 SMPL humanoid에 통합해서 강화학습을 돌릴 때, 두 가지 심각한 문제가 연쇄적으로 발생했다.

1. **NaN 폭발** — 첫 물리 스텝에서 일부 환경이 완전히 발산
2. **학습 완전 정지** — 10,000 epoch 내내 eps_len=2.0, rwd=6.2에서 변화 없음

원인을 추적한 결과, **CALM의 Ligament 모델이 SMPL의 axis-angle DOF 표현과 근본적으로 호환되지 않아** 정상 포즈에서도 exponential 복원 토크를 발생시키는 것이 근본 원인이었다. 임시로 ligament를 비활성화하고 torque_limit을 추가하자, 100 epoch만에 eps_len 2→22, rwd 6→69로 극적으로 개선됐다.

하지만 ligament는 생체역학적으로 필수 요소이므로, **SMPL axis-angle 호환 버전의 재구현이 필요하다.**

---

## 문제 상세

### Issue A: 첫 물리 스텝 NaN

학습 시작 직후 `torch.distributions.Normal(mu, sigma)`에서 ValueError:
```
ValueError: Expected parameter loc ... to satisfy the constraint Real(), but found invalid values
```

**추적 결과**:
- `obs_buf`의 일부 환경 (10/512)에서 NaN 발생
- `_rigid_body_pos`, `_dof_pos`, `_dof_vel` 전부 NaN
- `progress_buf = 1`에서 발생 → 첫 물리 스텝 직후

**원인 1: torque_limits = FLT_MAX**
- SMPL MJCF에 effort limit이 정의되어 있지 않음
- IsaacGym이 기본값으로 `torque_limits = FLT_MAX` (3.4e+38) 설정
- 사실상 토크 clamp가 작동하지 않음

**원인 2: CALM Ligament 모델의 exponential 폭발**

`standard_human_model/core/ligament_model.py`의 핵심 공식:
```python
tau_upper = -k_lig * (exp(clamp(alpha * excess_upper, max=50)) - 1)
tau_lower =  k_lig * (exp(clamp(alpha * excess_lower, max=50)) - 1)
```

기본 파라미터: `k_lig=50`, `alpha=10`

excess가 1 rad만 되어도:
- `tau = 50 * (exp(10) - 1) ≈ 1,100,000 Nm`

excess가 2 rad면:
- `tau = 50 * (exp(20) - 1) ≈ 24,000,000,000 Nm`

이 거대한 토크가 `torque_limits=FLT_MAX`라 clamp 안 되고 그대로 시뮬레이터에 주입 → 일부 환경이 즉시 발산 → NaN.

**임시 fix (완료)**:
- `torque_limits = 500 Nm`으로 override
- 근본 문제인 ligament는 아직 활성 상태

### Issue B: 학습 완전 정지

위 NaN fix 후 학습은 돌지만, 10,000 epoch 내내 다음 상태에서 변하지 않음:
- `rwd = 6.0~6.3` (변화 없음)
- `eps_len = 2.0` (변화 없음)

**2 step = 66ms만에 episode 종료**는 물리적으로 이상하다.

**진단 결과** (humanoid_im_calm.py에 임시 프린트 추가):

Reset 직후:
```
pelvis_z = 0.933 (정상, ~93cm)
|dof_pos|max = 0.88 rad (정상 walking pose)
pos_err = 0.029 (reference와 거의 일치)
```

첫 substep 직후:
```
dof_vel_max = 0.857 rad/s  (정상)
calm_lower_max = 170 Nm    (정상)
pd_upper_max = 2.7 Nm       (정상)
```

**두 번째 substep**:
```
dof_vel_max = 10.767 rad/s   ← 16.7ms 만에 0.857 → 10.77!
calm_lower_max = 89 Nm
pd_upper_max = 132 Nm
```

즉 ligament 모델이 첫 substep에서 매우 큰 토크를 내고 (500 Nm로 clamp됨), 이것이 작은 limb(발, 손 등)에 가해지면서 즉시 회전 폭주 → `||body_pos - ref_body_pos|| > 0.25m` termination 조건 위반 → early termination.

---

## 근본 원인: SMPL axis-angle vs Euler 불일치

### SMPL DOF 표현
SMPL은 각 관절을 **axis-angle 벡터**로 표현한다:
```
dof_pos[joint] = [x, y, z]  # 회전축 * 회전각
```
- `|dof_pos|` = 전체 회전각 (rad)
- `dof_pos / |dof_pos|` = 회전축 (단위벡터)

예) `L_Hip = [0.5, 0.0, 0.0]`은 "x축 방향으로 0.5 rad 회전"을 의미.

### CALM의 skeleton.py joint limits

`standard_human_model/core/skeleton.py`:
```python
_JOINT_LIMITS_DEG = {
    "L_Hip":  [(-30, 120), (-45, 45), (-45, 30)],  # (x, y, z) 각 축
    "L_Knee": [(0, 145), (-5.625, 5.625), (-5.625, 5.625)],
    ...
}
```

이것은 **Euler-like 독립 축 한계**를 가정한다:
- "x축은 -30°~120°, y축은 -45°~45°, z축은 ..."

### 불일치

SMPL의 `dof_pos = [0.3, 0.3, 0.3]`은:
- **SMPL 해석**: axis=(0.577,0.577,0.577) 방향으로 0.52 rad 회전 (단일 회전)
- **CALM 해석**: x축=0.3, y축=0.3, z축=0.3 (세 개 독립 회전각) → 각각 Euler 한계와 비교

이 mismatch 때문에 **정상 SMPL 포즈도 CALM 기준으로는 "한계 초과"로 해석**되어, ligament가 거대한 exponential 토크를 생성한다.

### 예: L_Knee가 정상인데도 폭발

```python
L_Knee 제약: (0, 145) deg = (0, 2.53) rad
soft_lower = 0.190 (margin 0.85 적용)

SMPL 정상 walking pose: L_Knee = [1.0, 0.1, 0.05]
  → |rotation| = sqrt(1.0² + 0.1² + 0.05²) ≈ 1.006 rad (정상 범위)
  
CALM 해석: dof_pos[L_Knee_y] = 0.1 < soft_lower(0.190)
  → excess_lower = 0.09 rad
  → tau = 50 * (exp(10*0.09) - 1) = 50 * 1.46 = 73 Nm

이게 누적되면서 결국 한계 초과 → 몇 만 Nm까지 폭발.
```

---

## 적용한 임시 수정

### 1. Torque limits override
`humanoid_im_calm.py` 8단계:
```python
self.torque_limits = torch.ones(self.num_dof) * 200.0
```

### 2. Ligament 비활성화
`humanoid_im_calm.py` 4b단계:
```python
self.calm_body.ligament.k_lig.zero_()
self.calm_body.ligament.damping.zero_()
```

### 3. NaN safety
CALM 출력에 safety guard:
```python
tau_calm = torch.where(torch.isfinite(tau_calm), tau_calm, torch.zeros_like(tau_calm))
```

### 4. Termination 완화
`env_im_walk_calm.yaml`:
```yaml
terminationDistance: 0.5  # 0.25에서 완화
```

### 결과 (100 epoch 검증)

| 지표 | 이전 (10,000 epoch) | 수정 후 (100 epoch) |
|------|-------|-------|
| Ep 1 rwd | 6.3 | **58.9** |
| Ep 100 rwd | 6.2 | **66.5** |
| Ep 1 eps_len | 2.0 | **18.7** |
| Ep 100 eps_len | 2.0 | **21.2** |
| 학습 추세 | 정지 | rwd/eps_len 모두 상승 |

**100 epoch만에 eps_len 10배, rwd 10배 개선.**

---

## 해결 방향: SMPL axis-angle 호환 Ligament 구현

ligament는 관절 한계 근처에서 복원 토크를 생성하는 **생체역학적 필수 요소**이므로 완전히 빼면 안 된다. SMPL에 호환되는 방식으로 재구현해야 한다.

### 옵션 1: Axis-Angle Magnitude 기반 soft limit (권장)

관절별로 **전체 회전각의 크기**를 single scalar soft limit과 비교.

```python
# 각 관절의 회전각 크기
angle_mag = torch.norm(dof_pos[joint_slice], dim=-1)  # (num_envs,)

# 관절별 스칼라 soft limit (예: L_Knee는 2.0 rad)
soft_limit = 2.0
excess = torch.clamp(angle_mag - soft_limit * margin, min=0)
```

**장점**:
- SMPL axis-angle 의미론과 일치
- 회전축 방향에 관계없이 일관된 동작
- skeleton.py 수정 최소화

**단점**:
- 방향별로 다른 ROM (예: L_Hip은 flexion 120°, extension 30°)을 반영 못 함
- 단일 soft_limit으로 모든 방향에 동일한 제약

**개선**: axis-direction-aware soft limit
```python
# dof_pos 방향을 4개 주방향(flexion+/-, abduction+/-)에 투영해서
# 방향별 soft limit 적용
```

### 옵션 2: Forward Kinematics로 해부학적 각도 복원

SMPL dof_pos → rotation matrix → 해부학적 각도 (flexion, abduction, rotation) 분해.

```python
rot_matrix = axis_angle_to_matrix(dof_pos[joint])
flexion_angle = extract_flexion(rot_matrix)
abduction_angle = extract_abduction(rot_matrix)
rotation_angle = extract_rotation(rot_matrix)
# 각 해부학적 축에 Euler-like 제약 적용
```

**장점**:
- 해부학적으로 가장 정확
- 기존 skeleton.py 한계값 재사용 가능

**단점**:
- Axis-angle → Euler 분해는 gimbal lock, 다양성(axis ordering) 문제
- 계산 복잡도 증가
- Singularity 근처에서 불안정

### 옵션 3: 지수 → 이차 (Quadratic) 완화

공식 자체는 유지하되 폭발을 막는 방식.

```python
# 기존: tau = k_lig * (exp(alpha * excess) - 1)
# 수정: tau = k_lig * excess²
```

**장점**: 구현 매우 간단
**단점**: 생체역학적 정확도 저하, 엄밀한 ROM 끝에서의 강한 복원력 부족

### 옵션 4: k_lig / alpha 대폭 감소 + excess cap

```python
k_lig = 5.0      # 50 → 5
alpha = 3.0      # 10 → 3
max_exp_arg = 5  # 50 → 5 (tau 상한 ≈ 742 Nm)
```

**장점**: 최소 수정, 기존 구조 유지
**단점**: Hacky. SMPL/Euler 불일치 근본 원인 해결 안 됨

### 권장 접근

**단기**: 옵션 4로 빠르게 완화 → 학습 진행 확인

**중기**: 옵션 1 (axis-angle magnitude) 구현 → `ligament_model.py`에 새 메서드 추가
```python
class LigamentModel:
    def compute_torque_axis_angle(self, dof_pos, dof_vel, joint_slices, soft_limits):
        """SMPL axis-angle 호환 버전."""
        ...
```

**장기**: 옵션 2 (forward kinematics) 정확 구현 → validation suite로 검증

### Non-trivial 고려사항

1. **관절별 soft_limit 스칼라 값**을 어떻게 정할 것인가?
   - skeleton.py의 `_JOINT_LIMITS_DEG`에서 각 축의 max 값으로?
   - 또는 해부학적 문헌 기반 총 ROM?

2. **토크 방향**을 어떻게 결정할 것인가?
   - axis-angle magnitude는 scalar라서 "어느 축으로 복원할지" 모호
   - 가장 직관적인 방향: `-dof_pos / |dof_pos|` 방향으로 scalar 복원력 적용
   - 즉 `tau = -k_lig * excess * (dof_pos / |dof_pos|)` (3D 벡터)

3. **Bi-directional limits**
   - 옵션 1은 "0(중립)에서 얼마나 떨어졌나"만 체크
   - Flexion과 extension의 한계가 다르면 반영 불가
   - → 방향 투영 필요

---

## 다음 작업

1. **[단기]** 옵션 4로 ligament 파라미터 완화 → 재활성화 후 100 epoch 재검증
2. **[중기]** `ligament_model.py`에 `compute_torque_axis_angle()` 메서드 추가
   - 옵션 1 구현
   - CALM 단위 테스트로 동작 검증
   - `humanoid_im_calm.py`에서 새 메서드 호출하도록 switch
3. **[장기]** 옵션 2 forward kinematics 기반 해부학적 분해
   - CALM validation suite L2(단일 관절 pendulum)로 정확도 검증
   - 방향별 ROM 차이 반영

## 참고

- 현재 `torque_limit = 200 Nm`는 임시값. Ligament를 올바르게 구현하면 더 높여도 될 것.
- `terminationDistance = 0.5`는 CALM의 jerky 특성 때문에 완화한 것. 학습이 안정되면 0.3 정도로 조정 가능.
- 시각화 코드 (`_draw_task`의 근육 라인)는 이미 추가됨 → 학습 끝난 후 viewer에서 근육 활성도 시각화 가능.

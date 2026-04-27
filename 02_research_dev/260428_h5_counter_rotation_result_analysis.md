# H5 변환 파이프라인 — 카운터-로테이션 적용 후 결과 분석

**작성일**: 2026-04-28
**최종 산출물**: `sample_data/h5_walk_S001_lv0_trial01_v2_counterrot.pkl`
**핵심 변경**: `--counter_rotation` 플래그 추가, `solve_foot_ik_trajectory` 가속도 스무딩

---

## 1. 배경 — 직전 단계 요약

### 1.1 Trajectory IK (가속도 스무딩)

이전 단계에서 stance foot 슬라이딩과 swing 디스컨티뉴이티 문제를 해결하기 위해 trajectory-level Inverse Kinematics를 구현했다.

핵심 결정:
- **Per-frame IK 폐기**: scipy least_squares per-frame은 swing→stance 천이 시 frame-to-frame jump가 32-64cm까지 발생.
- **PyTorch Adam trajectory IK**: 모든 프레임을 동시 최적화. 펠비스 trans + 양 다리 (hip/knee/ankle, T×6×3) 최적화.
- **속도 스무딩 → 가속도 스무딩 교체**: `||x[t+1]-x[t]||²`은 정상 motion까지 감쇠시킴 (swing arc 평탄화). `||x[t+1]-2x[t]+x[t-1]||²`로 변경하여 자연스러운 swing 보존.

결과 metrics (v2_footik_traj):
- 펠비스 frame-to-frame max jump: 32.10 → **5.46 cm/frame** (baseline cop의 4.05와 비슷)
- L_Ankle peak velocity: 8.90 → **11.17 cm/frame** (cop baseline 11.77 회복)

### 1.2 사용자 피드백 — "팔이 너무 많이 움직인다"

Trajectory IK 비디오 확인 후 사용자가 지적: 상지가 AMASS 대비 과도하게 움직임. Trunk가 거의 고정되어야 하는데 흔들림이 보인다.

---

## 2. 진단 — Joint Index 버그 발견

### 2.1 Spine joint angular speed 비교 (잘못된 인덱스)

처음에 BONE order 인덱스(표준 SMPL)로 측정:
- pose_aa[:, 9] = Chest (BONE order)
- 결과: H5 chest pitch ±5.2° vs AMASS ±0.7° → 7배 차이 → 명확한 spine 노이즈

이 결론으로 spine smoothing을 제안했으나, 사용자가 push-back ("trunk orientation 보고 있는 거 맞아?").

### 2.2 인덱스 불일치 발견

`pose_aa`는 **BONE order** (16=L_Shoulder, 9=Chest)로 저장되지만,
`pose_quat_global`은 **MUJOCO order** (15=L_Shoulder, 11=Chest)로 저장된다.

`SkeletonState.from_rotation_and_root_translation(sk_mjcf, pose_aa)`로 FK를 돌리면 인덱스가 어긋나서 잘못된 관절을 분석하게 된다.

수정: `pose_quat_global`을 직접 사용하고 MJCF 인덱스로 분석.

### 2.3 정정된 비교 — Spine은 사실 AMASS와 비슷

올바른 인덱스로 chest-relative-to-pelvis 측정:

| | sag flex | twist | lat flex |
|---|---|---|---|
| H5 (cop & traj) | +11.7° ± 1.7 | -1.0° ± 3.2 | +3.8° ± 2.5 |
| AMASS | +11.9° ± 3.7 | -2.0° ± 6.7 | -2.9° ± 2.2 |

**상대 회전은 AMASS보다 오히려 작음**. Spine 자체는 노이지하지 않다.

---

## 3. 진짜 원인 — Pelvis-Chest 디커플링 부재

### 3.1 Joint global angular speed

| Joint | H5 (cop & traj) | AMASS | 비율 |
|---|---|---|---|
| Pelvis | 1.72 | 1.71 | 동일 |
| Torso (joint 9) | **3.87** | 1.44 | **H5 2.7배** |
| Spine (joint 10) | **3.58** | 2.07 | H5 1.7배 |
| Chest (joint 11) | **3.58** | 1.92 | **H5 1.9배** |

Pelvis는 동일한데 chest 글로벌 각속도는 ~2배. 상대 회전은 AMASS보다 작음.

### 3.2 해석

- AMASS: chest가 pelvis와 **반대로 회전(counter-rotation)** — 자연스러운 팔 흔들기 디커플링. Pelvis가 오른쪽으로 yaw하면 chest는 살짝 왼쪽으로 yaw → 월드 기준 chest는 안정.
- H5: chest가 pelvis와 **같이** 강체처럼 회전 → pelvis 회전이 chest 회전에 그대로 더해짐.

### 3.3 PiG 트렁크 채널 상관관계

| 채널 | mean ± std | pelvis_z(yaw)와 상관 |
|---|---|---|
| spine_z_L (lat) | -0.66 ± 4.23 | **+0.970** (pelvis와 동기, 추가 시 증폭) |
| **thorax_z_L** (lat) | +4.25 ± 1.59 | **-0.771** (자연 카운터-로테이션) |

기존 변환은 `spine/x`, `thorax/x` (시상면 굴곡)만 사용하고 y/z를 버린다. 그런데 `thorax_z_L`이 정확히 **자연 카운터-로테이션**을 인코딩하고 있었다.

---

## 4. 해결 — `--counter_rotation` 플래그

### 4.1 구현

`scripts/data/h5_to_motion_lib_v2.py`:

```python
# convert_trial signature
def convert_trial(..., counter_rotation=False, ...):

# LEG_SPINE_MAP loop 조건 수정
if (spine_3axis and smpl_idx in (3, 6)) or (counter_rotation and smpl_idx == 6):
    # y, z 채널 추가 적용
    ...
```

핵심: **Spine joint(6)에만** thorax y/z 채널 추가 (carry counter-rotation). Torso joint(3)는 손대지 않음 — `spine_z`가 pelvis yaw와 +0.97 상관이라 추가하면 회전 증폭됨.

### 4.2 결과

| pkl | Pelvis max | Torso max | Spine max | **Chest max** |
|---|---|---|---|---|
| v2_cop / v2_footik_traj (이전) | 1.72 | 3.87 | 3.58 | **3.58** |
| **v2_counterrot (수정 후)** | 1.82 | 1.89 | 1.91 | **1.91** ✓ |
| amass (목표) | 1.71 | 1.44 | 2.07 | **1.92** |

**Chest 글로벌 각속도 3.58 → 1.91, AMASS의 1.92와 거의 동일.** Trunk wobble 문제 해결.

비디오: `output/h5_footik_check/counterrot_vs_amass.mp4`, `counterrot_vs_footik_traj.mp4`

---

## 5. 발목 모션 — 보류 중인 작은 이슈

사용자가 "ankle pop out larger"로 표현한 부수적 관찰. 측정 결과:

### 5.1 발목 글로벌 회전 / 위치

| 측정 | v2_counterrot | amass |
|---|---|---|
| 발목 글로벌 각속도 max | 8.09 | 14.06 |
| 발목 LOCAL pitch std | 0.6° | 3.4° |
| 발목 stance Z (5pct) | 0.078m | 0.048m |
| 발목 swing peak Z (95pct) | 0.227m | 0.193m |
| Swing arc 높이 | 14.9 cm | 14.5 cm |

H5 발은 AMASS보다 3cm 더 높이 떠 있음. Swing arc 자체는 거의 동일.

### 5.2 시도 — `--foot_ik_ankle_z_offset 0.07`

발목 anchor z에 +7cm offset 적용 (해부학적 발목 높이 보정). 결과: stance ankle Z 0.078 → 0.077 (변화 없음). IK의 smoothness/regularization 비용이 anchor를 압도, ankle은 이미 자연스러운 높이에 안착되어 있었음.

### 5.3 결론

- Swing arc는 AMASS와 일치 → 시각적 큰 문제 아님
- 발 전체가 3cm 떠 있는 정도 (작은 차이)
- 본 단계의 핵심 문제(trunk wobble)는 해결됨

추가 fix는 더 침습적인 변경 필요 (pelvis trans z 보정 또는 anchor weight 큰 폭 증가). 사용자 비주얼 컨펌 후 결정.

---

## 6. 현재 데이터 상태 (최종)

### 6.1 권장 변환 명령

```bash
python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --subjects S001 \
  --tasks level_100mps \
  --assist_level lv0 \
  --output sample_data/h5_walk_S001_lv0_trial01_v2_counterrot.pkl \
  --counter_rotation \
  --no_split \
  --foot_ik trajectory \
  --foot_ik_pitch_threshold_deg 5.0 \
  --foot_ik_anchor_weight 1.0 \
  --foot_ik_smoothness_weight 100.0 \
  --foot_ik_joint_reg_weight 0.01 \
  --foot_ik_pelvis_reg_weight 0.05 \
  --foot_ik_lr 0.005 \
  --foot_ik_max_iter 200 \
  --foot_ik_bound_weight 1.0 \
  --foot_ik_bounds_deg 15.0
```

### 6.2 적용된 fix 누적 요약

| Fix | 효과 |
|---|---|
| Trajectory IK (Adam, 가속도 스무딩) | Stance foot 슬라이딩 + swing 디스컨티뉴이티 해결 |
| FK frame Z-up 버그 수정 | 발목 0.75m 지하 매몰 해결 |
| Anchor 축 swap (`_swap` permutation) | Anchor 좌표계 정합 |
| `--counter_rotation` 플래그 | Trunk-pelvis 디커플링 → Chest 각속도 AMASS 수준 |

### 6.3 검증된 metrics

- 펠비스 frame-to-frame max jump: 5.46 cm/frame (cop baseline 4.05 ± 35%)
- L_Ankle peak velocity: 11.17 cm/frame (cop 11.77, swing 자연성 유지)
- Chest 글로벌 각속도 max: 1.91°/frame (AMASS 1.92, 동일)
- Trunk wobble: AMASS와 시각적으로 매칭

### 6.4 비디오

- `output/h5_footik_check/counterrot_vs_amass.mp4` — H5 v2_counterrot vs AMASS 참조
- `output/h5_footik_check/counterrot_vs_footik_traj.mp4` — counter-rotation 적용 전후 비교
- `output/h5_footik_check/traj_accel_vs_amass_compare.mp4` — Trajectory IK 단독 (counter-rotation 없음)

---

## 7. 다음 단계 후보

우선순위 순:

1. **VIC 학습 적용 테스트** — counter-rotation 변환 데이터로 VIC 학습 → AMASS 학습과 metric 비교 (av_reward, av_steps)
2. **발목 Z 3cm 차이 해결** (선택적) — pelvis trans z 보정 또는 anchor weight 강화
3. **다른 trial로 일반화 검증** — S001만 했음. 다른 subjects/levels에서도 counter-rotation이 작동하는지 확인
4. **기존 변환 데이터 마이그레이션** — h5_motion_library.pkl 등 기존 산출물에 counter-rotation 재적용

---

## 8. 변경 파일 목록

- `scripts/data/h5_to_motion_lib_v2.py` — `--counter_rotation`, `--foot_ik_ankle_z_offset` 플래그 추가, IK 블록에서 anchor z offset 적용
- `scripts/data/h5_conversion_helpers.py` — `solve_foot_ik_trajectory` 가속도 스무딩 적용 (이전 단계 작업)
- `01_research_docs/260425_h5_foot_anchor_ik_design.md` — 설계 문서
- `01_research_docs/260425_h5_foot_anchor_ik_plan.md` — 구현 계획

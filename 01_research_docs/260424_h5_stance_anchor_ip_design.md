# H5 Stance-Anchor Inverted-Pendulum Root Translation — 설계서

**날짜**: 2026-04-24
**작성자**: Jimin (brainstorming via Claude)
**관련 파일**: `scripts/data/h5_to_motion_lib_v2.py`, `scripts/data/h5_conversion_helpers.py`
**이전 참고**:
- `backups/260424_pre_v2/` — v2 작업 전 백업
- Session summary — GRF-based foot-lock 시도 후 "pose noise → root" amplification으로 실패 (2026-04-23)

---

## 1. 배경 및 문제 정의

### 현재 상태 (v2_full)
H5 변환 시 `compute_root_translation()`이 pelvis 위치를 Vicon CoM의 medio-lateral (y) 성분에서 직접 가져옴:
```python
trans[:, 0] = -(com_ml - com_ml[0])   # X = right (negate medio-lateral)
```
필터링 없음. 결과적으로 frontal view에서 pelvis lateral motion 관측:
- v2_full: std **4.89 cm**, range **19.58 cm**
- AMASS: std 3.53 cm, range 11.96 cm
- H5가 약 38% 큼

사용자 visual feedback: "whole body (like an offset) moves in frontal plane too much. when swinging, roll movement is big and whole-body is moving in roll direction not a single joint."

### 문제의 핵심
Root (pelvis) 레벨에서 translation + rotation이 모두 whole-body에 그대로 전달됨. `pelvis_obliq_scale=0.0`으로 roll을 제거해도 여전히 과도한 lateral motion 관측 → 원인은 translation 성분.

### Scale-down의 한계
`pelvis_lat_scale=0.5/0.3/0.0` 실험: lateral 진폭은 줄어들지만 stance phase 동안 foot이 여전히 미끄러짐 (실제 보행에서는 stance foot이 world frame에 고정되어야 함).

### 이전 시도의 실패 원인
2026-04-23 시도: GRF-based stance detection → raw Vicon marker foot position을 anchor로 고정 → marker jitter가 root로 amplify됨. 사용자: "i think we cannot use foot locking haha."

---

## 2. Goal

Stance foot을 world frame에 rolling-smoothed anchor로 고정하고, pelvis world position을 IP (Inverted Pendulum) constraint로 역산하여 전체 lateral + vertical 안정화. 이전 실패의 원인(raw marker noise)을 회피하는 설계 적용.

**Flag**: `--stance_anchor_ip` (store_true, default off → 기존 동작 유지 = no-op invariant).

---

## 3. 주요 설계 결정 (brainstorming 합의)

| # | 결정 사항 | 선택 | 근거 |
|---|---|---|---|
| Q1 | IP scope | **(C) Full 3D** | Lateral-only는 부분 해결; full 3D가 가장 완전 |
| Q2 | Stance detection | **(C) GRF OR velocity** | 단일 신호 실패 시 backup, robust |
| Q3 | Stance 중 anchor 모델 | **(B) Rolling mean** | Heel→toe roll 자연스럽고 noise 제거 양립 |
| Q4 | Double stance 처리 | **(A) GRF-weighted blend** | Weight transfer를 GRF로 직접 표현 |

**Forward motion 특이사항**: Treadmill 데이터에서 forward 방향은 외부 기준(belt 속도)이므로 IP 범위 밖. IP는 lateral + vertical 두 축만 결정. Forward는 기존 treadmill 적분 유지.

---

## 4. 아키텍처

```
[H5 input]
    │
    ├─▶ (A) Pose angles: 기존 v2 파이프라인 그대로 → pose_aa_local [T, 24, 3]
    │
    ├─▶ (B) Bootstrap translation: 기존 compute_root_translation() → trans_boot [T, 3]
    │       (CoM-based, reference only)
    │
    ├─▶ (C) Stance detection (NEW): h5 GRF + FK foot velocity from (A)+(B)
    │       → stance_L[T], stance_R[T], grf_L[T], grf_R[T]
    │
    ├─▶ (D) Foot anchor (NEW): FK foot_world from (A)+(B), rolling mean per stance
    │       → anchor_L[T], anchor_R[T]
    │
    ├─▶ (E) IP pelvis solver (NEW): GRF-weighted blend
    │       target = (grf_L*anchor_L + grf_R*anchor_R) / (grf_L+grf_R)
    │       pelvis = target - R_pelvis @ foot_offset_in_pelvis_frame
    │
    └─▶ (F) Final trans: forward(Z) = treadmill, lateral(X)+vertical(Y) = IP (E)
            → trans[T, 3]
```

**핵심 안전장치 3가지**:
1. **Bootstrap fallback**: 기존 CoM 방식을 여전히 계산 → stance 감지 실패 시 fallback 경로 존재
2. **Rolling mean**: window=9 frames (0.3s at 30 fps) → marker noise smoothing
3. **Flight-phase guard**: 양발 모두 swing 상태(GRF<20N & vel>0.3 m/s)가 >5 frames 지속되면 CoM-based로 대체

---

## 5. 알고리즘 상세

### 5.1 `detect_stance(grf_L, grf_R, vel_L, vel_R, ...)` → bool arrays

**Hysteresis (Schmitt trigger)** 두 신호 독립 적용:
- GRF path: on when `grf > 50 N`, off when `grf < 20 N`
- Velocity path: on when `|foot_vel| < 0.3 m/s`, off when `|foot_vel| > 0.5 m/s`
- Final: `GRF_stance OR velocity_stance`

### 5.2 `compute_foot_anchor(foot_world, stance_mask, window=9)` → anchor

**Stance episode** (연속된 stance frames) 별로:
- 해당 구간 `foot_world`의 centered rolling mean (window=9, 경계는 truncated)
- Swing 구간: 이전 stance end anchor와 다음 stance start anchor 사이 linear interp (연속성 유지, 실제로는 solver에서 사용 안 됨 — swing 중엔 해당 foot weight=0)

**Rolling window 9 frames = 0.3 s**: 실제 foot roll (heel-to-toe shift) timescale ~0.2-0.4 s와 일치, marker jitter (높은 주파수) 제거에 충분.

### 5.3 `solve_pelvis_ip(anchor_L, anchor_R, grf_L, grf_R, R_pelvis, foot_offset_L, foot_offset_R)` → pelvis_world

수식 (per frame t):
```
w_L = grf_L[t] / (grf_L[t] + grf_R[t] + ε)
w_R = 1 − w_L
target_anchor      = w_L * anchor_L[t] + w_R * anchor_R[t]
target_foot_offset = w_L * foot_offset_L[t] + w_R * foot_offset_R[t]
pelvis_world[t]    = target_anchor − R_pelvis[t] @ target_foot_offset
```

**Edge cases**:
- 양 GRF 모두 `<20 N` → flight phase → 이전 pelvis 위치 carry (vertical velocity 유지)
- 단일 stance (w_L=1 or w_R=1) → 수식 자연스럽게 축소

### 5.4 메인 파이프라인 통합

**Anchor joint 선택**: SMPL `L_Ankle` (index 3), `R_Ankle` (index 7) world position을 foot anchor로 사용. Toe는 stance 동안 heel→toe roll로 움직이므로 ankle이 더 안정적인 pivot.

**FK helper (신규)**: `fk_feet(pose_quat_global, trans_boot)` — poselib의 pose_quat_global는 이미 모든 joint world rotation을 담고 있으므로, SkeletonTree의 bone offsets을 적용해 world position을 계산. `h5_to_motion_lib_v2.py` 내부에 local 함수로 추가. 출력: `foot_L_world[T,3]`, `foot_R_world[T,3]`, `foot_offset_L[T,3]` (pelvis-local frame에서의 ankle 위치), `foot_offset_R[T,3]`, `R_pelvis[T,3,3]`.

**GRF reader (신규)**: `read_grf(f, trial_path, side='left')` — H5에서 `treadmill/{side}/grf_y` 또는 유사 경로 읽기 (실제 경로는 h5_file_spec.md 확인 필요). `h5_to_motion_lib_v2.py` 내부 local 함수.

`convert_trial()`에 조건부 블록 추가 (기존 `trans = compute_root_translation(...)` 직후):

```python
if stance_anchor_ip:
    # Pass 1: FK with bootstrap trans
    foot_L_w, foot_R_w, foff_L, foff_R, R_pel = fk_feet(pose_quat_global, trans)
    vel_L = np.linalg.norm(np.gradient(foot_L_w, axis=0), axis=1) * fps_out
    vel_R = np.linalg.norm(np.gradient(foot_R_w, axis=0), axis=1) * fps_out
    grf_L = read_grf(f, trial_path, side='left')
    grf_R = read_grf(f, trial_path, side='right')
    # Pass 2: stance + anchor + IP
    stance_L, stance_R = detect_stance(grf_L, grf_R, vel_L, vel_R)
    anc_L = compute_foot_anchor(foot_L_w, stance_L)
    anc_R = compute_foot_anchor(foot_R_w, stance_R)
    pelvis_ip = solve_pelvis_ip(anc_L, anc_R, grf_L, grf_R, R_pel, foff_L, foff_R)
    # Replace lateral + vertical (keep forward from treadmill)
    trans[:, 0] = pelvis_ip[:, 0]
    trans[:, 1] = pelvis_ip[:, 1]
```

---

## 6. 노이즈 Mitigation (이전 실패 회피 체크리스트)

| 노이즈 소스 | 완화 방법 |
|---|---|
| Raw marker jitter (H5 CoM) | **사용 안 함** — FK-computed foot 사용 |
| Foot position jitter (FK) | Rolling mean (9-frame window) per stance episode |
| Stance detection chatter | Schmitt-trigger hysteresis (GRF + velocity 각각) |
| GRF signal noise | Weight clip to `[0.05, 0.95]` 시 단일 stance로 취급 |
| Pelvis R 노이즈 | Task 2 baseline subtraction 이미 적용됨 |

---

## 7. Edge Cases

| 상황 | 처리 |
|---|---|
| t=0에서 stance 감지 안됨 | Bootstrap pelvis 유지 (CoM fallback) |
| 연속 >5 frames flight | CoM-based trans 사용, stance 재진입 시 3 frames smoothly blend |
| Trial 종료 시 stance 중 | 마지막 anchor 유지 |
| GRF 데이터 없음 (legacy) | Velocity-only stance detection, WARNING |
| IP 결과 ↔ CoM 차이 >30 cm | Clip + WARNING |
| Rolling mean window > stance 길이 | `min(window, stance_length)` 사용 |

---

## 8. 파일 구조 변경

**Modify**:
- `scripts/data/h5_conversion_helpers.py`: `detect_stance`, `compute_foot_anchor`, `solve_pelvis_ip` 추가
- `scripts/data/h5_to_motion_lib_v2.py`: `--stance_anchor_ip` 플래그 + `convert_trial` 조건부 블록 + FK helper `fk_feet()`
- `scripts/data/test_h5_conversion_fixes.py`: 3개 unit test 추가

**Create**: 없음 (기존 파일 연장)

---

## 9. Unit Tests

### 9.1 `test_detect_stance_hysteresis`
Synthetic GRF signal: 50→60→25→15→30 N. Hysteresis 없으면 매 threshold 교차마다 toggling. 있으면 `GRF>50` 첫 진입 → stance, `GRF<20` 첫 진출 → swing, 중간 25/30에서 계속 stance 유지.

### 9.2 `test_foot_anchor_rolling_smooths_noise`
Synthetic foot_world = constant + Gaussian noise (±5 mm). stance_mask = all True.
Output std < 1.5 mm (noise 1/3+ 감소 예상).

### 9.3 `test_solve_pelvis_ip_fixes_foot`
Synthetic: R_pelvis = identity, foot_offset = [0, 0, -1.0] (발 1m 아래), anchor = [0, 0, 0] (원점).
Expected pelvis_world = [0, 0, 1.0] (발 위 1m). 오차 < 1e-6.

GRF-weighted blend도 검증: grf_L=1, grf_R=1, anchor_L=[-0.1,0,0], anchor_R=[+0.1,0,0] → target=[0,0,0] → pelvis=[0,0,1.0].

---

## 10. Acceptance Metrics (빼기/넣기 모두 확인)

### 10.1 정량 (자동화 가능)
- Pelvis lateral std: 4.89 cm → **3-4 cm** (AMASS 범위 내)
- Pelvis lateral range: 19.58 cm → **11-14 cm** (AMASS: 11.96 cm)
- FK stance foot world-position std during single stance: <2 cm (regression: was ~19 cm 보였음)

### 10.2 정성 (사용자 시각 확인)
- Frontal view mp4: stance foot이 world에 거의 고정된 채 pelvis만 roll
- Sagittal view mp4: gait cycle 자연스럽게 유지 (회귀 방지)
- Isometric view mp4: whole-body rigid roll 사라지고 joint articulation 살아남

### 10.3 Invariant
- Flag off (`--baseline_s 3.0 --spine_3axis --upper_body`만) → 기존 v2_full.pkl과 numerically identical (np.allclose)

---

## 11. Rollback Plan

실패 시 즉시 flag off로 기존 v2_full 복원 가능 (설계상 분기 없음, 플래그 1개 토글).
v2_full.pkl, 기존 `compute_root_translation()` 모두 untouched. 최악의 경우 `scripts/data/h5_to_motion_lib_v2.py`의 추가 블록만 제거하면 됨.

---

## 12. Out of Scope

- Non-treadmill 데이터로 일반화 (향후 필요 시 `--no_treadmill_forward` 플래그로 확장)
- Leg joint angle adjustment (hip adduction 재분배 등) — 별도 이슈
- Foot IK (ankle/toe 세부 pose 추정) — ankle world position만 anchor로 사용
- VIC 학습 적용 (별도 task, 먼저 data 검증 후)

# H5→SMPL Foot-Anchor IK Design

**Date**: 2026-04-25
**Status**: Approved (brainstorming → design)
**Owner**: Jimin / Claude
**Related**: `260424_h5_stance_anchor_ip_design.md` (이전 시도, FK-anchor circular flaw)

---

## 1. Background

### 1.1. 문제

H5 (Vicon Plug-in Gait) 측정 데이터를 SMPL 24-joint humanoid 모션으로 변환할 때, **stance foot이 world frame에서 sliding** 한다. RL imitation learning에 사용하면 policy가 자연스러운 보행을 학습하기 어렵다.

### 1.2. 이전 시도들

| 접근 | lat std [cm] | 한계 |
|---|---|---|
| v2_full (CoM 기반 baseline) | 4.16 | trunk + foot 모두 과한 sliding |
| v2_ip (FK-derived foot anchor IP) | 4.13 | circular: FK foot ← pelvis_CoM, IP가 같은 pelvis 보정 → no-op |
| v2_cop (CoP를 lateral anchor로 IP) | 1.60 | trunk lateral은 잡히지만 stance foot 자체는 여전히 미끄러짐 (joint angle noise가 발끝에서 증폭) |
| AMASS walking (target) | 2.82 | — |

### 1.3. 근본 원인

Pelvis lateral 보정만으로는 stance foot fixity를 보장 못 함. PiG로 측정된 hip/knee/ankle joint angle에 작은 noise가 있으면, FK chain 끝의 foot 위치에서 cm 단위 sliding으로 증폭된다. **Pelvis와 leg joint angle을 함께 조정하는 IK가 필요.**

---

## 2. Goal

Stance phase 동안 **stance foot world position을 (거의) 고정**시키는 full-IK pipeline 추가. Foot이 ground에 붙어 있는 동안:
- Heel-only sub-phase: heel position fixed (ankle을 heel proxy로 사용)
- Full-contact sub-phase: heel + toe 둘 다 fixed (foot rigid block flat on ground)
- Toe-only sub-phase: toe position fixed

기존 path (v2_cop)는 backward-compat과 비교 baseline으로 보존.

### 2.1. Acceptance criteria

1. **Full-contact 동안 ankle world position std < 1.0 cm** per stance episode (현재 cm 단위 sliding)
2. Joint angles가 측정값 대비 ±10° 이내 (95% of frames) — physiological signal 보존
3. IK 수렴률 > 95% frames
4. Visual frontal 영상에서 stance foot sliding 없음

---

## 3. Design Decisions (brainstorm summary)

| Q | 결정 | 근거 |
|---|---|---|
| Anchor source | **CoP (forceplate Center of Pressure)** | 측정 기반, pelvis와 독립 (circularity 깨짐) |
| Sub-phase 검출 | **Hybrid: GRF stance + foot pitch sub-phase** | GRF는 stance/swing에 robust, foot pitch는 ankle joint angle 직접 반영 |
| IK DoF | **Full IK: pelvis_trans + stance leg joint angles** | hip/knee 측정 noise가 ankle에 누적되면 ankle만 조정으로 부족 |
| Heel landmark | **Ankle joint as heel proxy** | SMPL에 heel 명시 joint 없음. ankle은 talus 위치 → heel과 가까움. ~5cm offset은 작아서 무시 가능 |
| Constraint per sub-phase | **Heel-only: ankle anchor; Full: ankle + toe; Toe-only: toe anchor** | rigid foot 모델에서 생리학적으로 정확 |
| Backward-compat | **새 flag `--foot_ik {none, full}` (default none)** | 기존 코드 path 보존, 새 helpers는 additive |

---

## 4. Architecture

기존 `convert_trial()` pipeline 안의 **기존 IP block 다음, pose_quat 빌드 이전**에 IK block 삽입.

```
[1] Read H5 measurements (PiG joint angles, CoM, GRF, CoP)
[2] Build initial pose_aa_local + pelvis trans (existing pipeline through compute_root_translation)
[3] If --stance_anchor_ip: 기존 IP block (CoP-anchor or FK-anchor) — 기존 동작
[4] If --foot_ik full:
    a. FK pass: ankle/toe world positions from current pose+trans
    b. Sub-phase classifier: per-foot, per-frame label
    c. Anchor builder: H, T per stance episode per foot
    d. Per-frame IK solve: corrected pelvis_trans + stance leg angles
    e. Write back: trans, pose_aa_local
[5] Build pose_quat_global, pose_quat (existing)
[6] Save pkl as *_v2_footik.pkl
```

---

## 5. Components

### 5a. Sub-phase classifier

**Input**:
- `grf_L`, `grf_R`: [T] GRF magnitudes (Newton)
- `ankle_L_w`, `toe_L_w`, `ankle_R_w`, `toe_R_w`: [T,3] world positions (Z-up frame, from FK)

**Algorithm**:
1. Stance/swing per foot: GRF Schmitt trigger (on=50N, off=20N) — 기존 `_schmitt`/`detect_stance` 재사용
2. Within stance, sub-phase from foot pitch:
   - `foot_pitch[t] = arctan2(toe_z - ankle_z, ||toe_xy - ankle_xy||)` in degrees
   - pitch > +5° → `heel-only`
   - |pitch| ≤ 5° → `full-contact`
   - pitch < -5° → `toe-only`
3. Output: `phase_L`, `phase_R` ∈ {0=swing, 1=heel-only, 2=full-contact, 3=toe-only}

**Threshold tunable** via `--foot_ik_pitch_threshold_deg` (default 5.0).

### 5b. Anchor builder

**Input**:
- `cop_L_xy`, `cop_R_xy`: [T,2] CoP medio-lateral + anterior-posterior in lab frame (m)
- `phase_L`, `phase_R`: from classifier
- Stance episodes (start, end) per foot — derived from phase ≠ swing

**Algorithm**:
For each stance episode `(s, e)` of foot `F`:
- `heel_frames = {t ∈ [s,e] : phase_F[t] == heel-only}`
- `toe_frames  = {t ∈ [s,e] : phase_F[t] == toe-only}`
- `H_F = mean(cop_F[heel_frames])` (xy lab frame, z=0)
  - Fallback if `len(heel_frames) < 2`: `H_F = cop_F[s]` (first stance frame)
- `T_F = mean(cop_F[toe_frames])`
  - Fallback if `len(toe_frames) < 2`: `T_F = cop_F[e]` (last stance frame)
- If `(e - s) < 10` frames: skip episode (mark as IK_skip, IK uses identity for this range)

**Output**: per-frame `anchor_H_L`, `anchor_T_L`, `anchor_H_R`, `anchor_T_R` (constant within each stance episode), and `ik_active_L`, `ik_active_R` masks (False = swing or skip).

### 5c. Lab frame ↔ world frame alignment

CoP는 lab frame, IK는 변환된 world frame (treadmill-integrated forward axis 포함). Lateral과 vertical 정렬:
- Lab x (medio-lateral) → world Y (lateral) with sign convention from `cop_replace_lateral` (auto-detected)
- Lab z (vertical) → world Z (vertical), z=0 → ground
- Lab y (anterior-posterior) → forward axis: **CoP의 y는 lab frame에서 bounded** (treadmill 위 위치), 우리 world frame은 **integrated forward** (unbounded). 따라서 anchor의 forward 좌표는 측정된 ankle/toe FK position의 stance episode 평균을 사용 (anchor stable in world forward).

즉:
- `anchor_x` (world lateral) = sign × CoP_x + offset (기존 `cop_replace_lateral` 로직 재사용)
- `anchor_y` (world forward) = mean(FK foot forward) over heel-only/toe-only frames
- `anchor_z` (world vertical) = 0

### 5d. IK solver (per frame)

**Decision variables** per frame:
- pelvis_trans: 3 DoF (world lateral + forward + vertical)
- L stance leg joints (if L in stance): hip (3) + knee (1) + ankle (3) = 7 DoF
- R stance leg joints (if R in stance): 7 DoF
- Total: 3, 10, or 17 DoF depending on single/double stance

**Constraint equations** (per sub-phase, applied as soft cost with weight `w_anchor=1e3`):
- L heel-only: `ankle_L_world(pose) - H_L = 0` (3 eq)
- L full-contact: `ankle_L_world(pose) - H_L = 0` AND `toe_L_world(pose) - T_L = 0` (6 eq)
- L toe-only: `toe_L_world(pose) - T_L = 0` (3 eq)
- (Same for R if in stance)

**Soft cost terms**:
- Joint angle deviation: `w_joint × (joint_angle - measured_angle)^2` per joint, default `w_joint=0.1`
- Pelvis trans deviation: `w_pelvis × (trans - measured_trans)^2`, default `w_pelvis=0.01`
- Frame-to-frame smoothness: `w_smooth × (joint_angle(t) - joint_angle(t-1))^2`, default `w_smooth=0.5`

**Solver**:
- `scipy.optimize.least_squares` with `method='trf'` (Trust Region Reflective)
- **Warm start**: measured values (zero IK adjustment as initial guess)
- **Bounds**: joint angles within ±20° of measured (avoid wild solutions)
- **Max iter**: 50 per frame (tunable via `--foot_ik_max_iter`)
- **Numerical Jacobian** (scipy default; analytical Jacobian as future optimization if too slow)

**Per-frame computation cost**: ~10-50ms × 5800 frames = ~1-5 min per trial. Acceptable for offline conversion.

**Convergence failure handling**:
- If solver fails to converge: log warning, retain measured values for that frame (graceful degradation)
- Tracked via `ik_converged` flag per frame

### 5e. Output writeback

After IK loop:
- `trans` (Y-up frame, before upright.apply): write back IK-corrected pelvis_trans
- `pose_aa_local` (per-frame axis-angle, BONE order): write back IK-corrected joint angles for L/R hip, knee, ankle
- Pose for non-stance leg unchanged
- Pelvis rotation unchanged (kept as PiG measurement)

Downstream pose_quat / pose_quat_global pipeline runs unchanged.

---

## 6. CLI

### 6a. Primary flag
- `--foot_ik {none, full}` (default `none`)
  - `none`: 기존 동작 (with optional `--stance_anchor_source cop`)
  - `full`: foot-anchor IK 활성화

### 6b. Tuning flags
- `--foot_ik_pitch_threshold_deg` (default 5.0): sub-phase 분류 threshold
- `--foot_ik_joint_reg_weight` (default 0.1): joint angle deviation cost
- `--foot_ik_smoothness_weight` (default 0.5): frame-to-frame smoothness cost
- `--foot_ik_pelvis_reg_weight` (default 0.01): pelvis trans deviation cost
- `--foot_ik_anchor_weight` (default 1e3): anchor satisfaction weight (effectively hard)
- `--foot_ik_max_iter` (default 50): solver iterations per frame
- `--foot_ik_bounds_deg` (default 20.0): joint angle bounds = measured ± this

### 6c. Output naming
- `--output sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl` (user-specified)
- Convention: `*_v2_footik.pkl`

---

## 7. Files affected

### 7a. New helpers in `scripts/data/h5_conversion_helpers.py`
- `classify_sub_phases(grf_L, grf_R, ankle_L, toe_L, ankle_R, toe_R, pitch_threshold_deg=5.0)` → `(phase_L, phase_R)`
- `build_foot_anchors(cop_L_xy, cop_R_xy, ankle_L_w, toe_L_w, ankle_R_w, toe_R_w, phase_L, phase_R, fps)` → `(anchors, ik_active)` dict
- `solve_foot_ik_frame(pose_measured, trans_measured, anchors_t, phase_t, weights, bounds, sk_tree)` → `(pose_corrected, trans_corrected, converged)`

### 7b. Modified `scripts/data/h5_to_motion_lib_v2.py`
- Add `--foot_ik` flag (and tuning flags) in `argparse`
- Add `foot_ik` parameter to `convert_trial()`
- Insert IK block after existing IP block (line ~691), before pose_quat construction
- Pass new args through `convert_trial()` call

### 7c. Tests `scripts/data/test_h5_conversion_fixes.py`
- `test_classify_sub_phases_synthetic`: hand-built foot pitch trajectory (heel-strike → flat → toe-off), assert correct labels
- `test_build_foot_anchors_synthetic`: synthetic CoP trajectory with known H, T, assert mean recovery
- `test_solve_foot_ik_frame_identity`: identity initial pose, anchor at FK position → IK solution = initial (zero adjustment)
- `test_solve_foot_ik_frame_lateral_offset`: anchor 5cm lateral of FK → IK adjusts pelvis trans by ~5cm

---

## 8. Testing strategy

### 8a. Unit tests (TDD per superpowers)
모든 새 helper는 failing test 먼저, 구현 후 pass.

### 8b. Integration test
- Run conversion: `python scripts/data/h5_to_motion_lib_v2.py --h5 data/combined_data_from_csv.h5 --output sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl --subjects S001 --tasks level_100mps --assist_level lv0 --baseline_s 2.0 --spine_3axis --upper_body --foot_ik full --no_split --trim_start 2.0`
- Measure: ankle world position std during full-contact, per stance episode
  - Pass: < 1.0 cm
  - Compare: v2_cop (~5cm visual sliding), v2_full (similar), AMASS (similar)

### 8c. Visual verification
- `compare_pkl_metrics.py --pkl_a v2_footik --pkl_b v2_cop --view all` → frontal에서 stance foot 비교
- `compare_pkl_metrics.py --pkl_a v2_footik --pkl_b AMASS --view all` → naturalness 비교

### 8d. RL training smoke test (optional, post-merge)
- Convert with `--foot_ik full`, train VIC pipeline 1k epochs, check `eval_success_rate` 차이

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| IK 수렴 실패 (solver stuck in local min) | Medium | Warm start from measured + bounds + per-frame fallback to measured |
| Joint angles 측정값에서 너무 멀어짐 (unphysical) | Medium | `w_joint=0.1` + `bounds=±20°` cap |
| 처리 시간 너무 길음 (>10 min per trial) | Low | Trial당 5800 frames × ~30ms = ~3min 예상. 문제 시 analytical Jacobian or 병렬화 |
| Sub-phase 검출 불안정 (pitch threshold 부적절) | Medium | Tunable flag + 추후 GRF profile shape도 합산하는 hybrid로 upgrade |
| CoP 정렬 (lab→world) 부정확 | Low | `cop_replace_lateral`의 sign auto-detect 재사용 |

---

## 10. Rollback

- `--foot_ik none` (default) → 새 helpers 호출 안 됨, 기존 동작 100% 유지
- 새 helpers는 별도 함수로 추가만 (기존 함수 modify 없음)
- Git revert: 마지막 IP commit (5f9d132 또는 그 이전) 으로 단순 revert 가능

---

## 11. Out of scope

- Multi-trial batched IK (per-trial optimization만)
- Cross-frame trajectory optimization (per-frame IK + smoothness regularizer만)
- SMPL shape parameter (`betas`) optimization (neutral 유지)
- Toe joint articulation (frozen 유지)
- Heel-bone offset 정확 모델링 (ankle = heel proxy)
- Knee abduction/adduction 추가 DoF (SMPL knee = 1 DoF hinge 그대로)
- Multi-pass IK with constraint relaxation

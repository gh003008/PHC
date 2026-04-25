# H5 Foot-Anchor IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full foot-anchor IK to the H5→SMPL conversion pipeline that anchors stance feet to measured forceplate CoP, jointly adjusting pelvis trans + stance-leg joint angles. Stance foot world position becomes (nearly) constant during stance.

**Architecture:** New helpers in `h5_conversion_helpers.py` (sub-phase classifier, anchor builder, per-frame IK solver). Wired into `convert_trial()` via new `--foot_ik {none, full}` CLI flag (default `none` for backward-compat). Per-frame `scipy.optimize.least_squares` over (pelvis_trans, stance leg joint angles) with sub-phase-aware constraints (heel-only / full-contact / toe-only).

**Tech Stack:** numpy, scipy.optimize.least_squares + scipy.spatial.transform.Rotation, h5py, joblib, poselib SkeletonTree (for local_translation only).

**Reference:** `01_research_docs/260425_h5_foot_anchor_ik_design.md` (design spec).

**Conda environment:** All commands run inside `conda activate phc`.

---

## Codebase facts (memorize before starting)

- **BONE order leg chain**: L: 0(Pelvis)→1(L_Hip)→4(L_Knee)→7(L_Ankle)→10(L_Toe); R: 0→2(R_Hip)→5(R_Knee)→8(R_Ankle)→11(R_Toe)
- **MUJOCO order leg chain** (used by `sk_tree.local_translation`): L: 0→1→2→3→4; R: 0→5→6→7→8
- **`pose_aa_local`** in `convert_trial`: shape (T, 24, 3), BONE order, axis-angle
- **`trans`** in `convert_trial` before line ~772: Y-up frame (X=right, Y=up, Z=forward)
- **`upright = sRot.from_quat([0.5,0.5,0.5,0.5])`**: Y-up → Z-up rotation. Applied to `trans` at the end (line ~772)
- **Existing IP block**: lines ~569-734, working in Z-up internally via `_upright.apply(trans)`
- **CoP H5 paths**: `forceplate/cop/{left,right}/{x,y,z}` — 100Hz, mm
- **GRF H5 paths**: `forceplate/grf/{left,right}/z` — 100Hz, Newtons
- **`read_grf`** (existing helper, line 168): reads `forceplate/grf/{side}/z`, returns abs values, NaN-interpolated, in Newtons
- **`read_cop`** (existing helper, line 168ish): reads `forceplate/cop/{side}/x` only, returns meters
- **All new helpers** go into `scripts/data/h5_conversion_helpers.py`
- **All new tests** go into `scripts/data/test_h5_conversion_fixes.py`
- **Run tests**: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py -v`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `scripts/data/h5_conversion_helpers.py` | Modify | Add 3 pure helpers: `classify_sub_phases`, `build_foot_anchors`, `solve_foot_ik_frame`, plus 2 internal helpers (`_extract_leg_local_offsets`, `_fk_leg_world_xyz`) |
| `scripts/data/h5_to_motion_lib_v2.py` | Modify | Add `read_cop_xy` helper (CoP x AND y), add `--foot_ik` flag + tuning flags, add IK block in `convert_trial`, pass new args |
| `scripts/data/test_h5_conversion_fixes.py` | Modify | Add 4 tests for new helpers (sub-phase, anchors, FK leg, IK frame) |

---

## Task 1: classify_sub_phases helper

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py` (append new function after `cop_replace_lateral`)
- Modify: `scripts/data/test_h5_conversion_fixes.py` (append new test)

**What this does:** Per-foot, per-frame label `0=swing, 1=heel-only, 2=full-contact, 3=toe-only` from GRF (stance/swing) and foot pitch (sub-phase within stance).

- [ ] **Step 1.1: Write the failing test**

Append to `scripts/data/test_h5_conversion_fixes.py`:

```python
def test_classify_sub_phases_synthetic():
    """Synthetic foot pitch sweep across one full gait cycle.

    GRF: 0..30 swing → 30..70 stance → 70..100 swing. (single foot used for test)
    Pitch trajectory during stance: starts at +15° (heel down, toe up),
    sweeps through 0° (full contact mid-stance), ends at -15° (toe down, heel up).
    """
    from data.h5_conversion_helpers import classify_sub_phases
    T = 100
    grf_L = np.zeros(T)
    grf_L[30:70] = 200.0  # 200N stance
    grf_R = np.zeros(T)   # R always swing
    # Foot pitch: ankle below toe = positive pitch (heel-down).
    # Build synthetic ankle/toe so pitch sweeps +15 → 0 → -15 across stance.
    ankle_L = np.zeros((T, 3))
    toe_L = np.zeros((T, 3))
    foot_len = 0.16  # m
    for t in range(T):
        if 30 <= t < 70:
            # pitch_deg = +15° at t=30, 0° at t=50, -15° at t=70
            pitch_deg = 15.0 - (t - 30) * (30.0 / 40.0)
            pitch_rad = np.radians(pitch_deg)
            # ankle at (0, 0, 0), toe in +x direction with vertical offset = foot_len * sin(pitch)
            ankle_L[t] = [0.0, 0.0, 0.0]
            toe_L[t] = [foot_len * np.cos(pitch_rad), 0.0, foot_len * np.sin(pitch_rad)]
        else:
            ankle_L[t] = [0.0, 0.0, 0.5]   # foot in air
            toe_L[t] = [foot_len, 0.0, 0.5]
    ankle_R = np.zeros((T, 3))
    toe_R = np.zeros((T, 3))
    toe_R[:, 0] = foot_len

    phase_L, phase_R = classify_sub_phases(
        grf_L, grf_R, ankle_L, toe_L, ankle_R, toe_R,
        pitch_threshold_deg=5.0,
    )

    # Swing frames are 0
    assert np.all(phase_L[:30] == 0), "pre-stance swing should be 0"
    assert np.all(phase_L[70:] == 0), "post-stance swing should be 0"
    # Stance: heel-only → full-contact → toe-only as pitch sweeps
    # pitch +15 → 0 → -15 across t=30..70
    # pitch > +5°: t in [30, ~43] → heel-only (label 1)
    # |pitch| <= 5°: t in [~44, ~57] → full-contact (label 2)
    # pitch < -5°: t in [~58, 70] → toe-only (label 3)
    # Verify at least 3 frames of each sub-phase exist:
    n_heel = int((phase_L == 1).sum())
    n_full = int((phase_L == 2).sum())
    n_toe = int((phase_L == 3).sum())
    assert n_heel >= 3, f"expected >=3 heel-only frames, got {n_heel}"
    assert n_full >= 3, f"expected >=3 full-contact frames, got {n_full}"
    assert n_toe >= 3, f"expected >=3 toe-only frames, got {n_toe}"
    # R foot all swing
    assert np.all(phase_R == 0), "R foot should be all swing"
```

- [ ] **Step 1.2: Run the test, verify FAIL**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_classify_sub_phases_synthetic -v`
Expected: FAIL with `ImportError` (function not yet defined).

- [ ] **Step 1.3: Implement classify_sub_phases**

Append to `scripts/data/h5_conversion_helpers.py` (after `cop_replace_lateral`, before `solve_pelvis_ip`):

```python
def classify_sub_phases(grf_L, grf_R, ankle_L, toe_L, ankle_R, toe_R,
                       pitch_threshold_deg=5.0,
                       grf_on=50.0, grf_off=20.0):
    """Per-foot, per-frame stance sub-phase classification.

    Stance/swing from GRF Schmitt trigger. Within stance, sub-phase from foot
    pitch (angle of ankle→toe vector above horizontal):
      pitch > +threshold:  heel-only (heel down, toe up)
      |pitch| <= threshold: full-contact
      pitch < -threshold:  toe-only (toe down, heel up)

    Args:
        grf_L, grf_R: [T] GRF magnitudes (Newtons).
        ankle_L, toe_L, ankle_R, toe_R: [T, 3] world positions (Z-up frame).
        pitch_threshold_deg: sub-phase split threshold (degrees).
        grf_on, grf_off: Schmitt trigger thresholds for stance (Newtons).

    Returns:
        phase_L, phase_R: [T] int8 arrays. 0=swing, 1=heel-only, 2=full-contact, 3=toe-only.
    """
    T = len(grf_L)
    stance_L = _schmitt(grf_L, grf_on, grf_off, on_when_above=True)
    stance_R = _schmitt(grf_R, grf_on, grf_off, on_when_above=True)

    def _foot_pitch_deg(ankle, toe):
        v = toe - ankle                                    # (T, 3)
        horiz = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)        # (T,)
        return np.degrees(np.arctan2(v[:, 2], horiz))       # (T,)

    pitch_L = _foot_pitch_deg(ankle_L, toe_L)
    pitch_R = _foot_pitch_deg(ankle_R, toe_R)

    def _classify(stance, pitch):
        out = np.zeros(T, dtype=np.int8)
        in_stance = stance
        out[in_stance & (pitch > pitch_threshold_deg)] = 1
        out[in_stance & (np.abs(pitch) <= pitch_threshold_deg)] = 2
        out[in_stance & (pitch < -pitch_threshold_deg)] = 3
        return out

    return _classify(stance_L, pitch_L), _classify(stance_R, pitch_R)
```

- [ ] **Step 1.4: Run test, verify PASS**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_classify_sub_phases_synthetic -v`
Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py scripts/data/test_h5_conversion_fixes.py
git commit -m "feat(h5): classify_sub_phases for foot-anchor IK"
```

---

## Task 2: build_foot_anchors helper

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py` (append after `classify_sub_phases`)
- Modify: `scripts/data/test_h5_conversion_fixes.py` (append new test)

**What this does:** For each stance episode per foot, compute heel anchor `H` (mean CoP over heel-only frames) and toe anchor `T` (mean CoP over toe-only frames). Returns per-frame anchor arrays + active masks.

- [ ] **Step 2.1: Write failing test**

```python
def test_build_foot_anchors_synthetic():
    """Synthetic stance with known CoP at heel/toe — anchors should recover means.

    Stance L: t in [30, 70). Phase: heel-only [30, 40), full [40, 60), toe-only [60, 70).
    CoP_L: x=0.10 during heel-only, x sweeps 0.10→0.25 during full, x=0.25 during toe-only.
    Expected: H_L_x ≈ 0.10, T_L_x ≈ 0.25.
    """
    from data.h5_conversion_helpers import build_foot_anchors
    T = 100
    cop_L_xy = np.zeros((T, 2))
    cop_R_xy = np.zeros((T, 2))
    cop_L_xy[30:40, 0] = 0.10                          # heel-only
    cop_L_xy[40:60, 0] = np.linspace(0.10, 0.25, 20)   # full sweep
    cop_L_xy[60:70, 0] = 0.25                          # toe-only

    # FK fwd (lab y) — simulate stance ankle world forward = treadmill-integrated value
    # During stance, "forward" component of foot increases linearly with time
    ankle_L = np.zeros((T, 3))
    toe_L = np.zeros((T, 3))
    ankle_L[:, 1] = np.linspace(0, 5.0, T)   # forward axis grows
    toe_L[:, 1] = ankle_L[:, 1] + 0.16
    ankle_R = np.zeros((T, 3))
    toe_R = np.zeros((T, 3))

    phase_L = np.zeros(T, dtype=np.int8)
    phase_L[30:40] = 1   # heel-only
    phase_L[40:60] = 2   # full
    phase_L[60:70] = 3   # toe-only
    phase_R = np.zeros(T, dtype=np.int8)

    anchors = build_foot_anchors(
        cop_L_xy, cop_R_xy, ankle_L, toe_L, ankle_R, toe_R,
        phase_L, phase_R, fps=30, min_episode_frames=10,
    )
    # anchors['H_L'] is per-frame [T, 3] (constant within episode), with NaN outside
    # During the stance episode (t in 30..70):
    H_L = anchors["H_L"]
    T_L = anchors["T_L"]
    ik_active_L = anchors["ik_active_L"]
    # anchor x: lateral (CoP-derived)
    assert abs(H_L[35, 0] - 0.10) < 1e-3, f"H_L lateral expected 0.10, got {H_L[35, 0]}"
    assert abs(T_L[65, 0] - 0.25) < 1e-3, f"T_L lateral expected 0.25, got {T_L[65, 0]}"
    # anchor z: ground level
    assert abs(H_L[35, 2]) < 1e-9, f"H_L z expected 0, got {H_L[35, 2]}"
    assert abs(T_L[65, 2]) < 1e-9
    # ik_active during stance, inactive during swing
    assert ik_active_L[35] and ik_active_L[65]
    assert not ik_active_L[10] and not ik_active_L[90]
    # R foot: no stance → not active
    assert not anchors["ik_active_R"].any()
```

- [ ] **Step 2.2: Run test, verify FAIL**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_build_foot_anchors_synthetic -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 2.3: Implement build_foot_anchors**

Append to `scripts/data/h5_conversion_helpers.py`:

```python
def build_foot_anchors(cop_L_xy, cop_R_xy, ankle_L, toe_L, ankle_R, toe_R,
                      phase_L, phase_R, fps, min_episode_frames=10):
    """Per-stance-episode heel/toe anchors from sub-phase-averaged CoP.

    Anchor = (lateral from CoP mean over sub-phase frames,
              forward from FK foot mean over sub-phase frames,
              vertical = 0 [ground]).
    Per-frame arrays have anchor value during the stance episode, NaN elsewhere.

    Args:
        cop_L_xy, cop_R_xy: [T, 2] CoP medio-lateral (lab x → world lateral, sign-aligned)
            and anterior-posterior (lab y, but UNUSED — forward comes from FK).
        ankle_L, toe_L, ankle_R, toe_R: [T, 3] world positions (Z-up).
        phase_L, phase_R: [T] sub-phase labels from classify_sub_phases.
        fps: sampling rate (used only for short-episode skip).
        min_episode_frames: stance episodes shorter than this are skipped (ik_active=False).

    Returns:
        dict with keys:
          H_L, T_L, H_R, T_R: [T, 3] per-frame anchor (NaN outside stance/skipped).
          ik_active_L, ik_active_R: [T] bool, True if IK should run on that frame.
    """
    T = len(phase_L)
    out = {
        "H_L": np.full((T, 3), np.nan), "T_L": np.full((T, 3), np.nan),
        "H_R": np.full((T, 3), np.nan), "T_R": np.full((T, 3), np.nan),
        "ik_active_L": np.zeros(T, dtype=bool),
        "ik_active_R": np.zeros(T, dtype=bool),
    }

    def _episodes(phase):
        """Yield (start, end_exclusive) for each contiguous stance episode (phase != 0)."""
        in_ep = False
        s = -1
        for t in range(T):
            stance = phase[t] != 0
            if stance and not in_ep:
                s = t; in_ep = True
            elif not stance and in_ep:
                yield (s, t); in_ep = False
        if in_ep:
            yield (s, T)

    def _build_one(phase, cop_xy, ankle, toe, H_out, T_out, active_out):
        for (s, e) in _episodes(phase):
            if (e - s) < min_episode_frames:
                continue
            sub = phase[s:e]
            heel_idx = s + np.flatnonzero(sub == 1)
            full_idx = s + np.flatnonzero(sub == 2)
            toe_idx = s + np.flatnonzero(sub == 3)
            # Heel anchor: mean CoP over heel-only frames; fallback to first stance frame
            if len(heel_idx) >= 2:
                H_lat = float(cop_xy[heel_idx, 0].mean())
                H_fwd = float(ankle[heel_idx, 1].mean())
            else:
                H_lat = float(cop_xy[s, 0])
                H_fwd = float(ankle[s, 1])
            # Toe anchor: mean CoP over toe-only frames; fallback to last stance frame
            if len(toe_idx) >= 2:
                T_lat = float(cop_xy[toe_idx, 0].mean())
                T_fwd = float(toe[toe_idx, 1].mean())
            else:
                T_lat = float(cop_xy[e - 1, 0])
                T_fwd = float(toe[e - 1, 1])
            # Write per-frame anchor (constant within episode)
            for t in range(s, e):
                H_out[t] = [H_lat, H_fwd, 0.0]
                T_out[t] = [T_lat, T_fwd, 0.0]
            active_out[s:e] = True

    _build_one(phase_L, cop_L_xy, ankle_L, toe_L,
               out["H_L"], out["T_L"], out["ik_active_L"])
    _build_one(phase_R, cop_R_xy, ankle_R, toe_R,
               out["H_R"], out["T_R"], out["ik_active_R"])
    return out
```

- [ ] **Step 2.4: Run test, verify PASS**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_build_foot_anchors_synthetic -v`
Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py scripts/data/test_h5_conversion_fixes.py
git commit -m "feat(h5): build_foot_anchors per stance episode from sub-phase CoP"
```

---

## Task 3: _fk_leg_world primitive (low-level FK)

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py` (append `_extract_leg_local_offsets`, `_fk_leg_world_xyz`)
- Modify: `scripts/data/test_h5_conversion_fixes.py` (append test)

**What this does:** Direct numpy FK for one leg chain (pelvis → hip → knee → ankle → toe). Pure function, no SkeletonState rebuild. Critical for IK speed — called inside scipy least_squares cost.

- [ ] **Step 3.1: Write failing test**

```python
def test_fk_leg_world_zero_pose_matches_local_translations():
    """Zero pose (identity rotations) → world ankle = sum of local translations along chain."""
    import torch
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from scripts.data.h5_to_motion_lib_v2 import get_skeleton_tree
    from data.h5_conversion_helpers import _extract_leg_local_offsets, _fk_leg_world_xyz

    sk_tree = get_skeleton_tree()
    offsets_L = _extract_leg_local_offsets(sk_tree, side='L')
    # Zero joint angles, identity pelvis rotation, pelvis at origin
    pelvis_trans = np.zeros(3)
    R_pelvis = np.eye(3)
    hip_aa = np.zeros(3)
    knee_aa = np.zeros(3)
    ankle_aa = np.zeros(3)

    ankle_world, toe_world = _fk_leg_world_xyz(
        pelvis_trans, R_pelvis, hip_aa, knee_aa, ankle_aa, offsets_L,
    )
    # Expected: sum of local translations (all rotations identity)
    expected_ankle = offsets_L["hip"] + offsets_L["knee"] + offsets_L["ankle"]
    expected_toe = expected_ankle + offsets_L["toe"]
    assert np.allclose(ankle_world, expected_ankle, atol=1e-6), \
        f"ankle_world {ankle_world} != expected {expected_ankle}"
    assert np.allclose(toe_world, expected_toe, atol=1e-6)


def test_fk_leg_world_pelvis_translates_chain():
    """Translating pelvis by Δ should translate ankle/toe by Δ (zero rotations)."""
    from scripts.data.h5_to_motion_lib_v2 import get_skeleton_tree
    from data.h5_conversion_helpers import _extract_leg_local_offsets, _fk_leg_world_xyz

    sk_tree = get_skeleton_tree()
    offsets_L = _extract_leg_local_offsets(sk_tree, side='L')
    delta = np.array([1.0, 2.0, 3.0])

    a0, t0 = _fk_leg_world_xyz(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), offsets_L)
    a1, t1 = _fk_leg_world_xyz(delta, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), offsets_L)
    assert np.allclose(a1 - a0, delta, atol=1e-6)
    assert np.allclose(t1 - t0, delta, atol=1e-6)
```

- [ ] **Step 3.2: Run test, verify FAIL**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_fk_leg_world_zero_pose_matches_local_translations test_h5_conversion_fixes.py::test_fk_leg_world_pelvis_translates_chain -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3.3: Implement helpers**

Append to `scripts/data/h5_conversion_helpers.py`:

```python
def _extract_leg_local_offsets(skeleton_tree, side='L'):
    """Extract local translations along the leg chain from poselib SkeletonTree.

    sk_tree.local_translation is in MUJOCO order. Indices:
      L: Pelvis(0)→L_Hip(1)→L_Knee(2)→L_Ankle(3)→L_Toe(4)
      R: Pelvis(0)→R_Hip(5)→R_Knee(6)→R_Ankle(7)→R_Toe(8)

    Returns dict with keys 'hip', 'knee', 'ankle', 'toe', each (3,) np.float64.
    """
    lt = skeleton_tree.local_translation.numpy().astype(np.float64)  # (24, 3)
    if side == 'L':
        return {
            "hip":   lt[1],
            "knee":  lt[2],
            "ankle": lt[3],
            "toe":   lt[4],
        }
    elif side == 'R':
        return {
            "hip":   lt[5],
            "knee":  lt[6],
            "ankle": lt[7],
            "toe":   lt[8],
        }
    raise ValueError(f"side must be 'L' or 'R', got {side}")


def _fk_leg_world_xyz(pelvis_trans, R_pelvis_world,
                     hip_aa, knee_aa, ankle_aa, leg_local_offsets):
    """Forward kinematics for one leg: world position of ankle and toe.

    Standard FK: each joint's world rotation = parent world rotation × local rotation.
    Each joint's world position = parent world position + parent world rotation × local offset.

    Args:
        pelvis_trans: (3,) pelvis world position.
        R_pelvis_world: (3, 3) pelvis world rotation matrix.
        hip_aa, knee_aa, ankle_aa: (3,) axis-angle joint rotations (BONE order semantics).
        leg_local_offsets: dict from _extract_leg_local_offsets.

    Returns:
        ankle_world: (3,) np.float64
        toe_world:   (3,) np.float64
    """
    from scipy.spatial.transform import Rotation as sRot
    R_hip = sRot.from_rotvec(hip_aa).as_matrix()
    R_knee = sRot.from_rotvec(knee_aa).as_matrix()
    R_ankle = sRot.from_rotvec(ankle_aa).as_matrix()

    hip_world = pelvis_trans + R_pelvis_world @ leg_local_offsets["hip"]
    R_hip_w = R_pelvis_world @ R_hip
    knee_world = hip_world + R_hip_w @ leg_local_offsets["knee"]
    R_knee_w = R_hip_w @ R_knee
    ankle_world = knee_world + R_knee_w @ leg_local_offsets["ankle"]
    R_ankle_w = R_knee_w @ R_ankle
    toe_world = ankle_world + R_ankle_w @ leg_local_offsets["toe"]
    return ankle_world.astype(np.float64), toe_world.astype(np.float64)
```

- [ ] **Step 3.4: Run tests, verify PASS**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_fk_leg_world_zero_pose_matches_local_translations test_h5_conversion_fixes.py::test_fk_leg_world_pelvis_translates_chain -v`
Expected: PASS both.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py scripts/data/test_h5_conversion_fixes.py
git commit -m "feat(h5): _fk_leg_world_xyz numpy FK primitive for IK"
```

---

## Task 4: solve_foot_ik_frame helper

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py` (append `solve_foot_ik_frame`)
- Modify: `scripts/data/test_h5_conversion_fixes.py` (append 2 tests)

**What this does:** Per-frame scipy nonlinear least-squares solve. Given measured pose, anchors, and sub-phase, returns IK-corrected pelvis_trans + stance leg joint angles.

- [ ] **Step 4.1: Write failing tests**

```python
def test_solve_foot_ik_frame_identity_no_adjustment():
    """If anchor = current FK foot position, IK should converge with ~zero adjustment."""
    from scripts.data.h5_to_motion_lib_v2 import get_skeleton_tree
    from data.h5_conversion_helpers import (
        _extract_leg_local_offsets, _fk_leg_world_xyz, solve_foot_ik_frame,
    )

    sk_tree = get_skeleton_tree()
    offsets_L = _extract_leg_local_offsets(sk_tree, side='L')
    offsets_R = _extract_leg_local_offsets(sk_tree, side='R')

    # Measured pose: small non-zero hip flex (10° forward = pose_aa around X axis)
    pose_measured = np.zeros((24, 3))
    pose_measured[1] = [np.radians(10), 0, 0]   # L_HIP small flex
    pose_measured[2] = [np.radians(10), 0, 0]   # R_HIP small flex
    trans_measured = np.array([0.0, 0.9, 0.0])  # ~waist height
    R_pelvis = np.eye(3)

    # FK to compute current ankle/toe positions
    ankle_L, toe_L = _fk_leg_world_xyz(trans_measured, R_pelvis,
                                       pose_measured[1], pose_measured[4], pose_measured[7], offsets_L)
    ankle_R, toe_R = _fk_leg_world_xyz(trans_measured, R_pelvis,
                                       pose_measured[2], pose_measured[5], pose_measured[8], offsets_R)
    # Use FK positions as anchors (consistent solution = no adjustment)
    anchors_t = {
        "H_L": ankle_L.copy(), "T_L": toe_L.copy(),
        "H_R": ankle_R.copy(), "T_R": toe_R.copy(),
    }
    # Both legs in full-contact
    phase_t = {"L": 2, "R": 2}

    pose_corr, trans_corr, converged = solve_foot_ik_frame(
        pose_measured, trans_measured, R_pelvis,
        anchors_t, phase_t,
        offsets_L, offsets_R,
        weights={"anchor": 1e3, "joint": 0.1, "smooth": 0.0, "pelvis": 0.01},
        bounds_deg=20.0, max_iter=50,
        pose_prev=None,
    )
    assert converged
    assert np.allclose(trans_corr, trans_measured, atol=1e-3)
    assert np.allclose(pose_corr[1], pose_measured[1], atol=np.radians(0.5))


def test_solve_foot_ik_frame_lateral_anchor_shifts_pelvis():
    """Anchor 5cm lateral of FK position → IK should shift pelvis trans laterally by ~5cm."""
    from scripts.data.h5_to_motion_lib_v2 import get_skeleton_tree
    from data.h5_conversion_helpers import (
        _extract_leg_local_offsets, _fk_leg_world_xyz, solve_foot_ik_frame,
    )

    sk_tree = get_skeleton_tree()
    offsets_L = _extract_leg_local_offsets(sk_tree, side='L')
    offsets_R = _extract_leg_local_offsets(sk_tree, side='R')

    pose_measured = np.zeros((24, 3))
    trans_measured = np.array([0.0, 0.9, 0.0])
    R_pelvis = np.eye(3)
    # Only L in stance (full-contact); R swing (no anchor)
    ankle_L, toe_L = _fk_leg_world_xyz(trans_measured, R_pelvis,
                                       pose_measured[1], pose_measured[4], pose_measured[7], offsets_L)
    # Shift L anchors laterally by +5cm (world Y axis = lateral in our convention; here use index 1)
    delta_lat = 0.05
    H_L_shifted = ankle_L.copy(); H_L_shifted[1] += delta_lat
    T_L_shifted = toe_L.copy(); T_L_shifted[1] += delta_lat
    anchors_t = {
        "H_L": H_L_shifted, "T_L": T_L_shifted,
        "H_R": np.full(3, np.nan), "T_R": np.full(3, np.nan),
    }
    phase_t = {"L": 2, "R": 0}

    pose_corr, trans_corr, converged = solve_foot_ik_frame(
        pose_measured, trans_measured, R_pelvis,
        anchors_t, phase_t,
        offsets_L, offsets_R,
        weights={"anchor": 1e3, "joint": 0.1, "smooth": 0.0, "pelvis": 0.01},
        bounds_deg=20.0, max_iter=50,
        pose_prev=None,
    )
    assert converged
    # With only joint reg + pelvis reg as costs, the cheapest solution is to shift pelvis trans
    # laterally. Tolerance loose because joint regularizer can also absorb some.
    assert (trans_corr[1] - trans_measured[1]) > 0.02, \
        f"pelvis lateral should shift positive, got Δ={trans_corr[1] - trans_measured[1]:.4f}"
    assert (trans_corr[1] - trans_measured[1]) <= delta_lat + 1e-3
```

- [ ] **Step 4.2: Run tests, verify FAIL**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_solve_foot_ik_frame_identity_no_adjustment test_h5_conversion_fixes.py::test_solve_foot_ik_frame_lateral_anchor_shifts_pelvis -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4.3: Implement solve_foot_ik_frame**

Append to `scripts/data/h5_conversion_helpers.py`:

```python
def solve_foot_ik_frame(pose_measured, trans_measured, R_pelvis_world,
                       anchors_t, phase_t,
                       offsets_L, offsets_R,
                       weights, bounds_deg, max_iter,
                       pose_prev=None):
    """Single-frame IK solve.

    Decision variables (packed into a flat vector x):
      [pelvis_trans (3), L_hip_aa (3), L_knee_aa (3), L_ankle_aa (3),
       R_hip_aa (3), R_knee_aa (3), R_ankle_aa (3)]
      Total 21 DoF, but if a leg is in swing, its 9 DoF are pinned to measured (zero residual).

    Cost (residuals returned to scipy.optimize.least_squares):
      anchor: sqrt(w_anchor) * (FK_anchor_position - measured_anchor)  per active anchor
      joint:  sqrt(w_joint)  * (joint_aa - measured_aa)                per leg joint axis
      pelvis: sqrt(w_pelvis) * (pelvis_trans - measured_trans)         3 components
      smooth: sqrt(w_smooth) * (joint_aa - prev_joint_aa)              per leg joint axis (if pose_prev given)

    Args:
        pose_measured: (24, 3) BONE order axis-angle.
        trans_measured: (3,) pelvis trans.
        R_pelvis_world: (3, 3).
        anchors_t: dict with H_L, T_L, H_R, T_R (each (3,) world position; NaN if inactive).
        phase_t: dict {'L': 0|1|2|3, 'R': 0|1|2|3}.
        offsets_L, offsets_R: from _extract_leg_local_offsets.
        weights: dict with 'anchor', 'joint', 'smooth', 'pelvis' float weights.
        bounds_deg: float, joint angle bounds = measured ± this (degrees).
        max_iter: scipy max iterations.
        pose_prev: (24, 3) or None — previous frame's pose for smoothness term.

    Returns:
        pose_corrected: (24, 3) — only L/R hip/knee/ankle modified, rest = measured.
        trans_corrected: (3,)
        converged: bool
    """
    from scipy.optimize import least_squares

    # BONE indices
    L_HIP, L_KNEE, L_ANKLE = 1, 4, 7
    R_HIP, R_KNEE, R_ANKLE = 2, 5, 8

    # Pack measured into x0
    x0 = np.concatenate([
        trans_measured,
        pose_measured[L_HIP], pose_measured[L_KNEE], pose_measured[L_ANKLE],
        pose_measured[R_HIP], pose_measured[R_KNEE], pose_measured[R_ANKLE],
    ])  # (21,)

    # Bounds: ±bounds_deg around measured (joint angles), no bound on trans
    b_rad = np.radians(bounds_deg)
    lb = x0.copy() - b_rad
    ub = x0.copy() + b_rad
    lb[:3] = -np.inf
    ub[:3] = +np.inf

    sw_a = np.sqrt(weights["anchor"])
    sw_j = np.sqrt(weights["joint"])
    sw_p = np.sqrt(weights["pelvis"])
    sw_s = np.sqrt(weights["smooth"]) if pose_prev is not None else 0.0

    L_in_stance = phase_t["L"] != 0
    R_in_stance = phase_t["R"] != 0

    # Pre-extract anchors (use NaN-safe; inactive anchors not included in residuals)
    H_L = anchors_t["H_L"]; T_L = anchors_t["T_L"]
    H_R = anchors_t["H_R"]; T_R = anchors_t["T_R"]

    if pose_prev is not None:
        prev_LHIP = pose_prev[L_HIP]; prev_LKNEE = pose_prev[L_KNEE]; prev_LANK = pose_prev[L_ANKLE]
        prev_RHIP = pose_prev[R_HIP]; prev_RKNEE = pose_prev[R_KNEE]; prev_RANK = pose_prev[R_ANKLE]

    def residuals(x):
        ptr = x[:3]
        l_hip, l_knee, l_ank = x[3:6], x[6:9], x[9:12]
        r_hip, r_knee, r_ank = x[12:15], x[15:18], x[18:21]
        res = []
        # FK
        if L_in_stance:
            ankle_L_w, toe_L_w = _fk_leg_world_xyz(ptr, R_pelvis_world, l_hip, l_knee, l_ank, offsets_L)
            ph = phase_t["L"]
            if ph == 1 or ph == 2:   # heel-only or full-contact: ankle anchored to H_L
                res.append(sw_a * (ankle_L_w - H_L))
            if ph == 2 or ph == 3:   # full-contact or toe-only: toe anchored to T_L
                res.append(sw_a * (toe_L_w - T_L))
        if R_in_stance:
            ankle_R_w, toe_R_w = _fk_leg_world_xyz(ptr, R_pelvis_world, r_hip, r_knee, r_ank, offsets_R)
            ph = phase_t["R"]
            if ph == 1 or ph == 2:
                res.append(sw_a * (ankle_R_w - H_R))
            if ph == 2 or ph == 3:
                res.append(sw_a * (toe_R_w - T_R))
        # Joint angle deviation regularizer
        res.append(sw_j * (l_hip - pose_measured[L_HIP]))
        res.append(sw_j * (l_knee - pose_measured[L_KNEE]))
        res.append(sw_j * (l_ank - pose_measured[L_ANKLE]))
        res.append(sw_j * (r_hip - pose_measured[R_HIP]))
        res.append(sw_j * (r_knee - pose_measured[R_KNEE]))
        res.append(sw_j * (r_ank - pose_measured[R_ANKLE]))
        # Pelvis trans regularizer
        res.append(sw_p * (ptr - trans_measured))
        # Smoothness (if pose_prev available)
        if pose_prev is not None and sw_s > 0.0:
            res.append(sw_s * (l_hip - prev_LHIP))
            res.append(sw_s * (l_knee - prev_LKNEE))
            res.append(sw_s * (l_ank - prev_LANK))
            res.append(sw_s * (r_hip - prev_RHIP))
            res.append(sw_s * (r_knee - prev_RKNEE))
            res.append(sw_s * (r_ank - prev_RANK))
        return np.concatenate(res)

    try:
        sol = least_squares(
            residuals, x0, bounds=(lb, ub), method='trf', max_nfev=max_iter,
        )
        converged = sol.status > 0
        x = sol.x
    except Exception:
        converged = False
        x = x0

    pose_corrected = pose_measured.copy()
    trans_corrected = x[:3]
    pose_corrected[L_HIP] = x[3:6]
    pose_corrected[L_KNEE] = x[6:9]
    pose_corrected[L_ANKLE] = x[9:12]
    pose_corrected[R_HIP] = x[12:15]
    pose_corrected[R_KNEE] = x[15:18]
    pose_corrected[R_ANKLE] = x[18:21]
    return pose_corrected, trans_corrected, converged
```

- [ ] **Step 4.4: Run tests, verify PASS**

Run: `cd scripts/data && conda run -n phc python -m pytest test_h5_conversion_fixes.py::test_solve_foot_ik_frame_identity_no_adjustment test_h5_conversion_fixes.py::test_solve_foot_ik_frame_lateral_anchor_shifts_pelvis -v`
Expected: PASS both.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py scripts/data/test_h5_conversion_fixes.py
git commit -m "feat(h5): solve_foot_ik_frame nonlinear LS solver"
```

---

## Task 5: Wire CLI flag + IK block into convert_trial

**Files:**
- Modify: `scripts/data/h5_to_motion_lib_v2.py`

**What this does:** Add new flags, add `read_cop_xy` helper (CoP x AND y, current `read_cop` only reads x), thread `foot_ik` + tuning params through `convert_trial()`, insert IK block before pose_quat construction.

- [ ] **Step 5.1: Read current flag block + convert_trial signature + IP block end**

Read these regions for context:
- `scripts/data/h5_to_motion_lib_v2.py:168-202` (existing read_grf, read_cop)
- `scripts/data/h5_to_motion_lib_v2.py:330-350` (convert_trial signature)
- `scripts/data/h5_to_motion_lib_v2.py:680-740` (end of IP block, where IK insertion goes)
- `scripts/data/h5_to_motion_lib_v2.py:840-895` (CLI flags)

- [ ] **Step 5.2: Add read_cop_xy helper**

In `scripts/data/h5_to_motion_lib_v2.py`, immediately after the existing `read_cop` function (around line 200), add:

```python
def read_cop_xy(f, trial_path, side):
    """Read CoP medio-lateral (x) AND anterior-posterior (y) in meters.

    Returns (cop_xy, valid) where:
      cop_xy: [T, 2] float64. Column 0 = lab x (medio-lateral), column 1 = lab y (AP).
      valid: bool — False if H5 lacks CoP data.

    NaN values are interpolated. Outside stance the CoP electronics return garbage —
    caller must mask by stance.
    """
    paths = (
        f"{trial_path}/forceplate/cop/{side}/x",
        f"{trial_path}/forceplate/cop/{side}/y",
    )
    out = np.zeros((0, 2), dtype=np.float64)
    arrs = []
    for path in paths:
        try:
            v = np.array(f[path], dtype=np.float64) / 1000.0   # mm → m
            nn = np.isnan(v)
            if nn.any() and not nn.all():
                v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
            elif nn.all():
                return out, False
            arrs.append(v)
        except KeyError:
            return out, False
    cop_xy = np.column_stack(arrs)
    return cop_xy, True
```

- [ ] **Step 5.3: Add CLI flags**

In `scripts/data/h5_to_motion_lib_v2.py`, immediately after the existing `--stance_anchor_source` flag (around line 832-835), add:

```python
    parser.add_argument("--foot_ik", type=str, default="none", choices=["none", "full"],
                        help="Full foot-anchor IK over pelvis_trans + stance leg joints. "
                             "'none' = disabled (default, backward-compat). "
                             "'full' = scipy nonlinear LS per frame, anchor = sub-phase-averaged CoP. "
                             "See 01_research_docs/260425_h5_foot_anchor_ik_design.md.")
    parser.add_argument("--foot_ik_pitch_threshold_deg", type=float, default=5.0,
                        help="Foot pitch threshold (degrees) for sub-phase classification.")
    parser.add_argument("--foot_ik_anchor_weight", type=float, default=1e3,
                        help="IK cost weight for anchor satisfaction (large = effectively hard).")
    parser.add_argument("--foot_ik_joint_reg_weight", type=float, default=0.1,
                        help="IK cost weight for joint angle deviation from measured.")
    parser.add_argument("--foot_ik_smoothness_weight", type=float, default=0.5,
                        help="IK cost weight for frame-to-frame joint angle smoothness.")
    parser.add_argument("--foot_ik_pelvis_reg_weight", type=float, default=0.01,
                        help="IK cost weight for pelvis trans deviation from measured.")
    parser.add_argument("--foot_ik_max_iter", type=int, default=50,
                        help="Max scipy least_squares iterations per frame.")
    parser.add_argument("--foot_ik_bounds_deg", type=float, default=20.0,
                        help="Joint angle bounds: measured ± this (degrees).")
```

- [ ] **Step 5.4: Update convert_trial signature**

Change the signature at line ~349:

OLD (one line):
```python
def convert_trial(f, trial_path, fps_in=100, fps_out=30, baseline_s=0.0, spine_3axis=False, upper_body=False, elbow_offset_deg=0.0, pelvis_obliq_scale=1.0, pelvis_lat_scale=1.0, stance_anchor_ip=False, stance_anchor_source="fk"):
```

NEW (one line):
```python
def convert_trial(f, trial_path, fps_in=100, fps_out=30, baseline_s=0.0, spine_3axis=False, upper_body=False, elbow_offset_deg=0.0, pelvis_obliq_scale=1.0, pelvis_lat_scale=1.0, stance_anchor_ip=False, stance_anchor_source="fk", foot_ik="none", foot_ik_kwargs=None):
```

- [ ] **Step 5.5: Update convert_trial caller**

At line ~878, change:

OLD:
```python
                    result = convert_trial(f, trial_path, fps_in, args.fps_out, baseline_s=args.baseline_s, spine_3axis=args.spine_3axis, upper_body=args.upper_body, elbow_offset_deg=args.elbow_offset_deg, pelvis_obliq_scale=args.pelvis_obliq_scale, pelvis_lat_scale=args.pelvis_lat_scale, stance_anchor_ip=args.stance_anchor_ip, stance_anchor_source=args.stance_anchor_source)
```

NEW:
```python
                    result = convert_trial(f, trial_path, fps_in, args.fps_out, baseline_s=args.baseline_s, spine_3axis=args.spine_3axis, upper_body=args.upper_body, elbow_offset_deg=args.elbow_offset_deg, pelvis_obliq_scale=args.pelvis_obliq_scale, pelvis_lat_scale=args.pelvis_lat_scale, stance_anchor_ip=args.stance_anchor_ip, stance_anchor_source=args.stance_anchor_source, foot_ik=args.foot_ik, foot_ik_kwargs={"pitch_threshold_deg": args.foot_ik_pitch_threshold_deg, "anchor_weight": args.foot_ik_anchor_weight, "joint_reg_weight": args.foot_ik_joint_reg_weight, "smoothness_weight": args.foot_ik_smoothness_weight, "pelvis_reg_weight": args.foot_ik_pelvis_reg_weight, "max_iter": args.foot_ik_max_iter, "bounds_deg": args.foot_ik_bounds_deg})
```

- [ ] **Step 5.6: Insert IK block in convert_trial**

In `convert_trial()`, find the end of the existing IP block — the last lines are around line 738 (after `print(f"  [stance_anchor_ip] done: ...")`). Insert immediately after that print (and before the `# ----- Build PHC-format motion using poselib (mirrors convert_amass_isaac.py) -----` comment at line ~742).

Insert this code:

```python
    if foot_ik == "full":
        from h5_conversion_helpers import (
            classify_sub_phases, build_foot_anchors, solve_foot_ik_frame,
            _extract_leg_local_offsets, _fk_leg_world_xyz, cop_replace_lateral,
        )
        if foot_ik_kwargs is None:
            foot_ik_kwargs = {}
        ik_pitch_thr = float(foot_ik_kwargs.get("pitch_threshold_deg", 5.0))
        ik_anchor_w = float(foot_ik_kwargs.get("anchor_weight", 1e3))
        ik_joint_w = float(foot_ik_kwargs.get("joint_reg_weight", 0.1))
        ik_smooth_w = float(foot_ik_kwargs.get("smoothness_weight", 0.5))
        ik_pelvis_w = float(foot_ik_kwargs.get("pelvis_reg_weight", 0.01))
        ik_max_iter = int(foot_ik_kwargs.get("max_iter", 50))
        ik_bounds_deg = float(foot_ik_kwargs.get("bounds_deg", 20.0))

        _T = T_out_final
        _sk = get_skeleton_tree()
        _offsets_L = _extract_leg_local_offsets(_sk, side='L')
        _offsets_R = _extract_leg_local_offsets(_sk, side='R')
        # Z-up frame for IK (consistent with FK pipeline). Convert trans + pelvis rotation.
        _upright_ik = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
        _upright_ik_inv = _upright_ik.inv()
        trans_zup_ik = _upright_ik.apply(trans)            # (T, 3) in Z-up
        # Per-frame Z-up pelvis rotation matrix (apply upright on the right of measured global rotation).
        # We have pelvis local axis-angle in pose_aa_local[:, 0] (BONE order). Convert to Z-up world R.
        _R_pel_local = sRot.from_rotvec(pose_aa_local[:, 0])               # (T,)
        _R_pel_world_zup = (_R_pel_local * _upright_ik_inv).as_matrix()    # (T, 3, 3)

        # FK pass: ankle/toe world (Z-up) per frame, using current pose
        ankle_L_zup = np.zeros((_T, 3))
        toe_L_zup = np.zeros((_T, 3))
        ankle_R_zup = np.zeros((_T, 3))
        toe_R_zup = np.zeros((_T, 3))
        for t in range(_T):
            ankle_L_zup[t], toe_L_zup[t] = _fk_leg_world_xyz(
                trans_zup_ik[t], _R_pel_world_zup[t],
                pose_aa_local[t, 1], pose_aa_local[t, 4], pose_aa_local[t, 7], _offsets_L,
            )
            ankle_R_zup[t], toe_R_zup[t] = _fk_leg_world_xyz(
                trans_zup_ik[t], _R_pel_world_zup[t],
                pose_aa_local[t, 2], pose_aa_local[t, 5], pose_aa_local[t, 8], _offsets_R,
            )

        # Read GRF + CoP (re-resample to fps_out)
        _grf_L_raw = read_grf(f, trial_path, side='left')
        _grf_R_raw = read_grf(f, trial_path, side='right')
        if _grf_L_raw is None or _grf_R_raw is None:
            print("  [foot_ik] no GRF — IK requires forceplate data, skipping IK block.")
        else:
            def _resample(arr, f_in, f_out, n_out):
                if len(arr) == n_out:
                    return arr
                t_in = np.arange(len(arr)) / f_in
                t_out = np.arange(n_out) / f_out
                return np.interp(t_out, t_in, arr)
            _grf_L = _resample(_grf_L_raw, fps_in, fps_out, _T)
            _grf_R = _resample(_grf_R_raw, fps_in, fps_out, _T)
            _cop_L_xy_raw, _cop_L_ok = read_cop_xy(f, trial_path, side='left')
            _cop_R_xy_raw, _cop_R_ok = read_cop_xy(f, trial_path, side='right')
            if not (_cop_L_ok and _cop_R_ok):
                print("  [foot_ik] no CoP — IK requires forceplate CoP data, skipping IK block.")
            else:
                _cop_L_xy = np.column_stack([
                    _resample(_cop_L_xy_raw[:, 0], fps_in, fps_out, _T),
                    _resample(_cop_L_xy_raw[:, 1], fps_in, fps_out, _T),
                ])
                _cop_R_xy = np.column_stack([
                    _resample(_cop_R_xy_raw[:, 0], fps_in, fps_out, _T),
                    _resample(_cop_R_xy_raw[:, 1], fps_in, fps_out, _T),
                ])
                # Sign-align CoP lateral with FK lateral (same logic as cop_replace_lateral)
                # Use Y-axis (index 1) of Z-up frame as lateral.
                _stance_L_mask = _grf_L > 50.0
                _stance_R_mask = _grf_R > 50.0
                if _stance_L_mask.sum() > 3 and _stance_R_mask.sum() > 3:
                    _fk_L = ankle_L_zup[_stance_L_mask, 1].mean()
                    _fk_R = ankle_R_zup[_stance_R_mask, 1].mean()
                    _cop_L_mean = _cop_L_xy[_stance_L_mask, 0].mean()
                    _cop_R_mean = _cop_R_xy[_stance_R_mask, 0].mean()
                    _sign = 1.0 if (_fk_L - _fk_R) * (_cop_L_mean - _cop_R_mean) > 0 else -1.0
                    _cop_L_xy[:, 0] = _sign * _cop_L_xy[:, 0] + (_fk_L - _sign * _cop_L_mean)
                    _cop_R_xy[:, 0] = _sign * _cop_R_xy[:, 0] + (_fk_R - _sign * _cop_R_mean)

                # Sub-phase classification
                phase_L, phase_R = classify_sub_phases(
                    _grf_L, _grf_R,
                    ankle_L_zup, toe_L_zup, ankle_R_zup, toe_R_zup,
                    pitch_threshold_deg=ik_pitch_thr,
                )

                # Anchors
                anchors = build_foot_anchors(
                    _cop_L_xy, _cop_R_xy,
                    ankle_L_zup, toe_L_zup, ankle_R_zup, toe_R_zup,
                    phase_L, phase_R, fps=fps_out, min_episode_frames=10,
                )

                # Per-frame IK loop
                weights = {
                    "anchor": ik_anchor_w, "joint": ik_joint_w,
                    "smooth": ik_smooth_w, "pelvis": ik_pelvis_w,
                }
                trans_zup_corrected = trans_zup_ik.copy()
                pose_aa_corrected = pose_aa_local.copy()
                _converged_count = 0
                _ik_frame_count = 0
                _pose_prev = None
                for t in range(_T):
                    if not (anchors["ik_active_L"][t] or anchors["ik_active_R"][t]):
                        _pose_prev = pose_aa_corrected[t]
                        continue
                    anchors_t = {
                        "H_L": anchors["H_L"][t], "T_L": anchors["T_L"][t],
                        "H_R": anchors["H_R"][t], "T_R": anchors["T_R"][t],
                    }
                    phase_t = {"L": int(phase_L[t]), "R": int(phase_R[t])}
                    pose_corr, trans_corr, converged = solve_foot_ik_frame(
                        pose_aa_corrected[t], trans_zup_corrected[t], _R_pel_world_zup[t],
                        anchors_t, phase_t,
                        _offsets_L, _offsets_R,
                        weights=weights, bounds_deg=ik_bounds_deg, max_iter=ik_max_iter,
                        pose_prev=_pose_prev,
                    )
                    pose_aa_corrected[t] = pose_corr
                    trans_zup_corrected[t] = trans_corr
                    _ik_frame_count += 1
                    if converged:
                        _converged_count += 1
                    _pose_prev = pose_corr

                # Write back: convert trans_zup_corrected back to Y-up
                trans = _upright_ik_inv.apply(trans_zup_corrected)
                pose_aa_local = pose_aa_corrected
                _conv_pct = (100.0 * _converged_count / max(_ik_frame_count, 1))
                print(f"  [foot_ik] done: ik_active_frames={_ik_frame_count}/{_T}  "
                      f"converged={_converged_count} ({_conv_pct:.1f}%)")
```

- [ ] **Step 5.7: Quick smoke test (no expected stance fix yet, just verify it runs)**

Run a tiny conversion to ensure no crashes:

```bash
conda run -n phc python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --output sample_data/h5_walk_S001_lv0_trial01_v2_footik_smoke.pkl \
  --subjects S001 --tasks level_100mps --assist_level lv0 \
  --baseline_s 2.0 --spine_3axis --upper_body \
  --foot_ik full \
  --no_split --trim_start 2.0
```

Expected: completes without error, prints `[foot_ik] done: ik_active_frames=...  converged=... (XX.X%)`. Convergence ≥ 80% acceptable.

- [ ] **Step 5.8: Commit**

```bash
git add scripts/data/h5_to_motion_lib_v2.py
git commit -m "feat(h5): wire --foot_ik full into convert_trial"
```

---

## Task 6: Validate stance ankle std + render comparison videos

**Files:**
- (no source code change; validation only)

**What this does:** Run the conversion, measure stance ankle std (acceptance criterion), render side-by-side comparison videos vs v2_cop and AMASS, hand off to user for visual review.

- [ ] **Step 6.1: Full conversion**

```bash
conda run -n phc python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --output sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl \
  --subjects S001 --tasks level_100mps --assist_level lv0 \
  --baseline_s 2.0 --spine_3axis --upper_body \
  --foot_ik full \
  --no_split --trim_start 2.0 2>&1 | tail -10
```

Expected: completes; reports IK convergence ≥ 80%.

- [ ] **Step 6.2: Measure stance ankle std**

Write `/tmp/measure_ankle_std.py`:

```python
import joblib
import numpy as np
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
import os, sys
sys.path.append(os.getcwd())
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

def build_sk_tree():
    cfg = {"mesh": False, "rel_joint_lm": False, "upright_start": True,
           "remove_toe": False, "real_weight_porpotion_capsules": True,
           "real_weight_porpotion_boxes": True, "model": "smpl", "big_ankle": True,
           "freeze_hand": False, "box_body": True, "body_params": {}, "joint_params": {},
           "geom_params": {}, "actuator_params": {}}
    smpl = SMPL_Robot(cfg, data_dir="data/smpl")
    smpl.load_from_skeleton(betas=torch.zeros(1, 16), gender=torch.zeros(1), objs_info=None)
    xml = "/tmp/smpl/measure_ankle_std.xml"
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    smpl.write_xml(xml)
    return SkeletonTree.from_mjcf(xml)


def stance_ankle_std(pkl_path):
    """Approximate stance ankle position std using FK on output pose+trans."""
    d = joblib.load(pkl_path)
    k = list(d.keys())[0]
    clip = d[k]
    sk_tree = build_sk_tree()
    from poselib.poselib.skeleton.skeleton3d import SkeletonState
    rt = clip["root_trans_offset"]
    if hasattr(rt, "numpy"):
        rt = rt.numpy()
    rt = torch.from_numpy(rt).float()
    pq = torch.from_numpy(clip["pose_quat"]).float()      # local
    state = SkeletonState.from_rotation_and_root_translation(sk_tree, r=pq, t=rt, is_local=True)
    gt = state.global_translation.numpy()
    names = list(sk_tree.node_names)
    L_ANK = names.index("L_Ankle"); R_ANK = names.index("R_Ankle")
    # Approximate stance via Z position threshold (proxy for GRF since pkl has no GRF)
    L_z = gt[:, L_ANK, 2]
    R_z = gt[:, R_ANK, 2]
    L_stance = L_z < (L_z.min() + 0.03)   # 3cm above min = "near ground"
    R_stance = R_z < (R_z.min() + 0.03)
    L_xy = gt[L_stance][:, L_ANK, :2]
    R_xy = gt[R_stance][:, R_ANK, :2]
    return {
        "L_stance_n": int(L_stance.sum()),
        "L_xy_std": (L_xy.std(axis=0) * 100).tolist(),     # cm
        "R_stance_n": int(R_stance.sum()),
        "R_xy_std": (R_xy.std(axis=0) * 100).tolist(),
    }

for label, p in [
    ("v2_full",   "sample_data/h5_walk_S001_lv0_trial01_v2_full.pkl"),
    ("v2_cop",    "sample_data/h5_walk_S001_lv0_trial01_v2_cop.pkl"),
    ("v2_footik", "sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl"),
]:
    s = stance_ankle_std(p)
    print(f"{label:12s}  L stance n={s['L_stance_n']:>5d}  L xy std [cm]={s['L_xy_std']}    R xy std [cm]={s['R_xy_std']}")
```

Run: `conda run -n phc python /tmp/measure_ankle_std.py`
Expected: v2_footik L/R xy std should be smaller than v2_cop / v2_full (acceptance: < 1.0 cm during near-ground frames).

- [ ] **Step 6.3: Render comparison videos**

```bash
mkdir -p output/h5_footik_check

conda run -n phc python scripts/vis/compare_pkl_metrics.py \
  --pkl_a sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl --label_a "v2_footik (full IK)" \
  --pkl_b sample_data/h5_walk_S001_lv0_trial01_v2_cop.pkl --label_b "v2_cop (CoP only)" \
  --out_prefix output/h5_footik_check/footik_vs_cop \
  --t_start_a 5 --t_end_a 20 --t_start_b 5 --t_end_b 20 \
  --view all 2>&1 | tail -5

conda run -n phc python scripts/vis/compare_pkl_metrics.py \
  --pkl_a sample_data/h5_walk_S001_lv0_trial01_v2_footik.pkl --label_a "v2_footik (full IK)" \
  --pkl_b sample_data/amass_isaac_walking_primitive.pkl --label_b "AMASS walking" \
  --out_prefix output/h5_footik_check/footik_vs_amass \
  --t_start_a 5 --t_end_a 20 \
  --view all 2>&1 | tail -5
```

Expected: 6 mp4 files written to `output/h5_footik_check/`.

- [ ] **Step 6.4: Hand off to user**

Report to user:
- Acceptance check result (stance ankle std numbers)
- IK convergence rate (from Step 5.7/6.1 logs)
- Six video files for visual review (frontal is the key one)

**STOP here per `feedback_visualization_workflow.md`** — do not write analysis docs or propose further code changes until user reviews videos and gives direction.

- [ ] **Step 6.5: Commit any remaining changes (code only, no commit if no source changed)**

If there were any final tweaks to source files needed for measurement script reuse, commit them. Otherwise skip.

---

## Self-Review Notes

### Spec coverage
- Section 4 (Architecture): Tasks 5.6 (IK block insertion in correct location)
- Section 5a (Sub-phase classifier): Task 1
- Section 5b (Anchor builder): Task 2
- Section 5c (Lab→world frame alignment): Task 5.6 sign-align block (CoP_L_xy[:, 0] sign + offset)
- Section 5d (IK solver): Task 4
- Section 5e (Output writeback): Task 5.6 final lines (`trans = _upright_ik_inv.apply(...)`, `pose_aa_local = pose_aa_corrected`)
- Section 6 (CLI): Task 5.3
- Section 7 (Files affected): all tasks
- Section 8 (Testing): Tasks 1-4 (unit) + Task 6 (integration + visual)
- Section 9 (Risks): convergence failure → graceful (Task 4 try/except), measurement script (Task 6.2) catches `<1cm` acceptance

### Type consistency
- `phase_L`/`phase_R` are `int8`, used as int comparisons in `phase_t["L"] != 0` etc. — consistent.
- `anchors_t["H_L"]` is `(3,)` np.float64 (NaN if inactive); checked via `ik_active_L` mask before passing — consistent.
- `_extract_leg_local_offsets` returns dict with keys `"hip", "knee", "ankle", "toe"` — used identically in `_fk_leg_world_xyz` and `solve_foot_ik_frame` — consistent.

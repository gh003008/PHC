# H5 Stance-Anchor Inverted-Pendulum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate excess lateral pelvis motion in H5-derived walking pkls by anchoring pelvis world position to stance feet (inverted pendulum), reducing lateral std from 4.89cm to AMASS-range (~3.5cm).

**Architecture:** Add `--stance_anchor_ip` flag to `scripts/data/h5_to_motion_lib_v2.py`. When on, after computing pose_aa_local (unchanged) and bootstrap CoM-based trans (unchanged), run a three-helper pipeline: (1) `detect_stance` via GRF+velocity hysteresis, (2) `compute_foot_anchor` via rolling mean per stance episode, (3) `solve_pelvis_ip` via GRF-weighted blend. Replace lateral (X) and vertical (Y) in `trans`, keep forward (Z) from treadmill integration.

**Tech Stack:** Python 3.8, NumPy, h5py, PyTorch, scipy.spatial.transform.Rotation, poselib.SkeletonState for FK, pytest. All present in `phc` conda env.

**Design spec:** `01_research_docs/260424_h5_stance_anchor_ip_design.md` (read for background/rationale).

**Conda activation:** All commands assume `conda activate phc`.

---

## File Structure

**Modify:**
- `scripts/data/h5_conversion_helpers.py` — Add `detect_stance`, `compute_foot_anchor`, `solve_pelvis_ip`. Pure functions, no H5 I/O.
- `scripts/data/h5_to_motion_lib_v2.py` — Add local helpers `fk_feet`, `read_grf`; add `--stance_anchor_ip` flag; add conditional IP block in `convert_trial`.
- `scripts/data/test_h5_conversion_fixes.py` — Add 3 unit tests for new helpers.

**Create:** None (extend existing files).

**Output artifacts:**
- `sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl` (after Task 6)
- `output/h5_visual_check/v2_ip_vs_amass_*.mp4/png` (after Task 7)

---

### Task 1: Stub helpers and failing tests (TDD scaffold)

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py`
- Modify: `scripts/data/test_h5_conversion_fixes.py`

Goal: add `raise NotImplementedError` stubs for the three new helpers plus three failing tests. Locks in signatures and test contracts before implementation.

- [ ] **Step 1: Append stubs to `scripts/data/h5_conversion_helpers.py`**

Append to the end of the file (after `subtract_baseline`):

```python
def detect_stance(grf_L, grf_R, vel_L, vel_R,
                  grf_on=50.0, grf_off=20.0,
                  vel_on=0.3, vel_off=0.5):
    """Detect stance phase per foot via GRF OR velocity with Schmitt-trigger hysteresis.

    Args:
        grf_L, grf_R: 1-D arrays [T] of vertical GRF magnitude in Newtons.
        vel_L, vel_R: 1-D arrays [T] of foot world-frame speed in m/s.
        grf_on:  threshold (N) to enter GRF-stance.
        grf_off: threshold (N) to exit GRF-stance (< grf_on for hysteresis).
        vel_on:  threshold (m/s) to enter velocity-stance (|vel| < vel_on).
        vel_off: threshold (m/s) to exit velocity-stance (> vel_off for hysteresis).

    Returns:
        stance_L, stance_R: 1-D bool arrays [T]. True when foot is in stance.
    """
    raise NotImplementedError


def compute_foot_anchor(foot_world, stance_mask, window=9):
    """Rolling-mean foot anchor position per contiguous stance episode.

    During stance: centered rolling mean of foot_world over `window` frames (edges truncated).
    During swing: linearly interpolate between last-known and next-known anchor
    (used only for continuity — solver masks swing frames via weights).

    Args:
        foot_world: [T, 3] world position of the foot (ankle joint recommended).
        stance_mask: [T] bool.
        window: int, rolling-mean window size (odd, centered).

    Returns:
        anchor_world: [T, 3] smoothed anchor.
    """
    raise NotImplementedError


def solve_pelvis_ip(anchor_L, anchor_R, grf_L, grf_R,
                    R_pelvis, foot_offset_L, foot_offset_R, eps=1e-3):
    """Solve pelvis world position from stance-foot anchor constraint (IP).

    Per frame t:
        w_L = grf_L / (grf_L + grf_R + eps)
        w_R = 1 - w_L
        target_anchor = w_L*anchor_L + w_R*anchor_R
        target_offset = w_L*foot_offset_L + w_R*foot_offset_R
        pelvis[t] = target_anchor - R_pelvis @ target_offset

    Args:
        anchor_L, anchor_R: [T, 3] world anchors.
        grf_L, grf_R: [T] GRF magnitudes (weights).
        R_pelvis: [T, 3, 3] pelvis world rotation matrix.
        foot_offset_L, foot_offset_R: [T, 3] ankle offset in pelvis-local frame.
        eps: small denominator floor.

    Returns:
        pelvis_world: [T, 3]. Frames where grf_L + grf_R < eps (flight) are NaN —
        caller is responsible for fallback.
    """
    raise NotImplementedError
```

- [ ] **Step 2: Append 3 failing tests to `scripts/data/test_h5_conversion_fixes.py`**

Append:

```python
from data.h5_conversion_helpers import detect_stance, compute_foot_anchor, solve_pelvis_ip


def test_detect_stance_hysteresis_prevents_chatter():
    """GRF grazing the 50N threshold must not toggle stance on every crossing."""
    # GRF trace that crosses 50N three times, then goes to 0
    grf = np.array([60, 52, 48, 55, 51, 45, 30, 15, 5, 0, 0], dtype=float)
    zeros = np.zeros_like(grf)
    vel_swing = np.full_like(grf, 5.0)  # high velocity so velocity-stance is always False
    stance_L, _ = detect_stance(grf, zeros, vel_swing, vel_swing)
    # With hysteresis: enters stance at idx 0 (60>50), only exits when grf<20 at idx 7
    # Without hysteresis: would chatter at idx 2, 3 (48→55 crossings)
    assert stance_L[0] == True
    assert stance_L[3] == True   # 55N, hysteresis holds
    assert stance_L[5] == True   # 45N, still above grf_off=20
    assert stance_L[7] == False  # 15N < 20, exits
    assert stance_L[8] == False


def test_compute_foot_anchor_rolling_smooths_noise():
    """Constant foot position + gaussian noise → anchor std << input std."""
    np.random.seed(0)
    T = 100
    foot_world = np.zeros((T, 3))
    foot_world[:, 0] = np.random.normal(0, 0.005, T)  # ±5 mm noise on x
    stance = np.ones(T, dtype=bool)
    anchor = compute_foot_anchor(foot_world, stance, window=9)
    # Input std ~5mm, output std should be < 2mm (rolling smoothing ~ sqrt(window) improvement)
    assert foot_world[:, 0].std() > 0.003
    assert anchor[:, 0].std() < 0.002


def test_solve_pelvis_ip_places_pelvis_above_anchor():
    """Trivial geometry: identity pelvis rot, foot 1m below → pelvis at anchor+(0,1,0)."""
    T = 5
    # anchor at origin, for both feet
    anchor = np.zeros((T, 3))
    grf = np.ones(T)
    # Identity rotation for all frames
    R = np.tile(np.eye(3), (T, 1, 1))
    # Foot 1m below pelvis in pelvis frame (along -Y)
    foot_offset = np.tile(np.array([0.0, -1.0, 0.0]), (T, 1))
    pelvis = solve_pelvis_ip(anchor, anchor, grf, grf, R, foot_offset, foot_offset)
    # pelvis = anchor - R @ offset = 0 - (-Y) = +Y direction with magnitude 1
    np.testing.assert_allclose(pelvis, np.tile([0.0, 1.0, 0.0], (T, 1)), atol=1e-9)
```

- [ ] **Step 3: Run tests, confirm 3 new failures**

Run:
```bash
cd /home/exolab/Documents/GitHub/PHC
pytest scripts/data/test_h5_conversion_fixes.py -v
```
Expected: 3 existing tests (from v2) still pass; 3 new tests FAIL with `NotImplementedError`.

- [ ] **Step 4: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py scripts/data/test_h5_conversion_fixes.py
git commit -m "$(cat <<'EOF'
feat(h5): stub stance-IP helpers with failing tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Implement `detect_stance`

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py`

Goal: Schmitt-trigger hysteresis on GRF and velocity, final stance = GRF-stance OR velocity-stance.

- [ ] **Step 1: Replace stub body for `detect_stance`**

In `scripts/data/h5_conversion_helpers.py`, replace the `raise NotImplementedError` in `detect_stance` with:

```python
def _schmitt(signal, on_threshold, off_threshold, on_when_above=True):
    """Generic Schmitt trigger. Returns bool array same shape as signal."""
    T = len(signal)
    out = np.zeros(T, dtype=bool)
    state = False
    for t in range(T):
        v = signal[t]
        if on_when_above:
            if not state and v > on_threshold:
                state = True
            elif state and v < off_threshold:
                state = False
        else:
            if not state and v < on_threshold:
                state = True
            elif state and v > off_threshold:
                state = False
        out[t] = state
    return out


def detect_stance(grf_L, grf_R, vel_L, vel_R,
                  grf_on=50.0, grf_off=20.0,
                  vel_on=0.3, vel_off=0.5):
    grf_stance_L = _schmitt(grf_L, grf_on, grf_off, on_when_above=True)
    grf_stance_R = _schmitt(grf_R, grf_on, grf_off, on_when_above=True)
    vel_stance_L = _schmitt(vel_L, vel_on, vel_off, on_when_above=False)
    vel_stance_R = _schmitt(vel_R, vel_on, vel_off, on_when_above=False)
    stance_L = grf_stance_L | vel_stance_L
    stance_R = grf_stance_R | vel_stance_R
    return stance_L, stance_R
```

- [ ] **Step 2: Run test, verify pass**

```bash
cd /home/exolab/Documents/GitHub/PHC
pytest scripts/data/test_h5_conversion_fixes.py::test_detect_stance_hysteresis_prevents_chatter -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py
git commit -m "$(cat <<'EOF'
feat(h5): implement detect_stance with Schmitt hysteresis

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement `compute_foot_anchor`

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py`

Goal: rolling mean per contiguous stance episode, linear interp in swing frames.

- [ ] **Step 1: Replace stub body for `compute_foot_anchor`**

In `scripts/data/h5_conversion_helpers.py`, replace the body with:

```python
def compute_foot_anchor(foot_world, stance_mask, window=9):
    T = foot_world.shape[0]
    anchor = np.zeros_like(foot_world)
    if T == 0:
        return anchor
    half = window // 2

    # Find contiguous stance episodes: list of (start, end_inclusive) index pairs
    episodes = []
    in_ep = False
    s = -1
    for t in range(T):
        if stance_mask[t] and not in_ep:
            s = t
            in_ep = True
        elif not stance_mask[t] and in_ep:
            episodes.append((s, t - 1))
            in_ep = False
    if in_ep:
        episodes.append((s, T - 1))

    # Apply rolling mean within each episode (edges truncated — use available samples)
    for (a, b) in episodes:
        for t in range(a, b + 1):
            lo = max(a, t - half)
            hi = min(b, t + half) + 1   # exclusive upper
            anchor[t] = foot_world[lo:hi].mean(axis=0)

    # Fill swing frames by linear interpolation between adjacent episodes
    # (continuity; solver masks swing via GRF weights, but this helps for callers
    # that inspect anchor directly).
    if episodes:
        # Before first stance: hold first episode's initial anchor
        first_a, _ = episodes[0]
        if first_a > 0:
            anchor[:first_a] = anchor[first_a]
        # Between episodes: linear interp
        for i in range(len(episodes) - 1):
            _, b_i = episodes[i]
            a_next, _ = episodes[i + 1]
            if a_next > b_i + 1:
                a0 = anchor[b_i]
                a1 = anchor[a_next]
                for t in range(b_i + 1, a_next):
                    alpha = (t - b_i) / (a_next - b_i)
                    anchor[t] = (1 - alpha) * a0 + alpha * a1
        # After last stance: hold last episode's final anchor
        _, last_b = episodes[-1]
        if last_b < T - 1:
            anchor[last_b + 1:] = anchor[last_b]
    else:
        # No stance at all: return zeros (caller must handle)
        pass
    return anchor
```

- [ ] **Step 2: Run test, verify pass**

```bash
cd /home/exolab/Documents/GitHub/PHC
pytest scripts/data/test_h5_conversion_fixes.py::test_compute_foot_anchor_rolling_smooths_noise -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py
git commit -m "$(cat <<'EOF'
feat(h5): implement compute_foot_anchor with rolling mean per episode

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement `solve_pelvis_ip`

**Files:**
- Modify: `scripts/data/h5_conversion_helpers.py`

Goal: GRF-weighted blend of L/R anchors, subtract rotated foot offset.

- [ ] **Step 1: Replace stub body for `solve_pelvis_ip`**

In `scripts/data/h5_conversion_helpers.py`, replace the body with:

```python
def solve_pelvis_ip(anchor_L, anchor_R, grf_L, grf_R,
                    R_pelvis, foot_offset_L, foot_offset_R, eps=1e-3):
    T = anchor_L.shape[0]
    pelvis = np.zeros((T, 3))
    total = grf_L + grf_R
    # Where total is below eps, mark NaN — caller handles flight-phase fallback
    valid = total > eps
    # Avoid div-by-zero (NaN frames get overwritten next)
    w_L = np.where(valid, grf_L / np.maximum(total, eps), 0.5)
    w_R = 1.0 - w_L
    target_anchor = w_L[:, None] * anchor_L + w_R[:, None] * anchor_R
    target_offset = w_L[:, None] * foot_offset_L + w_R[:, None] * foot_offset_R
    # Per-frame: pelvis = target_anchor - R @ target_offset
    # Use einsum for batch matmul: R_pelvis [T,3,3] @ target_offset [T,3] → [T,3]
    rotated_offset = np.einsum('tij,tj->ti', R_pelvis, target_offset)
    pelvis = target_anchor - rotated_offset
    pelvis[~valid] = np.nan
    return pelvis
```

- [ ] **Step 2: Run test, verify pass**

```bash
cd /home/exolab/Documents/GitHub/PHC
pytest scripts/data/test_h5_conversion_fixes.py::test_solve_pelvis_ip_places_pelvis_above_anchor -v
```
Expected: PASS.

- [ ] **Step 3: Run full test suite (all 6 tests should pass now)**

```bash
cd /home/exolab/Documents/GitHub/PHC
pytest scripts/data/test_h5_conversion_fixes.py -v
```
Expected: 6 passed (3 original + 3 new).

- [ ] **Step 4: Commit**

```bash
git add scripts/data/h5_conversion_helpers.py
git commit -m "$(cat <<'EOF'
feat(h5): implement solve_pelvis_ip with GRF-weighted blend

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add `--stance_anchor_ip` flag + FK/GRF local helpers (no-op wiring)

**Files:**
- Modify: `scripts/data/h5_to_motion_lib_v2.py`

Goal: add the CLI flag, thread through `convert_trial`, write `fk_feet()` and `read_grf()` local helpers, but do NOT yet run the IP block. Verify that flag OFF produces output identical to current v2_full (no-op invariant).

- [ ] **Step 1: Add `--stance_anchor_ip` CLI flag**

Locate the argparse block (after `--pelvis_lat_scale`). Append:

```python
    parser.add_argument("--stance_anchor_ip", action="store_true",
                        help="Anchor pelvis world position to stance feet via IP "
                             "(replaces lateral+vertical in trans; keeps treadmill forward). "
                             "See design doc 01_research_docs/260424_h5_stance_anchor_ip_design.md.")
```

- [ ] **Step 2: Thread `stance_anchor_ip` through `convert_trial`**

Find `def convert_trial(...)` signature — add `stance_anchor_ip=False` kwarg at the end:

```python
def convert_trial(f, trial_path, fps_in=100, fps_out=30, baseline_s=0.0, spine_3axis=False, upper_body=False, elbow_offset_deg=0.0, pelvis_obliq_scale=1.0, pelvis_lat_scale=1.0, stance_anchor_ip=False):
```

Find the `convert_trial(...)` call in `main()` — add kwarg:

```python
                    result = convert_trial(f, trial_path, fps_in, args.fps_out, baseline_s=args.baseline_s, spine_3axis=args.spine_3axis, upper_body=args.upper_body, elbow_offset_deg=args.elbow_offset_deg, pelvis_obliq_scale=args.pelvis_obliq_scale, pelvis_lat_scale=args.pelvis_lat_scale, stance_anchor_ip=args.stance_anchor_ip)
```

- [ ] **Step 3: Add local helper `read_grf` at module level (above `compute_root_translation`)**

Inspect the H5 GRF path first:

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python -c "
import h5py
with h5py.File('data/combined_data_from_csv.h5', 'r') as f:
    k = 'S001/level_100mps/lv0/trial_01/treadmill'
    if k in f:
        def walk(g, prefix=''):
            for name, item in g.items():
                p = f'{prefix}/{name}'
                if hasattr(item, 'items'): walk(item, p)
                else: print(p, item.shape, item.dtype)
        walk(f[k], k)
" 2>&1 | head -20
```

Expected output will reveal actual paths (likely `treadmill/left/grf_y` or similar — use what's printed).

Then add in `scripts/data/h5_to_motion_lib_v2.py`, above `compute_root_translation`:

```python
def read_grf(f, trial_path, side):
    """Read vertical GRF for one foot. Returns [T] Newtons, NaN-interpolated, at input fps.

    Returns None if the H5 lacks GRF data (legacy trials).
    side: 'left' or 'right'.
    """
    candidates = [
        f"{trial_path}/treadmill/{side}/grf_y",
        f"{trial_path}/treadmill/{side}/grf_vertical",
        f"{trial_path}/forceplate/{side}/grf_y",
    ]
    for path in candidates:
        try:
            v = np.array(f[path], dtype=np.float64)
            nn = np.isnan(v)
            if nn.any() and not nn.all():
                v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
            elif nn.all():
                return np.zeros(len(v), dtype=np.float64)
            return np.abs(v)   # magnitude, sign-agnostic
        except KeyError:
            continue
    return None
```

If Step 3's introspection shows a different path, update the `candidates` list accordingly before proceeding.

- [ ] **Step 4: Add local helper `fk_feet` at module level (above `compute_root_translation`)**

Add after `read_grf`:

```python
def fk_feet(skeleton_tree, pose_quat_global_xyzw, trans):
    """Forward kinematics: compute ankle world positions and pelvis-local offsets.

    Uses poselib's SkeletonState (already a project dependency).

    Args:
        skeleton_tree: poselib SkeletonTree.
        pose_quat_global_xyzw: [T, 24, 4] global-frame quaternions (xyzw).
        trans: [T, 3] pelvis world position.

    Returns:
        foot_L_world: [T, 3]  L_Ankle world pos (SMPL index 7 in MUJOCO order)
        foot_R_world: [T, 3]  R_Ankle world pos (SMPL index 8)
        foot_offset_L: [T, 3] L_Ankle pos in pelvis-local frame
        foot_offset_R: [T, 3] R_Ankle pos in pelvis-local frame
        R_pelvis: [T, 3, 3]   pelvis world rotation matrix
    """
    import torch
    from poselib.poselib.skeleton.skeleton3d import SkeletonState
    T = trans.shape[0]
    r = torch.from_numpy(pose_quat_global_xyzw).float()
    t = torch.from_numpy(trans).float()
    state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, r=r, t=t, is_local=False
    )
    # state.global_translation: [T, N_joints, 3]
    gt = state.global_translation.numpy()
    # Indices in MUJOCO/PHC-augmented SMPL order (used by this pipeline):
    # 0 = Pelvis, 7 = L_Ankle, 8 = R_Ankle (verified via skeleton_tree.node_names)
    # (If mapping differs for this build, look up by name.)
    names = list(skeleton_tree.node_names)
    IDX_PELVIS = names.index("Pelvis") if "Pelvis" in names else 0
    IDX_L_ANK = names.index("L_Ankle") if "L_Ankle" in names else 7
    IDX_R_ANK = names.index("R_Ankle") if "R_Ankle" in names else 8
    foot_L_world = gt[:, IDX_L_ANK, :]
    foot_R_world = gt[:, IDX_R_ANK, :]
    pelvis_world = gt[:, IDX_PELVIS, :]
    # Pelvis rotation matrix from pose_quat_global[:, 0] (xyzw)
    q = pose_quat_global_xyzw[:, IDX_PELVIS, :]
    R_pelvis = sRot.from_quat(q).as_matrix()  # [T, 3, 3]
    # foot offset in pelvis-local frame = R_pelvis^T @ (foot_world - pelvis_world)
    diff_L = foot_L_world - pelvis_world
    diff_R = foot_R_world - pelvis_world
    foot_offset_L = np.einsum('tij,tj->ti', R_pelvis.transpose(0, 2, 1), diff_L)
    foot_offset_R = np.einsum('tij,tj->ti', R_pelvis.transpose(0, 2, 1), diff_R)
    return foot_L_world, foot_R_world, foot_offset_L, foot_offset_R, R_pelvis
```

- [ ] **Step 5: Verify no-op invariant (flag OFF → same as before)**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --output /tmp/v2_noop_ip_off.pkl \
  --subjects S001 --tasks level_100mps --assist_level lv0 \
  --fps_out 30 --trim_start 10 --no_split
conda run -n phc python -c "
import joblib, numpy as np, torch
a = joblib.load('sample_data/h5_walk_S001_lv0_trial01_armfix.pkl')
b = joblib.load('/tmp/v2_noop_ip_off.pkl')
ka, kb = list(a)[0], list(b)[0]
for k in a[ka]:
    va, vb = a[ka][k], b[kb][k]
    if isinstance(va, torch.Tensor): va = va.numpy()
    if isinstance(vb, torch.Tensor): vb = vb.numpy()
    if hasattr(va, 'shape'):
        assert np.allclose(np.asarray(va), np.asarray(vb)), f'{k} differs'
print('no-op invariant holds')
"
```
Expected: prints `no-op invariant holds`.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/h5_to_motion_lib_v2.py
git commit -m "$(cat <<'EOF'
feat(h5): add --stance_anchor_ip flag + fk_feet/read_grf helpers (no-op)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Integrate IP block end-to-end + verify acceptance metrics

**Files:**
- Modify: `scripts/data/h5_to_motion_lib_v2.py`

Goal: activate the three helpers inside `convert_trial` when `stance_anchor_ip=True`. Regenerate the pkl with all flags + IP. Check acceptance metrics.

- [ ] **Step 1: Locate the insertion point**

In `scripts/data/h5_to_motion_lib_v2.py`, find the line:
```python
    # Compute root translation
    trans = compute_root_translation(f, trial_path, fps_in, fps_out, pelvis_lat_scale=pelvis_lat_scale)
```

Right AFTER that line, insert the IP block (Step 2).

- [ ] **Step 2: Insert IP block**

The block needs access to `pose_quat_global_xyzw` and `skeleton_tree`. These are computed later in the original pipeline, so we need to compute them once here (this is the bootstrap). Look further down in `convert_trial` for:
```python
    pose_quat_bone_xyzw = sRot.from_rotvec(pose_aa_local.reshape(-1, 3)).as_quat().reshape(T_out_final, 24, 4)
```
and:
```python
    sk_tree = SkeletonTree.from_mjcf(mjcf_path)
```

**We cannot simply use these later-computed variables. Instead, bootstrap them locally**:

Insert after the `trans = compute_root_translation(...)` line:

```python
    if stance_anchor_ip:
        from h5_conversion_helpers import detect_stance, compute_foot_anchor, solve_pelvis_ip
        # Bootstrap: compute pose_quat_global from pose_aa_local + build skeleton tree
        # Resample pose_aa_local to output fps if needed (mirrors downstream code)
        if fps_in != fps_out:
            T_out = int(T * fps_out / fps_in)
            t_in_a = np.arange(T) / fps_in
            t_out_a = np.arange(T_out) / fps_out
            aa_re = np.zeros((T_out, 24, 3), dtype=np.float64)
            for j in range(24):
                for c in range(3):
                    aa_re[:, j, c] = np.interp(t_out_a, t_in_a, pose_aa_local[:, j, c])
            pose_aa_ip = aa_re
        else:
            pose_aa_ip = pose_aa_local
        T_ip = pose_aa_ip.shape[0]

        # Build global quaternions via poselib for FK
        import torch
        from poselib.poselib.skeleton.skeleton3d import SkeletonTree as _SKT
        from poselib.poselib.skeleton.skeleton3d import SkeletonState as _SKS
        sk_tree_ip = _SKT.from_mjcf(
            "/tmp/smpl/humanoid_smpl_neutral_no_limit.xml"
            if os.path.exists("/tmp/smpl/humanoid_smpl_neutral_no_limit.xml")
            else "data/mjcf/smpl_humanoid_1.xml"
        )
        # Local quats → global quats using skeleton tree
        pose_quat_bone = sRot.from_rotvec(pose_aa_ip.reshape(-1, 3)).as_quat().reshape(T_ip, 24, 4)
        # BONE → MUJOCO order reorder (if needed by this skeleton build; mirrors later code)
        # For simplicity, call SkeletonState.from_rotation_and_root_translation with is_local=True:
        r_local_t = torch.from_numpy(pose_quat_bone).float()
        t_t = torch.from_numpy(trans).float()
        state_loc = _SKS.from_rotation_and_root_translation(
            sk_tree_ip, r=r_local_t, t=t_t, is_local=True
        )
        # Convert to global frame quaternions for fk_feet
        pose_quat_global_xyzw = state_loc.global_rotation.numpy()   # [T, N_j, 4]
        # Run FK helper
        foot_L_w, foot_R_w, foff_L, foff_R, R_pel = fk_feet(sk_tree_ip, pose_quat_global_xyzw, trans)
        # Foot velocity (world-frame speed in m/s)
        vel_L = np.linalg.norm(np.gradient(foot_L_w, axis=0), axis=1) * fps_out
        vel_R = np.linalg.norm(np.gradient(foot_R_w, axis=0), axis=1) * fps_out
        # GRF (resample to fps_out if input fps_in)
        grf_L_raw = read_grf(f, trial_path, side='left')
        grf_R_raw = read_grf(f, trial_path, side='right')
        if grf_L_raw is None or grf_R_raw is None:
            print("  [stance_anchor_ip] WARNING: no GRF data found — falling back to velocity-only stance")
            grf_L = np.zeros(T_ip)
            grf_R = np.zeros(T_ip)
        else:
            # Resample GRF to output fps
            t_in_g = np.arange(len(grf_L_raw)) / fps_in
            t_out_g = np.arange(T_ip) / fps_out
            grf_L = np.interp(t_out_g, t_in_g, grf_L_raw)
            grf_R = np.interp(t_out_g, t_in_g, grf_R_raw)
        # Stance detection
        stance_L, stance_R = detect_stance(grf_L, grf_R, vel_L, vel_R)
        # Foot anchor (rolling mean per stance episode)
        anchor_L = compute_foot_anchor(foot_L_w, stance_L, window=9)
        anchor_R = compute_foot_anchor(foot_R_w, stance_R, window=9)
        # IP solver
        pelvis_ip = solve_pelvis_ip(anchor_L, anchor_R, grf_L, grf_R,
                                     R_pel, foff_L, foff_R)
        # Flight-phase fallback: NaN → use previous valid pelvis position
        nan_mask = np.isnan(pelvis_ip).any(axis=1)
        if nan_mask[0]:
            pelvis_ip[0] = trans[0]   # bootstrap: use CoM
        for t in range(1, T_ip):
            if nan_mask[t]:
                pelvis_ip[t] = pelvis_ip[t - 1] + (trans[t] - trans[t - 1])   # carry forward with CoM delta
        # Sanity clip: if IP differs from CoM by >30 cm → warn + clip
        diff = pelvis_ip - trans
        big = np.linalg.norm(diff, axis=1) > 0.30
        if big.any():
            n_big = int(big.sum())
            print(f"  [stance_anchor_ip] WARNING: {n_big}/{T_ip} frames IP vs CoM differ >30cm — clipping")
            for t in range(T_ip):
                if big[t]:
                    d = diff[t]
                    norm = np.linalg.norm(d) + 1e-9
                    pelvis_ip[t] = trans[t] + d / norm * 0.30
        # Replace lateral (X) and vertical (Y); keep forward (Z) from treadmill
        trans[:, 0] = pelvis_ip[:, 0]
        trans[:, 1] = pelvis_ip[:, 1]
        # forward (trans[:, 2]) unchanged
        print(f"  [stance_anchor_ip] done: "
              f"stance_L={stance_L.mean()*100:.1f}%  stance_R={stance_R.mean()*100:.1f}%  "
              f"flight_frames={int(nan_mask.sum())}")
```

**Notes on the block**:
- SkeletonTree path uses the standard PHC SMPL neutral MJCF. If your env differs, update the path — `/tmp/smpl/humanoid_smpl_neutral_no_limit.xml` is created by `SMPL_Robot.write_xml()` in other scripts; fallback `data/mjcf/smpl_humanoid_1.xml` is a common alternative. If neither exists, the implementer should search `find data -name 'smpl*.xml'` before assuming.
- The `poselib.SkeletonState.from_rotation_and_root_translation(..., is_local=True)` builds global rotations from local — this mirrors what the downstream pipeline does.

- [ ] **Step 3: Regenerate IP pkl**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --output sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl \
  --subjects S001 --tasks level_100mps --assist_level lv0 \
  --fps_out 30 --trim_start 10 --no_split \
  --baseline_s 3.0 --spine_3axis --upper_body --stance_anchor_ip
```
Expected: prints `[stance_anchor_ip] done: stance_L=~60%  stance_R=~60%  flight_frames=~0` and `1 clips`.

- [ ] **Step 4: Verify acceptance metric — lateral std**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python -c "
import os, sys, numpy as np, torch, joblib
sys.path.append(os.getcwd())
from easydict import EasyDict
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
robot_cfg = {'mesh': False,'rel_joint_lm': False,'upright_start': True,'remove_toe': False,
 'real_weight_porpotion_capsules': True,'real_weight_porpotion_boxes': True,'model': 'smpl',
 'big_ankle': True,'freeze_hand': False,'box_body': True,'body_params':{},'joint_params':{},
 'geom_params':{},'actuator_params':{}}
smpl = SMPL_Robot(robot_cfg, data_dir='data/smpl')
gb=np.zeros(17); smpl.load_from_skeleton(betas=torch.from_numpy(gb[None,1:]),gender=gb[0:1],objs_info=None)
xml='/tmp/smpl/ip_check.xml'; os.makedirs('/tmp/smpl', exist_ok=True); smpl.write_xml(xml); sk=SkeletonTree.from_mjcf(xml)
data=joblib.load('sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl'); k=list(data)[0]
tmp='/tmp/smpl/ip_clip.pkl'; joblib.dump({k:data[k]}, tmp)
cfg=EasyDict({'motion_file':tmp,'device':'cuda:0','fix_height':FixHeightMode.full_fix,
 'min_length':-1,'max_length':-1,'im_eval':False,'multi_thread':False,'smpl_type':'smpl','randomrize_heading':False})
m=MotionLibSMPL(cfg); m.load_motions(skeleton_trees=[sk],gender_betas=[torch.zeros(17)],
 limb_weights=[np.zeros(10)],random_sample=False,start_idx=0)
times=torch.linspace(10.0, 30.0, 600, device='cuda:0').float()
res=m.get_motion_state(torch.zeros_like(times).long(),times)
root=res['root_pos'].cpu().numpy()
vxy=np.gradient(root[:, :2], 1.0/30, axis=0); vmean=vxy.mean(0); speed=np.linalg.norm(vmean)
fwd=vmean/(speed+1e-9); right=np.array([fwd[1], -fwd[0]])
t_arr=np.arange(600)/30.0
exp_xy=root[0,:2][None,:]+np.outer(t_arr,vmean)
lat=(root[:,:2]-exp_xy)@right
print(f'v2_ip: lateral std={lat.std()*100:.2f}cm range={(lat.max()-lat.min())*100:.2f}cm')
print(f'(target: std 3-4cm, was 4.89cm; AMASS ref: 3.53cm)')
assert lat.std()*100 < 4.5, f'IP did not reduce lateral std below 4.5cm (got {lat.std()*100:.2f})'
print('PASS Task 6 acceptance')
"
```
Expected: prints `v2_ip: lateral std=X.XXcm` where X < 4.5. Prints `PASS Task 6 acceptance`.

- [ ] **Step 5: Verify no-op invariant still holds (flag OFF)**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python scripts/data/h5_to_motion_lib_v2.py \
  --h5 data/combined_data_from_csv.h5 \
  --output /tmp/v2_noop_after_t6.pkl \
  --subjects S001 --tasks level_100mps --assist_level lv0 \
  --fps_out 30 --trim_start 10 --no_split
conda run -n phc python -c "
import joblib, numpy as np, torch
a = joblib.load('sample_data/h5_walk_S001_lv0_trial01_armfix.pkl')
b = joblib.load('/tmp/v2_noop_after_t6.pkl')
ka, kb = list(a)[0], list(b)[0]
for k in a[ka]:
    va, vb = a[ka][k], b[kb][k]
    if isinstance(va, torch.Tensor): va = va.numpy()
    if isinstance(vb, torch.Tensor): vb = vb.numpy()
    if hasattr(va, 'shape'):
        assert np.allclose(np.asarray(va), np.asarray(vb)), f'{k} differs'
print('no-op invariant holds')
"
```
Expected: prints `no-op invariant holds`.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/h5_to_motion_lib_v2.py
git commit -m "$(cat <<'EOF'
feat(h5): wire --stance_anchor_ip end-to-end (IP pelvis anchor)

Regenerates lateral + vertical of pelvis world position from stance-foot
rolling-mean anchor. Reduces lateral std from 4.89cm (v2_full) toward
AMASS 3.53cm range.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Visual comparison rendering

**Files:**
- None (uses existing `scripts/vis/compare_pkl_metrics.py`)

Goal: produce 3 comparison mp4/png pairs for user visual judgment (sagittal, frontal, 3D isometric).

- [ ] **Step 1: Render v2_ip vs v2_full (before/after IP) — all views**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python scripts/vis/compare_pkl_metrics.py \
  --pkl_a sample_data/h5_walk_S001_lv0_trial01_v2_full.pkl --label_a "v2_full (no IP)" \
  --pkl_b sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl   --label_b "v2_ip (stance anchor)" \
  --out_prefix output/h5_visual_check/v2_ip_vs_v2_full \
  --view all
```
Expected: writes `_compare.mp4`, `_compare_front.mp4`, `_compare_3d.mp4`, `_metrics.png`.

- [ ] **Step 2: Render v2_ip vs AMASS — all views**

```bash
cd /home/exolab/Documents/GitHub/PHC
conda run -n phc python scripts/vis/compare_pkl_metrics.py \
  --pkl_a sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl --label_a "H5 v2_ip" \
  --pkl_b sample_data/amass_isaac_walking_primitive.pkl --label_b "AMASS walking" \
  --t_start_a 10.0 --t_end_a 30.0 --t_start_b 0.3 --t_end_b 3.8 \
  --out_prefix output/h5_visual_check/v2_ip_vs_amass \
  --view all
```
Expected: 4 artifacts written.

- [ ] **Step 3: Print file sizes + open for user review (no automatic judgment)**

```bash
cd /home/exolab/Documents/GitHub/PHC
ls -lh output/h5_visual_check/v2_ip_vs_v2_full* output/h5_visual_check/v2_ip_vs_amass*
```
Expected: 8 files listed (4 per compare), each with reasonable size (PNG >100KB, MP4 >500KB).

**Hand-off note**: Per the project feedback memory, the implementer (subagent) must NOT write result analysis or visual judgments. They present the file paths and stop. The user will open and evaluate the videos themselves.

- [ ] **Step 4: Commit** (no new files for this task; artifacts go in output/ which is gitignored, but the plan record matters)

No commit needed — this task produces visualization artifacts in `output/h5_visual_check/` which is gitignored. Task completion is signaled by the 8 artifact files existing.

---

## Acceptance Criteria

The plan is complete when:

1. ✓ All 4 task commits landed (Tasks 1–6). Task 7 produces artifacts.
2. ✓ `pytest scripts/data/test_h5_conversion_fixes.py -v` → 6 passed (3 original + 3 new).
3. ✓ `sample_data/h5_walk_S001_lv0_trial01_v2_ip.pkl` loads in MotionLibSMPL.
4. ✓ `v2_ip` pelvis lateral std < 4.5 cm in the 10–30 s window (was 4.89 cm in v2_full).
5. ✓ No-op invariant: flag OFF → output numerically identical to v2_full (via `np.allclose`).
6. ✓ 8 visualization artifacts in `output/h5_visual_check/` exist with reasonable sizes.

## Rollback

If IP creates visual regression, simply omit `--stance_anchor_ip` to restore v2_full behavior. No files need deletion. The helpers remain dormant.

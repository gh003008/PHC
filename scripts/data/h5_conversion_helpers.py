"""Pure helpers for H5→SMPL conversion fixes. Unit-testable without HDF5 I/O."""
import numpy as np
from scipy.spatial.transform import Rotation as sRot


def subtract_baseline(signal: np.ndarray, fps: int, baseline_s: float) -> np.ndarray:
    """Subtract the mean of the first `baseline_s` seconds from `signal`.

    Handles NaN via nanmean over the baseline window. If baseline_s <= 0 or
    window shorter than signal's available valid samples, returns signal unchanged.

    Args:
        signal: 1-D time series in degrees (same convention as PiG channels).
        fps: sampling rate of `signal`.
        baseline_s: window length (seconds) at the start of `signal` to use.

    Returns:
        New array, same shape as `signal`, with the baseline mean subtracted.
    """
    if baseline_s <= 0:
        return signal
    window = int(round(baseline_s * fps))
    window = min(window, len(signal))
    if window <= 0:
        return signal
    base = np.nanmean(signal[:window])
    if not np.isfinite(base):
        return signal
    return signal - base


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
    grf_stance_L = _schmitt(grf_L, grf_on, grf_off, on_when_above=True)
    grf_stance_R = _schmitt(grf_R, grf_on, grf_off, on_when_above=True)
    vel_stance_L = _schmitt(vel_L, vel_on, vel_off, on_when_above=False)
    vel_stance_R = _schmitt(vel_R, vel_on, vel_off, on_when_above=False)
    stance_L = grf_stance_L | vel_stance_L
    stance_R = grf_stance_R | vel_stance_R
    return stance_L, stance_R


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


def cop_replace_lateral(fk_foot_L, fk_foot_R, stance_L, stance_R,
                        cop_L_lat_m, cop_R_lat_m, fk_lat_axis=1):
    """Replace the lateral component of FK foot anchors with CoP (independent measurement).

    Forceplate CoP gives the true stance-foot lateral position in lab frame, independent
    of the CoM-based pelvis translation. Using it as the IP anchor breaks the circular
    dependency where FK foot positions derive from the same pelvis the IP is trying to
    correct.

    During stance frames: anchor[lat] = CoP lateral + centering offset.
    During swing frames: anchor[lat] = FK foot lateral (unchanged).

    Sign alignment is auto-detected: if FK (L-stance mean lateral) vs (R-stance mean)
    has opposite sign from CoP (L-stance mean) vs (R-stance mean), CoP is negated.

    Args:
        fk_foot_L, fk_foot_R: [T, 3] FK-derived foot world positions (Z-up).
        stance_L, stance_R: [T] bool stance masks.
        cop_L_lat_m, cop_R_lat_m: [T] CoP medio-lateral (lab frame x), in meters.
            NaN/invalid values outside stance are tolerated (those frames unused).
        fk_lat_axis: axis index for lateral in Z-up (default 1 = Y).

    Returns:
        anchor_L, anchor_R: [T, 3], FK foot with lateral axis replaced.
    """
    anchor_L = fk_foot_L.copy()
    anchor_R = fk_foot_R.copy()

    if stance_L.sum() < 3 or stance_R.sum() < 3:
        # Not enough stance frames to align — fall back to FK.
        return anchor_L, anchor_R

    fk_L_mean = float(np.nanmean(fk_foot_L[stance_L, fk_lat_axis]))
    fk_R_mean = float(np.nanmean(fk_foot_R[stance_R, fk_lat_axis]))
    cop_L_mean = float(np.nanmean(cop_L_lat_m[stance_L]))
    cop_R_mean = float(np.nanmean(cop_R_lat_m[stance_R]))

    fk_diff = fk_L_mean - fk_R_mean
    cop_diff = cop_L_mean - cop_R_mean
    sign = 1.0 if fk_diff * cop_diff > 0 else -1.0

    cop_L_aligned = sign * cop_L_lat_m
    cop_R_aligned = sign * cop_R_lat_m
    cop_L_mean_s = sign * cop_L_mean
    cop_R_mean_s = sign * cop_R_mean

    # Centering offset: align CoP mean with FK mean, per foot (preserves base-of-support width)
    offset_L = fk_L_mean - cop_L_mean_s
    offset_R = fk_R_mean - cop_R_mean_s

    anchor_L[stance_L, fk_lat_axis] = cop_L_aligned[stance_L] + offset_L
    anchor_R[stance_R, fk_lat_axis] = cop_R_aligned[stance_R] + offset_R
    return anchor_L, anchor_R


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


def build_foot_anchors(cop_L_xy, cop_R_xy, ankle_L, toe_L, ankle_R, toe_R,
                      phase_L, phase_R, min_episode_frames=10):
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


def solve_foot_ik_frame(pose_measured, trans_measured, R_pelvis_world,
                        anchors_t, phase_t,
                        offsets_L, offsets_R,
                        weights, bounds_deg, max_nfev,
                        pose_prev=None, trans_prev=None):
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
        max_nfev: scipy max function evaluations (≈ max_iter × (21+1) for finite-diff Jacobian).
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
        # Pelvis trans regularizer (deviation from measured CoM-derived trans)
        res.append(sw_p * (ptr - trans_measured))
        # Frame-to-frame smoothness on joint angles AND pelvis trans (if previous given).
        # Pelvis trans smoothness is critical at swing→stance boundaries — without it,
        # IK can teleport pelvis tens of cm in a single frame to satisfy a freshly-
        # activated anchor.
        if pose_prev is not None and sw_s > 0.0:
            res.append(sw_s * (l_hip - prev_LHIP))
            res.append(sw_s * (l_knee - prev_LKNEE))
            res.append(sw_s * (l_ank - prev_LANK))
            res.append(sw_s * (r_hip - prev_RHIP))
            res.append(sw_s * (r_knee - prev_RKNEE))
            res.append(sw_s * (r_ank - prev_RANK))
        if trans_prev is not None and sw_s > 0.0:
            res.append(sw_s * (ptr - trans_prev))
        return np.concatenate(res)

    try:
        sol = least_squares(
            residuals, x0, bounds=(lb, ub), method='trf', max_nfev=max_nfev,
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


def solve_pelvis_ip(anchor_L, anchor_R, grf_L, grf_R,
                    R_pelvis, foot_offset_L, foot_offset_R, eps=1e-3):
    """Solve pelvis world position from stance-foot anchor constraint (IP).

    Per frame t:
        w_L = grf_L / max(grf_L + grf_R, eps)
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
    T = anchor_L.shape[0]
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


def _aa_to_matrix_torch(aa):
    """Axis-angle (..., 3) → rotation matrix (..., 3, 3) via Rodrigues. Differentiable.

    Handles the small-angle case via Taylor expansion to avoid 0/0 gradient at θ=0.
    """
    import torch
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp(min=1e-12)        # (..., 1)
    k = aa / theta                                                              # (..., 3) unit axis
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)
    s = torch.sin(theta).unsqueeze(-1)                                          # (..., 1, 1)
    c = (1.0 - torch.cos(theta)).unsqueeze(-1)
    return eye + s * K + c * (K @ K)


def _fk_leg_world_xyz_torch(pelvis_trans, R_pelvis_world,
                             hip_aa, knee_aa, ankle_aa, leg_local_offsets):
    """Vectorized PyTorch FK for one leg over a trajectory. Differentiable.

    Args (all torch tensors, dtype float32 or float64):
        pelvis_trans: (T, 3) pelvis world position.
        R_pelvis_world: (T, 3, 3) pelvis world rotation matrix.
        hip_aa, knee_aa, ankle_aa: (T, 3) axis-angle joint rotations.
        leg_local_offsets: dict with 'hip', 'knee', 'ankle', 'toe' (each (3,) torch tensor).

    Returns:
        ankle_world: (T, 3)
        toe_world:   (T, 3)
    """
    import torch
    R_hip = _aa_to_matrix_torch(hip_aa)                                          # (T, 3, 3)
    R_knee = _aa_to_matrix_torch(knee_aa)
    R_ankle = _aa_to_matrix_torch(ankle_aa)

    def _apply(R, v):
        # R: (T, 3, 3), v: (3,) → (T, 3)
        return torch.einsum('tij,j->ti', R, v)

    hip_world = pelvis_trans + _apply(R_pelvis_world, leg_local_offsets["hip"])
    R_hip_w = R_pelvis_world @ R_hip
    knee_world = hip_world + _apply(R_hip_w, leg_local_offsets["knee"])
    R_knee_w = R_hip_w @ R_knee
    ankle_world = knee_world + _apply(R_knee_w, leg_local_offsets["ankle"])
    R_ankle_w = R_knee_w @ R_ankle
    toe_world = ankle_world + _apply(R_ankle_w, leg_local_offsets["toe"])
    return ankle_world, toe_world


def solve_foot_ik_trajectory(pose_measured, trans_measured, R_pelvis_world,
                              anchors, phase_L, phase_R,
                              offsets_L, offsets_R,
                              weights, bounds_deg, lr, max_iter, tol=1e-4,
                              device="cpu", verbose=False):
    """Trajectory-wide foot-anchor IK via PyTorch Adam.

    Optimizes pelvis_trans + L/R hip/knee/ankle joint angles over the FULL
    trajectory simultaneously. Cross-frame smoothness in the cost prevents the
    discrete swing→stance jumps that plague per-frame IK.

    Args:
        pose_measured: (T, 24, 3) BONE order axis-angle (numpy).
        trans_measured: (T, 3) pelvis world trans in Z-up (numpy).
        R_pelvis_world: (T, 3, 3) pelvis world rotation matrix in Z-up (numpy).
        anchors: dict with H_L, T_L, H_R, T_R (each (T, 3) np world position; NaN where inactive).
        phase_L, phase_R: (T,) int8 — 0=swing, 1=heel-only, 2=full-contact, 3=toe-only.
        offsets_L, offsets_R: from _extract_leg_local_offsets (np dicts; converted to torch internally).
        weights: dict with 'anchor', 'smooth', 'joint_reg', 'pelvis_reg', 'bound' float weights.
        bounds_deg: float — soft bound radius in degrees.
        lr: Adam learning rate.
        max_iter: max optimizer iterations.
        tol: early-stop loss-change tolerance.
        device: 'cpu' or 'cuda'.
        verbose: print loss every 50 iters.

    Returns:
        pose_corrected: (T, 24, 3) numpy — only L/R hip/knee/ankle modified.
        trans_corrected: (T, 3) numpy.
        info: dict with 'final_loss', 'iters', 'converged'.
    """
    import torch
    T = pose_measured.shape[0]
    L_HIP, L_KNEE, L_ANKLE = 1, 4, 7
    R_HIP, R_KNEE, R_ANKLE = 2, 5, 8

    dtype = torch.float32
    dev = torch.device(device)

    # Convert inputs to torch
    trans_m = torch.tensor(trans_measured, dtype=dtype, device=dev)
    R_pel = torch.tensor(R_pelvis_world, dtype=dtype, device=dev)
    pose_m = torch.tensor(pose_measured, dtype=dtype, device=dev)
    off_L = {k: torch.tensor(v, dtype=dtype, device=dev) for k, v in offsets_L.items()}
    off_R = {k: torch.tensor(v, dtype=dtype, device=dev) for k, v in offsets_R.items()}

    # Anchor masks per sub-phase
    pL = torch.tensor(phase_L.astype(np.int64), dtype=torch.long, device=dev)
    pR = torch.tensor(phase_R.astype(np.int64), dtype=torch.long, device=dev)
    mask_L_ankle = ((pL == 1) | (pL == 2)).float()        # (T,) — anchor heel/ankle active
    mask_L_toe   = ((pL == 2) | (pL == 3)).float()
    mask_R_ankle = ((pR == 1) | (pR == 2)).float()
    mask_R_toe   = ((pR == 2) | (pR == 3)).float()

    # Anchor positions (NaN-replaced with zero for safety; mask zeros out invalid)
    def _safe(arr):
        a = np.where(np.isfinite(arr), arr, 0.0)
        return torch.tensor(a, dtype=dtype, device=dev)
    H_L = _safe(anchors["H_L"]); T_L_anc = _safe(anchors["T_L"])
    H_R = _safe(anchors["H_R"]); T_R_anc = _safe(anchors["T_R"])

    # Decision variables (warm-start from measured)
    trans_var = trans_m.clone().requires_grad_(True)
    pose_var = torch.stack([
        pose_m[:, L_HIP],  pose_m[:, L_KNEE],  pose_m[:, L_ANKLE],
        pose_m[:, R_HIP],  pose_m[:, R_KNEE],  pose_m[:, R_ANKLE],
    ], dim=1).clone().requires_grad_(True)                 # (T, 6, 3)

    # Measured pose-leg snapshot for regularizer
    pose_m_legs = torch.stack([
        pose_m[:, L_HIP],  pose_m[:, L_KNEE],  pose_m[:, L_ANKLE],
        pose_m[:, R_HIP],  pose_m[:, R_KNEE],  pose_m[:, R_ANKLE],
    ], dim=1)

    bound_rad = float(np.radians(bounds_deg))
    w_anchor = float(weights["anchor"])
    w_smooth = float(weights["smooth"])
    w_jreg   = float(weights["joint_reg"])
    w_preg   = float(weights["pelvis_reg"])
    w_bound  = float(weights["bound"])

    optimizer = torch.optim.Adam([trans_var, pose_var], lr=lr)

    prev_loss = None
    converged = False
    iters_done = 0

    for it in range(max_iter):
        optimizer.zero_grad()

        # FK
        l_hip, l_knee, l_ank = pose_var[:, 0], pose_var[:, 1], pose_var[:, 2]
        r_hip, r_knee, r_ank = pose_var[:, 3], pose_var[:, 4], pose_var[:, 5]
        ankle_L_w, toe_L_w = _fk_leg_world_xyz_torch(trans_var, R_pel, l_hip, l_knee, l_ank, off_L)
        ankle_R_w, toe_R_w = _fk_leg_world_xyz_torch(trans_var, R_pel, r_hip, r_knee, r_ank, off_R)

        # Anchor losses (mean of squared error, masked by sub-phase activation)
        def _anchor_loss(pred, target, mask):
            err2 = ((pred - target) ** 2).sum(dim=-1)              # (T,)
            return (mask * err2).sum() / (mask.sum().clamp(min=1.0))

        L_anchor = (_anchor_loss(ankle_L_w, H_L, mask_L_ankle)
                    + _anchor_loss(toe_L_w, T_L_anc, mask_L_toe)
                    + _anchor_loss(ankle_R_w, H_R, mask_R_ankle)
                    + _anchor_loss(toe_R_w, T_R_anc, mask_R_toe))

        # Smoothness on acceleration (second difference): penalize jerk, not velocity.
        # Velocity smoothing flattens swing arcs (constant motion incurs penalty).
        # Acceleration smoothing allows constant velocity → preserves natural swing curves
        # while still suppressing sudden jumps. Standard trajectory-optimization choice.
        L_smooth_pose = ((pose_var[2:] - 2.0 * pose_var[1:-1] + pose_var[:-2]) ** 2).mean()
        L_smooth_trans = ((trans_var[2:] - 2.0 * trans_var[1:-1] + trans_var[:-2]) ** 2).mean()
        L_smooth = L_smooth_pose + L_smooth_trans

        # Regularizers (deviation from measured)
        L_jreg = ((pose_var - pose_m_legs) ** 2).mean()
        L_preg = ((trans_var - trans_m) ** 2).mean()

        # Soft bounds on joint angles (only joints — trans unbounded)
        excess = (pose_var - pose_m_legs).abs() - bound_rad
        L_bound = (excess.clamp(min=0.0) ** 2).mean()

        loss = (w_anchor * L_anchor
                + w_smooth * L_smooth
                + w_jreg   * L_jreg
                + w_preg   * L_preg
                + w_bound  * L_bound)

        loss.backward()
        optimizer.step()

        if verbose and (it % 50 == 0 or it == max_iter - 1):
            print(f"    iter {it:4d}  loss={loss.item():.6f}  "
                  f"anchor={L_anchor.item():.4f}  smooth={L_smooth.item():.6f}  "
                  f"jreg={L_jreg.item():.4f}  preg={L_preg.item():.4f}  bound={L_bound.item():.6f}")

        iters_done = it + 1
        if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
            converged = True
            break
        prev_loss = loss.item()

    # Write back to pose_corrected (BONE order)
    pose_corrected = pose_measured.copy()
    pose_var_np = pose_var.detach().cpu().numpy()
    pose_corrected[:, L_HIP]   = pose_var_np[:, 0]
    pose_corrected[:, L_KNEE]  = pose_var_np[:, 1]
    pose_corrected[:, L_ANKLE] = pose_var_np[:, 2]
    pose_corrected[:, R_HIP]   = pose_var_np[:, 3]
    pose_corrected[:, R_KNEE]  = pose_var_np[:, 4]
    pose_corrected[:, R_ANKLE] = pose_var_np[:, 5]
    trans_corrected = trans_var.detach().cpu().numpy()

    info = {"final_loss": float(prev_loss if prev_loss is not None else 0.0),
            "iters": iters_done, "converged": converged}
    return pose_corrected, trans_corrected, info

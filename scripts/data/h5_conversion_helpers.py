"""Pure helpers for H5→SMPL conversion fixes. Unit-testable without HDF5 I/O."""
import numpy as np


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

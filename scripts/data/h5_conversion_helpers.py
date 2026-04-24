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

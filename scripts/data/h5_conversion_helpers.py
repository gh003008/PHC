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
    raise NotImplementedError

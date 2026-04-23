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

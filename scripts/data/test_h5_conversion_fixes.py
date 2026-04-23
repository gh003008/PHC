"""Unit tests for H5 conversion helpers (scripts/data/h5_conversion_helpers.py)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.h5_conversion_helpers import subtract_baseline


def test_baseline_subtraction_removes_constant_offset():
    """Constant 8° standing followed by ±5° oscillation → mean in baseline window ≈ 0,
    oscillation amplitude preserved."""
    fps = 100
    standing = np.full(5 * fps, 8.0)                                     # 5 s at 8°
    walking = 8.0 + 5.0 * np.sin(np.arange(5 * fps) * 0.1)               # 5 s oscillation
    signal = np.concatenate([standing, walking])
    result = subtract_baseline(signal, fps=fps, baseline_s=3.0)
    # Baseline window should be zero
    assert abs(result[:3 * fps].mean()) < 0.1
    # Walking mean should also be ~0 (since we subtracted the true offset)
    assert abs(result[5 * fps:].mean()) < 0.5
    # Oscillation amplitude preserved (±5°)
    osc = result[5 * fps:]
    assert 4.5 < (osc.max() - osc.min()) / 2 < 5.5


def test_baseline_subtraction_handles_nans_in_baseline_window():
    """NaN in baseline window should be ignored (nanmean) not propagate."""
    fps = 100
    signal = np.full(500, 5.0)
    signal[50:60] = np.nan  # NaN chunk inside baseline window
    result = subtract_baseline(signal, fps=fps, baseline_s=3.0)
    # Non-NaN positions should have ~0 mean
    finite = result[np.isfinite(result)]
    assert abs(finite[:200].mean()) < 0.05


def test_baseline_subtraction_zero_seconds_is_noop():
    """baseline_s=0 leaves signal unchanged."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    result = subtract_baseline(signal, fps=100, baseline_s=0.0)
    assert np.array_equal(result, signal)

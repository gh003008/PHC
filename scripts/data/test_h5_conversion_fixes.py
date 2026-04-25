"""Unit tests for H5 conversion helpers (scripts/data/h5_conversion_helpers.py)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.h5_conversion_helpers import subtract_baseline, detect_stance, compute_foot_anchor, solve_pelvis_ip, classify_sub_phases


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
    # Input std ~5mm, output std should be < 3mm (rolling smoothing ~ sqrt(window) improvement with edge truncation)
    assert foot_world[:, 0].std() > 0.003
    assert anchor[:, 0].std() < 0.003  # 3mm — robust to edge truncation with window=9


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


def test_classify_sub_phases_synthetic():
    """Synthetic foot pitch sweep across one full gait cycle.

    GRF: 0..30 swing → 30..70 stance → 70..100 swing. (single foot used for test)
    Pitch trajectory during stance: starts at +15° (heel down, toe up),
    sweeps through 0° (full contact mid-stance), ends at -15° (toe down, heel up).
    """
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
    n_heel = int((phase_L == 1).sum())
    n_full = int((phase_L == 2).sum())
    n_toe = int((phase_L == 3).sum())
    assert n_heel >= 3, f"expected >=3 heel-only frames, got {n_heel}"
    assert n_full >= 3, f"expected >=3 full-contact frames, got {n_full}"
    assert n_toe >= 3, f"expected >=3 toe-only frames, got {n_toe}"
    # R foot all swing
    assert np.all(phase_R == 0), "R foot should be all swing"

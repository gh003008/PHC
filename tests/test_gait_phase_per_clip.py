"""Unit test for gait-phase detector — CPU-only logic, but the module
under test (humanoid_im_vic) pulls phc.utils.torch_utils -> isaacgym
at import time, which must be loaded before torch. We therefore import
isaacgym before anything else, then exercise only the pure-function
helpers."""
import isaacgym  # noqa: F401  — must be before torch import
import math
import pytest
import torch

from phc.env.tasks.humanoid_im_vic import (
    _detect_heel_strikes_static,
    _build_phase_from_hs_static,
)


def _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30):
    t = torch.linspace(0, duration_s - 1e-4, int(duration_s * fps))
    z = 0.5 - 0.2 * torch.cos(2 * math.pi * t / stride_s)
    return z, t


def test_detect_heel_strikes_finds_expected_count():
    z, t = _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert 4 <= len(hs) <= 5


def test_detect_heel_strikes_filters_close_minima():
    # 1 Hz base (5 true minima over 5 s) + 10 Hz ripple. The min_interval
    # filter collapses ripple minima that fall < 0.5 s from the last accepted
    # one, but ripple peaks spaced exactly >0.5s apart still survive.
    # Upper bound here is a loose sanity check (≤50 raw local minima collapse
    # to ~10), not a precise count.
    t = torch.linspace(0, 5 - 1e-4, 150)
    z = 0.5 - 0.2 * torch.cos(2 * math.pi * t) + 0.05 * torch.cos(2 * math.pi * t / 0.1)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert len(hs) <= 12


def test_build_phase_from_hs_monotone_in_cycle():
    z, t = _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    phase = _build_phase_from_hs_static(t, hs, num_samples=len(t), device="cpu")
    assert phase.min() >= -1e-4
    assert phase.max() <= 1.0 + 1e-4
    for i in range(len(hs) - 1):
        idx_lo, idx_hi = hs[i], hs[i + 1]
        seg = phase[idx_lo:idx_hi]
        if len(seg) >= 2:
            diffs = seg[1:] - seg[:-1]
            assert (diffs >= -1e-3).all()


def test_detect_heel_strikes_returns_empty_on_flat():
    t = torch.linspace(0, 5 - 1e-4, 150)
    z = torch.full_like(t, 0.5)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert len(hs) == 0

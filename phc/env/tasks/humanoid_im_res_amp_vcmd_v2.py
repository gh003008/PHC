"""V2 task class: residual + AMP + v_cmd on top of frozen phc_3.

V2 design (post 2026-05-02 failure analysis):
  - Built from gates: each gate is the smallest possible addition that we
    verify with the diagnostic (PHC_ZERO_RESIDUAL=1, dynamics off → walks).
  - This is GATE 1: empty subclass of HumanoidIm. No overrides, no v_cmd,
    no multi-clip, no retime, no custom reward. Just verifies inheritance
    + Hydra-style env yaml + phc_3 ckpt loading walks.

Subsequent gates (added in separate commits):
  Gate 2: phc_3 RunningMeanStd freeze
  Gate 3: v_cmd in obs (residual head only)
  Gate 4: multi-clip switch (motion_lib level)
  Gate 5: per-step retime
  Gate 6: residual head random init
  Gate 7: r_track + r_im reward (no r_survive) → start training

Spec: docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md
Failure analysis: 02_research_dev/260502_res_amp_vcmd_failure_analysis.md
"""
from __future__ import annotations
import isaacgym  # noqa: F401  # must precede torch
from phc.env.tasks.humanoid_im import HumanoidIm


class HumanoidImResAMPVCmdV2(HumanoidIm):
    """Gate 1: empty subclass. Inherits all HumanoidIm behavior unchanged."""
    pass

"""IsaacGym interactive demo for VIC4_VCMD policies.

Run:
  conda activate phc
  python scripts/vic4_demo.py

To switch models, edit the SLOT/EPOCH constants below. The yaml dir is
auto-resolved from SLOT (S4-S8 -> 260427, S9-S11 -> 260428_v8,
S12-S13 -> 260428_v9).

Controls (live, in the IsaacGym window):
  Up   / Down  : v_cmd +/- 0.05 (clamped to learned range)
  1..9         : v_cmd snap to 9 levels across learned range
  R            : hard reset humanoid to standing pose at origin
  Space        : pause / resume sim
  Esc          : quit

ImGui slider is best-effort: rendered if the IsaacGym binding exposes
a value-mutating UI callback, otherwise falls back to keyboard only.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

# === EDIT THESE TWO CONSTANTS TO SWITCH MODELS ===
SLOT = "S10"           # one of S4..S13
EPOCH = -1             # -1 = auto-pick latest output/VIC4_VCMD_<SLOT>_*.pth
# =================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for _p in (_PHC_PKG, _PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np
import torch
import yaml


def resolve_exp_dir(slot: str) -> str:
    if slot in ('S12', 'S13'):
        return 'exp_config/forward_walking/260428_VIC4_VCMD_v9'
    if slot in ('S9', 'S10', 'S11'):
        return 'exp_config/forward_walking/260428_VIC4_VCMD_v8'
    return 'exp_config/forward_walking/260427_VIC4_VCMD'


def resolve_checkpoint(slot: str, epoch: int) -> tuple[str, int]:
    """Return (path, resolved_epoch). epoch=-1 means auto-pick latest."""
    if epoch >= 0:
        path = f"output/VIC4_VCMD_{slot}_{epoch:08d}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint at {path}. Pull from server with:\n"
                f"  rsync -avzP server1-jiminyoun:PHC/output/VIC4_VCMD_{slot}_{epoch:08d}.pth output/"
            )
        return path, epoch
    # auto-pick latest
    candidates = sorted(glob.glob(f"output/VIC4_VCMD_{slot}_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching output/VIC4_VCMD_{slot}_*.pth\n"
            f"Pull from server with:\n"
            f"  rsync -avzP 'server1-jiminyoun:PHC/output/VIC4_VCMD_{slot}_*.pth' output/"
        )
    path = candidates[-1]
    ep_str = os.path.basename(path).split('_')[-1].replace('.pth', '')
    return path, int(ep_str)

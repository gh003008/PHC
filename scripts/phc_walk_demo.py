"""IsaacGym interactive demo: PHC pretrained phc_3 + continuous v_cmd walking.

No training. Multi-clip + per-step retime via _motion_start_times_offset
accumulation. The pretrained phc_3 imitates the active retimed walking clip;
v_cmd controls the playback rate.

Run:
  conda activate phc
  python scripts/phc_walk_demo.py
  # optional 30s recording:
  python scripts/phc_walk_demo.py --record_seconds 30

Controls (in the IsaacGym window):
  Up   / Down  : v_cmd ±0.02 (clamped to [0.76, 1.23])
  1..9         : v_cmd snap (linear interp between V_CMD_MIN and V_CMD_MAX)
  R            : episode reset
  P            : pause / resume
  Q            : quit

The Tkinter panel (auto-spawned) supports numeric input, slider, and
Apply/Reset/Pause buttons. v_cmd is smoothed by max_accel=0.5 m/s².
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys

EXP_NAME = "phc_3"
LEARNING = "im_pnn_big"
ENV_CFG_NAME = "env_im_pnn"
MOTION_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9.pkl"
DIRMETA_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json"

V_NATURAL = (0.897, 0.975, 1.068)
V_CMD_INIT = 1.0
V_CMD_MIN = 0.76
V_CMD_MAX = 1.23
V_CMD_KEY_STEP = 0.02
V_CMD_MAX_ACCEL = 0.5
HYSTERESIS = 0.02
MIDPOINT_AB = (V_NATURAL[0] + V_NATURAL[1]) / 2.0
MIDPOINT_BC = (V_NATURAL[1] + V_NATURAL[2]) / 2.0

NUM_ENVS = 2
ENV_SPACING = 50

PANEL_STATE_PATH = "/tmp/phc_walk_state.json"
PANEL_INPUT_PATH = "/tmp/phc_walk_input.json"
ARROW_LEN_AT_VHI = 1.5
LOG_STEP_INTERVAL = 60

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, "phc")
for _p in (_PHC_PKG, _PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record_seconds", type=int, default=0,
                    help="If > 0, capture frames and save mp4 then exit.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", default="videos")
    args = ap.parse_args()

    sys.argv = [
        "run_hydra.py",
        f"learning={LEARNING}",
        f"env={ENV_CFG_NAME}",
        "robot=smpl_humanoid",
        f"exp_name={EXP_NAME}",
        "epoch=-1",
        "test=True",
        "headless=False",
        "no_virtual_display=True",
        f"env.num_envs={NUM_ENVS}",
        f"env.env_spacing={ENV_SPACING}",
        f"env.motion_file={MOTION_FILE}",
        "env.cycle_motion=True",
        "env.episode_length=300",
    ]

    print("=" * 70)
    print(f"[demo] EXP={EXP_NAME} LEARNING={LEARNING}")
    print(f"[demo] motion={MOTION_FILE}")
    print(f"[demo] v_cmd range: [{V_CMD_MIN:.2f}, {V_CMD_MAX:.2f}]  init={V_CMD_INIT}")
    print(f"[demo] keys:  ↑/↓ ±{V_CMD_KEY_STEP}  |  1..9 snap  |  R reset  |  P pause  |  Q quit")
    print("=" * 70)

    import runpy
    try:
        runpy.run_path("phc/run_hydra.py", run_name="__main__")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()

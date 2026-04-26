"""Subset 3 speed-spaced KIT_425 walking clips for VIC4+v_cmd training.

Selects exactly 3 clips from the 50-clip walking primitive pkl, per spec
section 5 (option B-1):
    slow:   0-KIT_425_walking_03_poses        (T=195, |v_x|=0.312)
    medium: 0-KIT_425_walking_medium08_poses  (T=162, |v_x|=0.528)
    fast:   0-KIT_425_walking_medium05_poses  (T=127, |v_x|=0.683)

All 3 are KIT_425 user with v_x_signed < 0 (forward in dataset frame).
Output is consumed by S5/S6 of the VIC4+v_cmd 3-slot training plan.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np

REPO_ROOT = "/home/exolab/Documents/GitHub/PHC"
SRC_PKL = os.path.join(REPO_ROOT, "sample_data", "amass_isaac_walking_primitive.pkl")
DST_PKL = os.path.join(REPO_ROOT, "sample_data", "amass_walking_3clips_speedspaced.pkl")

# Exact dict keys to keep, in order: slow -> medium -> fast.
SELECTED_KEYS = [
    "0-KIT_425_walking_03_poses",         # slow:   T=195, |v_x|=0.312
    "0-KIT_425_walking_medium08_poses",   # medium: T=162, |v_x|=0.528
    "0-KIT_425_walking_medium05_poses",   # fast:   T=127, |v_x|=0.683
]
EXPECTED = {
    "0-KIT_425_walking_03_poses":        {"frames": 195, "v_x_abs": 0.312},
    "0-KIT_425_walking_medium08_poses":  {"frames": 162, "v_x_abs": 0.528},
    "0-KIT_425_walking_medium05_poses":  {"frames": 127, "v_x_abs": 0.683},
}
FPS = 30.0
TOL = 0.001


def compute_v_x_abs(trans: np.ndarray, fps: float = FPS) -> float:
    """|v_x| = |trans[-1, 0] - trans[0, 0]| / (T / fps).

    `trans` is expected to be (T, 3) root translation in meters.
    """
    T = int(trans.shape[0])
    if T < 2:
        return 0.0
    duration = T / fps
    return float(abs(trans[-1, 0] - trans[0, 0]) / duration)


def main() -> int:
    print(f"[subset_3clips] loading source pkl: {SRC_PKL}")
    src = joblib.load(SRC_PKL)
    print(f"[subset_3clips] source contains {len(src)} clips")

    # Validate all selected keys exist (exact match, no substring).
    missing = [k for k in SELECTED_KEYS if k not in src]
    if missing:
        raise KeyError(
            f"Missing required clip keys in source pkl: {missing}. "
            f"Aborting subset build."
        )

    # Build new dict in the requested order: slow -> medium -> fast.
    subset: dict = {}
    for k in SELECTED_KEYS:
        subset[k] = src[k]

    print(f"[subset_3clips] saving subset pkl: {DST_PKL}")
    joblib.dump(subset, DST_PKL)

    # Re-load and verify.
    print(f"[subset_3clips] reloading {DST_PKL} for verification")
    reloaded = joblib.load(DST_PKL)
    n = len(reloaded)
    if n != 3:
        raise RuntimeError(f"Expected 3 clips in saved pkl, got {n}")

    print(f"[subset_3clips] OK: saved pkl has {n} clips")
    print("=" * 78)
    print(f"{'name':<42s} {'frames':>7s} {'v_x_abs':>10s}  expected")
    print("-" * 78)

    all_ok = True
    for name, entry in reloaded.items():
        trans = entry["trans"] if "trans" in entry else entry["root_trans_offset"]
        trans = np.asarray(trans)
        T = int(trans.shape[0])
        v = compute_v_x_abs(trans, FPS)

        exp = EXPECTED[name]
        frames_ok = (T == exp["frames"])
        v_ok = abs(v - exp["v_x_abs"]) <= TOL
        flag = " " if (frames_ok and v_ok) else "!"
        print(
            f"{flag} {name:<40s} {T:>7d} {v:>10.3f}  "
            f"(T={exp['frames']}, |v_x|={exp['v_x_abs']:.3f})"
        )
        if not (frames_ok and v_ok):
            all_ok = False

    print("=" * 78)
    if not all_ok:
        print("[subset_3clips] VERIFICATION FAILED")
        return 1

    size_bytes = os.path.getsize(DST_PKL)
    print(f"[subset_3clips] file size: {size_bytes} bytes ({size_bytes / 1024:.1f} KiB)")
    print("[subset_3clips] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

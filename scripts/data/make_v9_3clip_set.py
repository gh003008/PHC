"""Generate v9 — 3-clip seamless training pkl (medium08 + medium05 + KIT_11).

KIT_11_WalkingStraightForwards05 (raw v_x ~0.67 m/s) added to widen the
v_cmd range below v8's [0.82, 1.23]. Subject pelvic height matches v8
within +0.2cm so no kinematic mismatch.
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_seamless_long_clip import (  # noqa: E402
    FPS,
    TARGET_DURATION_S,
    build_seamless_loop,
)

SOURCE_PKL = "sample_data/amass_isaac_walking_primitive.pkl"
OUT_PKL = "sample_data/amass_walking_3clips_seamless_60s_v9.pkl"
KEEP_FULL = [
    "0-KIT_425_walking_medium08_poses",
    "0-KIT_425_walking_medium05_poses",
    "0-KIT_11_WalkingStraightForwards05_poses",
]


def main():
    src = joblib.load(SOURCE_PKL)
    n_frames = int(TARGET_DURATION_S * FPS)
    out = {}
    for name in KEEP_FULL:
        if name not in src:
            raise KeyError(f"clip {name} not found in {SOURCE_PKL}")
        clip = src[name]
        T = clip["pose_quat_global"].shape[0]
        print(f"\n=== Processing {name} (T={T} frames) ===")
        seamless = build_seamless_loop(clip, n_frames)
        out_key = f"{name}_seamless_60s"
        out[out_key] = seamless
    joblib.dump(out, OUT_PKL)
    print(f"\nWrote {OUT_PKL} ({len(out)} clips x {n_frames} frames each)")
    for k in out.keys():
        print(f"  - {k}")


if __name__ == "__main__":
    main()

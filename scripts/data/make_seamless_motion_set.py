"""Generate seamless 60s loops for each clip in the 3-clip training pkl.

Reuses build_seamless_loop from make_seamless_long_clip.py, applies it to
each entry of amass_walking_3clips_speedspaced.pkl, and writes a single
multi-clip pkl where each entry is a 60-second seamless loop.

Output is parallel to the training pkl (same 3 keys, suffixed with
'_seamless_60s'). Used post-training for demo: feed the policy a
v_cmd-conditioned seamless reference at any of the 3 walking speeds.

Usage:
  conda activate phc
  python scripts/data/make_seamless_motion_set.py
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

SOURCE_PKL = "sample_data/amass_walking_3clips_speedspaced.pkl"
OUT_PKL = "sample_data/amass_walking_3clips_seamless_60s.pkl"


def main():
    src = joblib.load(SOURCE_PKL)
    n_frames = int(TARGET_DURATION_S * FPS)
    out = {}
    for name, clip in src.items():
        T = clip["pose_quat_global"].shape[0]
        print(f"\n=== Processing {name} (T={T} frames) ===")
        seamless = build_seamless_loop(clip, n_frames)
        out_key = f"{name}_seamless_60s"
        out[out_key] = seamless
    joblib.dump(out, OUT_PKL)
    print(f"\nWrote {OUT_PKL} ({len(out)} clips × {n_frames} frames each)")
    for k in out.keys():
        print(f"  - {k}")


if __name__ == "__main__":
    main()

"""Generate v8 — 2-clip seamless training pkl (medium08 + medium05 only).

walking_03 dropped due to source-data jerkiness in upper limb. Result:
clean 2-clip set with natural speeds 0.97-1.07 m/s (retime ±15% covers
v_cmd [0.82, 1.23]).
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
OUT_PKL = "sample_data/amass_walking_2clips_seamless_60s_v8.pkl"
KEEP = ["medium08", "medium05"]


def main():
    src = joblib.load(SOURCE_PKL)
    n_frames = int(TARGET_DURATION_S * FPS)
    out = {}
    for name, clip in src.items():
        if not any(k in name for k in KEEP):
            print(f"\n=== Skipping {name} (not in KEEP) ===")
            continue
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

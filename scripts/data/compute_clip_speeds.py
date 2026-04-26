"""Compute natural forward velocity (v_x) for every clip in a motion pkl.
Outputs: stdout table + JSON file at sample_data/clip_speeds_<source>.json.

Usage: python scripts/data/compute_clip_speeds.py sample_data/amass_isaac_walking_primitive.pkl
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl_path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min_v", type=float, default=0.30,
                    help="filter forward-walking only")
    ap.add_argument("--min_frames", type=int, default=60)
    ap.add_argument("--max_frames", type=int, default=250)
    args = ap.parse_args()

    data = joblib.load(args.pkl_path)
    print(f"Loaded {len(data)} clips from {args.pkl_path}")
    print(f"{'clip_name':<60s} {'frames':>6s} {'v_x_nat (m/s)':>15s}")
    print("-" * 85)

    rows = []
    for name, clip in data.items():
        if "trans_orig" in clip:
            trans = clip["trans_orig"]
        elif "root_trans_offset" in clip:
            trans = clip["root_trans_offset"]
            if hasattr(trans, "cpu"):
                trans = trans.cpu().numpy()
        else:
            continue

        T = trans.shape[0]
        if T < args.min_frames or T > args.max_frames:
            continue

        v_x = float((trans[T - 1, 0] - trans[0, 0]) / (T / args.fps))
        if v_x < args.min_v:
            continue

        rows.append({"name": name, "frames": int(T), "v_x_nat": v_x})
        print(f"{name:<60s} {T:>6d} {v_x:>15.3f}")

    rows.sort(key=lambda r: r["v_x_nat"])
    out_path = os.path.splitext(args.pkl_path)[0] + "_speeds.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path} ({len(rows)} forward-walking clips)")


if __name__ == "__main__":
    main()

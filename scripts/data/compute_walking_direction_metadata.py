"""Compute signed forward velocity & yaw rate per clip, filter to forward-straight walking.

Input:
    sample_data/amass_isaac_walking_primitive.pkl      (50 clips, all 'walking'
                                                        but mixed direction: forward,
                                                        backward, clockwise circle,
                                                        counter-clockwise circle)

Output:
    sample_data/amass_isaac_walking_primitive_dirmeta.json
        per-clip {
            duration_s, fps, n_frames,
            v_x_mean_mid, v_y_mean_mid,                    # signed (PHC +x = forward)
            v_mag_mid,                                     # | (v_x, v_y) |
            yaw_rate_mean_mid,                             # rad/s, signed
            dir_class: "forward_straight" | "backward_straight" | "turning" | "other",
            is_forward_straight: bool,
        }

    sample_data/amass_isaac_walking_primitive_fwd_only.json
        subset limited to forward_straight clips, same schema,
        ordered by v_x_mean_mid ascending.

Filter criteria (forward_straight):
    v_x_mean_mid        <  -V_FWD_MIN       (pelvis moves in -x direction, which is
                                             "forward" in the PHC world-frame
                                             convention — confirmed via
                                             walking_forward_single.pkl, v_x ≈ -0.44 m/s)
    |v_y_mean_mid|      <   V_LATERAL_MAX   (no strong sideways drift)
    |yaw_rate_mean_mid| <   YAW_RATE_MAX    (not turning)

Defaults chosen conservatively so the first multi-clip pure-speed experiment sees
only clips that are essentially straight forward walks; these can be tightened
further if a later run still shows conflated behavior.

Middle segment = frames in [skip, n_frames-skip), skip = int(0.5 * fps),
so we ignore initial startup and final deceleration parts of each clip.

Usage (on server, inside phc conda env):
    conda activate phc
    python scripts/data/compute_walking_direction_metadata.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch


# --- Tuning knobs -----------------------------------------------------------
V_FWD_MIN = 0.15           # m/s  — lower bound on forward velocity
V_LATERAL_MAX = 0.20       # m/s  — upper bound on lateral drift
YAW_RATE_MAX = 0.30        # rad/s — ~17 deg/s, excludes circle / turning clips
SKIP_EDGE_SEC = 0.5         # skip this many seconds at start/end for mid-segment
# ----------------------------------------------------------------------------

SRC = Path("sample_data/amass_isaac_walking_primitive.pkl")
OUT_FULL = Path("sample_data/amass_isaac_walking_primitive_dirmeta.json")
OUT_FWD = Path("sample_data/amass_isaac_walking_primitive_fwd_only.json")


def _yaw_from_quat(quat):
    """quat: [T, 4] in (x,y,z,w) convention (PHC default). Returns yaw angle [T]."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # yaw around +z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _unwrap_yaw(yaw):
    """Unwrap discontinuities so finite differencing is valid."""
    yaw = yaw.clone()
    diffs = yaw[1:] - yaw[:-1]
    jumps = torch.where(diffs > np.pi, -2 * np.pi, torch.where(diffs < -np.pi, 2 * np.pi, torch.zeros_like(diffs)))
    yaw[1:] += torch.cumsum(jumps, dim=0)
    return yaw


def classify_direction(v_x, v_y, yaw_rate):
    """Classify a clip's direction from its middle-segment signed means."""
    v_lat = abs(v_y)
    if abs(yaw_rate) >= YAW_RATE_MAX:
        return "turning"
    # PHC world-frame convention: forward walking has v_x < 0 (verified via
    # walking_forward_single.pkl: v_x_mean ≈ -0.44 m/s over its peak phase).
    if v_x < -V_FWD_MIN and v_lat < V_LATERAL_MAX:
        return "forward_straight"
    if v_x > +V_FWD_MIN and v_lat < V_LATERAL_MAX:
        return "backward_straight"
    return "other"


def compute_for_clip(entry):
    """entry is a dict with PHC-standard keys. Returns direction metadata dict."""
    fps = float(entry.get("fps", 30))
    pose_quat_global = entry.get("pose_quat_global", None)  # [T, J, 4]
    root_trans_offset = entry.get("root_trans_offset", None)  # [T, 3]

    if root_trans_offset is None or pose_quat_global is None:
        raise RuntimeError("clip entry missing root_trans_offset / pose_quat_global")

    root_trans = torch.as_tensor(root_trans_offset, dtype=torch.float32)  # [T, 3]
    root_quat = torch.as_tensor(pose_quat_global[:, 0, :], dtype=torch.float32)  # [T, 4]

    T = root_trans.shape[0]
    skip = max(1, int(SKIP_EDGE_SEC * fps))
    if T <= 2 * skip + 2:
        skip = 1  # degenerate: use almost full clip

    # Pelvis horizontal velocity (finite diff) over mid segment
    pos_xy = root_trans[:, :2]                                # [T, 2]
    vel_xy = (pos_xy[1:] - pos_xy[:-1]) * fps                 # [T-1, 2]
    mid_lo = skip
    mid_hi = max(mid_lo + 1, T - 1 - skip)
    vel_mid = vel_xy[mid_lo:mid_hi]
    v_x_mean = vel_mid[:, 0].mean().item()
    v_y_mean = vel_mid[:, 1].mean().item()
    v_mag = (v_x_mean ** 2 + v_y_mean ** 2) ** 0.5

    # Yaw rate over mid segment
    yaw = _unwrap_yaw(_yaw_from_quat(root_quat))              # [T]
    yaw_rate_series = (yaw[1:] - yaw[:-1]) * fps              # [T-1]
    yaw_rate_mid = yaw_rate_series[mid_lo:mid_hi]
    yaw_rate_mean = yaw_rate_mid.mean().item()

    dir_class = classify_direction(v_x_mean, v_y_mean, yaw_rate_mean)
    return {
        "duration_s": float(T / fps),
        "fps": fps,
        "n_frames": T,
        "v_x_mean_mid": v_x_mean,
        "v_y_mean_mid": v_y_mean,
        "v_mag_mid": v_mag,
        "yaw_rate_mean_mid": yaw_rate_mean,
        "dir_class": dir_class,
        "is_forward_straight": dir_class == "forward_straight",
    }


def main():
    if not SRC.exists():
        raise RuntimeError(f"{SRC} not found")
    print(f"Loading {SRC} ...")
    data = joblib.load(SRC)
    print(f"  {len(data)} clips")

    full_meta = {}
    cls_counts = {"forward_straight": 0, "backward_straight": 0, "turning": 0, "other": 0}
    for k, entry in data.items():
        try:
            meta = compute_for_clip(entry)
        except Exception as e:
            print(f"  [skip] {k}: {e}")
            continue
        full_meta[k] = meta
        cls_counts[meta["dir_class"]] += 1

    print()
    print("Direction classification (V_FWD_MIN=%.2f V_LAT_MAX=%.2f YAW_RATE_MAX=%.2f rad/s):" %
          (V_FWD_MIN, V_LATERAL_MAX, YAW_RATE_MAX))
    for k, c in cls_counts.items():
        print(f"  {k:20s}: {c}")
    print()

    with open(OUT_FULL, "w") as f:
        json.dump(full_meta, f, indent=2)
    print(f"Wrote {OUT_FULL}")

    # Forward-only subset, ordered by ascending forward-speed magnitude |v_x|
    # (forward v_x is negative, so smallest |v_x| = slowest forward walk).
    fwd = {k: v for k, v in full_meta.items() if v["is_forward_straight"]}
    fwd_sorted = dict(sorted(fwd.items(), key=lambda kv: abs(kv[1]["v_x_mean_mid"])))
    with open(OUT_FWD, "w") as f:
        json.dump(fwd_sorted, f, indent=2)
    print(f"Wrote {OUT_FWD}  ({len(fwd_sorted)} forward-straight clips)")
    if fwd_sorted:
        vs = [v["v_x_mean_mid"] for v in fwd_sorted.values()]
        print(f"  v_x_mean_mid range: [{min(vs):.3f}, {max(vs):.3f}] m/s")
        print(f"  Clip list (by ascending v_x):")
        for k, v in fwd_sorted.items():
            print(f"    {k:60s}  v_x={v['v_x_mean_mid']:+.3f}  v_y={v['v_y_mean_mid']:+.3f}  "
                  f"yaw_rate={v['yaw_rate_mean_mid']:+.3f}")


if __name__ == "__main__":
    main()

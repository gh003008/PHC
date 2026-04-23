"""Dry-run of R_Ankle heel-strike detection on the 22 forward clips.

Recreates the logic of HumanoidImVIC._precompute_gait_phase() without IsaacGym.

For each forward clip:
    1. Reconstruct R_Ankle world-frame position from
       root_trans_offset + bone-chain global rotations + MJCF T-pose offsets.
    2. Sample 1000 equi-spaced time points over the clip duration.
    3. Run the same local-minimum HS detector (> 0.5 s separation filter).
    4. Report success/fail, n_hs, mean stride period, R_Ankle z range.

No training, no IsaacGym. Pure CPU torch + joblib.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

# --- SMPL MJCF bone offsets (right leg chain) -------------------------------
# From phc/data/assets/mjcf/smpl_humanoid.xml
D_HIP   = torch.tensor([-0.0677, -0.0905, -0.0043], dtype=torch.float32)
D_KNEE  = torch.tensor([-0.0383, -0.3826, -0.0089], dtype=torch.float32)
D_ANKLE = torch.tensor([ 0.0158, -0.3984, -0.0423], dtype=torch.float32)

# SMPL joint order (standard 24-joint)
PELVIS_IDX = 0
R_HIP_IDX  = 2
R_KNEE_IDX = 5
R_ANKLE_IDX = 8

FPS_ASSUMED = 30
NUM_SAMPLES = 1000
HS_MIN_INTERVAL_S = 0.5  # same as HumanoidImVIC._precompute_gait_phase


def quat_rotate(q, v):
    """q: [..., 4] (x,y,z,w), v: [..., 3] -> rotated v [..., 3]."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # v' = v + 2 * q_xyz × (q_xyz × v + q_w * v)
    t = 2.0 * torch.cross(q[..., :3], v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(q[..., :3], t, dim=-1)


def reconstruct_r_ankle_z(pose_quat_global, root_trans):
    """pose_quat_global: [T, 24, 4] (x,y,z,w). root_trans: [T, 3].
    Returns R_Ankle z-coord [T].
    """
    T = pose_quat_global.shape[0]
    q_pelvis  = pose_quat_global[:, PELVIS_IDX, :]
    q_r_hip   = pose_quat_global[:, R_HIP_IDX, :]
    q_r_knee  = pose_quat_global[:, R_KNEE_IDX, :]

    # If pose_quat_global is truly GLOBAL (already accumulated), then
    #   R_Hip_pos   = root_trans + q_pelvis @ d_hip
    #   R_Knee_pos  = R_Hip_pos  + q_r_hip  @ d_knee
    #   R_Ankle_pos = R_Knee_pos + q_r_knee @ d_ankle
    d_hip   = D_HIP.expand(T, 3)
    d_knee  = D_KNEE.expand(T, 3)
    d_ankle = D_ANKLE.expand(T, 3)

    r_hip   = root_trans + quat_rotate(q_pelvis, d_hip)
    r_knee  = r_hip      + quat_rotate(q_r_hip, d_knee)
    r_ankle = r_knee     + quat_rotate(q_r_knee, d_ankle)
    return r_ankle[:, 2]  # z-component


def detect_heel_strikes(z, times):
    """z: [T] R_Ankle z-height samples. times: [T] same-length time stamps.
    Returns List[int] indices of detected heel strikes (filtered by min
    interval). Replicates HumanoidImVIC._precompute_gait_phase detector.
    """
    h = z
    local_min = (h[1:-1] < h[:-2]) & (h[1:-1] <= h[2:])
    min_indices = torch.where(local_min)[0] + 1
    if len(min_indices) == 0:
        return []
    filtered = [min_indices[0].item()]
    for i in range(1, len(min_indices)):
        if times[min_indices[i]] - times[filtered[-1]] > HS_MIN_INTERVAL_S:
            filtered.append(min_indices[i].item())
    return filtered


def analyze_clip(entry, clip_key):
    """Run HS detection for one clip. Return dict of metrics."""
    fps = float(entry.get("fps", FPS_ASSUMED))
    pose_quat_global = torch.as_tensor(np.asarray(entry["pose_quat_global"]), dtype=torch.float32)
    if isinstance(entry["root_trans_offset"], torch.Tensor):
        root_trans = entry["root_trans_offset"].to(torch.float32)
    else:
        root_trans = torch.as_tensor(np.asarray(entry["root_trans_offset"]), dtype=torch.float32)

    T = pose_quat_global.shape[0]
    duration = T / fps

    r_ankle_z_raw = reconstruct_r_ankle_z(pose_quat_global, root_trans)  # [T]

    # Resample to 1000 points like the actual precompute
    sample_times = torch.linspace(0, duration - 1e-4, NUM_SAMPLES)
    src_times = torch.arange(T, dtype=torch.float32) / fps
    # linear interpolation
    idx_f = (sample_times / (src_times[-1] if src_times[-1] > 0 else 1.0) * (T - 1)).clamp(0, T - 1)
    idx_lo = idx_f.floor().long().clamp(0, T - 1)
    idx_hi = (idx_lo + 1).clamp(0, T - 1)
    w = (idx_f - idx_lo.float()).clamp(0, 1)
    z_resampled = r_ankle_z_raw[idx_lo] * (1 - w) + r_ankle_z_raw[idx_hi] * w

    hs_indices = detect_heel_strikes(z_resampled, sample_times)
    n_hs = len(hs_indices)
    success = n_hs >= 2
    if success:
        hs_times = sample_times[hs_indices]
        strides = hs_times[1:] - hs_times[:-1]
        stride_mean = strides.mean().item()
        stride_std  = strides.std().item() if len(strides) > 1 else 0.0
    else:
        stride_mean = float("nan")
        stride_std  = float("nan")

    return {
        "clip_key": clip_key,
        "duration_s": duration,
        "T_frames": T,
        "z_min": r_ankle_z_raw.min().item(),
        "z_max": r_ankle_z_raw.max().item(),
        "z_range": (r_ankle_z_raw.max() - r_ankle_z_raw.min()).item(),
        "n_hs": n_hs,
        "hs_detect_ok": success,
        "stride_period_mean_s": stride_mean,
        "stride_period_std_s": stride_std,
    }


def main():
    pkl_path = Path("sample_data/amass_isaac_walking_primitive.pkl")
    fwd_json = Path("sample_data/amass_isaac_walking_primitive_fwd_only.json")

    print(f"Loading {pkl_path} ...")
    data = joblib.load(pkl_path)
    print(f"  {len(data)} clips in pkl")

    with open(fwd_json) as f:
        fwd_meta = json.load(f)
    print(f"  {len(fwd_meta)} forward-straight clips in metadata json")

    rows = []
    for key in fwd_meta.keys():
        if key not in data:
            print(f"  [miss] {key} not in pkl")
            continue
        meta = analyze_clip(data[key], key)
        meta["v_x_mean_mid"] = fwd_meta[key]["v_x_mean_mid"]
        meta["abs_v_x"] = abs(fwd_meta[key]["v_x_mean_mid"])
        rows.append(meta)

    # sort by ascending |v_x|
    rows.sort(key=lambda r: r["abs_v_x"])

    # Banner
    print()
    print("Per-clip heel-strike detection results (R_Ankle z, local-minima, >0.5s filter)")
    print("=" * 120)
    print(f"{'idx':>3}  {'|v_x|':>6}  {'dur_s':>6}  {'z_min':>6}  {'z_max':>6}  {'rng':>5}  "
          f"{'n_hs':>4}  {'OK':>3}  {'stride_s (m+/-s)':>18}  clip_key")
    print("-" * 120)
    for i, r in enumerate(rows):
        stride_str = (f"{r['stride_period_mean_s']:.3f}±{r['stride_period_std_s']:.3f}"
                      if r['hs_detect_ok'] else "      —       ")
        print(f"{i:>3}  {r['abs_v_x']:>6.3f}  {r['duration_s']:>6.2f}  "
              f"{r['z_min']:>6.3f}  {r['z_max']:>6.3f}  {r['z_range']:>5.3f}  "
              f"{r['n_hs']:>4}  {'Y' if r['hs_detect_ok'] else 'N':>3}  "
              f"{stride_str:>16}  {r['clip_key']}")

    print()
    # Aggregate
    n_total = len(rows)
    n_ok    = sum(r['hs_detect_ok'] for r in rows)
    n_fail  = n_total - n_ok
    print(f"Aggregate: {n_ok}/{n_total} detection success ({100*n_ok/n_total:.1f}%), "
          f"{n_fail} fallback.")

    # Per-speed-bin
    print("\nBinned by |v_x|:")
    bins = [(0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["abs_v_x"] < hi]
        if not sub:
            print(f"  [{lo:.2f}, {hi:.2f}): 0 clips")
            continue
        n_ok_b = sum(r['hs_detect_ok'] for r in sub)
        pct = 100 * n_ok_b / len(sub)
        stride_mean_b = np.nanmean([r['stride_period_mean_s'] for r in sub])
        print(f"  [{lo:.2f}, {hi:.2f}): {n_ok_b}/{len(sub)} ok ({pct:.0f}%)   "
              f"mean stride ≈ {stride_mean_b:.3f}s")

    # Sanity check — if z_range is tiny (< 0.05m), the convention is wrong
    print("\nConvention sanity check:")
    bad = [r for r in rows if r["z_range"] < 0.05]
    if bad:
        print(f"  WARN: {len(bad)} clips with R_Ankle z range < 0.05 m")
        print(f"  suggests coord convention mismatch. Typical range should be ~0.1-0.3 m.")
    else:
        typical_range = np.mean([r['z_range'] for r in rows])
        print(f"  OK: mean R_Ankle z range = {typical_range:.3f} m across clips.")


if __name__ == "__main__":
    main()

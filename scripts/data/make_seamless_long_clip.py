"""Make a 60s seamless walking clip by extracting one gait cycle from KIT_11
and looping it phase-aligned, with slerp blending at loop boundaries.

Detection: heel-strikes via root z-velocity zero-crossings (going positive).
Loop construction: take cycle from HS_i to HS_j, repeat until target duration.
Boundary smoothing: replace BLEND_W frames straddling each loop seam with a
spherical-lerp interpolation between the pose just-before and just-after the
window. Trans is left untouched (already continuous via cycle_dx accumulation).
"""
import numpy as np
import torch
import joblib

SOURCE_PKL = "sample_data/amass_isaac_walking_forward_single.pkl"
OUT_PKL = "sample_data/amass_walking_kit11_seamless_60s.pkl"
TARGET_DURATION_S = 60.0
FPS = 30
BLEND_W = 5  # frames straddling each loop seam to slerp-blend


def detect_heel_strikes(root_trans, fps=30):
    """Return frame indices of heel-strike events via root vertical
    velocity sign-change (going up after being down, i.e. low point)."""
    z = root_trans[:, 2]
    vz = np.diff(z, prepend=z[0])
    hs = np.where((vz[:-1] < 0) & (vz[1:] >= 0))[0] + 1
    return hs


def slerp_quat(q0, q1, t):
    """Spherical lerp between two batches of unit quaternions.
    q0, q1: (..., 4) in [x, y, z, w] format. t: scalar in [0, 1]."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1c = np.where(dot < 0, -q1, q1)
    dot_abs = np.abs(dot)
    # Linear fallback for very close quats (avoids div-by-zero)
    close = dot_abs > 0.9995
    linear = q0 + t * (q1c - q0)
    linear = linear / (np.linalg.norm(linear, axis=-1, keepdims=True) + 1e-12)
    # Spherical interp
    dot_clip = np.clip(dot_abs, -1.0, 1.0)
    theta_0 = np.arccos(dot_clip)
    sin_theta_0 = np.sin(theta_0)
    sin_safe = np.where(sin_theta_0 < 1e-8, 1.0, sin_theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot_clip * sin_theta / sin_safe
    s1 = sin_theta / sin_safe
    spherical = s0 * q0 + s1 * q1c
    return np.where(close, linear, spherical)


def smooth_loop_boundaries(out_pq, out_pql, out_aa, cycle_len, n_loops, blend_w):
    """Slerp-blend a window of `blend_w` frames straddling each loop seam.
    Anchors are the unmodified frames just outside the window: pre = boundary - half - 1,
    post = boundary + (blend_w - half). Modifies arrays in-place."""
    half = blend_w // 2
    n_total = out_pq.shape[0]
    seams_blended = 0
    for k in range(1, n_loops):
        boundary = k * cycle_len
        pre_idx = boundary - half - 1
        post_idx = boundary - half + blend_w
        if pre_idx < 0 or post_idx >= n_total:
            continue
        pre_pq = out_pq[pre_idx].copy()
        post_pq = out_pq[post_idx].copy()
        pre_pql = out_pql[pre_idx].copy()
        post_pql = out_pql[post_idx].copy()
        pre_aa = out_aa[pre_idx].copy()
        post_aa = out_aa[post_idx].copy()
        for i in range(blend_w):
            f = boundary - half + i
            if f < 0 or f >= n_total:
                continue
            t = (i + 1) / (blend_w + 1)
            out_pq[f] = slerp_quat(pre_pq, post_pq, t)
            out_pql[f] = slerp_quat(pre_pql, post_pql, t)
            out_aa[f] = (1 - t) * pre_aa + t * post_aa
        seams_blended += 1
    return seams_blended


def build_seamless_loop(clip, n_frames_target):
    pose_quat = np.asarray(clip["pose_quat_global"])
    pose_quat_local = np.asarray(clip["pose_quat"])
    pose_aa = np.asarray(clip["pose_aa"])
    trans = np.asarray(clip["trans_orig"])

    hs = detect_heel_strikes(trans, FPS)
    print(f"  detected {len(hs)} heel strikes at frames {hs.tolist()}")

    if len(hs) < 2:
        raise RuntimeError("not enough heel strikes for cycle extraction")

    # Full stride = same-foot HS to next same-foot HS = 2 step intervals.
    # Pick the modal stride length (most-common, robust to outliers like the
    # acceleration-from-standing first stride and deceleration-to-stop tail
    # strides which are shorter than steady walking).
    if len(hs) < 3:
        raise RuntimeError("need at least 3 heel strikes for full-stride cycle")
    stride_lengths = np.array([hs[k + 2] - hs[k] for k in range(len(hs) - 2)])
    unique, counts = np.unique(stride_lengths, return_counts=True)
    target_stride = int(unique[np.argmax(counts)])  # mode of walking strides
    mode_indices = np.where(stride_lengths == target_stride)[0]
    # Among strides matching mode, pick the median-index one (middle of walking
    # phase). This avoids both acceleration (first walking stride) and the
    # last walking stride which sits next to the deceleration phase.
    hs_a_idx = int(mode_indices[len(mode_indices) // 2])
    hs_b_idx = hs_a_idx + 2
    hs_a, hs_b = hs[hs_a_idx], hs[hs_b_idx]
    cycle_len = hs_b - hs_a
    print(f"  stride lengths (HS[k]→HS[k+2]): {stride_lengths.tolist()}")
    print(f"  modal stride={target_stride} (occurs {counts.max()}x), candidates={mode_indices.tolist()}, picked HS[{hs_a_idx}]→HS[{hs_b_idx}]")
    print(f"  cycle: frames [{hs_a}, {hs_b}) len={cycle_len} ({cycle_len/FPS:.2f}s) — full stride, steady-walk core")

    cycle_pq = pose_quat[hs_a:hs_b].copy()
    cycle_pql = pose_quat_local[hs_a:hs_b].copy()
    cycle_aa = pose_aa[hs_a:hs_b].copy()
    cycle_tr = trans[hs_a:hs_b].copy()

    cycle_tr -= cycle_tr[0]
    cycle_dx = cycle_tr[-1] - cycle_tr[0]

    n_loops = int(np.ceil(n_frames_target / cycle_len))
    out_pq = np.zeros((cycle_len * n_loops, 24, 4), dtype=cycle_pq.dtype)
    out_pql = np.zeros((cycle_len * n_loops, 24, 4), dtype=cycle_pql.dtype)
    out_aa = np.zeros((cycle_len * n_loops, 72), dtype=cycle_aa.dtype)
    out_tr = np.zeros((cycle_len * n_loops, 3), dtype=cycle_tr.dtype)

    for i in range(n_loops):
        s = i * cycle_len
        e = s + cycle_len
        out_pq[s:e] = cycle_pq
        out_pql[s:e] = cycle_pql
        out_aa[s:e] = cycle_aa
        out_tr[s:e] = cycle_tr + i * cycle_dx

    out_pq = out_pq[:n_frames_target]
    out_pql = out_pql[:n_frames_target]
    out_aa = out_aa[:n_frames_target]
    out_tr = out_tr[:n_frames_target]

    seams = smooth_loop_boundaries(out_pq, out_pql, out_aa, cycle_len, n_loops, BLEND_W)
    print(f"  slerp blend: {seams} seams smoothed (window={BLEND_W} frames each)")

    return {
        "pose_quat_global": out_pq,
        "pose_quat": out_pql,
        "pose_aa": out_aa,
        "trans_orig": out_tr,
        "root_trans_offset": torch.from_numpy(out_tr.copy()),
        "beta": clip["beta"],
        "gender": clip["gender"],
        "fps": FPS,
    }


def main():
    src = joblib.load(SOURCE_PKL)
    name, clip = next(iter(src.items()))
    print(f"Source: {name} ({clip['pose_quat_global'].shape[0]} frames)")

    n_frames = int(TARGET_DURATION_S * FPS)
    out_clip = build_seamless_loop(clip, n_frames)
    out = {f"{name}_seamless_60s": out_clip}
    joblib.dump(out, OUT_PKL)
    print(f"\nWrote {OUT_PKL} (1 clip, {n_frames} frames = {n_frames/FPS:.1f}s)")


if __name__ == "__main__":
    main()

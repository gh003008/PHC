"""
Diagnostic: empirically determine correct PiG → SMPL axis mapping.

Creates test motion libraries with controlled joint rotations:
1. T-pose (all identity) — baseline
2. Left hip flexion only (+30°)
3. Left hip abduction only (+30°)
4. Left knee flexion only (+60°)
5. Full walking motion from H5 (no axis swap, no base rotation)
6. Full walking motion from H5 (with axis swap via coordinate transform)
7. Full walking motion from H5 (original approach: direct Euler, 180° base rot)

Outputs: sample_data/debug_axis_test.pkl — load in vis_h5_motion_mj.py
"""
import os, sys, argparse
sys.path.append(os.getcwd())

import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation


SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]


def quat_multiply(q1, q2):
    """Multiply quaternions in wxyz format."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)


def local_to_global(pose_local, base_rot=None):
    """FK: local → global. base_rot applied to root if given."""
    T = pose_local.shape[0]
    pose_global = np.zeros_like(pose_local)

    if base_rot is not None:
        br = np.tile(base_rot, (T, 1))
        pose_global[:, 0] = quat_multiply(br, pose_local[:, 0])
    else:
        pose_global[:, 0] = pose_local[:, 0]

    for j in range(1, 24):
        parent = SMPL_PARENTS[j]
        pose_global[:, j] = quat_multiply(pose_global[:, parent], pose_local[:, j])

    norms = np.linalg.norm(pose_global, axis=-1, keepdims=True)
    pose_global /= np.maximum(norms, 1e-8)
    return pose_global


def make_identity_clip(T=60, fps=30):
    """T-pose: all identity quaternions, slight forward motion."""
    pose_local = np.zeros((T, 24, 4))
    pose_local[:, :, 0] = 1.0  # w=1 identity

    # 180° Y base rotation for SMPL convention
    base_rot = np.array([0.0, 0.0, 1.0, 0.0])  # Ry(180°) wxyz
    pose_global = local_to_global(pose_local, base_rot)

    trans = np.zeros((T, 3))
    trans[:, 0] = np.linspace(0, 2, T)  # forward
    trans[:, 2] = 0.93  # standing height

    return make_clip_dict(pose_global, pose_local, trans, fps)


def euler_to_quat_wxyz(euler_deg):
    """Euler XYZ (degrees) → quaternion wxyz."""
    rot = Rotation.from_euler('XYZ', np.deg2rad(euler_deg).reshape(-1, 3))
    q = rot.as_quat()  # xyzw
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def make_single_joint_clip(joint_idx, axis, angle_deg, T=60, fps=30):
    """One joint rotates, everything else identity."""
    pose_local = np.zeros((T, 24, 4))
    pose_local[:, :, 0] = 1.0

    euler = np.zeros((T, 3))
    # Ramp angle from 0 to angle_deg over the clip
    euler[:, axis] = np.linspace(0, angle_deg, T)

    pose_local[:, joint_idx] = euler_to_quat_wxyz(euler)

    base_rot = np.array([0.0, 0.0, 1.0, 0.0])
    pose_global = local_to_global(pose_local, base_rot)

    trans = np.zeros((T, 3))
    trans[:, 0] = np.linspace(0, 2, T)
    trans[:, 2] = 0.93

    return make_clip_dict(pose_global, pose_local, trans, fps)


def make_clip_dict(pose_global, pose_local, trans, fps):
    return {
        "pose_quat_global": pose_global,
        "pose_quat": pose_local.astype(np.float32),
        "trans_orig": trans,
        "root_trans_offset": torch.from_numpy(trans).double(),
        "pose_aa": np.zeros((len(trans), 72)),
        "beta": np.zeros(16),
        "gender": "neutral",
        "fps": fps,
    }


def make_h5_walking_clip(h5_path, trial_path="S001/level_100mps/lv0/trial_01",
                          fps_in=100, fps_out=30, mode="direct"):
    """Convert H5 walking data with different axis mapping modes.

    mode:
      "direct"    — PiG Euler XYZ used as SMPL Euler XYZ directly (no swap)
      "swap_xy"   — Swap PiG X↔Y: SMPL = (PiG_Y, PiG_X, PiG_Z), Euler XYZ
      "coord_transform" — Proper coordinate transform: R_smpl = P·R_pig·P^T
      "no_base_rot" — Like direct but without 180° Y base rotation
    """
    import h5py
    f = h5py.File(h5_path, "r")

    time_ms = np.array(f[f"{trial_path}/common/time"])
    T = len(time_ms)

    # Joint mapping (same as h5_to_motion_lib.py)
    VICON_MAP = {
        ("left", "hip"):      (1,  (1, 1, 1)),
        ("right", "hip"):     (2,  (1, -1, -1)),
        ("left", "knee"):     (4,  (1, 1, 1)),
        ("right", "knee"):    (5,  (1, -1, -1)),
        ("left", "ankle"):    (7,  (1, 1, 1)),
        ("right", "ankle"):   (8,  (1, -1, -1)),
        ("left", "pelvis"):   (0,  (1, 1, 1)),
        ("left", "spine"):    (3,  (1, 1, 1)),
        ("left", "thorax"):   (6,  (1, 1, 1)),
        ("left", "neck"):     (12, (1, 1, 1)),
        ("left", "head"):     (15, (1, 1, 1)),
        ("left", "shoulder"):  (16, (1, 1, 1)),
        ("right", "shoulder"): (17, (1, -1, -1)),
        ("left", "elbow"):    (18, (1, 1, 1)),
        ("right", "elbow"):   (19, (1, -1, -1)),
        ("left", "wrist"):    (20, (1, 1, 1)),
        ("right", "wrist"):   (21, (1, -1, -1)),
    }

    pose_local = np.zeros((T, 24, 4))
    pose_local[:, :, 0] = 1.0

    for (side, joint), (smpl_idx, signs) in VICON_MAP.items():
        try:
            ex = np.array(f[f"{trial_path}/mocap/angle/{side}/{joint}/x"])
            ey = np.array(f[f"{trial_path}/mocap/angle/{side}/{joint}/y"])
            ez = np.array(f[f"{trial_path}/mocap/angle/{side}/{joint}/z"])
        except KeyError:
            continue

        euler_deg = np.stack([ex, ey, ez], axis=-1)
        for i in range(3):
            nans = np.isnan(euler_deg[:, i])
            if nans.any() and not nans.all():
                euler_deg[nans, i] = np.interp(
                    np.where(nans)[0], np.where(~nans)[0], euler_deg[~nans, i])

        # Apply axis signs
        euler_deg[:, 0] *= signs[0]
        euler_deg[:, 1] *= signs[1]
        euler_deg[:, 2] *= signs[2]

        euler_rad = np.deg2rad(euler_deg)

        if mode == "direct" or mode == "no_base_rot":
            # PiG Euler XYZ used directly as SMPL Euler XYZ
            rot = Rotation.from_euler('XYZ', euler_rad)

        elif mode == "swap_xy":
            # Swap: SMPL = (PiG_Y, PiG_X, PiG_Z)
            swapped = np.stack([euler_rad[:, 1], euler_rad[:, 0], euler_rad[:, 2]], axis=-1)
            rot = Rotation.from_euler('XYZ', swapped)

        elif mode == "coord_transform":
            # Proper coordinate transform: R_smpl = P · R_pig · P^T
            # where P swaps X↔Y axes
            # This gives: Rotation.from_euler('YXZ', [-flex, -abd, -rot])
            # flex=euler[0], abd=euler[1], rot=euler[2] in PiG space
            transformed = np.stack([-euler_rad[:, 0], -euler_rad[:, 1], -euler_rad[:, 2]], axis=-1)
            rot = Rotation.from_euler('YXZ', transformed)

        elif mode == "pig_as_yxz":
            # What if PiG is actually YXZ (flex around Y, abd around X)?
            # Then we can use PiG angles directly with YXZ convention
            rot = Rotation.from_euler('YXZ', euler_rad)

        q = rot.as_quat()  # xyzw
        pose_local[:, smpl_idx] = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])

    f.close()

    # Resample
    T_out = int(T * fps_out / fps_in)
    t_in = np.arange(T) / fps_in
    t_out = np.arange(T_out) / fps_out
    pose_resampled = np.zeros((T_out, 24, 4))
    for j in range(24):
        for c in range(4):
            pose_resampled[:, j, c] = np.interp(t_out, t_in, pose_local[:, j, c])
        norms = np.linalg.norm(pose_resampled[:, j], axis=-1, keepdims=True)
        pose_resampled[:, j] /= np.maximum(norms, 1e-8)
    pose_local = pose_resampled

    # FK
    if mode == "no_base_rot":
        base_rot = None
    else:
        base_rot = np.array([0.0, 0.0, 1.0, 0.0])  # Ry(180°)
    pose_global = local_to_global(pose_local, base_rot)

    # Simple translation
    trans = np.zeros((T_out, 3))
    trans[:, 0] = np.linspace(0, 5, T_out)
    trans[:, 2] = 0.93

    return make_clip_dict(pose_global, pose_local, trans, fps_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/combined_data_from_csv.h5")
    parser.add_argument("--output", default="sample_data/debug_axis_test.pkl")
    parser.add_argument("--trial", default="S001/level_100mps/lv0/trial_01")
    args = parser.parse_args()

    clips = {}

    # 1. T-pose (identity)
    print("Creating T-pose clip...")
    clips["00_tpose"] = make_identity_clip()

    # 2. L_Hip X-axis rotation (+30°) — in SMPL, X = abduction
    print("Creating L_Hip X-rotation clip...")
    clips["01_lhip_X30"] = make_single_joint_clip(1, 0, 30)

    # 3. L_Hip Y-axis rotation (+30°) — in SMPL, Y = flexion
    print("Creating L_Hip Y-rotation clip...")
    clips["02_lhip_Y30"] = make_single_joint_clip(1, 1, 30)

    # 4. L_Hip Z-axis rotation (+30°)
    print("Creating L_Hip Z-rotation clip...")
    clips["03_lhip_Z30"] = make_single_joint_clip(1, 2, 30)

    # 5. L_Knee Y-axis (+60°) — flexion in SMPL
    print("Creating L_Knee Y-rotation clip...")
    clips["04_lknee_Y60"] = make_single_joint_clip(4, 1, 60)

    # 6-9. H5 walking with different mappings
    if os.path.exists(args.h5):
        for mode in ["direct", "swap_xy", "coord_transform", "no_base_rot"]:
            print(f"Creating H5 walking clip (mode={mode})...")
            clips[f"05_h5_{mode}"] = make_h5_walking_clip(
                args.h5, args.trial, mode=mode
            )
    else:
        print(f"H5 file not found: {args.h5}, skipping H5 clips")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(clips, args.output)
    print(f"\nSaved {len(clips)} clips to {args.output}")
    print("Clip names:")
    for name in sorted(clips.keys()):
        T = clips[name]["pose_quat_global"].shape[0]
        print(f"  {name}: {T} frames")

    # Print pelvis height for each H5 mode
    print("\n=== Pelvis height check (frame 30 for H5 clips) ===")
    for name in sorted(clips.keys()):
        if "h5" in name:
            pg = clips[name]["pose_quat_global"]
            t = clips[name]["trans_orig"]
            # The z translation is our set height
            print(f"  {name}: trans_z={t[30, 2]:.3f}")


if __name__ == "__main__":
    main()

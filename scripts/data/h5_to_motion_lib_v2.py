"""
H5 (Vicon Plug-in Gait) → PHC Motion Library PKL 변환 스크립트

Vicon Plug-in Gait의 Euler XYZ(Cardan) 관절각(degree)을
PHC가 요구하는 SMPL 24-joint quaternion + root translation 형식으로 변환.

Usage:
    python scripts/data/h5_to_motion_lib_v2.py \
        --h5 data/combined_data_from_csv.h5 \
        --output sample_data/h5_motion_library.pkl \
        --assist_level lv0 \
        --fps_out 30

    # 특정 subject만
    python scripts/data/h5_to_motion_lib_v2.py \
        --h5 data/combined_data_from_csv.h5 \
        --output sample_data/h5_motion_library.pkl \
        --subjects S001 S002

    # 특정 task만
    python scripts/data/h5_to_motion_lib_v2.py \
        --h5 data/combined_data_from_csv.h5 \
        --output sample_data/h5_motion_library.pkl \
        --tasks level_100mps level_125mps
"""

import argparse
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# Reorder bone-order (AMASS / our mocap mapping) → mujoco-order (PHC skeleton tree)
SMPL_2_MUJOCO = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
# Standard "upright" rotation that maps AMASS Y-up → PHC Z-up + face +X
UPRIGHT_QUAT_XYZW = np.array([0.5, 0.5, 0.5, 0.5])
SMPL_HUMANOID_MJCF = "phc/data/assets/mjcf/smpl_humanoid.xml"
_SK_TREE_CACHE = None

def get_skeleton_tree():
    global _SK_TREE_CACHE
    if _SK_TREE_CACHE is None:
        _SK_TREE_CACHE = SkeletonTree.from_mjcf(SMPL_HUMANOID_MJCF)
    return _SK_TREE_CACHE


# ============================================================
# SMPL 24-joint order (from smpl_sim)
# ============================================================
SMPL_JOINT_NAMES = [
    "Pelvis",       # 0
    "L_Hip",        # 1
    "R_Hip",        # 2
    "Torso",        # 3  (= Spine1 / lumbar)
    "L_Knee",       # 4
    "R_Knee",       # 5
    "Spine",        # 6  (= Spine2 / thoracolumbar)
    "L_Ankle",      # 7
    "R_Ankle",      # 8
    "Chest",        # 9  (= Spine3 / upper thorax)
    "L_Toe",        # 10
    "R_Toe",        # 11
    "Neck",         # 12
    "L_Thorax",     # 13 (= L_Collar / clavicle)
    "R_Thorax",     # 14 (= R_Collar / clavicle)
    "Head",         # 15
    "L_Shoulder",   # 16
    "R_Shoulder",   # 17
    "L_Elbow",      # 18
    "R_Elbow",      # 19
    "L_Wrist",      # 20
    "R_Wrist",      # 21
    "L_Hand",       # 22
    "R_Hand",       # 23
]

# ============================================================
# Vicon Plug-in Gait → SMPL joint mapping
# ============================================================
# Based on data/compare_smpl_human_suit.py (verified correct visualization).
#
# Key principles:
# 1. Lower body & spine: sagittal flexion ONLY (x channel), drop y/z.
# 2. Hip flexion sign is negated for SMPL convention.
# 3. Shoulders/elbows handled specially (SMPL T-pose arms along +x).
# 4. Right-side mirroring: negate y,z in get_mocap_euler (not in map).
# 5. Euler convention: extrinsic "xyz" (lowercase), NOT intrinsic "XYZ".
#
# Mapping: smpl_index → (side, h5_joint, axis_mask, sign_vec)
# axis_mask: (1,0,0) = use only x (sagittal flexion)
# sign_vec: per-axis sign flips applied before Euler conversion
# side="any": average left+right (for midline joints)
VICON_TO_SMPL_MAP = {
    # Lower body — sagittal flexion only
    1:  ("left",  "hip",    (1, 0, 0), (-1, +1, +1)),   # L_Hip: negate flexion
    2:  ("right", "hip",    (1, 0, 0), (-1, +1, +1)),   # R_Hip: negate flexion
    4:  ("left",  "knee",   (1, 0, 0), (+1, +1, +1)),   # L_Knee
    5:  ("right", "knee",   (1, 0, 0), (+1, +1, +1)),   # R_Knee
    7:  ("left",  "ankle",  (1, 0, 0), (+1, +1, +1)),   # L_Ankle
    8:  ("right", "ankle",  (1, 0, 0), (+1, +1, +1)),   # R_Ankle
    # Spine — sagittal flexion only, average L+R
    3:  ("any",   "spine",  (1, 0, 0), (+1, +1, +1)),   # Torso (Spine1)
    6:  ("any",   "thorax", (1, 0, 0), (+1, +1, +1)),   # Spine (Spine2)
    # Shoulders and elbows are NOT in this table — handled specially below.
    # Neck, head, wrist, pelvis: left as identity (not enough reliable data).
}

# Shoulder A-pose offset: rotate arms from T-pose down toward body.
# In SMPL local shoulder frame, rotation about local Z brings arm from T-pose.
SHOULDER_DOWN_RAD = np.deg2rad(85.0)


def get_mocap_euler(f, trial_path, side, joint, frame_slice, mask, sign):
    """Read mocap Euler angles with axis masking, sign flips, and R-side mirroring.

    Based on compare_smpl_human_suit.py:get_mocap_eul.
    Right-side mirroring: negate y,z for right side (biomech → SMPL symmetry).
    side="any": average left+right (for midline joints like spine).
    """
    T = frame_slice
    eul = np.zeros((T, 3), dtype=np.float64)
    for k, ax in enumerate("xyz"):
        if not mask[k]:
            continue
        if side == "any":
            try:
                vL = np.array(f[f"{trial_path}/mocap/angle/left/{joint}/{ax}"])
                vR = np.array(f[f"{trial_path}/mocap/angle/right/{joint}/{ax}"])
                vals = 0.5 * (vL + vR)
            except KeyError:
                continue
        else:
            try:
                vals = np.array(f[f"{trial_path}/mocap/angle/{side}/{joint}/{ax}"])
            except KeyError:
                continue
            # Right-side mirror: negate off-axis (y, z)
            if side == "right" and ax in ("y", "z"):
                vals = -vals
        # Handle NaN
        nans = np.isnan(vals)
        if nans.any() and not nans.all():
            vals[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), vals[~nans])
        elif nans.all():
            vals[:] = 0.0
        eul[:, k] = sign[k] * vals
    return eul


def euler_xyz_deg_to_axis_angle(euler_deg):
    """Convert Euler xyz (extrinsic, degrees) → axis-angle (T, 3).

    Uses lowercase "xyz" (extrinsic) per compare_smpl_human_suit.py convention.
    """
    rot = Rotation.from_euler("xyz", euler_deg, degrees=True)
    return rot.as_rotvec().astype(np.float64)


def compute_root_translation(f, trial_path, fps_in, fps_out):
    """Compute root translation from CoM and treadmill data.

    Treadmill walking: subject walks in place, but we need to generate
    forward translation. Use treadmill speed integrated over time for
    the forward (x) component. CoM y and z used directly.

    Args:
        f: h5py File
        trial_path: path to trial group
        fps_in: input Hz (100)
        fps_out: output Hz (30)

    Returns:
        (T_out, 3) root translation in meters
    """
    time_ms = np.array(f[f"{trial_path}/common/time"])
    time_s = (time_ms - time_ms[0]) / 1000.0
    T = len(time_s)

    # CoM in mm → meters (if available)
    # Per h5_file_spec.md: x=anterior-posterior, y=medio-lateral, z=vertical
    has_com = f"{trial_path}/mocap/com/x" in f
    if has_com:
        com_ap = np.array(f[f"{trial_path}/mocap/com/x"]) / 1000.0   # anterior-posterior
        com_ml = np.array(f[f"{trial_path}/mocap/com/y"]) / 1000.0   # medio-lateral
        com_vert = np.array(f[f"{trial_path}/mocap/com/z"]) / 1000.0  # vertical
        # Interpolate NaN values
        for arr in [com_ap, com_ml, com_vert]:
            mask = np.isnan(arr)
            if mask.any() and not mask.all():
                arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
            elif mask.all():
                arr[:] = 0.95 if arr is com_vert else 0.0
    else:
        # Fallback: use constant height, zero lateral
        com_ap = np.zeros(T)
        com_ml = np.zeros(T)
        com_vert = np.full(T, 0.95)  # ~950mm default standing height

    # Treadmill speed for forward translation
    try:
        speed_l = np.array(f[f"{trial_path}/treadmill/left/speed_leftbelt"])
        speed_r = np.array(f[f"{trial_path}/treadmill/right/speed_rightbelt"])
        belt_speed = (speed_l + speed_r) / 2.0  # m/s
    except KeyError:
        belt_speed = np.zeros(T)

    # Integrate belt speed for forward displacement
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.01
    forward_disp = np.cumsum(belt_speed * dt)  # meters

    # Root translation in Y-up frame: [x, y, z]
    # Y = up (height), Z = forward (facing +Z in T-pose), X = right
    # The upright rotation applied later will convert Y-up → Z-up.
    trans = np.zeros((T, 3))
    trans[:, 2] = forward_disp              # +Z = forward (character faces +Z in Y-up)
    trans[:, 0] = -(com_ml - com_ml[0])     # X = right (negate medio-lateral)
    trans[:, 1] = com_vert                  # Y = up (height, absolute ~0.9-1.0m)

    # Resample to output fps
    if fps_in != fps_out:
        t_in = np.arange(T) / fps_in
        T_out = int(T * fps_out / fps_in)
        t_out = np.arange(T_out) / fps_out
        trans_out = np.zeros((T_out, 3))
        for i in range(3):
            trans_out[:, i] = np.interp(t_out, t_in, trans[:, i])
        return trans_out
    return trans


def convert_trial(f, trial_path, fps_in=100, fps_out=30):
    """Convert a single H5 trial to PHC motion library format.

    Returns:
        dict with pose_quat_global, pose_quat, trans_orig, pose_aa, beta, gender, fps
        or None if conversion fails
    """
    try:
        time_ms = np.array(f[f"{trial_path}/common/time"])
    except KeyError:
        print(f"  SKIP (no common/time): {trial_path}")
        return None

    # Check if mocap angle data exists
    if f"{trial_path}/mocap/angle/left/hip/x" not in f:
        print(f"  SKIP (no mocap/angle data): {trial_path}")
        return None

    T = len(time_ms)

    # Build per-frame axis-angle pose (24, 3) following compare_smpl_human_suit.py
    pose_aa_local = np.zeros((T, 24, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Root (Pelvis) orientation — SMPL convention:
    # AMASS SMPL root rotation encodes the pelvis's 3D orientation in world.
    # For a person standing upright facing +X in Z-up world, the SMPL root
    # rotation must map body Y→world Z (up), body Z→world X (fwd),
    # body X→world Y (right). This IS the upright rotation [0.5,0.5,0.5,0.5].
    #
    # After the pipeline applies * upright_inv (step 4), this cancels out:
    #   upright * upright_inv = identity → humanoid stands upright in Isaac Gym.
    #
    # AMASS data naturally has this in the root pose_aa (from SMPL fitting).
    # For H5 data, we read Vicon PiG pelvis angles (tilt, obliquity, rotation)
    # and compose them with the upright rotation so that per-frame pelvis
    # dynamics are preserved.
    #
    # Vicon PiG pelvis angles (lab frame, Z-up after conversion):
    #   tilt (x):     anterior(+)/posterior(-), rotation about medio-lateral axis
    #   obliquity (y): left-up(+), rotation about antero-posterior axis
    #   rotation (z):  internal/left-forward(+), rotation about vertical axis
    #
    # In Z-up world (forward=+X, right=+Y, up=+Z):
    #   tilt     → Ry(-angle): positive anterior tilt leans upper body forward
    #   obliquity → Rx(angle): positive left-up tilts pelvis laterally
    #   rotation  → Rz(angle): positive rotates left side forward
    # ------------------------------------------------------------------
    upright = sRot.from_quat([0.5, 0.5, 0.5, 0.5])

    # Read Vicon pelvis angles (use "left" convention)
    pelvis_tilt = np.zeros(T, dtype=np.float64)
    pelvis_obliq = np.zeros(T, dtype=np.float64)
    pelvis_rot = np.zeros(T, dtype=np.float64)
    try:
        pelvis_tilt = np.array(f[f"{trial_path}/mocap/angle/left/pelvis/x"], dtype=np.float64)
        pelvis_obliq = np.array(f[f"{trial_path}/mocap/angle/left/pelvis/y"], dtype=np.float64)
        pelvis_rot = np.array(f[f"{trial_path}/mocap/angle/left/pelvis/z"], dtype=np.float64)
        # NaN interpolation
        for arr in [pelvis_tilt, pelvis_obliq, pelvis_rot]:
            nn = np.isnan(arr)
            if nn.any() and not nn.all():
                arr[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), arr[~nn])
            elif nn.all():
                arr[:] = 0.0
    except KeyError:
        pass  # No pelvis data available, keep zeros (pure upright)

    for t in range(T):
        # Pelvis rotation in Z-up world frame (small angles, ~±10°)
        R_pelvis_world = sRot.from_euler('XYZ', [
            np.deg2rad(pelvis_obliq[t]),    # Rx: lateral tilt
            np.deg2rad(pelvis_tilt[t]),       # Ry: anterior tilt(+) → +Ry → lean forward in Z-up world
            np.deg2rad(pelvis_rot[t]),       # Rz: yaw rotation
        ])
        # Compose: body_frame_root = R_pelvis_world * upright
        # After pipeline's * upright_inv: R_pelvis_world * upright * upright_inv = R_pelvis_world
        pose_aa_local[t, 0, :] = (R_pelvis_world * upright).as_rotvec()

    # ------------------------------------------------------------------
    # Joint authoring: local rotations in SMPL body frame.
    # The skeleton tree (smpl_humanoid.xml) is Y-up:
    #   Y = up (pelvis→head), X = right, Z = forward (facing +Z in T-pose)
    #   Legs extend in -Y, arms extend in ±X.
    #
    # Rotation axes for biomech sagittal flexion in body frame:
    #   - Legs / spine: about local X (rotating limb in Y-Z sagittal plane)
    #   - Shoulder offset (arms T-pose → arms down): about local Z
    #   - Shoulder swing: about local X, applied AFTER offset
    #   - Elbow flexion: about local Y
    # ------------------------------------------------------------------

    def _read_side(path):
        """Read and NaN-interpolate a 1-D channel, fall back to zeros."""
        try:
            v = np.array(f[path])
        except KeyError:
            return np.zeros(T, dtype=np.float64)
        v = v.astype(np.float64)
        nn = np.isnan(v)
        if nn.any() and not nn.all():
            v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
        elif nn.all():
            v[:] = 0.0
        return v

    # 1. Legs + spine — sagittal flex as rotation about world X (Y-up frame)
    #    In Y-up: legs hang in -Y, sagittal flex rotates about X axis
    LEG_SPINE_MAP = {
        # smpl_idx : (side, joint, sign)
        1: ("left",  "hip",    -1.0),   # L_Hip
        2: ("right", "hip",    -1.0),   # R_Hip
        4: ("left",  "knee",   +1.0),   # L_Knee
        5: ("right", "knee",   +1.0),   # R_Knee
        7: ("left",  "ankle",  +1.0),   # L_Ankle
        8: ("right", "ankle",  +1.0),   # R_Ankle
        3: ("any",   "spine",  +1.0),   # Torso (avg L+R)
        6: ("any",   "thorax", +1.0),   # Spine (avg L+R)
    }
    mapped_count = 0
    for smpl_idx, (side, joint, sign) in LEG_SPINE_MAP.items():
        if side == "any":
            vL = _read_side(f"{trial_path}/mocap/angle/left/{joint}/x")
            vR = _read_side(f"{trial_path}/mocap/angle/right/{joint}/x")
            vals = 0.5 * (vL + vR)
        else:
            vals = _read_side(f"{trial_path}/mocap/angle/{side}/{joint}/x")
        pose_aa_local[:, smpl_idx, 0] = sign * np.deg2rad(vals)   # Rx (Y-up sagittal)
        mapped_count += 1

    if mapped_count < 4:
        print(f"  SKIP (only {mapped_count} joints mapped): {trial_path}")
        return None

    # 2. Shoulders — A-pose offset about world Z (arms ±X → -Y in Y-up), then Rx swing.
    #   In Y-up frame: arms extend along ±X in T-pose. Rotating about Z brings arms
    #   from ±X toward -Y (hanging down). Then Rx swings arm forward (+Z).
    for smpl_idx, side, off_sign in [
        (16, "left",  -1.0),   # L_Shoulder: Rz(-85°) takes +X → -Y (arm down)
        (17, "right", +1.0),   # R_Shoulder: Rz(+85°) takes -X → -Y (arm down)
    ]:
        vals = _read_side(f"{trial_path}/mocap/angle/{side}/shoulder/x")
        R_off = Rotation.from_rotvec([0.0, 0.0, off_sign * SHOULDER_DOWN_RAD])
        swing_sign = -1.0   # ARMFIX: flipped from +1.0. Rx(+θ) takes -Y→-Z (backward), so PiG +shoulder_x (arm fwd) needs Rx(-)
        for t in range(T):
            R_swing = Rotation.from_rotvec([swing_sign * np.deg2rad(vals[t]), 0.0, 0.0])
            pose_aa_local[t, smpl_idx] = (R_swing * R_off).as_rotvec()

    # 3. Elbows — flexion about world Y (in Y-up frame, bends forearm from hanging
    #   position -Y toward +Z/forward). After shoulder A-pose, elbow is a child joint.
    for smpl_idx, side, sgn in [
        (18, "left",  -1.0),   # L_Elbow
        (19, "right", +1.0),   # R_Elbow
    ]:
        vals = _read_side(f"{trial_path}/mocap/angle/{side}/elbow/x")
        for t in range(T):
            pose_aa_local[t, smpl_idx, 1] = sgn * np.deg2rad(vals[t])   # Ry

    # Resample axis-angle to output fps (we recompute quaternions from pose_aa_local later)
    if fps_in != fps_out:
        T_out = int(T * fps_out / fps_in)
        t_in = np.arange(T) / fps_in
        t_out = np.arange(T_out) / fps_out
        aa_resampled = np.zeros((T_out, 24, 3), dtype=np.float64)
        for j in range(24):
            for c in range(3):
                aa_resampled[:, j, c] = np.interp(t_out, t_in, pose_aa_local[:, j, c])
        pose_aa_local = aa_resampled
        T_out_final = T_out
    else:
        T_out_final = T

    # Compute root translation
    trans = compute_root_translation(f, trial_path, fps_in, fps_out)
    # Ensure same length
    min_len = min(len(trans), len(pose_aa_local))
    trans = trans[:min_len]
    pose_aa_local = pose_aa_local[:min_len]
    T_out_final = min_len

    # ----- Build PHC-format motion using poselib (mirrors convert_amass_isaac.py) -----
    # 1. Local axis-angle (BONE order) → local quaternion xyzw (BONE order)
    pose_quat_bone_xyzw = sRot.from_rotvec(pose_aa_local.reshape(-1, 3)).as_quat().reshape(T_out_final, 24, 4)

    # 2. Reorder BONE → MUJOCO (PHC's skeleton tree uses mujoco order)
    pose_quat_mj = pose_quat_bone_xyzw[:, SMPL_2_MUJOCO]

    # 3. Build initial SkeletonState with local rotations
    sk_tree = get_skeleton_tree()
    root_trans_offset_t = torch.from_numpy(trans).double() + sk_tree.local_translation[0]

    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(pose_quat_mj).float(),
        root_trans_offset_t,
        is_local=True,
    )

    # 4. Apply upright correction (convert_amass_isaac.py pipeline):
    #    a) RIGHT-multiply global rotations by upright_inv (same as AMASS line 112).
    #    b) Rotate root translation from Y-up → Z-up using upright.
    #       AMASS trans_orig is already Z-up (from SMPL world coordinates).
    #       Our H5 trans is Y-up, so we must convert: upright.apply([x,y,z]) = [z,x,y]
    #       which maps [right, height, forward] → [forward, right, height] in Z-up.
    upright = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    upright_inv = upright.inv()
    pose_quat_global = (
        sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * upright_inv
    ).as_quat().reshape(T_out_final, -1, 4)

    # Root translation: rotate from Y-up to Z-up
    trans = upright.apply(trans)  # trans_orig in Z-up
    root_trans_offset_t = torch.from_numpy(trans).double() + sk_tree.local_translation[0]

    # 5. Rebuild skeleton state from corrected global rotations to get local rotations
    sk_state_corrected = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(pose_quat_global).float(),
        root_trans_offset_t.float(),
        is_local=False,
    )
    pose_quat_local_mj = sk_state_corrected.local_rotation.numpy()

    # pose_aa: store original Y-up frame (same as AMASS convention)
    pose_aa_flat = pose_aa_local.reshape(T_out_final, 72)

    # Default SMPL shape params (neutral body)
    beta = np.zeros(16, dtype=np.float64)

    result = {
        "pose_quat_global": pose_quat_global,           # MUJOCO order, xyzw
        "pose_quat": pose_quat_local_mj,                # MUJOCO order, xyzw, local
        "trans_orig": trans,
        "root_trans_offset": root_trans_offset_t,
        "pose_aa": pose_aa_flat,                        # BONE order (24*3=72), upright-corrected
        "beta": beta,
        "gender": "neutral",
        "fps": fps_out,
    }
    return result


def extract_command_from_path(task, level, f, trial_path):
    """Extract command metadata from task name and treadmill data."""
    # Speed from task name
    speed_map = {
        "level_075mps": 0.75,
        "level_100mps": 1.0,
        "level_125mps": 1.25,
    }
    v_cmd = speed_map.get(task, None)

    # If not in map, try to get from treadmill
    if v_cmd is None:
        try:
            speed = np.array(f[f"{trial_path}/treadmill/left/speed_leftbelt"])
            nonzero = speed > 0.01
            if nonzero.any():
                v_cmd = float(speed[nonzero].mean())
            else:
                v_cmd = 0.0
        except KeyError:
            v_cmd = 0.0

    # Incline from task name
    incline = 0.0
    if "incline" in task:
        incline = 10.0
    elif "decline" in task:
        incline = -5.0

    return {
        "v_cmd": v_cmd,
        "incline": incline,
        "task_type": task,
        "assist_level": level,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert H5 to PHC motion library")
    parser.add_argument("--h5", required=True, help="Input H5 file")
    parser.add_argument("--output", default="sample_data/h5_motion_library.pkl",
                        help="Output pkl file")
    parser.add_argument("--assist_level", default="lv0",
                        help="Filter assist level (default: lv0)")
    parser.add_argument("--fps_out", type=int, default=30,
                        help="Output FPS (default: 30)")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Filter subjects (default: all)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Filter tasks (default: all except accel_sine)")
    parser.add_argument("--clip_duration", type=float, default=5.0,
                        help="Split long trials into clips of this duration (seconds)")
    parser.add_argument("--clip_overlap", type=float, default=1.0,
                        help="Overlap between clips (seconds)")
    parser.add_argument("--no_split", action="store_true",
                        help="Don't split into clips, keep full trials")
    parser.add_argument("--trim_start", type=float, default=0.0,
                        help="Trim this many seconds from the start of each trial (skip warmup)")
    # TODO(Tasks 2-4): these three flags are currently no-ops. Tasks 2/3/4 will wire them
    # into baseline subtraction, spine_3axis mapping, and upper-body (neck/head) mapping.
    parser.add_argument("--baseline_s", type=float, default=0.0,
                        help="Subtract standing-mean baseline computed from first N seconds of each trial. 0 = off.")
    parser.add_argument("--spine_3axis", action="store_true",
                        help="Map all three PiG spine/thorax axes (x,y,z) to SMPL instead of sagittal only.")
    parser.add_argument("--upper_body", action="store_true",
                        help="Map PiG neck→SMPL joint 12 and head→SMPL joint 15 (instead of leaving them identity).")
    args = parser.parse_args()

    f = h5py.File(args.h5, "r")
    fps_in = 100  # H5 data is 100 Hz

    # Determine which tasks to skip
    skip_tasks = {"accel_sine"}  # Different structure

    motion_lib = {}
    metadata = {}
    total_clips = 0
    skipped = 0

    subjects = args.subjects if args.subjects else sorted(f.keys())

    for subj in subjects:
        if subj not in f:
            print(f"Subject {subj} not found, skipping")
            continue

        for task in sorted(f[subj].keys()):
            if task in skip_tasks:
                continue
            if args.tasks and task not in args.tasks:
                continue

            for level in sorted(f[subj][task].keys()):
                # Filter assist level
                if args.assist_level and args.assist_level != "all":
                    if "lv" in level and level != args.assist_level:
                        continue
                    if "trial" in level and args.assist_level != "lv0":
                        # trial_01 etc. under accel_sine don't have lv prefix
                        continue

                grp = f[f"{subj}/{task}/{level}"]
                trials = sorted(grp.keys())

                for trial in trials:
                    trial_path = f"{subj}/{task}/{level}/{trial}"
                    print(f"Processing: {trial_path}")

                    result = convert_trial(f, trial_path, fps_in, args.fps_out)
                    if result is None:
                        skipped += 1
                        continue

                    cmd = extract_command_from_path(task, level, f, trial_path)
                    T = result["pose_quat_global"].shape[0]

                    # Trim warmup frames from start
                    trim_frames = int(args.trim_start * args.fps_out)
                    if trim_frames > 0 and T > trim_frames + int(args.clip_duration * args.fps_out):
                        sk_tree_lt = get_skeleton_tree().local_translation[0]
                        result["pose_quat_global"] = result["pose_quat_global"][trim_frames:]
                        result["pose_quat"] = result["pose_quat"][trim_frames:]
                        result["trans_orig"] = result["trans_orig"][trim_frames:]
                        result["root_trans_offset"] = torch.from_numpy(
                            result["trans_orig"]).double() + sk_tree_lt
                        result["pose_aa"] = result["pose_aa"][trim_frames:]
                        T = result["pose_quat_global"].shape[0]
                        print(f"  Trimmed {args.trim_start}s ({trim_frames} frames), {T} frames remain")

                    if args.no_split or T <= int(args.clip_duration * args.fps_out * 1.5):
                        # Keep as single clip
                        clip_name = f"{subj}_{task}_{level}_{trial}"
                        motion_lib[clip_name] = result
                        metadata[clip_name] = {
                            "subject_id": subj,
                            **cmd,
                            "trial": trial,
                            "frames": T,
                        }
                        total_clips += 1
                    else:
                        # Split into overlapping clips
                        clip_frames = int(args.clip_duration * args.fps_out)
                        overlap_frames = int(args.clip_overlap * args.fps_out)
                        step = clip_frames - overlap_frames

                        idx = 0
                        clip_num = 0
                        while idx + clip_frames <= T:
                            clip_name = f"{subj}_{task}_{level}_{trial}_c{clip_num:03d}"

                            sk_tree_local_translation = get_skeleton_tree().local_translation[0]
                            clip_data = {
                                "pose_quat_global": result["pose_quat_global"][idx:idx+clip_frames],
                                "pose_quat": result["pose_quat"][idx:idx+clip_frames],
                                "trans_orig": result["trans_orig"][idx:idx+clip_frames],
                                "root_trans_offset": torch.from_numpy(
                                    result["trans_orig"][idx:idx+clip_frames]
                                ).double() + sk_tree_local_translation,
                                "pose_aa": result["pose_aa"][idx:idx+clip_frames],
                                "beta": result["beta"],
                                "gender": result["gender"],
                                "fps": result["fps"],
                            }
                            # Reset translation to start from origin for each clip
                            # Z-up frame after upright rotation: X=forward, Y=right, Z=up
                            t0 = clip_data["trans_orig"][0].copy()
                            clip_data["trans_orig"] = clip_data["trans_orig"] - t0
                            clip_data["trans_orig"][:, 2] += t0[2]  # Keep absolute height (Z-up)
                            clip_data["root_trans_offset"] = (
                                torch.from_numpy(clip_data["trans_orig"]).double()
                                + sk_tree_local_translation
                            )

                            motion_lib[clip_name] = clip_data
                            metadata[clip_name] = {
                                "subject_id": subj,
                                **cmd,
                                "trial": trial,
                                "clip_num": clip_num,
                                "frames": clip_frames,
                                "start_frame": idx,
                            }
                            total_clips += 1
                            clip_num += 1
                            idx += step

    f.close()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_lib, output_path)

    # Save metadata separately
    meta_path = output_path.with_suffix(".meta.pkl")
    joblib.dump(metadata, meta_path)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Total clips: {total_clips}")
    print(f"  Skipped trials: {skipped}")
    print(f"  Output: {output_path}")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")

    # Print summary of commands
    if metadata:
        v_cmds = sorted(set(m["v_cmd"] for m in metadata.values()))
        tasks = sorted(set(m["task_type"] for m in metadata.values()))
        subjects = sorted(set(m["subject_id"] for m in metadata.values()))
        print(f"  Subjects: {subjects}")
        print(f"  Tasks: {tasks}")
        print(f"  Velocities: {v_cmds}")
        print(f"  Clips per subject:")
        for s in subjects:
            n = sum(1 for m in metadata.values() if m["subject_id"] == s)
            print(f"    {s}: {n} clips")


if __name__ == "__main__":
    main()

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
# 5. Euler convention: extrinsic "xyz" used in get_mocap_euler / euler_xyz_deg_to_axis_angle.
#    Tasks 3+4 (spine/neck/head) use "XYZ" (scipy uppercase = extrinsic) directly.
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
    # Neck, head: mapped conditionally via --upper_body (UPPER_BODY_MAP in convert_trial).
    # Wrist, pelvis: left as identity (not enough reliable data).
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


def read_grf(f, trial_path, side):
    """Read vertical GRF for one foot. Returns [T] Newtons, NaN-interpolated, at input fps.

    Returns None if the H5 lacks GRF data (legacy trials).
    side: 'left' or 'right'.

    H5 forceplate structure (discovered via introspection):
      forceplate/grf/{side}/z  — vertical GRF (Z=up in PiG lab frame convention)
      forceplate/cop/{side}/x,y,z — Center of Pressure position

    Legacy treadmill-only paths are also checked as fallback.
    """
    candidates = [
        # Primary: forceplate vertical GRF (Z = up in PiG convention)
        f"{trial_path}/forceplate/grf/{side}/z",
        # Legacy treadmill paths (may not exist)
        f"{trial_path}/treadmill/{side}/grf_z",
        f"{trial_path}/treadmill/{side}/grf_vertical",
        f"{trial_path}/forceplate/{side}/grf_z",
        # Fallback: some datasets may label vertical as y
        f"{trial_path}/treadmill/{side}/grf_y",
        f"{trial_path}/forceplate/{side}/grf_y",
    ]
    for path in candidates:
        try:
            v = np.array(f[path], dtype=np.float64)
            nn = np.isnan(v)
            if nn.any() and not nn.all():
                v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
            elif nn.all():
                return np.zeros(len(v), dtype=np.float64)
            return np.abs(v)   # magnitude, sign-agnostic
        except KeyError:
            continue
    return None


def fk_feet(skeleton_tree, pose_quat_global_xyzw, trans):
    """Forward kinematics: compute ankle world positions and pelvis-local offsets.

    Uses poselib's SkeletonState (already a project dependency).

    Args:
        skeleton_tree: poselib SkeletonTree.
        pose_quat_global_xyzw: [T, 24, 4] global-frame quaternions (xyzw).
        trans: [T, 3] pelvis world position.

    Returns:
        foot_L_world: [T, 3]  L_Ankle world pos (SMPL index 7 in MUJOCO order)
        foot_R_world: [T, 3]  R_Ankle world pos (SMPL index 8)
        foot_offset_L: [T, 3] L_Ankle pos in pelvis-local frame
        foot_offset_R: [T, 3] R_Ankle pos in pelvis-local frame
        R_pelvis: [T, 3, 3]   pelvis world rotation matrix

    Note on coordinate frame:
        Inputs (`pose_quat_global_xyzw`, `trans`) and outputs are in whatever
        frame the caller passes. At the Task 6 insertion point, this is Y-up
        pre-upright space — the later `upright.apply()` converts to Z-up.
        Callers MUST pass data from before the upright transform.
    """
    import torch
    from poselib.poselib.skeleton.skeleton3d import SkeletonState
    r = torch.from_numpy(pose_quat_global_xyzw).float()
    t = torch.from_numpy(trans).float()
    state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, r=r, t=t, is_local=False
    )
    # state.global_translation: [T, N_joints, 3]
    gt = state.global_translation.numpy()
    # Indices in MUJOCO/PHC-augmented SMPL order (used by this pipeline):
    # 0 = Pelvis, 7 = L_Ankle, 8 = R_Ankle (verified via skeleton_tree.node_names)
    names = list(skeleton_tree.node_names)
    IDX_PELVIS = names.index("Pelvis") if "Pelvis" in names else 0
    IDX_L_ANK = names.index("L_Ankle") if "L_Ankle" in names else 7
    IDX_R_ANK = names.index("R_Ankle") if "R_Ankle" in names else 8
    foot_L_world = gt[:, IDX_L_ANK, :]
    foot_R_world = gt[:, IDX_R_ANK, :]
    pelvis_world = gt[:, IDX_PELVIS, :]
    # Pelvis rotation matrix from pose_quat_global[:, 0] (xyzw)
    q = pose_quat_global_xyzw[:, IDX_PELVIS, :]
    R_pelvis = sRot.from_quat(q).as_matrix()  # [T, 3, 3]
    # foot offset in pelvis-local frame = R_pelvis^T @ (foot_world - pelvis_world)
    diff_L = foot_L_world - pelvis_world
    diff_R = foot_R_world - pelvis_world
    foot_offset_L = np.einsum('tij,tj->ti', R_pelvis.transpose(0, 2, 1), diff_L)
    foot_offset_R = np.einsum('tij,tj->ti', R_pelvis.transpose(0, 2, 1), diff_R)
    return foot_L_world, foot_R_world, foot_offset_L, foot_offset_R, R_pelvis


def compute_root_translation(f, trial_path, fps_in, fps_out, pelvis_lat_scale=1.0):
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
    trans[:, 0] = -(com_ml - com_ml[0]) * pelvis_lat_scale   # X = right (negate medio-lateral); scale dampens lateral sway
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


def read_cop(f, trial_path, side):
    """Read CoP medio-lateral (x) in meters, at input fps. Returns None if missing.

    Forceplate CoP is in PiG lab frame: x=medio-lateral, y=anterior-posterior, z=vertical.
    Valid only during stance (when GRF > threshold); outside stance the values are garbage
    (huge range from unloaded plate electronics). Caller must mask by stance.
    """
    path = f"{trial_path}/forceplate/cop/{side}/x"
    try:
        v = np.array(f[path], dtype=np.float64) / 1000.0  # mm → m
        nn = np.isnan(v)
        if nn.any() and not nn.all():
            v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
        elif nn.all():
            return None
        return v
    except KeyError:
        return None


def read_cop_xy(f, trial_path, side):
    """Read CoP medio-lateral (x) AND anterior-posterior (y) in meters.

    Returns (cop_xy, valid) where:
      cop_xy: [T, 2] float64. Column 0 = lab x (medio-lateral), column 1 = lab y (AP).
      valid: bool — False if H5 lacks CoP data.

    NaN values are interpolated. Outside stance the CoP electronics return garbage —
    caller must mask by stance.
    """
    paths = (
        f"{trial_path}/forceplate/cop/{side}/x",
        f"{trial_path}/forceplate/cop/{side}/y",
    )
    arrs = []
    for path in paths:
        try:
            v = np.array(f[path], dtype=np.float64) / 1000.0   # mm → m
            nn = np.isnan(v)
            if nn.any() and not nn.all():
                v[nn] = np.interp(np.flatnonzero(nn), np.flatnonzero(~nn), v[~nn])
            elif nn.all():
                return np.zeros((0, 2), dtype=np.float64), False
            arrs.append(v)
        except KeyError:
            return np.zeros((0, 2), dtype=np.float64), False
    cop_xy = np.column_stack(arrs)
    return cop_xy, True


def convert_trial(f, trial_path, fps_in=100, fps_out=30, baseline_s=0.0, spine_3axis=False, upper_body=False, elbow_offset_deg=0.0, pelvis_obliq_scale=1.0, pelvis_lat_scale=1.0, stance_anchor_ip=False, stance_anchor_source="fk", foot_ik="none", foot_ik_kwargs=None):
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
        # Baseline subtraction for pelvis standing offset
        if baseline_s > 0:
            from h5_conversion_helpers import subtract_baseline
            pelvis_tilt = subtract_baseline(pelvis_tilt, fps=fps_in, baseline_s=baseline_s)
            pelvis_obliq = subtract_baseline(pelvis_obliq, fps=fps_in, baseline_s=baseline_s)
            pelvis_rot = subtract_baseline(pelvis_rot, fps=fps_in, baseline_s=baseline_s)
        # Scale pelvis obliquity (frontal roll) — 1.0 = unchanged, 0.0 = zero out
        if pelvis_obliq_scale != 1.0:
            pelvis_obliq = pelvis_obliq * pelvis_obliq_scale
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
        """Read and NaN-interpolate a 1-D channel, fall back to zeros.

        Note: baseline subtraction is applied to pelvis channels only (see above), NOT here.
        Spine standing offset is actual lumbar lordosis, not a calibration artefact, so
        subtracting it would inflate walking flexion (empirically: +14.64° trunk lean vs
        +5.83° with pelvis-only).
        """
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
        if spine_3axis and smpl_idx in (3, 6):
            # Read y, z in addition to x (already `vals`) and compose as extrinsic XYZ Euler
            # (scipy upper-case "XYZ" = extrinsic, matching PiG Cardan convention).
            # For midline joints (side=="any"), PiG reports left/right with opposite y,z signs
            # (mirror convention), so averaging cancels lateral/axial motion.
            # Use left channel directly for y,z — it gives the actual joint motion.
            # NOTE: The left-channel sign has not been independently confirmed against
            # anatomical convention — visually verify (Task 6 comparison mp4) before
            # trusting y,z magnitudes for quantitative analysis.
            vx = vals                                                 # already mean(L,R) x
            if side == "any":
                vy = _read_side(f"{trial_path}/mocap/angle/left/{joint}/y")
                vz = _read_side(f"{trial_path}/mocap/angle/left/{joint}/z")
            else:
                # Currently unreachable: LEG_SPINE_MAP entries for smpl_idx 3,6 both use side="any".
                # Kept as a forward-compat hook if a future 3-axis spine joint has a non-"any" side.
                vy = _read_side(f"{trial_path}/mocap/angle/{side}/{joint}/y")
                vz = _read_side(f"{trial_path}/mocap/angle/{side}/{joint}/z")
            eul = np.stack([sign * np.deg2rad(vx),
                            np.deg2rad(vy),
                            np.deg2rad(vz)], axis=-1)
            pose_aa_local[:, smpl_idx, :] = sRot.from_euler("XYZ", eul).as_rotvec()
        else:
            pose_aa_local[:, smpl_idx, 0] = sign * np.deg2rad(vals)   # existing behavior
        mapped_count += 1

    if mapped_count < 4:
        print(f"  SKIP (only {mapped_count} joints mapped): {trial_path}")
        return None

    if upper_body:
        # Neck (SMPL 12) and Head (SMPL 15): full Euler xyz from PiG neck/head channels.
        # x is averaged L+R (same sign both sides, midline sagittal flexion).
        # y,z use left channel only — PiG mirror convention flips y,z signs for midline
        # joints, so L+R average cancels to ~0 (verified on spine in Task 3).
        # NOTE: Sign direction for y,z not empirically confirmed against anatomical
        # convention — visually verify (Task 6 compare mp4) before trusting magnitudes.
        UPPER_BODY_MAP = {
            12: "neck",   # SMPL Neck
            15: "head",   # SMPL Head
        }
        for smpl_idx, joint_name in UPPER_BODY_MAP.items():
            vx = 0.5 * (_read_side(f"{trial_path}/mocap/angle/left/{joint_name}/x")
                        + _read_side(f"{trial_path}/mocap/angle/right/{joint_name}/x"))
            vy = _read_side(f"{trial_path}/mocap/angle/left/{joint_name}/y")
            vz = _read_side(f"{trial_path}/mocap/angle/left/{joint_name}/z")
            eul = np.stack([np.deg2rad(vx), np.deg2rad(vy), np.deg2rad(vz)], axis=-1)
            pose_aa_local[:, smpl_idx, :] = sRot.from_euler("XYZ", eul).as_rotvec()

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
        if elbow_offset_deg != 0.0:
            vals = vals - elbow_offset_deg                              # reduce flexion magnitude (positive = less bent)
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
    trans = compute_root_translation(f, trial_path, fps_in, fps_out, pelvis_lat_scale=pelvis_lat_scale)
    # Ensure same length
    min_len = min(len(trans), len(pose_aa_local))
    trans = trans[:min_len]
    pose_aa_local = pose_aa_local[:min_len]
    T_out_final = min_len

    if stance_anchor_ip:
        from h5_conversion_helpers import detect_stance, compute_foot_anchor, solve_pelvis_ip
        import torch as _torch
        from poselib.poselib.skeleton.skeleton3d import SkeletonState as _SKS
        # The IP solver needs FK in a gravity-aligned frame where the foot height is meaningful.
        # trans is currently Y-up (X=right, Y=up, Z=forward). The SMPL skeleton's root rotation
        # encodes `R_pelvis * upright`, where `upright` maps body Y → world Z (i.e., the body
        # is oriented to lie horizontal in Y-up world). FK in Y-up therefore gives wrong foot
        # heights for ground-contact detection.
        #
        # Strategy: convert to Z-up (the gravity-aligned frame) BEFORE FK:
        #   trans_zup = upright.apply(trans)   →  [fwd, lat, up]  (indices 0,1,2)
        #   pose_quat_global_zup = pose_quat_global_yup * upright_inv
        #     (same transform applied in the main pipeline at lines downstream)
        # All IP computations run in Z-up. Corrections are mapped back to Y-up:
        #   trans_yup[0] (lateral) ← pelvis_ip_zup[1]  (Z-up Y = lateral)
        #   trans_yup[1] (vertical) ← pelvis_ip_zup[2]  (Z-up Z = vertical/up)
        #   trans_yup[2] (forward) unchanged
        _T = T_out_final
        _upright = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
        _upright_inv = _upright.inv()
        # --- Build Z-up global rotations and translation ---
        # 1. Local quaternions in MUJOCO order → SkeletonState → global Y-up rotations
        pose_quat_bone_ip = sRot.from_rotvec(pose_aa_local.reshape(-1, 3)).as_quat().reshape(_T, 24, 4)
        pose_quat_mj_ip = pose_quat_bone_ip[:, SMPL_2_MUJOCO]
        sk_tree_ip = get_skeleton_tree()
        r_local_t = _torch.from_numpy(pose_quat_mj_ip).float()
        t_yup_t = _torch.from_numpy(trans).float()
        state_loc = _SKS.from_rotation_and_root_translation(
            sk_tree_ip, r=r_local_t, t=t_yup_t, is_local=True
        )
        # 2. Right-multiply by upright_inv to get Z-up global rotations
        pqg_yup = state_loc.global_rotation.numpy()   # (T, 24, 4) xyzw
        pqg_zup = (sRot.from_quat(pqg_yup.reshape(-1, 4)) * _upright_inv
                   ).as_quat().reshape(_T, -1, 4)
        # 3. Convert trans to Z-up
        trans_zup = _upright.apply(trans)              # (T, 3): [fwd, lat, up]
        # --- FK in Z-up frame ---
        foot_L_w, foot_R_w, foff_L, foff_R, R_pel = fk_feet(
            sk_tree_ip, pqg_zup, trans_zup
        )
        # Foot velocity: vertical (Z in Z-up, index 2) only for treadmill stance detection.
        # In Z-up: X=forward (belt dir, ~1 m/s during stance), Y=lateral, Z=vertical.
        # During stance, vertical foot velocity ≈ 0; during swing it peaks at ~0.5-1 m/s.
        _grad_L = np.gradient(foot_L_w, axis=0) * fps_out
        _grad_R = np.gradient(foot_R_w, axis=0) * fps_out
        vel_L = np.abs(_grad_L[:, 2])   # Z = vertical only (m/s)
        vel_R = np.abs(_grad_R[:, 2])
        # GRF — read_grf returns data at fps_in; resample to fps_out if needed.
        grf_L_raw = read_grf(f, trial_path, side='left')
        grf_R_raw = read_grf(f, trial_path, side='right')
        no_grf = (grf_L_raw is None or grf_R_raw is None)
        if no_grf:
            print("  [stance_anchor_ip] no GRF channels in H5 — using velocity-only stance detection")
            # Velocity-only detection: detect_stance uses vel hysteresis alone when grf is zero
            grf_L = np.zeros(_T)
            grf_R = np.zeros(_T)
        else:
            # Resample GRF from fps_in to fps_out if lengths differ
            def _resample_grf(arr, f_in, f_out, n_out):
                if len(arr) == n_out:
                    return arr
                t_in = np.arange(len(arr)) / f_in
                t_out = np.arange(n_out) / f_out
                return np.interp(t_out, t_in, arr)
            grf_L = _resample_grf(grf_L_raw, fps_in, fps_out, _T)
            grf_R = _resample_grf(grf_R_raw, fps_in, fps_out, _T)
        # Stance detection:
        # - When GRF is available: GRF-only Schmitt trigger (do NOT OR with velocity, because
        #   on a treadmill the velocity signal is noisy and would inflate stance fraction).
        #   grf_on=50N, grf_off=20N produces ~67% stance for this dual-belt treadmill.
        # - When no GRF: velocity-only (detect_stance fallback) using Z-up vertical velocity.
        if not no_grf:
            from h5_conversion_helpers import _schmitt
            stance_L = _schmitt(grf_L, 50.0, 20.0, on_when_above=True)
            stance_R = _schmitt(grf_R, 50.0, 20.0, on_when_above=True)
        else:
            stance_L, stance_R = detect_stance(grf_L, grf_R, vel_L, vel_R)
        # Optionally replace FK-derived foot lateral with CoP (independent measurement).
        # This breaks the circular dependency where foot_L_w comes from FK(pelvis_from_CoM)
        # and the IP then tries to correct the same pelvis — which collapses to near no-op.
        # CoP is the true stance-foot lateral position in lab frame (std ~2cm during stance).
        if stance_anchor_source == "cop":
            from h5_conversion_helpers import cop_replace_lateral
            cop_L_raw = read_cop(f, trial_path, side='left')
            cop_R_raw = read_cop(f, trial_path, side='right')
            if cop_L_raw is None or cop_R_raw is None:
                print("  [stance_anchor_ip] stance_anchor_source=cop requested but no CoP data — falling back to FK")
            else:
                def _resample_cop(arr, f_in, f_out, n_out):
                    if len(arr) == n_out:
                        return arr
                    t_in = np.arange(len(arr)) / f_in
                    t_out = np.arange(n_out) / f_out
                    return np.interp(t_out, t_in, arr)
                cop_L_m = _resample_cop(cop_L_raw, fps_in, fps_out, _T)
                cop_R_m = _resample_cop(cop_R_raw, fps_in, fps_out, _T)
                # Replace the lateral axis of FK foot positions with CoP (Z-up lateral = index 1).
                foot_L_anchor_in, foot_R_anchor_in = cop_replace_lateral(
                    foot_L_w, foot_R_w, stance_L, stance_R,
                    cop_L_m, cop_R_m, fk_lat_axis=1,
                )
                print(f"  [stance_anchor_ip] using CoP for lateral anchor "
                      f"(L stance mean: FK={foot_L_w[stance_L,1].mean():.3f} → "
                      f"CoP-based={foot_L_anchor_in[stance_L,1].mean():.3f})")
                foot_L_w = foot_L_anchor_in
                foot_R_w = foot_R_anchor_in
        # Foot anchor (rolling mean per stance episode, in Z-up)
        anchor_L = compute_foot_anchor(foot_L_w, stance_L, window=9)
        anchor_R = compute_foot_anchor(foot_R_w, stance_R, window=9)
        # For weighting: if no GRF, use stance bool → float (1.0 / 0.0) as pseudo-weights
        if no_grf:
            w_L_pseudo = stance_L.astype(np.float64)
            w_R_pseudo = stance_R.astype(np.float64)
        else:
            w_L_pseudo = grf_L
            w_R_pseudo = grf_R
        # IP solver (Z-up frame)
        pelvis_ip_zup = solve_pelvis_ip(
            anchor_L, anchor_R, w_L_pseudo, w_R_pseudo,
            R_pel, foff_L, foff_R
        )
        # Flight-phase fallback: NaN → previous valid position + CoM delta carry (in Z-up)
        nan_mask = np.isnan(pelvis_ip_zup).any(axis=1)
        if nan_mask[0]:
            pelvis_ip_zup[0] = trans_zup[0]
        for ti in range(1, _T):
            if nan_mask[ti]:
                pelvis_ip_zup[ti] = pelvis_ip_zup[ti - 1] + (trans_zup[ti] - trans_zup[ti - 1])
        # Sanity clip: if IP differs from CoM-based trans by >30 cm, clip to 30cm (in Z-up)
        diff_zup = pelvis_ip_zup - trans_zup
        big = np.linalg.norm(diff_zup, axis=1) > 0.30
        if big.any():
            n_big = int(big.sum())
            print(f"  [stance_anchor_ip] WARNING: {n_big}/{_T} frames IP vs CoM differ >30cm — clipping")
            for ti in range(_T):
                if big[ti]:
                    d = diff_zup[ti]
                    norm = np.linalg.norm(d) + 1e-9
                    pelvis_ip_zup[ti] = trans_zup[ti] + d / norm * 0.30
        # Write corrections back to Y-up trans (before upright.apply downstream).
        # Axis mapping (upright.apply maps [x_yup,y_yup,z_yup]→[z_yup,x_yup,y_yup]=Z-up[fwd,lat,up]):
        #   Z-up[0] = fwd = unchanged  (treadmill integration)
        #   Z-up[1] = lat = Y-up X    → trans[:, 0]
        #   Z-up[2] = up  = Y-up Y    → trans[:, 1]
        trans[:, 0] = pelvis_ip_zup[:, 1]   # lateral (Z-up Y → Y-up X)
        trans[:, 1] = pelvis_ip_zup[:, 2]   # vertical (Z-up Z → Y-up Y)
        # trans[:, 2] (forward) unchanged — treadmill integration
        print(f"  [stance_anchor_ip] done: "
              f"stance_L={stance_L.mean()*100:.1f}%  stance_R={stance_R.mean()*100:.1f}%  "
              f"flight_frames={int(nan_mask.sum())}")

    if foot_ik in ("full", "trajectory"):
        from h5_conversion_helpers import (
            classify_sub_phases, build_foot_anchors, solve_foot_ik_frame,
            _extract_leg_local_offsets, _fk_leg_world_xyz,
        )
        if foot_ik_kwargs is None:
            foot_ik_kwargs = {}
        ik_pitch_thr = float(foot_ik_kwargs.get("pitch_threshold_deg", 5.0))
        ik_anchor_w = float(foot_ik_kwargs.get("anchor_weight", 1e3))
        ik_joint_w = float(foot_ik_kwargs.get("joint_reg_weight", 0.1))
        ik_smooth_w = float(foot_ik_kwargs.get("smoothness_weight", 0.5))
        ik_pelvis_w = float(foot_ik_kwargs.get("pelvis_reg_weight", 0.01))
        ik_max_nfev = int(foot_ik_kwargs.get("max_nfev", 300))
        ik_bounds_deg = float(foot_ik_kwargs.get("bounds_deg", 20.0))

        _T = T_out_final
        _sk = get_skeleton_tree()
        _offsets_L = _extract_leg_local_offsets(_sk, side='L')
        _offsets_R = _extract_leg_local_offsets(_sk, side='R')
        # Z-up frame for IK (consistent with FK pipeline). Convert trans + pelvis rotation.
        _upright_ik = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
        _upright_ik_inv = _upright_ik.inv()
        trans_zup_ik = _upright_ik.apply(trans)            # (T, 3) in Z-up
        # Per-frame pelvis rotation matrix for FK in Z-up world.
        # pose_aa_local[:, 0] = R_pelvis_zup * upright (per line 466). FK uses Y-up
        # skeleton offsets, so the pelvis world rotation must include `* upright` to
        # transform Y-up local offsets into Z-up world. Use _R_pel_local directly.
        _R_pel_local = sRot.from_rotvec(pose_aa_local[:, 0])               # = R_pelvis * upright
        _R_pel_world_zup = _R_pel_local.as_matrix()                        # (T, 3, 3)

        # FK pass: ankle/toe world (Z-up) per frame, using current pose
        ankle_L_zup = np.zeros((_T, 3))
        toe_L_zup = np.zeros((_T, 3))
        ankle_R_zup = np.zeros((_T, 3))
        toe_R_zup = np.zeros((_T, 3))
        for t in range(_T):
            ankle_L_zup[t], toe_L_zup[t] = _fk_leg_world_xyz(
                trans_zup_ik[t], _R_pel_world_zup[t],
                pose_aa_local[t, 1], pose_aa_local[t, 4], pose_aa_local[t, 7], _offsets_L,
            )
            ankle_R_zup[t], toe_R_zup[t] = _fk_leg_world_xyz(
                trans_zup_ik[t], _R_pel_world_zup[t],
                pose_aa_local[t, 2], pose_aa_local[t, 5], pose_aa_local[t, 8], _offsets_R,
            )

        # Read GRF + CoP (re-resample to fps_out)
        _grf_L_raw = read_grf(f, trial_path, side='left')
        _grf_R_raw = read_grf(f, trial_path, side='right')
        if _grf_L_raw is None or _grf_R_raw is None:
            print("  [foot_ik] no GRF — IK requires forceplate data, skipping IK block.")
        else:
            def _resample(arr, f_in, f_out, n_out):
                if len(arr) == n_out:
                    return arr
                t_in = np.arange(len(arr)) / f_in
                t_out = np.arange(n_out) / f_out
                return np.interp(t_out, t_in, arr)
            _grf_L = _resample(_grf_L_raw, fps_in, fps_out, _T)
            _grf_R = _resample(_grf_R_raw, fps_in, fps_out, _T)
            _cop_L_xy_raw, _cop_L_ok = read_cop_xy(f, trial_path, side='left')
            _cop_R_xy_raw, _cop_R_ok = read_cop_xy(f, trial_path, side='right')
            if not (_cop_L_ok and _cop_R_ok):
                print("  [foot_ik] no CoP — IK requires forceplate CoP data, skipping IK block.")
            else:
                _cop_L_xy = np.column_stack([
                    _resample(_cop_L_xy_raw[:, 0], fps_in, fps_out, _T),
                    _resample(_cop_L_xy_raw[:, 1], fps_in, fps_out, _T),
                ])
                _cop_R_xy = np.column_stack([
                    _resample(_cop_R_xy_raw[:, 0], fps_in, fps_out, _T),
                    _resample(_cop_R_xy_raw[:, 1], fps_in, fps_out, _T),
                ])
                # Sign-align CoP lateral with FK lateral (same logic as cop_replace_lateral)
                # Use Y-axis (index 1) of Z-up frame as lateral.
                _stance_L_mask = _grf_L > 50.0
                _stance_R_mask = _grf_R > 50.0
                if _stance_L_mask.sum() > 3 and _stance_R_mask.sum() > 3:
                    _fk_L = ankle_L_zup[_stance_L_mask, 1].mean()
                    _fk_R = ankle_R_zup[_stance_R_mask, 1].mean()
                    _cop_L_mean = _cop_L_xy[_stance_L_mask, 0].mean()
                    _cop_R_mean = _cop_R_xy[_stance_R_mask, 0].mean()
                    _sign = 1.0 if (_fk_L - _fk_R) * (_cop_L_mean - _cop_R_mean) > 0 else -1.0
                    _cop_L_xy[:, 0] = _sign * _cop_L_xy[:, 0] + (_fk_L - _sign * _cop_L_mean)
                    _cop_R_xy[:, 0] = _sign * _cop_R_xy[:, 0] + (_fk_R - _sign * _cop_R_mean)

                # Sub-phase classification
                phase_L, phase_R = classify_sub_phases(
                    _grf_L, _grf_R,
                    ankle_L_zup, toe_L_zup, ankle_R_zup, toe_R_zup,
                    pitch_threshold_deg=ik_pitch_thr,
                )

                # Anchors. build_foot_anchors output convention is [lateral, forward, vertical]
                # (column 0 = CoP medio-lateral, column 1 = FK foot column 1 as "forward").
                # Our Z-up world convention is [forward, lateral, vertical] — column 0 IS forward.
                # So pass ankle/toe with columns 0 and 1 swapped (helper sees forward at col 1),
                # then swap anchor output columns back to Z-up [fwd, lat, vert] for IK.
                _swap = np.array([1, 0, 2])
                anchors = build_foot_anchors(
                    _cop_L_xy, _cop_R_xy,
                    ankle_L_zup[:, _swap], toe_L_zup[:, _swap],
                    ankle_R_zup[:, _swap], toe_R_zup[:, _swap],
                    phase_L, phase_R, min_episode_frames=10,
                )
                for _k in ("H_L", "T_L", "H_R", "T_R"):
                    anchors[_k] = anchors[_k][:, _swap]

                ik_active_either = anchors["ik_active_L"] | anchors["ik_active_R"]
                trans_zup_corrected = trans_zup_ik.copy()
                pose_aa_corrected = pose_aa_local.copy()

                if foot_ik == "trajectory":
                    from h5_conversion_helpers import solve_foot_ik_trajectory
                    ik_lr = float(foot_ik_kwargs.get("lr", 0.01))
                    ik_max_iter_traj = int(foot_ik_kwargs.get("max_iter", 500))
                    ik_bound_w = float(foot_ik_kwargs.get("bound_weight", 100.0))
                    weights_traj = {
                        "anchor": ik_anchor_w,
                        "smooth": ik_smooth_w,
                        "joint_reg": ik_joint_w,
                        "pelvis_reg": ik_pelvis_w,
                        "bound": ik_bound_w,
                    }
                    print(f"  [foot_ik trajectory] starting Adam optimization "
                          f"(T={_T}, lr={ik_lr}, max_iter={ik_max_iter_traj}, "
                          f"weights={weights_traj}, bounds_deg={ik_bounds_deg})")
                    pose_aa_corrected, trans_zup_corrected, info = solve_foot_ik_trajectory(
                        pose_aa_local, trans_zup_ik, _R_pel_world_zup,
                        anchors, phase_L, phase_R,
                        _offsets_L, _offsets_R,
                        weights=weights_traj, bounds_deg=ik_bounds_deg,
                        lr=ik_lr, max_iter=ik_max_iter_traj,
                        device="cpu", verbose=True,
                    )
                    print(f"  [foot_ik trajectory] done: iters={info['iters']}/{ik_max_iter_traj} "
                          f"converged={info['converged']}  final_loss={info['final_loss']:.6f}")
                else:
                    # foot_ik == "full" — per-frame scipy LS (deprecated; discontinuous)
                    # Build per-frame anchor weight ramp at stance edges to soften activation.
                    _RAMP_FRAMES = 5
                    anchor_w_per_frame = np.zeros(_T)
                    _in = False; _s = 0
                    for t in range(_T):
                        if ik_active_either[t] and not _in:
                            _s = t; _in = True
                        elif not ik_active_either[t] and _in:
                            _e = t
                            for tt in range(_s, _e):
                                edge = min(tt - _s, _e - 1 - tt)
                                ramp = min(1.0, (edge + 1) / _RAMP_FRAMES)
                                anchor_w_per_frame[tt] = ik_anchor_w * ramp
                            _in = False
                    if _in:
                        _e = _T
                        for tt in range(_s, _e):
                            edge = min(tt - _s, _e - 1 - tt)
                            ramp = min(1.0, (edge + 1) / _RAMP_FRAMES)
                            anchor_w_per_frame[tt] = ik_anchor_w * ramp

                    _converged_count = 0
                    _ik_frame_count = 0
                    _pose_prev = None
                    _trans_prev = None
                    for t in range(_T):
                        if not ik_active_either[t]:
                            _pose_prev = pose_aa_corrected[t]
                            _trans_prev = trans_zup_corrected[t]
                            continue
                        anchors_t = {
                            "H_L": anchors["H_L"][t], "T_L": anchors["T_L"][t],
                            "H_R": anchors["H_R"][t], "T_R": anchors["T_R"][t],
                        }
                        phase_t = {"L": int(phase_L[t]), "R": int(phase_R[t])}
                        weights = {
                            "anchor": anchor_w_per_frame[t],
                            "joint": ik_joint_w,
                            "smooth": ik_smooth_w,
                            "pelvis": ik_pelvis_w,
                        }
                        pose_corr, trans_corr, converged = solve_foot_ik_frame(
                            pose_aa_corrected[t], trans_zup_corrected[t], _R_pel_world_zup[t],
                            anchors_t, phase_t,
                            _offsets_L, _offsets_R,
                            weights=weights, bounds_deg=ik_bounds_deg, max_nfev=ik_max_nfev,
                            pose_prev=_pose_prev, trans_prev=_trans_prev,
                        )
                        pose_aa_corrected[t] = pose_corr
                        trans_zup_corrected[t] = trans_corr
                        _ik_frame_count += 1
                        if converged:
                            _converged_count += 1
                        _pose_prev = pose_corr
                        _trans_prev = trans_corr
                    _conv_pct = (100.0 * _converged_count / max(_ik_frame_count, 1))
                    print(f"  [foot_ik full] done: ik_active_frames={_ik_frame_count}/{_T}  "
                          f"converged={_converged_count} ({_conv_pct:.1f}%)")

                # Write back: convert trans_zup_corrected back to Y-up
                trans = _upright_ik_inv.apply(trans_zup_corrected)
                pose_aa_local = pose_aa_corrected

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
    # All three conversion fixes are now wired:
    # --baseline_s (pelvis channels only; see convert_trial).
    # --spine_3axis (adds y,z Euler axes to SMPL Torso/Spine2 via LEG_SPINE_MAP).
    # --upper_body (maps PiG neck/head to SMPL Neck(12)/Head(15) via UPPER_BODY_MAP).
    parser.add_argument("--baseline_s", type=float, default=0.0,
                        help="Subtract standing-mean baseline computed from first N seconds of each trial. 0 = off.")
    parser.add_argument("--spine_3axis", action="store_true",
                        help="Map all three PiG spine/thorax axes (x,y,z) to SMPL instead of sagittal only.")
    parser.add_argument("--upper_body", action="store_true",
                        help="Map PiG neck→SMPL joint 12 and head→SMPL joint 15 (instead of leaving them identity).")
    parser.add_argument("--elbow_offset_deg", type=float, default=0.0,
                        help="Subtract this many degrees from PiG elbow flexion (positive value = less bent). Experimental.")
    parser.add_argument("--pelvis_obliq_scale", type=float, default=1.0,
                        help="Scale pelvis obliquity (frontal roll). 1.0=unchanged, 0.0=zero out, 0.5=halve. Experimental.")
    parser.add_argument("--pelvis_lat_scale", type=float, default=1.0,
                        help="Scale root lateral translation (CoM medio-lateral). 1.0=unchanged, 0.0=straight forward. Experimental.")
    parser.add_argument("--stance_anchor_ip", action="store_true",
                        help="Anchor pelvis world position to stance feet via IP "
                             "(replaces lateral+vertical in trans; keeps treadmill forward). "
                             "See design doc 01_research_docs/260424_h5_stance_anchor_ip_design.md.")
    parser.add_argument("--stance_anchor_source", type=str, default="fk",
                        choices=["fk", "cop"],
                        help="IP anchor source. 'fk' = FK-derived foot (circular, near no-op). "
                             "'cop' = Forceplate CoP for lateral, FK for fwd/vert (breaks circularity).")
    parser.add_argument("--foot_ik", type=str, default="none", choices=["none", "full", "trajectory"],
                        help="Full foot-anchor IK over pelvis_trans + stance leg joints. "
                             "'none' = disabled (default, backward-compat). "
                             "'full' = scipy nonlinear LS per frame (deprecated; discontinuous). "
                             "'trajectory' = PyTorch Adam over full trajectory (smooth, recommended). "
                             "See 01_research_docs/260425_h5_foot_anchor_ik_design.md.")
    parser.add_argument("--foot_ik_lr", type=float, default=0.01,
                        help="[trajectory] Adam learning rate.")
    parser.add_argument("--foot_ik_max_iter", type=int, default=500,
                        help="[trajectory] Max Adam iterations.")
    parser.add_argument("--foot_ik_bound_weight", type=float, default=100.0,
                        help="[trajectory] Soft bound penalty weight (joint angle outside ± bounds_deg).")
    parser.add_argument("--foot_ik_pitch_threshold_deg", type=float, default=5.0,
                        help="Foot pitch threshold (degrees) for sub-phase classification.")
    parser.add_argument("--foot_ik_anchor_weight", type=float, default=1e3,
                        help="IK cost weight for anchor satisfaction (large = effectively hard).")
    parser.add_argument("--foot_ik_joint_reg_weight", type=float, default=0.1,
                        help="IK cost weight for joint angle deviation from measured.")
    parser.add_argument("--foot_ik_smoothness_weight", type=float, default=0.5,
                        help="IK cost weight for frame-to-frame joint angle smoothness.")
    parser.add_argument("--foot_ik_pelvis_reg_weight", type=float, default=0.01,
                        help="IK cost weight for pelvis trans deviation from measured.")
    parser.add_argument("--foot_ik_max_nfev", type=int, default=300,
                        help="Max scipy least_squares function evaluations per frame (≈ outer iter × 22 for 21-DoF finite-diff).")
    parser.add_argument("--foot_ik_bounds_deg", type=float, default=20.0,
                        help="Joint angle bounds: measured ± this (degrees).")
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

                    result = convert_trial(f, trial_path, fps_in, args.fps_out, baseline_s=args.baseline_s, spine_3axis=args.spine_3axis, upper_body=args.upper_body, elbow_offset_deg=args.elbow_offset_deg, pelvis_obliq_scale=args.pelvis_obliq_scale, pelvis_lat_scale=args.pelvis_lat_scale, stance_anchor_ip=args.stance_anchor_ip, stance_anchor_source=args.stance_anchor_source, foot_ik=args.foot_ik, foot_ik_kwargs={"pitch_threshold_deg": args.foot_ik_pitch_threshold_deg, "anchor_weight": args.foot_ik_anchor_weight, "joint_reg_weight": args.foot_ik_joint_reg_weight, "smoothness_weight": args.foot_ik_smoothness_weight, "pelvis_reg_weight": args.foot_ik_pelvis_reg_weight, "max_nfev": args.foot_ik_max_nfev, "bounds_deg": args.foot_ik_bounds_deg, "lr": args.foot_ik_lr, "max_iter": args.foot_ik_max_iter, "bound_weight": args.foot_ik_bound_weight})
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

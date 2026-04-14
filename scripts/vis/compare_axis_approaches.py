"""
Quick comparison: different PiG→SMPL axis mapping approaches.
Renders side-by-side at walking frames (t=15s+) to compare forward lean.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from easydict import EasyDict

from phc.utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21), (20, 22), (21, 23),
]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

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


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def local_to_global(pose_local):
    T = pose_local.shape[0]
    pose_global = np.zeros_like(pose_local)
    base_rot = np.array([0.0, 0.0, 1.0, 0.0])  # Ry(180°)
    pose_global[:, 0] = quat_multiply(np.tile(base_rot, (T, 1)), pose_local[:, 0])
    for j in range(1, 24):
        pose_global[:, j] = quat_multiply(pose_global[:, SMPL_PARENTS[j]], pose_local[:, j])
    norms = np.linalg.norm(pose_global, axis=-1, keepdims=True)
    pose_global /= np.maximum(norms, 1e-8)
    return pose_global


def read_h5_angles(h5_path, trial_path, fps_in=100, fps_out=30):
    """Read raw H5 angles and resample."""
    f = h5py.File(h5_path, "r")
    T = len(np.array(f[f"{trial_path}/common/time"]))

    raw_angles = {}
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
        raw_angles[(side, joint)] = (smpl_idx, euler_deg)
    f.close()

    # Resample
    T_out = int(T * fps_out / fps_in)
    t_in = np.arange(T) / fps_in
    t_out = np.arange(T_out) / fps_out
    resampled = {}
    for key, (smpl_idx, euler_deg) in raw_angles.items():
        euler_out = np.zeros((T_out, 3))
        for i in range(3):
            euler_out[:, i] = np.interp(t_out, t_in, euler_deg[:, i])
        resampled[key] = (smpl_idx, euler_out)
    return resampled, T_out


def make_clip(raw_angles, T, approach_fn, fps=30):
    """Build pose_quat_local using given approach function, then FK→global."""
    pose_local = np.zeros((T, 24, 4))
    pose_local[:, :, 0] = 1.0

    for (side, joint), (smpl_idx, euler_deg) in raw_angles.items():
        euler_rad = np.deg2rad(euler_deg[:T])
        quat_wxyz = approach_fn(euler_rad, smpl_idx, side, joint)
        pose_local[:, smpl_idx, :len(quat_wxyz[0])] = quat_wxyz[:T]

    pose_global = local_to_global(pose_local)

    trans = np.zeros((T, 3))
    trans[:, 0] = np.linspace(0, 5, T)
    trans[:, 2] = 0.93

    # pose_aa with Y-up→Z-up base
    pose_aa = np.zeros((T, 72))
    base_smpl = Rotation.from_euler('Y', 180, degrees=True) * Rotation.from_euler('X', -90, degrees=True)
    root_xyzw = np.column_stack([pose_local[:, 0, 1], pose_local[:, 0, 2],
                                  pose_local[:, 0, 3], pose_local[:, 0, 0]])
    root_rot = Rotation.from_quat(root_xyzw)
    pose_aa[:, 0:3] = (base_smpl * root_rot).as_rotvec()
    for j in range(1, 24):
        xyzw = np.column_stack([pose_local[:, j, 1], pose_local[:, j, 2],
                                 pose_local[:, j, 3], pose_local[:, j, 0]])
        pose_aa[:, j*3:j*3+3] = Rotation.from_quat(xyzw).as_rotvec()

    return {
        "pose_quat_global": pose_global,
        "pose_quat": pose_local.astype(np.float32),
        "trans_orig": trans,
        "root_trans_offset": torch.from_numpy(trans).double(),
        "pose_aa": pose_aa,
        "beta": np.zeros(16),
        "gender": "neutral",
        "fps": fps,
    }


# === Approach functions ===

def approach_direct(euler_rad, smpl_idx, side, joint):
    """PiG XYZ directly as SMPL XYZ."""
    rot = Rotation.from_euler('XYZ', euler_rad)
    q = rot.as_quat()  # xyzw
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def approach_rz90(euler_rad, smpl_idx, side, joint):
    """Rz(-90°) coordinate transform (current approach)."""
    R_pig = Rotation.from_euler('XYZ', euler_rad)
    M = Rotation.from_euler('z', np.full(len(euler_rad), -np.pi/2))
    M_inv = Rotation.from_euler('z', np.full(len(euler_rad), np.pi/2))
    R_smpl = M * R_pig * M_inv
    q = R_smpl.as_quat()
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def approach_rz90_knee_fix(euler_rad, smpl_idx, side, joint):
    """Rz(-90°) but negate flexion for knees (idx 4,5)."""
    e = euler_rad.copy()
    if smpl_idx in (4, 5):  # knees
        e[:, 0] = -e[:, 0]  # negate flexion before transform
    R_pig = Rotation.from_euler('XYZ', e)
    M = Rotation.from_euler('z', np.full(len(e), -np.pi/2))
    M_inv = Rotation.from_euler('z', np.full(len(e), np.pi/2))
    R_smpl = M * R_pig * M_inv
    q = R_smpl.as_quat()
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def approach_neg_flex(euler_rad, smpl_idx, side, joint):
    """Direct XYZ but negate flexion (X) to fix forward/backward."""
    e = euler_rad.copy()
    e[:, 0] = -e[:, 0]
    rot = Rotation.from_euler('XYZ', e)
    q = rot.as_quat()
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def approach_yxz(euler_rad, smpl_idx, side, joint):
    """PiG (flex, abd, rot) interpreted as YXZ Euler: Ry(flex)·Rx(abd)·Rz(rot)."""
    rot = Rotation.from_euler('YXZ', euler_rad)
    q = rot.as_quat()
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def setup_smpl():
    robot_cfg = {
        "mesh": False, "rel_joint_lm": False, "upright_start": True,
        "remove_toe": False, "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, "model": "smpl",
        "big_ankle": True, "freeze_hand": False, "box_body": True,
        "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gender_beta = np.zeros(17)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(gender_beta[None, 1:]),
        gender=gender_beta[0:1], objs_info=None
    )
    test_xml = "/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_xml)
    return SkeletonTree.from_mjcf(test_xml)


def get_positions(motion_file, clip_idx, frame, sk_tree, device):
    cfg = EasyDict({
        "motion_file": motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": 'smpl', "randomrize_heading": False,
    })
    mlib = MotionLibSMPL(cfg)
    mlib.load_motions(
        skeleton_trees=[sk_tree], gender_betas=[torch.zeros(17)],
        limb_weights=[np.zeros(10)], random_sample=False, start_idx=clip_idx
    )
    motion_len = mlib.get_motion_length(0).item()
    t = min(frame / 30.0, motion_len - 0.01)
    res = mlib.get_motion_state(
        torch.tensor([0]).to(device), torch.tensor([t]).to(device)
    )
    return res["rg_pos"][0].cpu().numpy()


def plot_skeleton(ax, rb_pos, color='blue', alpha=1.0):
    ax.scatter(rb_pos[:, 0], rb_pos[:, 1], rb_pos[:, 2],
               c=color, s=15, alpha=alpha, zorder=5)
    for i, j in SMPL_BONES:
        if i < len(rb_pos) and j < len(rb_pos):
            ax.plot([rb_pos[i, 0], rb_pos[j, 0]],
                    [rb_pos[i, 1], rb_pos[j, 1]],
                    [rb_pos[i, 2], rb_pos[j, 2]],
                    c=color, linewidth=1.5, alpha=alpha)


def main():
    device = torch.device("cpu")
    h5_path = "data/combined_data_from_csv.h5"
    trial_path = "S001/level_100mps/lv0/trial_01"

    # Read raw angles once
    print("Reading H5 data...")
    raw_angles, T = read_h5_angles(h5_path, trial_path)

    approaches = {
        "A: Direct XYZ": approach_direct,
        "B: Rz(-90)": approach_rz90,
        "C: Rz(-90) knee fix": approach_rz90_knee_fix,
        "D: Negate flex": approach_neg_flex,
        "E: YXZ order": approach_yxz,
    }

    # Walking frames: t=15s → frame 450 at 30fps
    frames_to_show = [450, 465, 480, 495]  # ~15.0s, 15.5s, 16.0s, 16.5s

    sk_tree = setup_smpl()
    os.makedirs("output", exist_ok=True)

    n_approaches = len(approaches)
    n_frames = len(frames_to_show)
    fig = plt.figure(figsize=(4 * n_frames, 4 * n_approaches * 2))

    for ai, (name, fn) in enumerate(approaches.items()):
        print(f"Building clip: {name}...")
        clip = make_clip(raw_angles, T, fn)

        # Save temp pkl
        tmp_pkl = f"/tmp/test_approach_{ai}.pkl"
        joblib.dump({f"test_{ai}": clip}, tmp_pkl)

        for fi, frame in enumerate(frames_to_show):
            if frame >= T:
                continue
            rb_pos = get_positions(tmp_pkl, 0, frame, sk_tree, device)
            rb_pos[:, :2] -= rb_pos[0, :2]

            # Front view
            row = ai * 2
            ax = fig.add_subplot(n_approaches * 2, n_frames,
                                  row * n_frames + fi + 1, projection='3d')
            plot_skeleton(ax, rb_pos, 'blue')
            ax.view_init(elev=10, azim=0)
            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.1, 1.8)
            if fi == 0:
                ax.set_ylabel(f'{name}\nfront', fontsize=7)
            ax.set_title(f'f{frame}', fontsize=7)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

            # Side view
            ax2 = fig.add_subplot(n_approaches * 2, n_frames,
                                   (row + 1) * n_frames + fi + 1, projection='3d')
            plot_skeleton(ax2, rb_pos, 'blue')
            ax2.view_init(elev=10, azim=90)
            ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.8, 0.8); ax2.set_zlim(-0.1, 1.8)
            if fi == 0:
                ax2.set_ylabel(f'{name}\nside', fontsize=7)
            ax2.set_xticklabels([]); ax2.set_yticklabels([]); ax2.set_zticklabels([])

            # Print heights
            if fi == 0:
                print(f"  {name} f{frame}: pelvis_z={rb_pos[0,2]:.3f}, head_z={rb_pos[15,2]:.3f}, "
                      f"L_ankle_z={rb_pos[7,2]:.3f}, R_ankle_z={rb_pos[8,2]:.3f}")

    plt.suptitle("PiG→SMPL axis mapping comparison (walking frames)", fontsize=12)
    plt.tight_layout()
    output = "output/compare_axis_approaches.png"
    plt.savefig(output, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

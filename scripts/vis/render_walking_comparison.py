"""
Compare H5 walking motion: direct vs coord_transform vs AMASS reference.
Renders multiple frames to show leg swing pattern.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


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


def get_multi_frame_positions(motion_file, clip_idx, frames, sk_tree, device):
    """Get positions at multiple frames for one clip."""
    cfg = EasyDict({
        "motion_file": motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": 'smpl', "randomrize_heading": False,
    })
    mlib = MotionLibSMPL(cfg)
    mlib.load_motions(
        skeleton_trees=[sk_tree],
        gender_betas=[torch.zeros(17)],
        limb_weights=[np.zeros(10)],
        random_sample=False, start_idx=clip_idx
    )
    motion_len = mlib.get_motion_length(0).item()

    positions = []
    for frame in frames:
        t = min(frame / 30.0, motion_len - 0.01)
        res = mlib.get_motion_state(
            torch.tensor([0]).to(device), torch.tensor([t]).to(device)
        )
        positions.append(res["rg_pos"][0].cpu().numpy())
    return positions


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
    sk_tree = setup_smpl()

    debug_file = "sample_data/debug_axis_test.pkl"
    amass_file = "sample_data/amass_isaac_walking_primitive.pkl"

    debug_data = joblib.load(debug_file)
    clip_names = sorted(debug_data.keys())

    # Get clip indices for the H5 modes we want to compare
    h5_clips = {}
    for i, name in enumerate(clip_names):
        if "h5_direct" in name:
            h5_clips["direct"] = i
        elif "h5_coord_transform" in name:
            h5_clips["coord_transform"] = i

    # Frames to render (at 30fps): roughly every 0.5s to show gait cycle
    frames = [0, 15, 30, 45, 60, 75, 90]

    # Create figure: 3 rows (direct, coord_transform, AMASS) x len(frames) columns
    # Front view and side view for each
    n_frames = len(frames)
    fig = plt.figure(figsize=(4 * n_frames, 16))

    row_labels = ["H5 direct (front)", "H5 direct (side)",
                  "H5 coord_transform (front)", "H5 coord_transform (side)",
                  "AMASS ref (front)", "AMASS ref (side)"]

    for row_idx, (label, mode) in enumerate([
        ("direct_front", "direct"),
        ("direct_side", "direct"),
        ("coord_front", "coord_transform"),
        ("coord_side", "coord_transform"),
        ("amass_front", "amass"),
        ("amass_side", "amass"),
    ]):
        if mode == "amass":
            if not os.path.exists(amass_file):
                continue
            positions = get_multi_frame_positions(amass_file, 0, frames, sk_tree, device)
        else:
            clip_idx = h5_clips[mode]
            positions = get_multi_frame_positions(debug_file, clip_idx, frames, sk_tree, device)

        for fi, (frame, rb_pos) in enumerate(zip(frames, positions)):
            # Center at pelvis XY
            rb_pos = rb_pos.copy()
            rb_pos[:, :2] -= rb_pos[0, :2]

            ax = fig.add_subplot(6, n_frames, row_idx * n_frames + fi + 1, projection='3d')
            plot_skeleton(ax, rb_pos, 'blue' if mode != 'amass' else 'green')

            if "front" in label:
                ax.view_init(elev=10, azim=0)
            else:
                ax.view_init(elev=10, azim=90)

            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.2, 1.5)
            if fi == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=7)
            ax.set_title(f'f{frame}', fontsize=7)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    plt.suptitle("Walking comparison: H5 direct vs coord_transform vs AMASS", fontsize=12)
    plt.tight_layout()
    output = "output/debug_walking_comparison.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")

    # Also print L_Hip and L_Knee XYZ positions over time for each mode
    print("\n=== L_Hip (joint 1) X position over time (should swing forward/back for walking) ===")
    for mode in ["direct", "coord_transform"]:
        clip_idx = h5_clips[mode]
        positions = get_multi_frame_positions(debug_file, clip_idx, frames, sk_tree, device)
        hip_x = [p[1, 0] - p[0, 0] for p in positions]  # relative to pelvis
        hip_y = [p[1, 1] - p[0, 1] for p in positions]
        hip_z = [p[1, 2] - p[0, 2] for p in positions]
        print(f"  {mode} L_Hip relative to Pelvis:")
        print(f"    X (fwd): {['%.3f' % v for v in hip_x]}")
        print(f"    Y (lat): {['%.3f' % v for v in hip_y]}")
        print(f"    Z (up):  {['%.3f' % v for v in hip_z]}")

    if os.path.exists(amass_file):
        positions = get_multi_frame_positions(amass_file, 0, frames, sk_tree, device)
        hip_x = [p[1, 0] - p[0, 0] for p in positions]
        hip_y = [p[1, 1] - p[0, 1] for p in positions]
        hip_z = [p[1, 2] - p[0, 2] for p in positions]
        print(f"  AMASS L_Hip relative to Pelvis:")
        print(f"    X (fwd): {['%.3f' % v for v in hip_x]}")
        print(f"    Y (lat): {['%.3f' % v for v in hip_y]}")
        print(f"    Z (up):  {['%.3f' % v for v in hip_z]}")

    # L_Ankle position (foot swing is the most visible walking indicator)
    print("\n=== L_Ankle (joint 7) position relative to Pelvis ===")
    for mode in ["direct", "coord_transform"]:
        clip_idx = h5_clips[mode]
        positions = get_multi_frame_positions(debug_file, clip_idx, frames, sk_tree, device)
        ankle_x = [p[7, 0] - p[0, 0] for p in positions]
        ankle_y = [p[7, 1] - p[0, 1] for p in positions]
        print(f"  {mode} L_Ankle:")
        print(f"    X (fwd): {['%.3f' % v for v in ankle_x]}")
        print(f"    Y (lat): {['%.3f' % v for v in ankle_y]}")

    if os.path.exists(amass_file):
        positions = get_multi_frame_positions(amass_file, 0, frames, sk_tree, device)
        ankle_x = [p[7, 0] - p[0, 0] for p in positions]
        ankle_y = [p[7, 1] - p[0, 1] for p in positions]
        print(f"  AMASS L_Ankle:")
        print(f"    X (fwd): {['%.3f' % v for v in ankle_x]}")
        print(f"    Y (lat): {['%.3f' % v for v in ankle_y]}")


if __name__ == "__main__":
    main()

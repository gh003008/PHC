"""
Debug H5 motion pose — render skeleton joint positions as 3D stick figure.
Compares H5 converted data vs AMASS reference.

Usage:
    python scripts/vis/debug_h5_pose.py
    python scripts/vis/debug_h5_pose.py --clip_idx 100 --frame 30
"""
import os, sys, argparse
sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easydict import EasyDict

from phc.utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

# SMPL bone connections for visualization
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),      # Pelvis → L_Hip, R_Hip, Torso
    (1, 4), (2, 5),               # Hips → Knees
    (4, 7), (5, 8),               # Knees → Ankles
    (7, 10), (8, 11),             # Ankles → Toes
    (3, 6), (6, 9),               # Torso → Spine → Chest
    (9, 12),                       # Chest → Neck
    (12, 15),                      # Neck → Head
    (9, 13), (9, 14),             # Chest → L/R Thorax
    (13, 16), (14, 17),           # Thorax → Shoulders
    (16, 18), (17, 19),           # Shoulders → Elbows
    (18, 20), (19, 21),           # Elbows → Wrists
    (20, 22), (21, 23),           # Wrists → Hands
]

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


def get_rb_positions(motion_file, clip_idx, frame, sk_tree, device):
    """Load motion and return rigid body positions at a given frame."""
    motion_lib_cfg = EasyDict({
        "motion_file": motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": 'smpl', "randomrize_heading": False,
    })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[sk_tree], gender_betas=[torch.zeros(17)],
        limb_weights=[np.zeros(10)], random_sample=False, start_idx=clip_idx
    )

    motion_len = motion_lib.get_motion_length(0).item()
    motion_time = min(frame / 30.0, motion_len - 0.01)

    motion_res = motion_lib.get_motion_state(
        torch.tensor([0]).to(device), torch.tensor([motion_time]).to(device)
    )

    rb_pos = motion_res["rg_pos"][0].cpu().numpy()  # (24, 3)
    return rb_pos


def plot_skeleton(ax, rb_pos, color, label, alpha=1.0):
    """Plot 3D stick figure skeleton."""
    # Plot joints
    ax.scatter(rb_pos[:, 0], rb_pos[:, 1], rb_pos[:, 2],
               c=color, s=30, alpha=alpha, zorder=5)

    # Plot bones
    for i, j in SMPL_BONES:
        if i < len(rb_pos) and j < len(rb_pos):
            ax.plot([rb_pos[i, 0], rb_pos[j, 0]],
                    [rb_pos[i, 1], rb_pos[j, 1]],
                    [rb_pos[i, 2], rb_pos[j, 2]],
                    c=color, linewidth=2, alpha=alpha)

    # Label key joints
    for idx in [0, 4, 5, 7, 8, 15, 16, 17, 20, 21]:
        if idx < len(rb_pos):
            ax.text(rb_pos[idx, 0], rb_pos[idx, 1], rb_pos[idx, 2] + 0.03,
                    SMPL_JOINT_NAMES[idx], fontsize=5, color=color, alpha=alpha)

    # Mark head with larger dot
    if 15 < len(rb_pos):
        ax.scatter([rb_pos[15, 0]], [rb_pos[15, 1]], [rb_pos[15, 2]],
                   c=color, s=100, marker='o', alpha=alpha, edgecolors='black')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", default="sample_data/h5_motion_library.pkl")
    parser.add_argument("--amass_file", default="sample_data/amass_isaac_walking_primitive.pkl")
    parser.add_argument("--clip_idx", type=int, default=100)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--output", default="output/debug_h5_pose.png")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Setup SMPL robot
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
    sk_tree = SkeletonTree.from_mjcf(test_xml)

    # Get positions for multiple frames
    frames_to_check = [0, 15, 30, 45]

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"H5 (red) vs AMASS (blue) — Clip {args.clip_idx}", fontsize=16)

    for fi, frame in enumerate(frames_to_check):
        print(f"Loading frame {frame}...")

        h5_pos = get_rb_positions(args.motion_file, args.clip_idx, frame, sk_tree, device)
        amass_pos = get_rb_positions(args.amass_file, 0, frame, sk_tree, device)

        # Center both at origin (x,y) for comparison
        h5_center = h5_pos[0, :2].copy()
        amass_center = amass_pos[0, :2].copy()
        h5_pos[:, :2] -= h5_center
        amass_pos[:, :2] -= amass_center

        # Front view
        ax = fig.add_subplot(2, 4, fi + 1, projection='3d')
        plot_skeleton(ax, h5_pos, 'red', 'H5')
        plot_skeleton(ax, amass_pos, 'blue', 'AMASS', alpha=0.5)
        ax.set_title(f'Frame {frame} - Front', fontsize=10)
        ax.view_init(elev=10, azim=0)
        ax.set_xlabel('X (fwd)')
        ax.set_ylabel('Y (lat)')
        ax.set_zlabel('Z (up)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)

        # Side view
        ax2 = fig.add_subplot(2, 4, fi + 5, projection='3d')
        plot_skeleton(ax2, h5_pos, 'red', 'H5')
        plot_skeleton(ax2, amass_pos, 'blue', 'AMASS', alpha=0.5)
        ax2.set_title(f'Frame {frame} - Side', fontsize=10)
        ax2.view_init(elev=10, azim=90)
        ax2.set_xlabel('X (fwd)')
        ax2.set_ylabel('Y (lat)')
        ax2.set_zlabel('Z (up)')
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(0, 2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")
    print("Top row: front view, Bottom row: side view")
    print("Red = H5 converted, Blue = AMASS reference")

    # Also print joint position comparison for frame 30
    h5_pos = get_rb_positions(args.motion_file, args.clip_idx, 30, sk_tree, device)
    amass_pos = get_rb_positions(args.amass_file, 0, 30, sk_tree, device)
    h5_pos[:, :2] -= h5_pos[0, :2].copy()
    amass_pos[:, :2] -= amass_pos[0, :2].copy()

    print(f"\n{'Joint':<14} {'H5 (x,y,z)':>30} {'AMASS (x,y,z)':>30} {'Diff':>8}")
    print("-" * 85)
    for j in range(24):
        h = h5_pos[j]
        a = amass_pos[j]
        diff = np.linalg.norm(h - a)
        print(f"{SMPL_JOINT_NAMES[j]:<14} ({h[0]:6.3f},{h[1]:6.3f},{h[2]:6.3f})"
              f"  ({a[0]:6.3f},{a[1]:6.3f},{a[2]:6.3f})  {diff:6.3f}")


if __name__ == "__main__":
    main()

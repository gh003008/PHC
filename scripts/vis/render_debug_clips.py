"""
Render debug axis test clips as static 3D skeleton images.
Uses the motion library pipeline (MotionLibSMPL) to get proper rigid body positions.

Usage:
    python scripts/vis/render_debug_clips.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    """Setup SMPL robot and skeleton tree."""
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
    return sk_tree


def get_rb_pos(motion_file, clip_idx, frame, sk_tree, device):
    """Get rigid body positions at a specific frame."""
    motion_lib_cfg = EasyDict({
        "motion_file": motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": 'smpl', "randomrize_heading": False,
    })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[sk_tree],
        gender_betas=[torch.zeros(17)],
        limb_weights=[np.zeros(10)],
        random_sample=False, start_idx=clip_idx
    )
    motion_len = motion_lib.get_motion_length(0).item()
    t = min(frame / 30.0, motion_len - 0.01)
    res = motion_lib.get_motion_state(
        torch.tensor([0]).to(device), torch.tensor([t]).to(device)
    )
    return res["rg_pos"][0].cpu().numpy()


def plot_skeleton(ax, rb_pos, color='blue', label='', alpha=1.0):
    """Plot 3D skeleton."""
    ax.scatter(rb_pos[:, 0], rb_pos[:, 1], rb_pos[:, 2],
               c=color, s=20, alpha=alpha, zorder=5)
    for i, j in SMPL_BONES:
        if i < len(rb_pos) and j < len(rb_pos):
            ax.plot([rb_pos[i, 0], rb_pos[j, 0]],
                    [rb_pos[i, 1], rb_pos[j, 1]],
                    [rb_pos[i, 2], rb_pos[j, 2]],
                    c=color, linewidth=1.5, alpha=alpha)
    # Mark head
    if 15 < len(rb_pos):
        ax.scatter([rb_pos[15, 0]], [rb_pos[15, 1]], [rb_pos[15, 2]],
                   c=color, s=80, marker='o', alpha=alpha, edgecolors='black')
    # Label key joints
    for idx in [0, 15, 7, 8, 20, 21]:
        if idx < len(rb_pos):
            ax.text(rb_pos[idx, 0], rb_pos[idx, 1], rb_pos[idx, 2] + 0.03,
                    JOINT_NAMES[idx], fontsize=5, color=color, alpha=alpha)


def main():
    device = torch.device("cpu")
    motion_file = "sample_data/debug_axis_test.pkl"

    if not os.path.exists(motion_file):
        print(f"Error: {motion_file} not found. Run debug_axis_mapping.py first.")
        return

    sk_tree = setup_smpl()

    # Load to get clip names
    data = joblib.load(motion_file)
    clip_names = sorted(data.keys())
    n_clips = len(clip_names)
    print(f"Found {n_clips} clips: {clip_names}")

    # Render each clip at frame 30 (or frame 15 for short clips)
    fig = plt.figure(figsize=(6 * min(n_clips, 5), 10 * ((n_clips + 4) // 5)))
    n_cols = min(n_clips, 5)
    n_rows = ((n_clips + n_cols - 1) // n_cols) * 2  # 2 rows per clip row (front + side)

    for ci, clip_name in enumerate(clip_names):
        clip_T = data[clip_name]["pose_quat_global"].shape[0]
        frame = min(30, clip_T - 1)

        print(f"Rendering {clip_name} at frame {frame}...")
        try:
            rb_pos = get_rb_pos(motion_file, ci, frame, sk_tree, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Center at origin XY
        rb_pos[:, :2] -= rb_pos[0, :2].copy()

        row_group = ci // n_cols
        col = ci % n_cols

        # Front view
        ax = fig.add_subplot(n_rows, n_cols, row_group * 2 * n_cols + col + 1, projection='3d')
        plot_skeleton(ax, rb_pos, 'blue')
        ax.set_title(f'{clip_name}\nfront (frame {frame})', fontsize=8)
        ax.view_init(elev=10, azim=0)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # Side view
        ax2 = fig.add_subplot(n_rows, n_cols, (row_group * 2 + 1) * n_cols + col + 1, projection='3d')
        plot_skeleton(ax2, rb_pos, 'blue')
        ax2.set_title(f'{clip_name}\nside (frame {frame})', fontsize=8)
        ax2.view_init(elev=10, azim=90)
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(0, 2)
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

        # Print key positions
        pelvis_z = rb_pos[0, 2]
        head_z = rb_pos[15, 2] if 15 < len(rb_pos) else 0
        l_ankle_z = rb_pos[7, 2] if 7 < len(rb_pos) else 0
        r_ankle_z = rb_pos[8, 2] if 8 < len(rb_pos) else 0
        print(f"  {clip_name}: pelvis_z={pelvis_z:.3f}, head_z={head_z:.3f}, "
              f"L_ankle_z={l_ankle_z:.3f}, R_ankle_z={r_ankle_z:.3f}")

    plt.tight_layout()
    output = "output/debug_axis_test.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

"""Quick render: H5 (direct mapping) vs AMASS reference at walking frames."""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
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


def plot_skeleton(ax, rb_pos, color='blue', alpha=1.0, label=''):
    ax.scatter(rb_pos[:, 0], rb_pos[:, 1], rb_pos[:, 2],
               c=color, s=20, alpha=alpha, zorder=5)
    for i, j in SMPL_BONES:
        if i < len(rb_pos) and j < len(rb_pos):
            ax.plot([rb_pos[i, 0], rb_pos[j, 0]],
                    [rb_pos[i, 1], rb_pos[j, 1]],
                    [rb_pos[i, 2], rb_pos[j, 2]],
                    c=color, linewidth=1.5, alpha=alpha)
    if 15 < len(rb_pos):
        ax.scatter([rb_pos[15, 0]], [rb_pos[15, 1]], [rb_pos[15, 2]],
                   c=color, s=80, marker='o', alpha=alpha, edgecolors='black')


def main():
    device = torch.device("cpu")
    sk_tree = setup_smpl()

    h5_file = "sample_data/h5_motion_library.pkl"
    amass_file = "sample_data/amass_isaac_walking_primitive.pkl"

    # Use clip index that corresponds to level_100mps walking
    # Clips are sorted alphabetically; level_100mps clips start around index ~100+
    import joblib
    data = joblib.load(h5_file)
    clip_names = sorted(data.keys())

    # Find a level_100mps clip that's in the walking part (not first few clips which are standing)
    h5_clip_idx = None
    for i, name in enumerate(clip_names):
        if "level_100mps" in name and "c010" in name:  # clip 10 = ~50s into trial
            h5_clip_idx = i
            print(f"Using H5 clip: {name} (idx={i})")
            break
    if h5_clip_idx is None:
        # fallback: first level_100mps clip with high index
        for i, name in enumerate(clip_names):
            if "level_100mps" in name:
                h5_clip_idx = i
                print(f"Fallback H5 clip: {name} (idx={i})")
                break
    if h5_clip_idx is None:
        h5_clip_idx = 50
        print(f"Using clip idx {h5_clip_idx}: {clip_names[h5_clip_idx]}")

    frames = [0, 15, 30, 45, 60, 90, 120]

    fig = plt.figure(figsize=(4 * len(frames), 16))

    for fi, frame in enumerate(frames):
        # H5 front
        h5_pos = get_positions(h5_file, h5_clip_idx, frame, sk_tree, device)
        h5_pos[:, :2] -= h5_pos[0, :2]

        ax = fig.add_subplot(4, len(frames), fi + 1, projection='3d')
        plot_skeleton(ax, h5_pos, 'red')
        ax.view_init(elev=10, azim=0)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.1, 1.8)
        if fi == 0: ax.set_ylabel('H5 front', fontsize=8)
        ax.set_title(f'f{frame}', fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

        # H5 side
        ax2 = fig.add_subplot(4, len(frames), len(frames) + fi + 1, projection='3d')
        plot_skeleton(ax2, h5_pos, 'red')
        ax2.view_init(elev=10, azim=90)
        ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.8, 0.8); ax2.set_zlim(-0.1, 1.8)
        if fi == 0: ax2.set_ylabel('H5 side', fontsize=8)
        ax2.set_xticklabels([]); ax2.set_yticklabels([]); ax2.set_zticklabels([])

        if fi == 0:
            print(f"H5 f{frame}: pelvis_z={h5_pos[0,2]:.3f}, head_z={h5_pos[15,2]:.3f}, "
                  f"L_ankle={h5_pos[7,2]:.3f}, R_ankle={h5_pos[8,2]:.3f}")

        # AMASS front
        if os.path.exists(amass_file):
            amass_pos = get_positions(amass_file, 0, frame, sk_tree, device)
            amass_pos[:, :2] -= amass_pos[0, :2]

            ax3 = fig.add_subplot(4, len(frames), 2*len(frames) + fi + 1, projection='3d')
            plot_skeleton(ax3, amass_pos, 'green')
            ax3.view_init(elev=10, azim=0)
            ax3.set_xlim(-0.8, 0.8); ax3.set_ylim(-0.8, 0.8); ax3.set_zlim(-0.1, 1.8)
            if fi == 0: ax3.set_ylabel('AMASS front', fontsize=8)
            ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])

            ax4 = fig.add_subplot(4, len(frames), 3*len(frames) + fi + 1, projection='3d')
            plot_skeleton(ax4, amass_pos, 'green')
            ax4.view_init(elev=10, azim=90)
            ax4.set_xlim(-0.8, 0.8); ax4.set_ylim(-0.8, 0.8); ax4.set_zlim(-0.1, 1.8)
            if fi == 0: ax4.set_ylabel('AMASS side', fontsize=8)
            ax4.set_xticklabels([]); ax4.set_yticklabels([]); ax4.set_zticklabels([])

            if fi == 0:
                print(f"AMASS f{frame}: pelvis_z={amass_pos[0,2]:.3f}, head_z={amass_pos[15,2]:.3f}, "
                      f"L_ankle={amass_pos[7,2]:.3f}, R_ankle={amass_pos[8,2]:.3f}")

    plt.suptitle("H5 Direct Mapping (red) vs AMASS Reference (green)", fontsize=14)
    plt.tight_layout()
    output = "output/h5_direct_vs_amass.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

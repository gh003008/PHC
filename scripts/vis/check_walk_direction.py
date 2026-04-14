"""
Lightweight stick-figure visualization to verify walking direction.

Loads the converted H5 motion library PKL, performs FK to get joint positions,
and plots a side-view (X-Z) stick figure at several frames across one gait cycle.
Also plots hip flexion angle over time alongside root X displacement.

Usage:
    python scripts/vis/check_walk_direction.py
    python scripts/vis/check_walk_direction.py --pkl sample_data/h5_motion_library.pkl
    python scripts/vis/check_walk_direction.py --amass sample_data/amass_copycat_take5_train.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# SMPL kinematic tree
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# Approximate SMPL bone lengths (meters) from parent to child
# These are rough — just for visualization, not exact SMPL mesh
BONE_LENGTHS = {
    0: 0.0,     # Pelvis (root)
    1: 0.10,    # L_Hip offset from pelvis
    2: 0.10,    # R_Hip offset from pelvis
    3: 0.10,    # Torso (spine1) up from pelvis
    4: 0.40,    # L_Knee (thigh length)
    5: 0.40,    # R_Knee (thigh length)
    6: 0.15,    # Spine (spine2)
    7: 0.42,    # L_Ankle (shank length)
    8: 0.42,    # R_Ankle (shank length)
    9: 0.15,    # Chest (spine3)
    10: 0.15,   # L_Toe
    11: 0.15,   # R_Toe
    12: 0.15,   # Neck
    13: 0.05,   # L_Thorax
    14: 0.05,   # R_Thorax
    15: 0.15,   # Head
    16: 0.15,   # L_Shoulder
    17: 0.15,   # R_Shoulder
    18: 0.28,   # L_Elbow (upper arm)
    19: 0.28,   # R_Elbow
    20: 0.25,   # L_Wrist (forearm)
    21: 0.25,   # R_Wrist
    22: 0.10,   # L_Hand
    23: 0.10,   # R_Hand
}

# Bone direction in SMPL T-pose (unit vector from parent toward child)
# In Z-up convention after Ry(180°): +X = forward, +Y = left, +Z = up
BONE_DIRS = {
    0: np.array([0, 0, 0]),
    1: np.array([0, 1, 0]),     # L_Hip: left of pelvis
    2: np.array([0, -1, 0]),    # R_Hip: right of pelvis
    3: np.array([0, 0, 1]),     # Torso: up
    4: np.array([0, 0, -1]),    # L_Knee: down (thigh)
    5: np.array([0, 0, -1]),    # R_Knee: down (thigh)
    6: np.array([0, 0, 1]),     # Spine: up
    7: np.array([0, 0, -1]),    # L_Ankle: down (shank)
    8: np.array([0, 0, -1]),    # R_Ankle: down (shank)
    9: np.array([0, 0, 1]),     # Chest: up
    10: np.array([1, 0, 0]),    # L_Toe: forward
    11: np.array([1, 0, 0]),    # R_Toe: forward
    12: np.array([0, 0, 1]),    # Neck: up
    13: np.array([0, 1, 0]),    # L_Thorax: left
    14: np.array([0, -1, 0]),   # R_Thorax: right
    15: np.array([0, 0, 1]),    # Head: up
    16: np.array([0, 1, 0]),    # L_Shoulder: left
    17: np.array([0, -1, 0]),   # R_Shoulder: right
    18: np.array([0, 0, -1]),   # L_Elbow: down (upper arm)
    19: np.array([0, 0, -1]),   # R_Elbow: down
    20: np.array([0, 0, -1]),   # L_Wrist: down (forearm)
    21: np.array([0, 0, -1]),   # R_Wrist: down
    22: np.array([0, 0, -1]),   # L_Hand
    23: np.array([0, 0, -1]),   # R_Hand
}


def fk_positions(global_quats, root_trans):
    """Compute joint positions from global rotations + root translation.

    Args:
        global_quats: (24, 4) wxyz global quaternions for one frame
        root_trans: (3,) root position
    Returns:
        (24, 3) joint positions
    """
    positions = np.zeros((24, 3))
    positions[0] = root_trans

    for j in range(1, 24):
        parent = SMPL_PARENTS[j]
        # Bone vector in parent's global frame
        bone_dir = BONE_DIRS[j]
        bone_len = BONE_LENGTHS[j]

        # Rotate bone direction by parent's global rotation
        pq = global_quats[parent]  # wxyz
        pq_xyzw = np.array([pq[1], pq[2], pq[3], pq[0]])
        rot = Rotation.from_quat(pq_xyzw)
        rotated_dir = rot.apply(bone_dir * bone_len)

        positions[j] = positions[parent] + rotated_dir

    return positions


def extract_hip_flexion(pose_quat_local, side="left"):
    """Extract hip flexion angle from local quaternions.

    Hip flexion in PiG/SMPL convention: rotation about local X-axis.
    """
    idx = 1 if side == "left" else 2
    quats = pose_quat_local[:, idx]  # (T, 4) wxyz
    # Convert to scipy
    xyzw = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])
    rots = Rotation.from_quat(xyzw)
    euler = rots.as_euler('XYZ', degrees=True)
    return euler[:, 0]  # X-axis = flexion


def load_amass_clip(amass_path):
    """Load one clip from AMASS pkl for comparison."""
    data = joblib.load(amass_path)
    # AMASS pkl is a dict of clip_name -> clip_data
    clip_name = list(data.keys())[0]
    clip = data[clip_name]
    print(f"AMASS clip: {clip_name}")

    pqg = clip.get("pose_quat_global", None)
    trans = clip.get("trans_orig", None)
    if trans is None:
        trans = clip.get("root_trans_offset", None)
        if hasattr(trans, 'numpy'):
            trans = trans.numpy()

    fps = clip.get("fps", 30)

    # Try to get local quats
    pq_local = clip.get("pose_quat", None)

    return pqg, pq_local, trans, fps, clip_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="sample_data/h5_motion_library.pkl")
    parser.add_argument("--amass", default=None, help="Optional AMASS pkl for comparison")
    parser.add_argument("--clip", default=None, help="Specific clip name")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to show")
    parser.add_argument("--out", default="output/walk_direction_check.png")
    args = parser.parse_args()

    # Load H5 data
    data = joblib.load(args.pkl)
    clip_name = args.clip if args.clip else list(data.keys())[0]
    clip = data[clip_name]

    pqg = clip["pose_quat_global"]  # (T, 24, 4) wxyz
    trans = clip["trans_orig"]       # (T, 3)
    pq_local = clip.get("pose_quat", None)
    fps = clip.get("fps", 30)
    T = pqg.shape[0]

    print(f"H5 clip: {clip_name}, {T} frames @ {fps} fps = {T/fps:.1f}s")
    print(f"Translation range: X [{trans[:, 0].min():.2f}, {trans[:, 0].max():.2f}], "
          f"Z [{trans[:, 2].min():.2f}, {trans[:, 2].max():.2f}]")

    # --- Figure layout ---
    fig = plt.figure(figsize=(20, 14))

    # Top row: stick figures (side view X-Z)
    ax_stick = fig.add_subplot(3, 1, 1)

    # Select frames evenly across the first 2 seconds (about one gait cycle)
    n_frames = args.frames
    cycle_end = min(int(2.0 * fps), T)
    frame_indices = np.linspace(0, cycle_end - 1, n_frames, dtype=int)

    # Skeleton connections for drawing
    skeleton_pairs = [
        (0, 1), (1, 4), (4, 7), (7, 10),   # Left leg
        (0, 2), (2, 5), (5, 8), (8, 11),   # Right leg
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Spine to head
        (9, 13), (13, 16), (16, 18), (18, 20),  # Left arm
        (9, 14), (14, 17), (17, 19), (19, 21),  # Right arm
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))

    for i, fi in enumerate(frame_indices):
        positions = fk_positions(pqg[fi], trans[fi])

        # Plot skeleton (side view: X = forward, Z = up)
        for (a, b) in skeleton_pairs:
            lw = 2.5 if b in [4, 5, 7, 8, 10, 11] else 1.5  # Thicker legs
            ax_stick.plot(
                [positions[a, 0], positions[b, 0]],
                [positions[a, 2], positions[b, 2]],
                color=colors[i], linewidth=lw, alpha=0.7
            )

        # Mark joints
        for j in [0, 4, 5, 7, 8, 10, 11, 15]:
            ax_stick.plot(positions[j, 0], positions[j, 2], 'o',
                         color=colors[i], markersize=4)

        # Time label at head
        ax_stick.text(positions[15, 0], positions[15, 2] + 0.05,
                     f"t={fi/fps:.2f}s", fontsize=7, ha='center',
                     color=colors[i])

    # Draw arrow showing forward direction
    x_start = trans[0, 0]
    x_end = trans[min(cycle_end-1, T-1), 0]
    ax_stick.annotate('', xy=(x_end, 0.05), xytext=(x_start, 0.05),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_stick.text((x_start + x_end) / 2, 0.10, 'Translation direction',
                 ha='center', color='red', fontsize=10)

    ax_stick.set_xlabel('X (forward)')
    ax_stick.set_ylabel('Z (up)')
    ax_stick.set_title(f'Side view: {clip_name} (first {cycle_end/fps:.1f}s)')
    ax_stick.set_aspect('equal')
    ax_stick.grid(True, alpha=0.3)

    # Middle row: hip flexion angles
    ax_hip = fig.add_subplot(3, 1, 2)
    if pq_local is not None:
        t_axis = np.arange(T) / fps
        l_hip_flex = extract_hip_flexion(pq_local, "left")
        r_hip_flex = extract_hip_flexion(pq_local, "right")
        ax_hip.plot(t_axis, l_hip_flex, 'b-', label='L_Hip flexion', linewidth=1.5)
        ax_hip.plot(t_axis, r_hip_flex, 'r-', label='R_Hip flexion', linewidth=1.5)
        ax_hip.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_hip.set_xlabel('Time (s)')
        ax_hip.set_ylabel('Angle (deg)')
        ax_hip.set_title('Hip flexion angle (+ = flexion in PiG convention)')
        ax_hip.legend()
        ax_hip.grid(True, alpha=0.3)
        ax_hip.set_xlim(0, min(4.0, T/fps))

    # Bottom row: root trajectory X over time + knee X positions
    ax_traj = fig.add_subplot(3, 1, 3)
    t_axis = np.arange(T) / fps
    ax_traj.plot(t_axis, trans[:, 0], 'k-', label='Root X', linewidth=2)

    # Also plot left and right ankle X positions to see leg swing
    l_ankle_x = []
    r_ankle_x = []
    for fi in range(min(T, int(4.0 * fps))):
        pos = fk_positions(pqg[fi], trans[fi])
        l_ankle_x.append(pos[7, 0])
        r_ankle_x.append(pos[8, 0])

    t_short = np.arange(len(l_ankle_x)) / fps
    ax_traj.plot(t_short, l_ankle_x, 'b--', label='L_Ankle X', linewidth=1, alpha=0.7)
    ax_traj.plot(t_short, r_ankle_x, 'r--', label='R_Ankle X', linewidth=1, alpha=0.7)

    ax_traj.set_xlabel('Time (s)')
    ax_traj.set_ylabel('X position (m)')
    ax_traj.set_title('Root & ankle X positions (ankles should swing AHEAD of root during stance)')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlim(0, min(4.0, T/fps))

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Print diagnostic info
    print(f"\n--- Diagnostic ---")
    print(f"Root X: starts {trans[0, 0]:.3f}, ends {trans[-1, 0]:.3f}, "
          f"delta = {trans[-1, 0] - trans[0, 0]:.3f}")
    print(f"Root Z (height): mean {trans[:, 2].mean():.3f}")

    if pq_local is not None:
        print(f"L_Hip flexion: min {l_hip_flex.min():.1f}°, max {l_hip_flex.max():.1f}°, "
              f"mean {l_hip_flex.mean():.1f}°")
        print(f"R_Hip flexion: min {r_hip_flex.min():.1f}°, max {r_hip_flex.max():.1f}°, "
              f"mean {r_hip_flex.mean():.1f}°")

    # Check: do ankle positions go AHEAD (larger X) of root during walking?
    # In correct forward walking, the foot swings ahead of CoM during swing phase
    l_ahead = np.array(l_ankle_x) - trans[:len(l_ankle_x), 0]
    r_ahead = np.array(r_ankle_x) - trans[:len(r_ankle_x), 0]
    print(f"L_Ankle relative to root: [{l_ahead.min():.3f}, {l_ahead.max():.3f}]")
    print(f"R_Ankle relative to root: [{r_ahead.min():.3f}, {r_ahead.max():.3f}]")

    if l_ahead.max() < 0 and r_ahead.max() < 0:
        print("\n** WARNING: Ankles never go AHEAD of root — legs swing BACKWARD! **")
        print("** Fix: negate hip flexion sign in axis_signs (change X sign from 1 to -1) **")
    elif l_ahead.min() > 0 and r_ahead.min() > 0:
        print("\n** WARNING: Ankles always ahead of root — may indicate backward lean **")
    else:
        print("\n** OK: Ankles swing both ahead and behind root (normal gait) **")


if __name__ == "__main__":
    main()

"""Quick single-frame posture visualization from H5 motion library PKL."""
import sys, os
sys.path.append(os.getcwd())

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# SMPL kinematic tree
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

SMPL_JOINT_NAMES = [
    "Pelvis","L_Hip","R_Hip","Torso","L_Knee","R_Knee",
    "Spine","L_Ankle","R_Ankle","Chest","L_Toe","R_Toe",
    "Neck","L_Thorax","R_Thorax","Head","L_Shoulder","R_Shoulder",
    "L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"
]

BONES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(12,15),(9,13),(9,14),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),(20,22),(21,23)
]

# Approximate SMPL rest-pose bone lengths (meters) from pelvis
BONE_LENGTHS = {
    (0,1): 0.09, (0,2): 0.09, (0,3): 0.12,
    (1,4): 0.41, (2,5): 0.41, (4,7): 0.40, (5,8): 0.40,
    (7,10): 0.18, (8,11): 0.18,
    (3,6): 0.12, (6,9): 0.12, (9,12): 0.18,
    (12,15): 0.12,
    (9,13): 0.08, (9,14): 0.08,
    (13,16): 0.12, (14,17): 0.12,
    (16,18): 0.28, (17,19): 0.28,
    (18,20): 0.25, (19,21): 0.25,
    (20,22): 0.10, (21,23): 0.10,
}

# Rest-pose local offsets (child relative to parent, in parent's frame)
# SMPL convention: Y-up in rest pose, but after base rotation we work in Z-up
# These are approximate unit direction vectors before scaling by bone length
REST_OFFSETS_UNIT = {
    (0,1): np.array([0, -1, 0]),    # L_Hip: left
    (0,2): np.array([0, 1, 0]),     # R_Hip: right
    (0,3): np.array([0, 0, 1]),     # Torso: up
    (1,4): np.array([0, 0, -1]),    # L_Knee: down
    (2,5): np.array([0, 0, -1]),    # R_Knee: down
    (4,7): np.array([0, 0, -1]),    # L_Ankle: down
    (5,8): np.array([0, 0, -1]),    # R_Ankle: down
    (7,10): np.array([1, 0, 0]),    # L_Toe: forward
    (8,11): np.array([1, 0, 0]),    # R_Toe: forward
    (3,6): np.array([0, 0, 1]),     # Spine: up
    (6,9): np.array([0, 0, 1]),     # Chest: up
    (9,12): np.array([0, 0, 1]),    # Neck: up
    (12,15): np.array([0, 0, 1]),   # Head: up
    (9,13): np.array([0, -1, 0]),   # L_Thorax: left
    (9,14): np.array([0, 1, 0]),    # R_Thorax: right
    (13,16): np.array([0, -1, 0]),  # L_Shoulder: left
    (14,17): np.array([0, 1, 0]),   # R_Shoulder: right
    (16,18): np.array([0, 0, -1]),  # L_Elbow: down
    (17,19): np.array([0, 0, -1]),  # R_Elbow: down
    (18,20): np.array([0, 0, -1]),  # L_Wrist: down
    (19,21): np.array([0, 0, -1]),  # R_Wrist: down
    (20,22): np.array([0, 0, -1]),  # L_Hand: down
    (21,23): np.array([0, 0, -1]),  # R_Hand: down
}


def fk_global_rotations(pose_quat_global, root_pos=np.array([0, 0, 0.95])):
    """Forward kinematics using global rotations + rest-pose offsets."""
    positions = np.zeros((24, 3))
    positions[0] = root_pos

    # Global rotation matrices
    rot_mats = np.zeros((24, 3, 3))
    for j in range(24):
        q_wxyz = pose_quat_global[j]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rot_mats[j] = Rotation.from_quat(q_xyzw).as_matrix()

    for parent, child in BONES:
        offset_dir = REST_OFFSETS_UNIT[(parent, child)]
        length = BONE_LENGTHS[(parent, child)]
        # Rotate rest-pose offset by PARENT's global rotation
        world_offset = rot_mats[parent] @ (offset_dir * length)
        positions[child] = positions[parent] + world_offset

    return positions


def plot_skeleton(ax, pos, color, label, alpha=1.0):
    """Plot a 3D stick figure."""
    # Left side = blue-ish, Right side = red-ish for clarity
    left_bones = [(0,1),(1,4),(4,7),(7,10),(9,13),(13,16),(16,18),(18,20),(20,22)]
    right_bones = [(0,2),(2,5),(5,8),(8,11),(9,14),(14,17),(17,19),(19,21),(21,23)]
    mid_bones = [(0,3),(3,6),(6,9),(9,12),(12,15)]

    for i, j in mid_bones:
        ax.plot([pos[i,0],pos[j,0]], [pos[i,1],pos[j,1]], [pos[i,2],pos[j,2]],
                c='green', linewidth=2.5, alpha=alpha)
    for i, j in left_bones:
        ax.plot([pos[i,0],pos[j,0]], [pos[i,1],pos[j,1]], [pos[i,2],pos[j,2]],
                c='blue', linewidth=2.5, alpha=alpha)
    for i, j in right_bones:
        ax.plot([pos[i,0],pos[j,0]], [pos[i,1],pos[j,1]], [pos[i,2],pos[j,2]],
                c='red', linewidth=2.5, alpha=alpha)

    # Joint dots
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='black', s=20, zorder=5)

    # Label key joints
    for idx in [0, 4, 5, 7, 8, 10, 11, 15, 16, 17, 20, 21]:
        ax.text(pos[idx,0], pos[idx,1], pos[idx,2]+0.03,
                SMPL_JOINT_NAMES[idx], fontsize=6, ha='center')


def main():
    pkl_path = "sample_data/h5_motion_library.pkl"
    print(f"Loading {pkl_path}...")
    data = joblib.load(pkl_path)

    clip_names = sorted(data.keys())
    print(f"Total clips: {len(clip_names)}")

    # Pick a clip near the middle for a representative walking frame
    clip_name = clip_names[len(clip_names)//2]
    clip = data[clip_name]
    print(f"Using clip: {clip_name}")

    pose_quat_global = clip["pose_quat_global"]  # (T, 24, 4) wxyz
    pose_quat_local = clip["pose_quat"]           # (T, 24, 4) wxyz
    trans = clip["trans_orig"]                     # (T, 3)
    T = pose_quat_global.shape[0]
    print(f"Frames: {T}")

    # Pick 3 frames: start, mid-stance, mid-swing
    frames = [0, T//4, T//2]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"H5 Posture: {clip_name}\nBlue=Left, Red=Right, Green=Spine", fontsize=13)

    for fi, frame in enumerate(frames):
        root_pos = trans[frame]
        pqg = pose_quat_global[frame]  # (24, 4)

        pos = fk_global_rotations(pqg, root_pos)

        # Front view
        ax = fig.add_subplot(2, 3, fi+1, projection='3d')
        plot_skeleton(ax, pos, 'black', 'H5')
        ax.set_title(f'Frame {frame} — Front', fontsize=11)
        ax.view_init(elev=5, azim=0)
        ax.set_xlabel('X (fwd)')
        ax.set_ylabel('Y (lat)')
        ax.set_zlabel('Z (up)')
        ax.set_xlim(root_pos[0]-0.8, root_pos[0]+0.8)
        ax.set_ylim(root_pos[1]-0.8, root_pos[1]+0.8)
        ax.set_zlim(0, 2.0)
        ax.set_aspect('equal')

        # Side view
        ax2 = fig.add_subplot(2, 3, fi+4, projection='3d')
        plot_skeleton(ax2, pos, 'black', 'H5')
        ax2.set_title(f'Frame {frame} — Side', fontsize=11)
        ax2.view_init(elev=5, azim=90)
        ax2.set_xlabel('X (fwd)')
        ax2.set_ylabel('Y (lat)')
        ax2.set_zlabel('Z (up)')
        ax2.set_xlim(root_pos[0]-0.8, root_pos[0]+0.8)
        ax2.set_ylim(root_pos[1]-0.8, root_pos[1]+0.8)
        ax2.set_zlim(0, 2.0)
        ax2.set_aspect('equal')

    plt.tight_layout()
    out = "output/h5_posture_check.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # Also print joint angles (Euler) for frame 0 to sanity-check
    print(f"\n--- Frame 0 local rotation (Euler XYZ degrees) ---")
    pql = pose_quat_local[frames[0]]
    for j in range(24):
        q = pql[j]
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        euler = Rotation.from_quat(q_xyzw).as_euler('XYZ', degrees=True)
        marker = " *" if np.any(np.abs(euler) > 5) else ""
        print(f"  [{j:2d}] {SMPL_JOINT_NAMES[j]:<14} x={euler[0]:7.2f}  y={euler[1]:7.2f}  z={euler[2]:7.2f}{marker}")


if __name__ == "__main__":
    main()

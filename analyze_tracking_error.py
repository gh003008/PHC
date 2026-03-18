"""Analyze per-body tracking error over first motion cycle.

Usage:
    python analyze_tracking_error.py --npy output/VIC_PHASE/tracking_error_log.npy
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# SMPL body index mapping
BODY_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
]

# L/R leg pairs: (name, L_idx, R_idx)
LEG_PAIRS = [
    ('Hip',   1, 5),
    ('Knee',  2, 6),
    ('Ankle', 3, 7),
    ('Toe',   4, 8),
]

DT = 1.0 / 30.0  # 30 Hz control


def analyze(npy_path, first_cycle_steps=132):
    data = np.load(npy_path, allow_pickle=True)

    # Find episode boundaries (where step resets)
    steps = np.array([d[0] for d in data])
    errors = np.array([d[1] for d in data])  # [N, 24]

    # Find reset points
    resets = [0] + [i for i in range(1, len(steps)) if steps[i] <= steps[i-1]]

    # Average first cycle across all episodes
    first_cycles = []
    for ep_start in resets:
        ep_end = ep_start + first_cycle_steps
        if ep_end <= len(data):
            first_cycles.append(errors[ep_start:ep_end])

    print(f"Found {len(first_cycles)} complete first-cycle episodes")
    avg_errors = np.mean(first_cycles, axis=0)  # [first_cycle_steps, 24]
    std_errors = np.std(first_cycles, axis=0)

    time_axis = np.arange(first_cycle_steps) * DT  # seconds

    out_dir = os.path.dirname(npy_path)

    # --- Plot 1: L vs R lower limb tracking error ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, (name, l_id, r_id) in enumerate(LEG_PAIRS):
        ax = axes[idx]
        l_mean = avg_errors[:, l_id] * 100  # convert to cm
        r_mean = avg_errors[:, r_id] * 100
        l_std = std_errors[:, l_id] * 100
        r_std = std_errors[:, r_id] * 100

        ax.plot(time_axis, l_mean, color='#2196F3', linewidth=2, label=f'L_{name}')
        ax.fill_between(time_axis, l_mean - l_std, l_mean + l_std, color='#2196F3', alpha=0.15)
        ax.plot(time_axis, r_mean, color='#F44336', linewidth=2, label=f'R_{name}')
        ax.fill_between(time_axis, r_mean - r_std, r_mean + r_std, color='#F44336', alpha=0.15)

        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Tracking Error (cm)')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[2].set_xlabel('Time (s)')
    axes[3].set_xlabel('Time (s)')
    fig.suptitle('L vs R Leg Tracking Error (First Motion Cycle, avg over episodes)', fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tracking_error_LR_legs.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"L/R leg tracking plot: {out_path}")
    plt.close()

    # --- Plot 2: All body groups summary ---
    groups = {
        'Pelvis': [0],
        'L_Leg (Hip+Knee+Ankle)': [1, 2, 3],
        'R_Leg (Hip+Knee+Ankle)': [5, 6, 7],
        'Spine+Chest': [9, 10, 11],
        'L_Arm': [15, 16, 17],
        'R_Arm': [20, 21, 22],
    }
    colors = ['#607D8B', '#2196F3', '#F44336', '#4CAF50', '#9C27B0', '#FF9800']

    fig, ax = plt.subplots(figsize=(12, 5))
    for (gname, gids), color in zip(groups.items(), colors):
        g_mean = avg_errors[:, gids].mean(axis=1) * 100
        ax.plot(time_axis, g_mean, linewidth=2, label=gname, color=color)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Tracking Error (cm)', fontsize=11)
    ax.set_title('Per-Group Tracking Error (First Motion Cycle)', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tracking_error_groups.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Group tracking plot: {out_path}")
    plt.close()

    # --- Print summary stats ---
    print("\n=== First Cycle Tracking Error Summary (cm) ===")
    print(f"{'Body':<15} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print("-" * 42)
    for name, l_id, r_id in LEG_PAIRS:
        l_mean_cm = avg_errors[:, l_id].mean() * 100
        l_max_cm = avg_errors[:, l_id].max() * 100
        l_std_cm = avg_errors[:, l_id].std() * 100
        r_mean_cm = avg_errors[:, r_id].mean() * 100
        r_max_cm = avg_errors[:, r_id].max() * 100
        r_std_cm = avg_errors[:, r_id].std() * 100
        print(f"L_{name:<11} {l_mean_cm:>7.2f}  {l_max_cm:>7.2f}  {l_std_cm:>7.2f}")
        print(f"R_{name:<11} {r_mean_cm:>7.2f}  {r_max_cm:>7.2f}  {r_std_cm:>7.2f}")
    pelvis_cm = avg_errors[:, 0].mean() * 100
    print(f"{'Pelvis':<15} {pelvis_cm:>7.2f}  {avg_errors[:, 0].max()*100:>7.2f}  {avg_errors[:, 0].std()*100:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', default='output/VIC_PHASE/tracking_error_log.npy')
    parser.add_argument('--steps', type=int, default=132, help='Steps in first motion cycle')
    args = parser.parse_args()
    analyze(args.npy, first_cycle_steps=args.steps)


if __name__ == '__main__':
    main()

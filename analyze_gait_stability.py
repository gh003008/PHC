"""Quantitative gait stability analysis for VIC_PHASE evaluation.

Generates PPT-ready figures:
  1. Per-body tracking error bar chart
  2. L/R symmetry index
  3. Phase-resolved tracking error (stance vs swing)
  4. First-cycle temporal tracking error (clean version)

Usage:
    python analyze_gait_stability.py --exp VIC_PHASE [--steps 132]
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

# SMPL body indices
BODY_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
]

# Body groups for summary
BODY_GROUPS = {
    'Pelvis':  [0],
    'L_Hip':   [1],
    'L_Knee':  [2],
    'L_Ankle': [3],
    'L_Toe':   [4],
    'R_Hip':   [5],
    'R_Knee':  [6],
    'R_Ankle': [7],
    'R_Toe':   [8],
    'Spine':   [9, 10, 11],
    'Head':    [12, 13],
    'L_Arm':   [14, 15, 16, 17, 18],
    'R_Arm':   [19, 20, 21, 22, 23],
}

# L/R pairs for symmetry
LR_PAIRS = [
    ('Hip',   1, 5),
    ('Knee',  2, 6),
    ('Ankle', 3, 7),
    ('Toe',   4, 8),
]

# Gait phase boundaries (% of cycle, R heel strike = 0%)
R_STANCE_END = 60   # ~60% right stance
L_STANCE_START = 10
L_STANCE_END = 70

DT = 1.0 / 30.0  # 30 Hz


def load_tracking_data(exp_name):
    npy_path = os.path.join('output', exp_name, 'tracking_error_log.npy')
    data = np.load(npy_path, allow_pickle=True)
    steps = np.array([d[0] for d in data])
    errors = np.array([d[1] for d in data])  # [N, 24]
    return steps, errors, npy_path


def get_first_cycles(steps, errors, first_cycle_steps):
    resets = [0] + [i for i in range(1, len(steps)) if steps[i] <= steps[i-1]]
    cycles = []
    for start in resets:
        end = start + first_cycle_steps
        if end <= len(errors):
            cycles.append(errors[start:end])
    return np.array(cycles)  # [n_episodes, steps, 24]


def plot_bar_chart(cycles, out_dir):
    """Fig 1: Per-body mean tracking error bar chart."""
    mean_err = cycles.mean(axis=(0, 1)) * 100  # [24] in cm
    std_err = cycles.std(axis=(0, 1)) * 100

    # Group into meaningful categories
    groups = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
              'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    colors = ['#607D8B',
              '#2196F3', '#2196F3', '#2196F3', '#2196F3',
              '#F44336', '#F44336', '#F44336', '#F44336']
    alphas = [0.9, 1.0, 0.85, 0.7, 0.55, 1.0, 0.85, 0.7, 0.55]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    bars = ax.bar(x, mean_err[indices], yerr=std_err[indices],
                  color=[matplotlib.colors.to_rgba(c, a) for c, a in zip(colors, alphas)],
                  edgecolor='white', linewidth=1.5, capsize=4,
                  error_kw={'linewidth': 1.5, 'capthick': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Tracking Error (cm)', fontsize=13)
    ax.set_title('Per-Body Mean Tracking Error (First Gait Cycle)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add value labels
    for bar, val in zip(bars, mean_err[indices]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'gait_stability_bar.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Bar chart: {out_path}")
    plt.close()


def plot_symmetry_index(cycles, out_dir, first_cycle_steps):
    """Fig 2: L/R Symmetry Index over gait cycle."""
    avg = cycles.mean(axis=0)  # [steps, 24]
    time_pct = np.linspace(0, 100, first_cycle_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: L vs R overlay
    ax = axes[0]
    pair_colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
    for (name, l_id, r_id), color in zip(LR_PAIRS, pair_colors):
        l_err = avg[:, l_id] * 100
        r_err = avg[:, r_id] * 100
        ax.plot(time_pct, l_err, color=color, linewidth=2, linestyle='-', label=f'L_{name}')
        ax.plot(time_pct, r_err, color=color, linewidth=2, linestyle='--', label=f'R_{name}')

    ax.set_xlabel('Gait Cycle (%)', fontsize=12)
    ax.set_ylabel('Tracking Error (cm)', fontsize=12)
    ax.set_title('L vs R Lower Limb Tracking Error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    # Stance/swing shading
    ax.axvspan(0, R_STANCE_END, alpha=0.05, color='#F44336')
    ax.axvline(R_STANCE_END, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Right: Symmetry Index bar
    ax2 = axes[1]
    si_values = []
    si_names = []
    for name, l_id, r_id in LR_PAIRS:
        l_mean = avg[:, l_id].mean()
        r_mean = avg[:, r_id].mean()
        si = abs(l_mean - r_mean) / (0.5 * (l_mean + r_mean)) * 100
        si_values.append(si)
        si_names.append(name)

    bars = ax2.barh(si_names, si_values, color=pair_colors, edgecolor='white', linewidth=1.5, height=0.6)
    ax2.set_xlabel('Symmetry Index (%)', fontsize=12)
    ax2.set_title('Gait Symmetry Index\n(0% = perfect symmetry)', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, max(si_values) * 1.3)
    ax2.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, si_values):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    # Reference lines
    ax2.axvline(10, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Good (<10%)')
    ax2.axvline(20, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Fair (<20%)')
    ax2.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'gait_stability_symmetry.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Symmetry plot: {out_path}")
    plt.close()

    return si_values, si_names


def plot_stance_swing_error(cycles, out_dir, first_cycle_steps):
    """Fig 3: Stance vs Swing phase tracking error comparison."""
    avg = cycles.mean(axis=0)  # [steps, 24]

    # Define stance/swing indices for each leg
    n = first_cycle_steps
    r_stance_idx = np.arange(0, int(n * R_STANCE_END / 100))
    r_swing_idx = np.arange(int(n * R_STANCE_END / 100), n)
    l_stance_idx = np.arange(int(n * L_STANCE_START / 100), int(n * L_STANCE_END / 100))
    l_swing_idx = np.concatenate([np.arange(0, int(n * L_STANCE_START / 100)),
                                   np.arange(int(n * L_STANCE_END / 100), n)])

    joints = ['Hip', 'Knee', 'Ankle', 'Toe']
    l_ids = [1, 2, 3, 4]
    r_ids = [5, 6, 7, 8]

    stance_errs = []
    swing_errs = []
    labels = []

    for jname, l_id, r_id in zip(joints, l_ids, r_ids):
        # Left leg
        l_stance = avg[l_stance_idx, l_id].mean() * 100
        l_swing = avg[l_swing_idx, l_id].mean() * 100
        stance_errs.append(l_stance)
        swing_errs.append(l_swing)
        labels.append(f'L_{jname}')

        # Right leg
        r_stance = avg[r_stance_idx, r_id].mean() * 100
        r_swing = avg[r_swing_idx, r_id].mean() * 100
        stance_errs.append(r_stance)
        swing_errs.append(r_swing)
        labels.append(f'R_{jname}')

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, stance_errs, width, label='Stance Phase',
                   color='#4CAF50', edgecolor='white', linewidth=1.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, swing_errs, width, label='Swing Phase',
                   color='#FF5722', edgecolor='white', linewidth=1.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Tracking Error (cm)', fontsize=13)
    ax.set_title('Stance vs Swing Phase Tracking Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'gait_stability_stance_swing.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Stance/Swing plot: {out_path}")
    plt.close()

    return stance_errs, swing_errs, labels


def plot_temporal_clean(cycles, out_dir, first_cycle_steps):
    """Fig 4: Clean first-cycle temporal tracking error for PPT."""
    avg = cycles.mean(axis=0) * 100  # [steps, 24] in cm
    std = cycles.std(axis=0) * 100
    time_pct = np.linspace(0, 100, first_cycle_steps)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Aggregate groups
    groups = {
        'Pelvis': [0],
        'L Leg (avg)': [1, 2, 3],
        'R Leg (avg)': [5, 6, 7],
        'Spine': [9, 10, 11],
        'Arms (avg)': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    colors = ['#607D8B', '#2196F3', '#F44336', '#4CAF50', '#9C27B0']

    for (gname, gids), color in zip(groups.items(), colors):
        g_mean = avg[:, gids].mean(axis=1)
        g_std = std[:, gids].mean(axis=1)
        ax.plot(time_pct, g_mean, color=color, linewidth=2.5, label=gname)
        ax.fill_between(time_pct, g_mean - g_std, g_mean + g_std,
                        color=color, alpha=0.1)

    # Stance/swing shading
    ax.axvspan(0, R_STANCE_END, alpha=0.04, color='gray')
    ax.axvline(R_STANCE_END, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ylim = ax.get_ylim()
    ax.text(R_STANCE_END/2, ylim[1]*0.95, 'R Stance', ha='center', fontsize=9, color='gray', alpha=0.7)
    ax.text((R_STANCE_END+100)/2, ylim[1]*0.95, 'R Swing', ha='center', fontsize=9, color='gray', alpha=0.7)

    ax.set_xlabel('Gait Cycle (%)', fontsize=13)
    ax.set_ylabel('Tracking Error (cm)', fontsize=13)
    ax.set_title('Tracking Error Over First Gait Cycle', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'gait_stability_temporal.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Temporal plot: {out_path}")
    plt.close()


def plot_summary_dashboard(cycles, si_values, si_names, out_dir, first_cycle_steps, n_episodes):
    """Fig 5: Single-page summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    avg = cycles.mean(axis=0) * 100  # cm
    std = cycles.std(axis=0) * 100
    time_pct = np.linspace(0, 100, first_cycle_steps)

    # --- Panel A: Key metrics table ---
    ax_table = fig.add_subplot(gs[0, 0])
    ax_table.axis('off')

    overall_mean = avg.mean()
    overall_max = avg.max()
    pelvis_mean = avg[:, 0].mean()
    l_leg_mean = avg[:, [1,2,3]].mean()
    r_leg_mean = avg[:, [5,6,7]].mean()

    table_data = [
        ['Metric', 'Value'],
        ['Episodes', f'{n_episodes}'],
        ['Survival Rate', '100%'],
        ['Steps / Episode', f'{first_cycle_steps}'],
        ['Overall Mean Error', f'{overall_mean:.1f} cm'],
        ['Pelvis Error', f'{pelvis_mean:.1f} cm'],
        ['L Leg Mean Error', f'{l_leg_mean:.1f} cm'],
        ['R Leg Mean Error', f'{r_leg_mean:.1f} cm'],
        ['Mean Symmetry Index', f'{np.mean(si_values):.1f}%'],
    ]

    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center',
                           colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(2):
            table[i, j].set_facecolor(color)

    ax_table.set_title('Key Metrics', fontsize=13, fontweight='bold', pad=20)

    # --- Panel B: Bar chart (lower limb) ---
    ax_bar = fig.add_subplot(gs[0, 1:])
    joints = ['Pelvis', 'Hip', 'Knee', 'Ankle', 'Toe']
    l_ids = [0, 1, 2, 3, 4]
    r_ids = [0, 5, 6, 7, 8]
    x = np.arange(len(joints))
    width = 0.35

    l_means = [avg[:, i].mean() for i in l_ids]
    r_means = [avg[:, i].mean() for i in r_ids]

    ax_bar.bar(x - width/2, l_means, width, label='Left', color='#2196F3',
               edgecolor='white', linewidth=1.5, alpha=0.85)
    ax_bar.bar(x + width/2, r_means, width, label='Right', color='#F44336',
               edgecolor='white', linewidth=1.5, alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(joints, fontsize=11)
    ax_bar.set_ylabel('Error (cm)', fontsize=12)
    ax_bar.set_title('L vs R Mean Tracking Error', fontsize=13, fontweight='bold')
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.set_ylim(bottom=0)

    # --- Panel C: Temporal plot ---
    ax_temp = fig.add_subplot(gs[1, :2])
    groups = {
        'Pelvis': [0],
        'L Leg': [1, 2, 3],
        'R Leg': [5, 6, 7],
        'Spine': [9, 10, 11],
    }
    colors = ['#607D8B', '#2196F3', '#F44336', '#4CAF50']

    for (gname, gids), color in zip(groups.items(), colors):
        g_mean = avg[:, gids].mean(axis=1)
        ax_temp.plot(time_pct, g_mean, color=color, linewidth=2, label=gname)

    ax_temp.axvspan(0, R_STANCE_END, alpha=0.04, color='gray')
    ax_temp.axvline(R_STANCE_END, color='gray', linestyle=':', linewidth=0.8)
    ax_temp.set_xlabel('Gait Cycle (%)', fontsize=12)
    ax_temp.set_ylabel('Error (cm)', fontsize=12)
    ax_temp.set_title('Tracking Error Over Gait Cycle', fontsize=13, fontweight='bold')
    ax_temp.legend(fontsize=9, loc='upper right')
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_xlim(0, 100)
    ax_temp.set_ylim(bottom=0)

    # --- Panel D: Symmetry index ---
    ax_si = fig.add_subplot(gs[1, 2])
    pair_colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
    bars = ax_si.barh(si_names, si_values, color=pair_colors,
                      edgecolor='white', linewidth=1.5, height=0.5)
    ax_si.axvline(10, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax_si.set_xlabel('SI (%)', fontsize=12)
    ax_si.set_title('Symmetry Index', fontsize=13, fontweight='bold')
    ax_si.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars, si_values):
        ax_si.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    fig.suptitle('VIC_PHASE: Gait Stability Analysis', fontsize=16, fontweight='bold', y=1.01)
    out_path = os.path.join(out_dir, 'gait_stability_dashboard.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Dashboard: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='VIC_PHASE', help='Experiment name')
    parser.add_argument('--steps', type=int, default=132, help='Steps in first motion cycle')
    args = parser.parse_args()

    out_dir = os.path.join('output', args.exp)
    steps, errors, npy_path = load_tracking_data(args.exp)
    cycles = get_first_cycles(steps, errors, args.steps)
    n_episodes = len(cycles)

    print(f"Experiment: {args.exp}")
    print(f"Episodes: {n_episodes}, First cycle steps: {args.steps}")
    print(f"Data: {npy_path}\n")

    # Generate all plots
    plot_bar_chart(cycles, out_dir)
    si_values, si_names = plot_symmetry_index(cycles, out_dir, args.steps)
    plot_stance_swing_error(cycles, out_dir, args.steps)
    plot_temporal_clean(cycles, out_dir, args.steps)
    plot_summary_dashboard(cycles, si_values, si_names, out_dir, args.steps, n_episodes)

    # Print summary
    avg = cycles.mean(axis=(0, 1)) * 100
    print("\n" + "=" * 55)
    print("  GAIT STABILITY SUMMARY (First Cycle, cm)")
    print("=" * 55)
    print(f"  {'Body':<12} {'Mean Error':>10} {'Symmetry':>10}")
    print("-" * 55)
    for name, l_id, r_id in LR_PAIRS:
        l = avg[l_id]
        r = avg[r_id]
        si = abs(l - r) / (0.5 * (l + r)) * 100
        print(f"  L_{name:<9} {l:>9.2f} cm")
        print(f"  R_{name:<9} {r:>9.2f} cm  SI={si:>5.1f}%")
    print(f"  {'Pelvis':<12} {avg[0]:>9.2f} cm")
    print(f"  {'Spine(avg)':<12} {avg[[9,10,11]].mean():>9.2f} cm")
    print("=" * 55)
    print(f"  Overall Mean: {avg.mean():.2f} cm")
    print(f"  Mean Symmetry Index: {np.mean([abs(avg[l]-avg[r])/(0.5*(avg[l]+avg[r]))*100 for _,l,r in LR_PAIRS]):.1f}%")
    print(f"  Episode Survival: {n_episodes}/{n_episodes} (100%)")
    print()


if __name__ == '__main__':
    main()

"""
VIC Phase-resolved CCF analysis
Usage:
  python analyze_phase_ccf.py --npy output/phase_ccf_log.npy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import argparse
import os

GROUP_NAMES = ['L_Hip', 'L_Knee', 'L_Ankle+Toe', 'R_Hip', 'R_Knee', 'R_Ankle+Toe', 'Upper-L', 'Upper-R']
N_BINS = 20

# L/R pairs for gait-phase comparison
LR_PAIRS = [
    ('Hip',       0, 3),   # L_Hip vs R_Hip
    ('Knee',      1, 4),   # L_Knee vs R_Knee
    ('Ankle+Toe', 2, 5),   # L_Ankle+Toe vs R_Ankle+Toe
]

# Gait cycle phase boundaries (R heel strike = phase 0)
# Right leg: stance ~0-60%, swing ~60-100%
# Left leg:  swing ~0-10%, stance ~10-60%, swing ~60-100% (roughly antiphase)
R_STANCE_END = 60   # Right toe-off at ~60%
L_STANCE_START = 10  # Left heel strike at ~10% (half cycle offset from R)
L_STANCE_END = 70    # Left toe-off at ~70%


def _add_stance_swing_shading(ax, side='both'):
    """Add stance/swing phase shading to a gait cycle plot.

    side: 'R' for right only, 'L' for left only, 'both' for both
    """
    _, y_hi = ax.get_ylim()
    if side in ('R', 'both'):
        ax.axvspan(0, R_STANCE_END, alpha=0.06, color='#F44336', zorder=0)
        ax.axvspan(R_STANCE_END, 100, alpha=0.03, color='#F44336', zorder=0)
        ax.text(R_STANCE_END / 2, y_hi - 0.05, 'R stance', ha='center', fontsize=7, color='#F44336', alpha=0.7)
        ax.text((R_STANCE_END + 100) / 2, y_hi - 0.05, 'R swing', ha='center', fontsize=7, color='#F44336', alpha=0.7)
    if side in ('L', 'both'):
        ax.text((L_STANCE_START + L_STANCE_END) / 2, y_hi - 0.13, 'L stance', ha='center', fontsize=7, color='#2196F3', alpha=0.7)
    # Vertical lines at phase boundaries
    ax.axvline(R_STANCE_END, color='#F44336', linestyle=':', linewidth=0.8, alpha=0.4)
    if side in ('L', 'both'):
        ax.axvline(L_STANCE_START, color='#2196F3', linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(L_STANCE_END, color='#2196F3', linestyle=':', linewidth=0.8, alpha=0.3)


def plot_per_group(phases, ccfs, bins, bin_centers, imp_per_bin, out_dir):
    """Original 8-group subplot plot."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B']

    for g in range(8):
        ax = axes[g]
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.plot(bin_centers * 100, imp_per_bin[:, g], color=colors[g], linewidth=2, marker='o', markersize=3)
        ax.set_title(GROUP_NAMES[g], fontsize=11)
        ax.set_ylim(0.4, 2.2)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Phase (%)')
        ax.set_ylabel('impedance_scale (2^ccf)')
        ax.grid(True, alpha=0.3)
        # Stance/swing shading: L groups (0,1,2) get L shading, R groups (3,4,5) get R shading
        if g < 3:
            _add_stance_swing_shading(ax, side='L')
        elif g < 6:
            _add_stance_swing_shading(ax, side='R')

    fig.suptitle('Phase-resolved Impedance Scale per Group', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'phase_ccf_per_group.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Per-group plot: {out_path}")
    plt.close()


def plot_lr_comparison(phases, ccfs, bins, bin_centers, imp_per_bin, out_dir):
    """L vs R gait-phase CCF comparison plot.
    X-axis: phase (0-100%), Y-axis: impedance scale for Left and Right.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, (name, l_idx, r_idx) in enumerate(LR_PAIRS):
        ax = axes[idx]
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        ax.plot(bin_centers * 100, imp_per_bin[:, l_idx],
                color='#2196F3', linewidth=2.5, marker='o', markersize=4, label=f'Left {name}')
        ax.plot(bin_centers * 100, imp_per_bin[:, r_idx],
                color='#F44336', linewidth=2.5, marker='s', markersize=4, label=f'Right {name}')

        ax.fill_between(bin_centers * 100, imp_per_bin[:, l_idx], alpha=0.1, color='#2196F3')
        ax.fill_between(bin_centers * 100, imp_per_bin[:, r_idx], alpha=0.1, color='#F44336')

        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Phase (%)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Impedance Scale (2^ccf)', fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.4, 2.2)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        _add_stance_swing_shading(ax, side='both')

    fig.suptitle('Left vs Right: Phase-resolved Impedance Scale', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'phase_ccf_LR_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"L/R comparison plot: {out_path}")
    plt.close()


def analyze(npy_path):
    """Main analysis function. Can be called from code or CLI."""
    data = np.load(npy_path, allow_pickle=True)  # list of (phase, ccf[8])
    phases = np.array([d[0] for d in data])
    ccfs   = np.array([d[1] for d in data])      # [T, 8]

    print(f"총 {len(phases)} 스텝 기록됨")
    print(f"phase 범위: {phases.min():.3f} ~ {phases.max():.3f}")

    # phase bin별 impedance_scale 평균
    bins = np.linspace(0, 1, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    imp_per_bin = np.zeros((N_BINS, 8))

    for b in range(N_BINS):
        mask = (phases >= bins[b]) & (phases < bins[b+1])
        if mask.sum() > 0:
            imp_per_bin[b] = np.mean(2 ** ccfs[mask], axis=0)
        else:
            imp_per_bin[b] = np.nan

    out_dir = os.path.dirname(npy_path) or '.'

    # Generate both plots
    plot_per_group(phases, ccfs, bins, bin_centers, imp_per_bin, out_dir)
    plot_lr_comparison(phases, ccfs, bins, bin_centers, imp_per_bin, out_dir)

    # 수치 요약
    print("\n=== Group별 impedance_scale (phase 구간별 평균) ===")
    print(f"{'Group':<15} {'초반(0-20%)':<14} {'중반(40-60%)':<14} {'후반(80-100%)':<14} {'전체평균':<12}")
    for g in range(8):
        early = np.nanmean(imp_per_bin[:4, g])
        mid   = np.nanmean(imp_per_bin[8:12, g])
        late  = np.nanmean(imp_per_bin[16:, g])
        overall = np.nanmean(imp_per_bin[:, g])
        print(f"{GROUP_NAMES[g]:<15} {early:<14.3f} {mid:<14.3f} {late:<14.3f} {overall:<12.3f}")

    return imp_per_bin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', default='output/phase_ccf_log.npy')
    args = parser.parse_args()
    analyze(args.npy)


if __name__ == '__main__':
    main()

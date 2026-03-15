"""
VIC_PHASE: Gait cycle phase별 CCF 시각화 (좌/우 분리, stride별 trace)
Usage:
  python analyze_phase_ccf_gait.py --npy output/phase_ccf_log.npy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

GROUP_NAMES = ['L_Hip', 'L_Knee', 'L_Ankle+Toe', 'R_Hip', 'R_Knee', 'R_Ankle+Toe', 'Upper-L', 'Upper-R']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', default='output/phase_ccf_log.npy')
    args = parser.parse_args()

    data = np.load(args.npy, allow_pickle=True)
    phases = np.array([float(d[0]) for d in data])
    ccfs = np.array([d[1] for d in data])  # [T, 8]
    imp_scales = 2 ** ccfs  # impedance scale

    # Detect stride boundaries (phase wraps)
    diffs = np.diff(phases)
    wrap_indices = np.where(diffs < -0.3)[0]

    # Extract individual strides
    boundaries = [0] + list(wrap_indices + 1) + [len(phases)]
    strides = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        p = phases[s:e]
        c = imp_scales[s:e]
        # Full stride: starts near 0 and has enough length
        if p[0] < 0.1 and len(p) > 50:
            strides.append((p, c))

    print(f"총 {len(phases)} 스텝, {len(wrap_indices)} stride boundaries")
    print(f"Full strides (phase 0부터 시작, 50+ steps): {len(strides)}")

    # ======== Plot 1: L/R 좌우 분리 (individual stride traces) ========
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)

    l_groups = [0, 1, 2]  # L_Hip, L_Knee, L_Ankle+Toe
    r_groups = [3, 4, 5]  # R_Hip, R_Knee, R_Ankle+Toe
    l_colors = ['#2196F3', '#1565C0', '#0D47A1']
    r_colors = ['#FF5722', '#D84315', '#BF360C']
    group_labels = ['Hip', 'Knee', 'Ankle+Toe']

    for col, (lg, rg) in enumerate(zip(l_groups, r_groups)):
        ax_l = axes[0, col]
        ax_r = axes[1, col]

        for si, (p, c) in enumerate(strides):
            alpha = 0.6 if len(strides) > 5 else 0.8
            lw = 1.0
            ax_l.plot(p, c[:, lg], color=l_colors[col], alpha=alpha, linewidth=lw,
                     label=f'Stride {si+1}' if si < 5 else None)
            ax_r.plot(p, c[:, rg], color=r_colors[col], alpha=alpha, linewidth=lw,
                     label=f'Stride {si+1}' if si < 5 else None)

        for ax, side, gname in [(ax_l, 'L', group_labels[col]), (ax_r, 'R', group_labels[col])]:
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_title(f'{side}_{gname}', fontsize=12, fontweight='bold')
            ax.set_ylim(0.3, 2.2)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Impedance Scale (2^ccf)')
            # Approximate stance/swing shading for gait cycle
            # Phase 0 = R heel strike. R stance ~0-0.6, R swing ~0.6-1.0
            # L is opposite: L swing ~0-0.5, L stance ~0.5-1.0 (approximately)
            ax.axvspan(0.0, 0.6, alpha=0.06, color='green')
            ax.axvspan(0.6, 1.0, alpha=0.06, color='orange')

        axes[1, col].set_xlabel('Gait Phase (0=R heel strike → 1=next R heel strike)')

    axes[0, 0].legend(fontsize=7, loc='upper right')
    axes[0, 0].set_ylabel('LEFT side\nImpedance Scale', fontsize=11)
    axes[1, 0].set_ylabel('RIGHT side\nImpedance Scale', fontsize=11)

    fig.suptitle(f'VIC_PHASE: CCF per Gait Cycle Phase (L/R separated, {len(strides)} full strides)',
                fontsize=14, y=1.02)
    plt.tight_layout()
    out1 = args.npy.replace('.npy', '_gait_LR.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"좌/우 분리 플롯 저장: {out1}")
    plt.close()

    # ======== Plot 2: All 8 groups, per-stride traces ========
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors_all = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B']

    for g in range(8):
        ax = axes[g]
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        for si, (p, c) in enumerate(strides):
            ax.plot(p, c[:, g], color=colors_all[g], alpha=0.5, linewidth=0.8)

        ax.set_title(GROUP_NAMES[g], fontsize=11, fontweight='bold')
        ax.set_ylim(0.3, 2.2)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axvspan(0.0, 0.6, alpha=0.06, color='green')
        ax.axvspan(0.6, 1.0, alpha=0.06, color='orange')

        if g >= 4:
            ax.set_xlabel('Gait Phase')
        if g % 4 == 0:
            ax.set_ylabel('Impedance Scale (2^ccf)')

    fig.suptitle(f'VIC_PHASE: All Groups CCF per Gait Phase ({len(strides)} full strides)\n'
                 'Green=R_stance(~0-0.6), Orange=R_swing(~0.6-1.0)',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    out2 = args.npy.replace('.npy', '_gait_all.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"전체 그룹 플롯 저장: {out2}")
    plt.close()

    # ======== 수치 요약: Stance vs Swing ========
    print("\n=== Gait Phase 구간별 Impedance Scale (R heel strike 기준) ===")
    print(f"  Phase 0.0-0.6 ≈ R_stance / L_swing")
    print(f"  Phase 0.6-1.0 ≈ R_swing  / L_stance")

    all_p = np.concatenate([p for p, c in strides])
    all_c = np.concatenate([c for p, c in strides])

    stance_mask = all_p < 0.6
    swing_mask = all_p >= 0.6

    print(f"\n{'Group':<15} {'R_stance(0-0.6)':<18} {'R_swing(0.6-1.0)':<18} {'차이':<10} {'전체':<10}")
    for g in range(8):
        st = np.mean(all_c[stance_mask, g])
        sw = np.mean(all_c[swing_mask, g])
        ov = np.mean(all_c[:, g])
        diff = sw - st
        print(f"{GROUP_NAMES[g]:<15} {st:<18.3f} {sw:<18.3f} {diff:+<10.3f} {ov:<10.3f}")

if __name__ == '__main__':
    main()

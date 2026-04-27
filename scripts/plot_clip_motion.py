"""Plot root x position, x velocity, and foot z heights over time for a clip
in a pkl. Shows whether motion has 'stops' (flat regions in vel plot).

Usage:
  python scripts/plot_clip_motion.py <pkl> <motion_idx> [--out path] [--seconds N]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('motion_idx', type=int)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    d = joblib.load(args.pkl)
    keys = list(d.keys())
    key = keys[args.motion_idx]
    clip = d[key]
    trans = np.asarray(clip['trans_orig'])
    pose_aa = np.asarray(clip['pose_aa']) if 'pose_aa' in clip else None
    fps = 30
    n = min(int(args.seconds * fps), trans.shape[0])

    t = np.arange(n) / fps
    root_x = trans[:n, 0]
    root_z = trans[:n, 2]

    # Per-frame velocity (signed, finite difference)
    root_vx = np.diff(root_x, prepend=root_x[0]) * fps   # m/s

    # Smoothed velocity for trend
    win = 5
    root_vx_smooth = np.convolve(root_vx, np.ones(win) / win, mode='same')

    # Foot z: approximate via pose_aa root height + leg motion if available
    # Easier — just plot root z (proxy for body up-down)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'Motion analysis: {os.path.basename(args.pkl)} idx={args.motion_idx}\n{key}', fontsize=11)

    axes[0].plot(t, root_x, lw=1.2)
    axes[0].set_ylabel('root x [m]')
    axes[0].set_title(f'Root forward position — should be straight line if continuous walking')
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, root_vx, lw=0.6, alpha=0.5, label='per-frame')
    axes[1].plot(t, root_vx_smooth, lw=1.5, color='C1', label=f'{win}-frame smoothed')
    axes[1].axhline(0, color='k', lw=0.5)
    mean_v = np.mean(root_vx)
    axes[1].axhline(mean_v, color='C2', ls='--', lw=0.8, label=f'mean = {mean_v:.3f}')
    axes[1].set_ylabel('root vx [m/s]')
    axes[1].set_title(f'Root forward velocity — flat dips/spikes = "stops"; should ripple around mean')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, root_z, lw=1.2)
    axes[2].set_ylabel('root z [m]')
    axes[2].set_title(f'Root height — should oscillate ±5cm at stride frequency, no drift')
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel('time [s]')

    out = args.out or f'videos/motion_plot_{os.path.splitext(os.path.basename(args.pkl))[0]}_idx{args.motion_idx}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f'[plot] saved {out}')
    print(f'[plot] mean vx = {mean_v:.3f} m/s, vx range = [{root_vx.min():.3f}, {root_vx.max():.3f}]')
    print(f'[plot] root z range = [{root_z.min():.4f}, {root_z.max():.4f}]')


if __name__ == '__main__':
    main()

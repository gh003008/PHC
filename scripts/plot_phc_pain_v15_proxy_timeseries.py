import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("analysis/plots/phc_pain_v15_joint_timeseries")
PRETRAINED = OUT_DIR / "pretrained_timeseries.json"
LATEST = OUT_DIR / "latest_timeseries.json"


PANELS = [
    ("pain_v1_right_knee_load", "Right Knee OA Load"),
    ("pain_v1_right_knee_state", "Right Knee Pain State"),
    ("pain_v1_right_knee_contact_load", "Right Knee Contact Load"),
    ("pain_v1_right_knee_moment_load", "Right Knee Moment Load"),
    ("pain_v1_right_knee_kam", "Right Knee KAM Proxy"),
    ("pain_v1_right_knee_kfm", "Right Knee KFM Proxy"),
    ("pain_v1_right_knee_compression", "Right Knee Compression"),
    ("pain_v1_right_knee_loaded_flex", "Right Knee Loaded Flexion"),
    ("pain_v1_right_knee_torque_load", "Right Knee Legacy Torque Load"),
    ("pain_v1_right_knee_tau_rms", "Right Knee Tau RMS"),
]

LEFT_RIGHT_PANELS = [
    ("pain_v1_left_knee_load", "pain_v1_right_knee_load", "Final OA Load"),
    ("pain_v1_left_knee_state", "pain_v1_right_knee_state", "Pain State"),
    ("pain_v1_left_knee_contact_load", "pain_v1_right_knee_contact_load", "Contact Load"),
    ("pain_v1_left_knee_moment_load", "pain_v1_right_knee_moment_load", "Moment Load"),
]


def load_series(path):
    with path.open() as f:
        return json.load(f)


def get(data, key, n=None):
    arr = np.asarray(data[key], dtype=float)
    return arr if n is None else arr[:n]


def common_n(*datasets):
    lengths = []
    for data in datasets:
        for key, _ in PANELS:
            lengths.append(len(data[key]))
        for left, right, _ in LEFT_RIGHT_PANELS:
            lengths.append(len(data[left]))
            lengths.append(len(data[right]))
    return min(lengths)


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_pretrained_vs_latest(pretrained, latest, n):
    t = np.arange(n) / 30.0
    fig, axes = plt.subplots(5, 2, figsize=(9.2, 11.0), sharex=True)
    axes = axes.ravel()
    summary = []

    for ax, (key, title) in zip(axes, PANELS):
        pre = get(pretrained, key, n)
        lat = get(latest, key, n)
        diff = lat - pre
        summary.append(
            (
                key,
                float(np.mean(pre)),
                float(np.mean(lat)),
                float(np.sqrt(np.mean(diff * diff))),
                float(np.max(pre)),
                float(np.max(lat)),
            )
        )
        ax.plot(t, pre, color="#333333", linewidth=1.5, label="PHC pretrained")
        ax.plot(t, lat, color="#C43B3B", linewidth=1.4, label="PHC-Pain v1.5 latest")
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.set_title(title)
        ax.grid(True, color="#E6E6E6", linewidth=0.7)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(
        "Pain Proxy Timeseries: PHC Pretrained vs PHC-Pain v1.5 Latest",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"phc_pain_v15_proxy_timeseries.{ext}", dpi=300)
    plt.close(fig)

    with (OUT_DIR / "pain_proxy_timeseries_summary.csv").open("w") as f:
        f.write("metric,pretrained_mean,latest_mean,rmse_diff,pretrained_max,latest_max\n")
        for row in summary:
            f.write(
                f"{row[0]},{row[1]:.8f},{row[2]:.8f},{row[3]:.8f},{row[4]:.8f},{row[5]:.8f}\n"
            )


def plot_left_right_latest(latest, n):
    t = np.arange(n) / 30.0
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8), sharex=True)
    axes = axes.ravel()

    for ax, (left_key, right_key, title) in zip(axes, LEFT_RIGHT_PANELS):
        left = get(latest, left_key, n)
        right = get(latest, right_key, n)
        ax.plot(t, left, color="#2D6F9F", linewidth=1.4, label="Left knee")
        ax.plot(t, right, color="#C43B3B", linewidth=1.4, label="Right knee")
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.set_title(title)
        ax.grid(True, color="#E6E6E6", linewidth=0.7)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Latest PHC-Pain v1.5: Left vs Right Knee Proxy Channels", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"phc_pain_v15_latest_left_right_proxy_timeseries.{ext}", dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained = load_series(PRETRAINED)
    latest = load_series(LATEST)
    n = common_n(pretrained, latest)
    apply_style()
    plot_pretrained_vs_latest(pretrained, latest, n)
    plot_left_right_latest(latest, n)


if __name__ == "__main__":
    main()

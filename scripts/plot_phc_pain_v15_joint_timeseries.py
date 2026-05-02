import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("analysis/plots/phc_pain_v15_joint_timeseries")
PRETRAINED = OUT_DIR / "pretrained_timeseries.json"
LATEST = OUT_DIR / "latest_timeseries.json"


JOINTS = [
    ("left_hip", "Left Hip"),
    ("right_hip", "Right Hip"),
    ("left_knee", "Left Knee"),
    ("right_knee", "Right Knee"),
    ("left_ankle", "Left Ankle"),
    ("right_ankle", "Right Ankle"),
]


def load_series(path):
    with path.open() as f:
        return json.load(f)


def get_deg(data, joint):
    key = f"lower_limb_{joint}_flex_mean_rad"
    return np.rad2deg(np.asarray(data[key], dtype=float))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained = load_series(PRETRAINED)
    latest = load_series(LATEST)
    n = min(
        min(len(pretrained[f"lower_limb_{joint}_flex_mean_rad"]) for joint, _ in JOINTS),
        min(len(latest[f"lower_limb_{joint}_flex_mean_rad"]) for joint, _ in JOINTS),
    )
    t = np.arange(n) / 30.0

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

    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.2), sharex=True)
    axes = axes.ravel()
    summary = []

    for ax, (joint, title) in zip(axes, JOINTS):
        pre = get_deg(pretrained, joint)[:n]
        lat = get_deg(latest, joint)[:n]
        diff = lat - pre
        summary.append((joint, float(np.mean(pre)), float(np.mean(lat)), float(np.sqrt(np.mean(diff * diff)))))

        ax.plot(t, pre, color="#333333", linewidth=1.6, label="PHC pretrained")
        ax.plot(t, lat, color="#C43B3B", linewidth=1.5, label="PHC-Pain v1.5 latest")
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("Flexion angle (deg)")
        ax.grid(True, color="#E6E6E6", linewidth=0.7)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(
        "Lower-Limb Joint Flexion Timeseries: PHC Pretrained vs PHC-Pain v1.5 Latest",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"phc_pain_v15_lower_limb_joint_timeseries.{ext}", dpi=300)
    plt.close(fig)

    with (OUT_DIR / "joint_timeseries_summary.csv").open("w") as f:
        f.write("joint,pretrained_mean_deg,latest_mean_deg,rmse_diff_deg\n")
        for row in summary:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n")


if __name__ == "__main__":
    main()

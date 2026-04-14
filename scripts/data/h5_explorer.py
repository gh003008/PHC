"""
H5 Explorer for combined_data_from_csv.h5
- H5 구조 탐색, trial 목록, 데이터 품질 검증
- 관절 각도 범위, GRF, CoP, CoM 시각화
- Vicon Plug-in Gait convention 확인

Usage:
    python scripts/data/h5_explorer.py --h5 data/combined_data_from_csv.h5
    python scripts/data/h5_explorer.py --h5 data/combined_data_from_csv.h5 --plot S001/level_100mps/lv0/trial_01
    python scripts/data/h5_explorer.py --h5 data/combined_data_from_csv.h5 --summary
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Joint names in H5 mocap/angle
JOINTS = ["hip", "knee", "ankle", "pelvis", "spine", "thorax",
          "neck", "head", "shoulder", "elbow", "wrist"]
LOWER_JOINTS = ["hip", "knee", "ankle"]
AXES = ["x", "y", "z"]


def list_all_trials(f):
    """List all trials in the H5 file with basic info."""
    print("=" * 80)
    print(f"{'Subject':<8} {'Task':<18} {'Level':<8} {'Trial':<12} {'Samples':>8} {'Duration(s)':>12}")
    print("-" * 80)
    total_trials = 0
    total_samples = 0
    for subj in sorted(f.keys()):
        for task in sorted(f[subj].keys()):
            for level in sorted(f[subj][task].keys()):
                grp = f[f"{subj}/{task}/{level}"]
                # level might be a trial directly or contain trials
                if isinstance(grp, h5py.Dataset):
                    continue
                trials = sorted(grp.keys())
                for trial in trials:
                    trial_path = f"{subj}/{task}/{level}/{trial}"
                    try:
                        time = np.array(f[f"{trial_path}/common/time"])
                        n = len(time)
                        dur = (time[-1] - time[0]) / 1000.0  # ms to s
                        print(f"{subj:<8} {task:<18} {level:<8} {trial:<12} {n:>8} {dur:>12.1f}")
                        total_trials += 1
                        total_samples += n
                    except KeyError:
                        print(f"{subj:<8} {task:<18} {level:<8} {trial:<12} {'ERROR':>8}")
    print("-" * 80)
    print(f"Total: {total_trials} trials, {total_samples} samples")
    print("=" * 80)


def check_data_quality(f, trial_path):
    """Check for NaN, inf, and outliers in a trial."""
    print(f"\n=== Data Quality: {trial_path} ===")
    issues = []

    # Check joint angles
    for side in ["left", "right"]:
        for joint in JOINTS:
            for axis in AXES:
                path = f"{trial_path}/mocap/angle/{side}/{joint}/{axis}"
                try:
                    data = np.array(f[path])
                    n_nan = np.isnan(data).sum()
                    n_inf = np.isinf(data).sum()
                    if n_nan > 0:
                        issues.append(f"  NaN: {side}/{joint}/{axis} ({n_nan} samples)")
                    if n_inf > 0:
                        issues.append(f"  Inf: {side}/{joint}/{axis} ({n_inf} samples)")
                except KeyError:
                    issues.append(f"  MISSING: {side}/{joint}/{axis}")

    # Check CoM
    for axis in AXES:
        path = f"{trial_path}/mocap/com/{axis}"
        try:
            data = np.array(f[path])
            if np.isnan(data).sum() > 0:
                issues.append(f"  NaN: com/{axis}")
        except KeyError:
            issues.append(f"  MISSING: com/{axis}")

    # Check GRF
    for side in ["left", "right"]:
        for axis in AXES:
            path = f"{trial_path}/forceplate/grf/{side}/{axis}"
            try:
                data = np.array(f[path])
                if np.isnan(data).sum() > 0:
                    issues.append(f"  NaN: grf/{side}/{axis}")
            except KeyError:
                issues.append(f"  MISSING: grf/{side}/{axis}")

    if issues:
        print(f"  Found {len(issues)} issues:")
        for iss in issues:
            print(iss)
    else:
        print("  No issues found.")
    return len(issues)


def print_angle_statistics(f, trial_path):
    """Print joint angle statistics to verify units and convention."""
    print(f"\n=== Joint Angle Statistics (degrees expected): {trial_path} ===")
    print(f"{'Side':<6} {'Joint':<10} {'Axis':<5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-" * 65)
    for side in ["left", "right"]:
        for joint in LOWER_JOINTS + ["shoulder", "elbow"]:
            for axis in AXES:
                path = f"{trial_path}/mocap/angle/{side}/{joint}/{axis}"
                try:
                    data = np.array(f[path])
                    data = data[~np.isnan(data)]
                    if len(data) == 0:
                        continue
                    print(f"{side:<6} {joint:<10} {axis:<5} "
                          f"{data.mean():>8.2f} {data.std():>8.2f} "
                          f"{data.min():>8.2f} {data.max():>8.2f} "
                          f"{data.max()-data.min():>8.2f}")
                except KeyError:
                    pass


def print_grf_statistics(f, trial_path):
    """Print GRF statistics."""
    print(f"\n=== GRF Statistics (N expected): {trial_path} ===")
    print(f"{'Side':<6} {'Axis':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for side in ["left", "right"]:
        for axis in AXES:
            path = f"{trial_path}/forceplate/grf/{side}/{axis}"
            try:
                data = np.array(f[path])
                data = data[~np.isnan(data)]
                print(f"{side:<6} {axis:<5} {data.mean():>10.2f} {data.std():>10.2f} "
                      f"{data.min():>10.2f} {data.max():>10.2f}")
            except KeyError:
                pass


def print_com_statistics(f, trial_path):
    """Print CoM statistics."""
    print(f"\n=== CoM Statistics (mm expected): {trial_path} ===")
    for axis in AXES:
        path = f"{trial_path}/mocap/com/{axis}"
        try:
            data = np.array(f[path])
            data = data[~np.isnan(data)]
            print(f"  {axis}: mean={data.mean():.2f}, std={data.std():.2f}, "
                  f"range=[{data.min():.2f}, {data.max():.2f}]")
        except KeyError:
            pass


def print_treadmill_info(f, trial_path):
    """Print treadmill speed and pitch."""
    print(f"\n=== Treadmill Info: {trial_path} ===")
    try:
        speed_l = np.array(f[f"{trial_path}/treadmill/left/speed_leftbelt"])
        speed_r = np.array(f[f"{trial_path}/treadmill/right/speed_rightbelt"])
        pitch = np.array(f[f"{trial_path}/treadmill/pitch"])
        # Remove initial zero phase
        nonzero = speed_l > 0.01
        if nonzero.any():
            speed_active = speed_l[nonzero]
            print(f"  Left belt speed (active): mean={speed_active.mean():.3f}, "
                  f"std={speed_active.std():.3f}, range=[{speed_active.min():.3f}, {speed_active.max():.3f}]")
        else:
            print(f"  Left belt speed: all zero/near-zero")
        print(f"  Pitch: mean={pitch.mean():.2f}, range=[{pitch.min():.2f}, {pitch.max():.2f}]")
    except KeyError as e:
        print(f"  Error: {e}")


def plot_trial(f, trial_path, save_dir=None):
    """Plot joint angles, GRF, CoP, CoM for a single trial."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    time_ms = np.array(f[f"{trial_path}/common/time"])
    time_s = (time_ms - time_ms[0]) / 1000.0

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"Trial: {trial_path}", fontsize=14)

    # 1. Lower limb angles (sagittal = x)
    for col, side in enumerate(["left", "right"]):
        ax = axes[0, col]
        for joint, color in zip(LOWER_JOINTS, ["r", "g", "b"]):
            path = f"{trial_path}/mocap/angle/{side}/{joint}/x"
            try:
                data = np.array(f[path])
                ax.plot(time_s, data, color=color, alpha=0.7, label=f"{joint}")
            except KeyError:
                pass
        ax.set_title(f"{side.capitalize()} Sagittal Angles (x)")
        ax.set_ylabel("Angle")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. GRF vertical (z)
    for col, side in enumerate(["left", "right"]):
        ax = axes[1, col]
        path = f"{trial_path}/forceplate/grf/{side}/z"
        try:
            data = np.array(f[path])
            ax.plot(time_s, data, "k-", alpha=0.7)
            ax.set_title(f"{side.capitalize()} GRF Vertical (z)")
            ax.set_ylabel("Force (N)")
        except KeyError:
            ax.set_title(f"{side.capitalize()} GRF - NOT FOUND")
        ax.grid(True, alpha=0.3)

    # 3. CoM trajectory (x,z)
    ax = axes[2, 0]
    try:
        com_x = np.array(f[f"{trial_path}/mocap/com/x"])
        com_z = np.array(f[f"{trial_path}/mocap/com/z"])
        ax.plot(time_s, com_x, "r-", alpha=0.7, label="CoM x")
        ax.set_ylabel("Position (mm)")
        ax2 = ax.twinx()
        ax2.plot(time_s, com_z, "b-", alpha=0.7, label="CoM z")
        ax2.set_ylabel("Height (mm)")
        ax.set_title("CoM x (red) and z height (blue)")
        ax.grid(True, alpha=0.3)
    except KeyError:
        ax.set_title("CoM - NOT FOUND")

    # 4. CoP trajectory
    ax = axes[2, 1]
    for side, color in zip(["left", "right"], ["b", "r"]):
        try:
            cop_x = np.array(f[f"{trial_path}/forceplate/cop/{side}/x"])
            cop_y = np.array(f[f"{trial_path}/forceplate/cop/{side}/y"])
            # Only plot when foot is in contact (GRF > threshold)
            grf_z = np.array(f[f"{trial_path}/forceplate/grf/{side}/z"])
            mask = np.abs(grf_z) > 20
            ax.scatter(cop_x[mask], cop_y[mask], s=0.5, alpha=0.3, color=color, label=f"{side}")
        except KeyError:
            pass
    ax.set_title("CoP (when in contact)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 5. Treadmill speed
    ax = axes[3, 0]
    try:
        speed = np.array(f[f"{trial_path}/treadmill/left/speed_leftbelt"])
        ax.plot(time_s, speed, "k-", alpha=0.7)
        ax.set_title("Treadmill Speed")
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
    except KeyError:
        ax.set_title("Treadmill - NOT FOUND")
    ax.grid(True, alpha=0.3)

    # 6. Robot hip torque (if available)
    ax = axes[3, 1]
    for side, color in zip(["left", "right"], ["b", "r"]):
        try:
            torque = np.array(f[f"{trial_path}/robot/{side}/torque"])
            ax.plot(time_s, torque, color=color, alpha=0.7, label=f"{side}")
        except KeyError:
            pass
    ax.set_title("Robot Hip Torque")
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is None:
        save_dir = Path("output/h5_explorer")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_name = trial_path.replace("/", "_")
    save_path = save_dir / f"{safe_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


def run_summary(f):
    """Run full summary across all subjects/tasks."""
    print("\n" + "=" * 80)
    print("FULL SUMMARY")
    print("=" * 80)

    # Pick one representative trial per task type for S001 lv0
    representative_trials = []
    subj = "S001"
    for task in sorted(f[subj].keys()):
        for level in sorted(f[subj][task].keys()):
            if "lv0" in level or "trial" in level:
                trials = sorted(f[f"{subj}/{task}/{level}"].keys())
                if trials:
                    trial_path = f"{subj}/{task}/{level}/{trials[0]}"
                    representative_trials.append(trial_path)
                break  # one level per task

    total_issues = 0
    for tp in representative_trials:
        print(f"\n{'='*60}")
        print(f"Trial: {tp}")
        print(f"{'='*60}")
        total_issues += check_data_quality(f, tp)
        print_angle_statistics(f, tp)
        print_grf_statistics(f, tp)
        print_com_statistics(f, tp)
        print_treadmill_info(f, tp)

    print(f"\n\nTotal issues across {len(representative_trials)} representative trials: {total_issues}")


def main():
    parser = argparse.ArgumentParser(description="Explore H5 motion data")
    parser.add_argument("--h5", required=True, help="Path to H5 file")
    parser.add_argument("--plot", type=str, default=None,
                        help="Trial path to plot (e.g., S001/level_100mps/lv0/trial_01)")
    parser.add_argument("--summary", action="store_true",
                        help="Run full summary with statistics")
    parser.add_argument("--list", action="store_true",
                        help="List all trials")
    parser.add_argument("--quality", type=str, default=None,
                        help="Check data quality for a trial path")
    parser.add_argument("--plot_all", action="store_true",
                        help="Plot all trials for S001 lv0")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save plots")
    args = parser.parse_args()

    f = h5py.File(args.h5, "r")

    if args.list or (not args.plot and not args.summary and not args.quality and not args.plot_all):
        list_all_trials(f)

    if args.quality:
        check_data_quality(f, args.quality)
        print_angle_statistics(f, args.quality)
        print_grf_statistics(f, args.quality)
        print_com_statistics(f, args.quality)
        print_treadmill_info(f, args.quality)

    if args.summary:
        run_summary(f)

    if args.plot:
        plot_trial(f, args.plot, save_dir=args.save_dir)

    if args.plot_all:
        subj = "S001"
        for task in sorted(f[subj].keys()):
            for level in sorted(f[subj][task].keys()):
                if "lv0" in level or "trial" in level:
                    trials = sorted(f[f"{subj}/{task}/{level}"].keys())
                    for trial in trials:
                        tp = f"{subj}/{task}/{level}/{trial}"
                        plot_trial(f, tp, save_dir=args.save_dir)
                    break

    f.close()


if __name__ == "__main__":
    main()

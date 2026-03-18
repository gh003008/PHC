"""
Exp5: Patient Profile Comparison

동일한 근골격계 구조에서 파라미터만 변경하여
healthy / stroke(spastic) / flaccid 환자의 차이를 정량 비교.

검증 포인트:
- Healthy: 보통 수준의 passive torque, reflex, impedance
- Stroke(spastic): 높은 stretch reflex, 높은 passive resistance
- Flaccid (SCI complete): reflex=0, 낮은 f_max → 낮은 impedance

비교 항목:
1. Passive torque at ROM boundary (관절별)
2. Reflex torque at fixed velocity (관절별)
3. Max co-contraction impedance (관절별)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import JOINT_DOF_RANGE, NUM_DOFS


def measure_passive_boundary_torque(body, joint_name, side="upper"):
    """ROM 경계 근처(90% 지점)에서의 passive torque 측정."""
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    x_dof = dof_start

    lower = body.joint_limits_lower[x_dof].item()
    upper = body.joint_limits_upper[x_dof].item()

    if side == "upper":
        angle = lower + (upper - lower) * 0.95  # 95% of ROM
    else:
        angle = lower + (upper - lower) * 0.05  # 5% of ROM

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    dof_pos[0, x_dof] = angle

    descending_cmd = torch.zeros(1, body.num_muscles)
    body.activation_dyn.reset()
    body.reflex.reset()
    tau = body.compute_torques(dof_pos, dof_vel, descending_cmd, dt=1/60)
    return abs(tau[0, x_dof].item())


def measure_reflex_torque(body, joint_name, velocity=2.0, dt=1/60):
    """일정 속도에서의 reflex torque 측정."""
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    x_dof = dof_start

    body.activation_dyn.reset()
    body.reflex.reset()

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    dof_vel[0, x_dof] = velocity

    # Run several steps for reflex to engage
    for _ in range(5):
        descending_cmd = torch.zeros(1, body.num_muscles)
        tau = body.compute_torques(dof_pos, dof_vel, descending_cmd, dt=dt)

    return abs(tau[0, x_dof].item())


def measure_max_impedance(body, joint_name, delta=0.01):
    """CC=100%에서의 impedance 측정."""
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    x_dof = dof_start

    R = body.moment_arm.R.numpy()

    activation = torch.zeros(1, body.num_muscles)
    for m_idx in range(body.num_muscles):
        if abs(R[m_idx, x_dof]) > 1e-6:
            activation[0, m_idx] = 1.0

    torques = []
    for offset in [-delta, delta]:
        dof_pos = torch.zeros(1, NUM_DOFS)
        dof_vel = torch.zeros(1, NUM_DOFS)
        dof_pos[0, x_dof] = offset

        ml = body.moment_arm.compute_muscle_length(dof_pos)
        mv = body.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)
        F = body.muscle_model.compute_force(activation, ml, mv)
        tau = body.moment_arm.forces_to_torques(F)
        torques.append(tau[0, x_dof].item())

    return abs(torques[1] - torques[0]) / (2 * delta)


def run(output_dir):
    profiles_dir = os.path.join(os.path.dirname(__file__), "../profiles")

    profile_configs = [
        ("Healthy", "healthy_baseline.yaml", False),
        ("Stroke (R Hemiplegia)", os.path.join(profiles_dir, "stroke/stroke_r_hemiplegia.yaml"), True),
        ("SCI Flaccid (T10)", os.path.join(profiles_dir, "sci/sci_t10_complete_flaccid.yaml"), True),
    ]

    # Check which profiles exist
    available_profiles = []
    for name, path, is_abs in profile_configs:
        if is_abs:
            if os.path.exists(path):
                available_profiles.append((name, path))
            else:
                print(f"  [SKIP] {name}: {path} not found")
        else:
            available_profiles.append((name, path))

    if len(available_profiles) < 2:
        print("  Need at least 2 profiles for comparison. Skipping Exp5.")
        return

    joints = [("L_Hip", "Hip"), ("L_Knee", "Knee"), ("L_Ankle", "Ankle")]

    # Collect data
    data = {}
    for profile_name, profile_path in available_profiles:
        is_abs = os.path.isabs(profile_path)
        body = HumanBody.from_config(
            "muscle_definitions.yaml",
            profile_path,
            num_envs=1, device="cpu",
        )

        data[profile_name] = {
            "passive": {},
            "reflex": {},
            "impedance": {},
        }

        for joint_name, label in joints:
            data[profile_name]["passive"][joint_name] = measure_passive_boundary_torque(body, joint_name)
            data[profile_name]["reflex"][joint_name] = measure_reflex_torque(body, joint_name, velocity=2.0)
            data[profile_name]["impedance"][joint_name] = measure_max_impedance(body, joint_name)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Exp5: Patient Profile Comparison (Healthy vs Pathological)", fontsize=14)

    metrics = [
        ("passive", "Passive Torque at ROM Boundary (Nm)", axes[0]),
        ("reflex", "Reflex Torque at v=2 rad/s (Nm)", axes[1]),
        ("impedance", "Max Impedance CC=100% (Nm/rad)", axes[2]),
    ]

    bar_width = 0.8 / len(available_profiles)
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    for metric_key, metric_title, ax in metrics:
        x = np.arange(len(joints))

        for p_idx, (profile_name, _) in enumerate(available_profiles):
            values = [data[profile_name][metric_key][j[0]] for j in joints]
            offset = (p_idx - len(available_profiles) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width * 0.9,
                          label=profile_name, color=colors[p_idx % len(colors)], alpha=0.8)

            # Value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([j[1] for j in joints])
        ax.set_ylabel(metric_title.split('(')[1].rstrip(')') if '(' in metric_title else '')
        ax.set_title(metric_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp5_patient_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")

    # Print summary table
    print(f"\n  {'Metric':<25} {'Joint':<10}", end="")
    for name, _ in available_profiles:
        print(f" {name:<25}", end="")
    print()
    print(f"  {'-'*100}")

    for metric_key, metric_title, _ in metrics:
        for joint_name, label in joints:
            print(f"  {metric_title[:24]:<25} {label:<10}", end="")
            for name, _ in available_profiles:
                val = data[name][metric_key][joint_name]
                print(f" {val:<25.2f}", end="")
            print()


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    run(output_dir)

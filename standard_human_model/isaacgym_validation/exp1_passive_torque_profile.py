"""
Exp1: Per-joint Passive Torque Profile

각 하지 관절(Hip, Knee, Ankle)의 주 운동축(x)을 ROM 전체에 걸쳐 sweep하며,
passive muscle force + ligament torque를 측정.

검증 포인트:
- ROM 중앙에서는 passive torque가 거의 0
- ROM 경계에서 지수적 증가 (J-shape curve)
- passive muscle force는 최적 길이 이상에서만 발생
- 좌/우 대칭성 확인
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import JOINT_DOF_RANGE, NUM_DOFS


def run(output_dir):
    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1,
        device="cpu",
    )

    joints_to_test = [
        ("L_Hip", "Hip Flexion/Extension"),
        ("L_Knee", "Knee Flexion/Extension"),
        ("L_Ankle", "Ankle Dorsi/Plantar Flexion"),
        ("R_Hip", "Hip Flexion/Extension (R)"),
        ("R_Knee", "Knee Flexion/Extension (R)"),
        ("R_Ankle", "Ankle Dorsi/Plantar Flexion (R)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Exp1: Per-Joint Passive Torque Profile (angle sweep, velocity=0)", fontsize=14)

    for idx, (joint_name, title) in enumerate(joints_to_test):
        row = idx % 3
        col = idx // 3
        ax = axes[row, col]

        dof_start, dof_end = JOINT_DOF_RANGE[joint_name]
        x_dof = dof_start  # primary axis (flexion/extension)

        # angle sweep: joint limits
        lower = body.joint_limits_lower[x_dof].item()
        upper = body.joint_limits_upper[x_dof].item()
        angles = np.linspace(lower, upper, 200)

        tau_muscle_list = []
        tau_ligament_list = []
        tau_total_list = []
        muscle_length_list = []

        for angle in angles:
            dof_pos = torch.zeros(1, NUM_DOFS)
            dof_vel = torch.zeros(1, NUM_DOFS)
            dof_pos[0, x_dof] = angle

            # Muscle passive force (descending_cmd=0, no active contraction)
            ml = body.moment_arm.compute_muscle_length(dof_pos)
            mv = body.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)

            # Zero activation → only passive + damping
            activation = torch.zeros(1, body.num_muscles)
            F_muscle = body.muscle_model.compute_force(activation, ml, mv)
            tau_muscle = body.moment_arm.forces_to_torques(F_muscle)

            # Ligament torque
            tau_lig = body.ligament.compute_torque(dof_pos, dof_vel)

            tau_muscle_list.append(tau_muscle[0, x_dof].item())
            tau_ligament_list.append(tau_lig[0, x_dof].item())
            tau_total_list.append(tau_muscle[0, x_dof].item() + tau_lig[0, x_dof].item())

        angles_deg = np.degrees(angles)
        tau_muscle_arr = np.array(tau_muscle_list)
        tau_ligament_arr = np.array(tau_ligament_list)
        tau_total_arr = np.array(tau_total_list)

        ax.plot(angles_deg, tau_muscle_arr, 'b-', linewidth=2, label='Passive Muscle')
        ax.plot(angles_deg, tau_ligament_arr, 'r-', linewidth=2, label='Ligament')
        ax.plot(angles_deg, tau_total_arr, 'k--', linewidth=1.5, label='Total')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=np.degrees(lower), color='gray', linestyle=':', alpha=0.5, label='ROM limit')
        ax.axvline(x=np.degrees(upper), color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Joint Angle (deg)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(f'{joint_name}: {title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print quantitative summary
        print(f"  {joint_name} ({title}):")
        print(f"    ROM: [{np.degrees(lower):.1f}, {np.degrees(upper):.1f}] deg")
        print(f"    Passive muscle torque range: [{tau_muscle_arr.min():.2f}, {tau_muscle_arr.max():.2f}] Nm")
        print(f"    Ligament torque range: [{tau_ligament_arr.min():.2f}, {tau_ligament_arr.max():.2f}] Nm")
        print(f"    Total passive torque at ROM center: {tau_total_arr[len(tau_total_arr)//2]:.3f} Nm")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp1_passive_torque_profile.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    run(output_dir)

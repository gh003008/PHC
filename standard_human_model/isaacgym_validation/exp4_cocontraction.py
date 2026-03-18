"""
Exp4: Co-contraction Impedance per Joint Group

길항근 쌍(agonist/antagonist)을 동시에 활성화하고,
외란(perturbation)에 대한 관절 강성(impedance) 변화를 측정.

검증 포인트:
- CC=0%: 수동 역학만 → 낮은 impedance
- CC=50%: 중간 impedance
- CC=100%: 최대 impedance
- impedance = delta_torque / delta_angle (수치 미분)
- 모든 하지 관절에서 CC 증가 → impedance 증가
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import JOINT_DOF_RANGE, NUM_DOFS


def compute_impedance(body, joint_name, cc_level, delta_angle=0.01):
    """공수축 수준별 관절 impedance 계산.

    impedance = |delta_tau / delta_angle|

    두 각도(neutral ± delta)에서 토크를 측정하고 수치 미분.
    """
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    x_dof = dof_start

    # Find agonist/antagonist muscles for this joint
    R = body.moment_arm.R.numpy()
    muscle_names = body.moment_arm.muscle_names

    # x축 moment arm 기준으로 agonist (+) / antagonist (-) 분류
    agonist_indices = []
    antagonist_indices = []
    for m_idx in range(body.num_muscles):
        ma = R[m_idx, x_dof]
        if ma > 1e-6:
            agonist_indices.append(m_idx)
        elif ma < -1e-6:
            antagonist_indices.append(m_idx)

    # Set co-contraction activation
    activation = torch.zeros(1, body.num_muscles)
    for m_idx in agonist_indices:
        activation[0, m_idx] = cc_level
    for m_idx in antagonist_indices:
        activation[0, m_idx] = cc_level

    # Measure torque at neutral and perturbed positions
    torques = []
    for angle_offset in [-delta_angle, delta_angle]:
        dof_pos = torch.zeros(1, NUM_DOFS)
        dof_vel = torch.zeros(1, NUM_DOFS)
        dof_pos[0, x_dof] = angle_offset

        ml = body.moment_arm.compute_muscle_length(dof_pos)
        mv = body.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)
        F = body.muscle_model.compute_force(activation, ml, mv)
        tau_muscle = body.moment_arm.forces_to_torques(F)
        tau_lig = body.ligament.compute_torque(dof_pos, dof_vel)
        tau_total = tau_muscle + tau_lig

        torques.append(tau_total[0, x_dof].item())

    impedance = abs(torques[1] - torques[0]) / (2 * delta_angle)
    return impedance


def run(output_dir):
    body = HumanBody.from_config(
        "muscle_definitions.yaml", "healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )

    joints = [
        ("L_Hip", "Hip"),
        ("L_Knee", "Knee"),
        ("L_Ankle", "Ankle"),
        ("R_Hip", "Hip (R)"),
        ("R_Knee", "Knee (R)"),
        ("R_Ankle", "Ankle (R)"),
    ]

    cc_levels = np.linspace(0, 1.0, 21)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exp4: Co-contraction Level → Joint Impedance (stiffness)", fontsize=14)

    summary_data = {}

    for idx, (joint_name, label) in enumerate(joints):
        ax = axes[idx // 3, idx % 3]

        impedances = []
        for cc in cc_levels:
            imp = compute_impedance(body, joint_name, cc)
            impedances.append(imp)

        impedances = np.array(impedances)
        summary_data[joint_name] = impedances

        ax.plot(cc_levels * 100, impedances, 'b-o', markersize=3, linewidth=2)
        ax.set_xlabel('Co-contraction Level (%)')
        ax.set_ylabel('Impedance (Nm/rad)')
        ax.set_title(f'{label} ({joint_name})')
        ax.grid(True, alpha=0.3)

        # Annotate key values
        ax.annotate(f'CC=0%: {impedances[0]:.1f}',
                    xy=(0, impedances[0]), fontsize=8,
                    xytext=(15, impedances[0] + (impedances[-1]-impedances[0])*0.1))
        ax.annotate(f'CC=100%: {impedances[-1]:.1f}',
                    xy=(100, impedances[-1]), fontsize=8,
                    xytext=(60, impedances[-1] - (impedances[-1]-impedances[0])*0.1))

        print(f"  {joint_name} ({label}):")
        print(f"    CC=0%: {impedances[0]:.2f} Nm/rad")
        print(f"    CC=50%: {impedances[10]:.2f} Nm/rad")
        print(f"    CC=100%: {impedances[-1]:.2f} Nm/rad")
        if impedances[0] > 0:
            print(f"    Ratio (CC100/CC0): {impedances[-1]/impedances[0]:.1f}x")
        print()

    # L/R symmetry check
    print("  L/R Symmetry check:")
    for side_pair in [("L_Hip", "R_Hip"), ("L_Knee", "R_Knee"), ("L_Ankle", "R_Ankle")]:
        l_imp = summary_data[side_pair[0]]
        r_imp = summary_data[side_pair[1]]
        max_diff_pct = np.abs(l_imp - r_imp).max() / (l_imp.max() + 1e-8) * 100
        print(f"    {side_pair[0]} vs {side_pair[1]}: max diff = {max_diff_pct:.2f}%")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp4_cocontraction_impedance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    run(output_dir)

"""
Exp3: Muscle Activation → Joint Torque Mapping (R Matrix Verification)

각 근육을 하나씩 활성화(activation=1.0)하고, 69 DOF 중 어디에 토크가
발생하는지 측정. R 행렬의 moment arm 구조가 정확한지 검증.

검증 포인트:
- 단관절근: 해당 관절 DOF에만 토크 발생
- 이관절근: 두 관절 모두에 토크 발생, 비율이 moment arm 비율과 일치
- 토크 부호가 moment arm 부호와 일치
- 비관련 관절에는 토크 = 0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import JOINT_DOF_RANGE, JOINT_NAMES, NUM_DOFS


def run(output_dir):
    body = HumanBody.from_config(
        "muscle_definitions.yaml", "healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )

    num_muscles = body.num_muscles
    muscle_names = body.moment_arm.muscle_names

    # 각 근육을 하나씩 활성화하여 토크 측정
    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)

    # Torque matrix: (num_muscles, num_dofs)
    torque_matrix = np.zeros((num_muscles, NUM_DOFS))

    for m_idx in range(num_muscles):
        activation = torch.zeros(1, num_muscles)
        activation[0, m_idx] = 1.0

        # Compute muscle force at neutral position
        ml = body.moment_arm.compute_muscle_length(dof_pos)
        mv = body.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)
        F_muscle = body.muscle_model.compute_force(activation, ml, mv)
        tau = body.moment_arm.forces_to_torques(F_muscle)

        torque_matrix[m_idx, :] = tau[0].numpy()

    # --- Plot 1: Heatmap of muscle→DOF torque matrix ---
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(torque_matrix, aspect='auto', cmap='RdBu_r',
                    vmin=-np.abs(torque_matrix).max(),
                    vmax=np.abs(torque_matrix).max())
    ax.set_yticks(range(num_muscles))
    ax.set_yticklabels(muscle_names, fontsize=8)
    ax.set_xlabel('DOF index (0-68)')
    ax.set_ylabel('Muscle')
    ax.set_title('Exp3: Muscle Activation (a=1.0) → Joint Torque Mapping')

    # Add joint name annotations on x-axis
    for joint_name in JOINT_NAMES[:8]:  # lower limb only for clarity
        s, e = JOINT_DOF_RANGE[joint_name]
        mid = (s + e) / 2
        ax.axvline(x=s-0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax.text(mid, -0.8, joint_name, ha='center', fontsize=6, rotation=45)

    plt.colorbar(im, ax=ax, label='Torque (Nm)', shrink=0.8)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp3_muscle_torque_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved: {save_path}")

    # --- Plot 2: Bi-articular verification ---
    print("\n  Bi-articular muscle verification:")
    print(f"  {'Muscle':<25} {'Coupled Joints':<30} {'Torque Ratio':<20} {'MA Ratio':<20} {'Match'}")
    print(f"  {'-'*115}")

    biarticular_muscles = []
    for m_idx, name in enumerate(muscle_names):
        nonzero_joints = []
        for joint_name in JOINT_NAMES:
            s, e = JOINT_DOF_RANGE[joint_name]
            if np.abs(torque_matrix[m_idx, s:e]).sum() > 1e-6:
                nonzero_joints.append((joint_name, torque_matrix[m_idx, s]))

        if len(nonzero_joints) >= 2:
            biarticular_muscles.append((name, nonzero_joints))
            j1_name, j1_tau = nonzero_joints[0]
            j2_name, j2_tau = nonzero_joints[1]

            # Get moment arm ratio
            j1_s, _ = JOINT_DOF_RANGE[j1_name]
            j2_s, _ = JOINT_DOF_RANGE[j2_name]
            ma1 = body.moment_arm.R[m_idx, j1_s].item()
            ma2 = body.moment_arm.R[m_idx, j2_s].item()

            if abs(j2_tau) > 1e-8 and abs(ma2) > 1e-8:
                torque_ratio = j1_tau / j2_tau
                ma_ratio = ma1 / ma2
                match = abs(torque_ratio - ma_ratio) < 0.01
                print(f"  {name:<25} {j1_name}({j1_tau:+.2f}), {j2_name}({j2_tau:+.2f})  "
                      f"tau_ratio={torque_ratio:+.3f}  ma_ratio={ma_ratio:+.3f}  {'OK' if match else 'MISMATCH'}")

    # --- Plot 3: Per-joint torque bar chart (agonist vs antagonist) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exp3: Per-Joint Agonist/Antagonist Torque at Full Activation (a=1.0)", fontsize=14)

    joints_plot = [
        ("L_Hip", "L Hip"),
        ("L_Knee", "L Knee"),
        ("L_Ankle", "L Ankle"),
        ("R_Hip", "R Hip"),
        ("R_Knee", "R Knee"),
        ("R_Ankle", "R Ankle"),
    ]

    for idx, (joint_name, label) in enumerate(joints_plot):
        ax = axes[idx // 3, idx % 3]
        s, _ = JOINT_DOF_RANGE[joint_name]
        x_torques = torque_matrix[:, s]

        # Only show muscles with nonzero torque at this joint
        active_mask = np.abs(x_torques) > 1e-6
        active_names = [muscle_names[i] for i in range(num_muscles) if active_mask[i]]
        active_torques = x_torques[active_mask]

        if len(active_names) > 0:
            colors = ['#2196F3' if t > 0 else '#F44336' for t in active_torques]
            bars = ax.barh(range(len(active_names)), active_torques, color=colors)
            ax.set_yticks(range(len(active_names)))
            ax.set_yticklabels([n.replace('_L', '').replace('_R', '') for n in active_names], fontsize=8)
            ax.axvline(x=0, color='gray', linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, active_torques):
                ax.text(val + (1 if val > 0 else -1), bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='left' if val > 0 else 'right', va='center', fontsize=7)

        ax.set_xlabel('Torque (Nm)')
        ax.set_title(f'{label} x-axis (flex/ext)')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp3_per_joint_torque_bar.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Bar chart saved: {save_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    run(output_dir)

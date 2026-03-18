"""
Exp2: Stretch Reflex Response per Joint

각 하지 관절의 주 운동축에 다양한 속도를 인가하고,
stretch reflex가 생성하는 반사 토크를 측정.

검증 포인트:
- 속도가 threshold 이하이면 reflex torque = 0
- 속도 증가 → reflex torque 증가 (양의 상관)
- reflex torque의 방향이 운동 반대 (저항)
- healthy vs spastic: spastic에서 gain 높고 threshold 낮음
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import JOINT_DOF_RANGE, NUM_DOFS


def compute_reflex_torque_sweep(body, joint_name, velocities, dt=1.0/60):
    """주어진 관절에 다양한 속도를 인가하고 reflex torque 측정."""
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    x_dof = dof_start

    torques = []
    for vel in velocities:
        body.activation_dyn.reset()
        body.reflex.reset()

        dof_pos = torch.zeros(1, NUM_DOFS)
        dof_vel = torch.zeros(1, NUM_DOFS)
        dof_vel[0, x_dof] = vel

        # Run multiple steps to let reflex settle (delay buffer)
        for _ in range(5):
            descending_cmd = torch.zeros(1, body.num_muscles)
            tau = body.compute_torques(dof_pos, dof_vel, descending_cmd, dt=dt)

        torques.append(tau[0, x_dof].item())

    return np.array(torques)


def run(output_dir):
    # Load healthy and spastic profiles
    body_healthy = HumanBody.from_config(
        "muscle_definitions.yaml", "healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )

    # Try spastic profile
    spastic_path = os.path.join(os.path.dirname(__file__),
                                 "../profiles/stroke/stroke_r_hemiplegia.yaml")
    if os.path.exists(spastic_path):
        body_spastic = HumanBody.from_config(
            "muscle_definitions.yaml", spastic_path,
            num_envs=1, device="cpu",
        )
        has_spastic = True
    else:
        has_spastic = False

    joints = [
        ("L_Hip", "Hip"),
        ("L_Knee", "Knee"),
        ("L_Ankle", "Ankle"),
    ]

    velocities = np.linspace(-5.0, 5.0, 100)  # rad/s

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Exp2: Stretch Reflex Response (velocity sweep → reflex torque)", fontsize=14)

    for idx, (joint_name, label) in enumerate(joints):
        ax = axes[idx]

        tau_healthy = compute_reflex_torque_sweep(body_healthy, joint_name, velocities)

        ax.plot(velocities, tau_healthy, 'b-', linewidth=2, label='Healthy')

        if has_spastic:
            tau_spastic = compute_reflex_torque_sweep(body_spastic, joint_name, velocities)
            ax.plot(velocities, tau_spastic, 'r-', linewidth=2, label='Spastic (stroke)')

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Joint Velocity (rad/s)')
        ax.set_ylabel('Reflex Torque (Nm)')
        ax.set_title(f'{label} ({joint_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Quantitative summary
        pos_vel_mask = velocities > 0.5
        neg_vel_mask = velocities < -0.5
        print(f"  {joint_name} ({label}):")
        print(f"    Healthy - positive vel torque range: [{tau_healthy[pos_vel_mask].min():.2f}, {tau_healthy[pos_vel_mask].max():.2f}] Nm")
        print(f"    Healthy - negative vel torque range: [{tau_healthy[neg_vel_mask].min():.2f}, {tau_healthy[neg_vel_mask].max():.2f}] Nm")
        if has_spastic:
            print(f"    Spastic - positive vel torque range: [{tau_spastic[pos_vel_mask].min():.2f}, {tau_spastic[pos_vel_mask].max():.2f}] Nm")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp2_stretch_reflex.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    run(output_dir)

"""
Demo: Knee Pendulum Test — 환자 프로파일별 비교 (IsaacGym 시각화)

3개의 SMPL 휴머노이드를 공중에 매달고, L_Knee만 자유롭게 둔 뒤
80° 굴곡 상태에서 놓아 중력으로 스윙시킨다.
Bio-mechanical 토크(수동 근육 + 반사 + 인대)만으로 저항.

시각적 결과:
  - Healthy (파랑): 적절한 감쇠, 2-3회 스윙 후 정지
  - Spastic/Stroke (빨강): 높은 저항, 거의 안 움직임
  - Flaccid/SCI (초록): 자유 진자처럼 계속 흔들림

사용법:
  python demo_knee_pendulum.py                      # 뷰어 + 플롯
  python demo_knee_pendulum.py --headless            # 플롯만
  python demo_knee_pendulum.py --pipeline cpu        # CPU 모드
"""

# CRITICAL: IsaacGym must be imported before torch
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import NUM_DOFS, JOINT_DOF_RANGE

# ============================================================
# Configuration
# ============================================================
TEST_JOINT = "L_Knee"
TEST_DOF_IDX = JOINT_DOF_RANGE[TEST_JOINT][0]  # x-axis (flexion/extension) = DOF 3
INITIAL_KNEE_ANGLE = 1.4  # ~80° flexion

# PD gains for holding non-test joints at neutral
HOLD_KP = 500.0
HOLD_KD = 50.0

DURATION = 8.0  # seconds
DT = 1.0 / 60.0
INITIAL_KICK_VEL = -5.0  # rad/s toward extension

PROFILES = [
    {
        "name": "Healthy",
        "color_rgb": (0.2, 0.6, 1.0),
        "modifications": {},
    },
    {
        "name": "Spastic (Stroke)",
        "color_rgb": (1.0, 0.3, 0.3),
        "modifications": {
            "reflex": {"stretch_gain": 8.0, "stretch_threshold": 0.02},
            "ligament": {"k_lig": 200.0, "damping": 25.0, "alpha": 15.0},
            "muscle": {"damping_scale": 3.0},
        },
    },
    {
        "name": "Flaccid (SCI)",
        "color_rgb": (0.3, 1.0, 0.3),
        "modifications": {
            "reflex": {"stretch_gain": 0.0, "stretch_threshold": 999.0},
            "ligament": {"k_lig": 5.0, "damping": 0.5, "alpha": 5.0},
            "muscle": {"f_max_scale": 0.05, "damping_scale": 0.1},
        },
    },
]


# ============================================================
# Helper functions
# ============================================================
def create_human_body(mods):
    """Create HumanBody and apply patient-specific parameter modifications."""
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    body = HumanBody.from_config(
        muscle_def_path=os.path.join(config_dir, "muscle_definitions.yaml"),
        param_path=os.path.join(config_dir, "healthy_baseline.yaml"),
        num_envs=1,
        device="cpu",
    )

    if not mods:
        return body

    # Reflex modifications
    if "reflex" in mods:
        r = mods["reflex"]
        if "stretch_gain" in r:
            body.reflex.stretch_gain[:] = r["stretch_gain"]
        if "stretch_threshold" in r:
            body.reflex.stretch_threshold[:] = r["stretch_threshold"]

    # Ligament modifications
    if "ligament" in mods:
        lg = mods["ligament"]
        if "k_lig" in lg:
            body.ligament.k_lig[:] = lg["k_lig"]
        if "damping" in lg:
            body.ligament.damping[:] = lg["damping"]
        if "alpha" in lg:
            body.ligament.alpha[:] = lg["alpha"]

    # Muscle modifications
    if "muscle" in mods:
        m = mods["muscle"]
        if "f_max_scale" in m:
            body.muscle_model.f_max *= m["f_max_scale"]
            body._f_max = body._f_max.float() * m["f_max_scale"]
        if "damping_scale" in m:
            body.muscle_model.damping *= m["damping_scale"]

    return body


def main():
    # Parse IsaacGym arguments
    custom_params = [
        {"name": "--headless", "action": "store_true", "default": False,
         "help": "Run headless (no viewer)"},
        {"name": "--test_joint", "type": str, "default": TEST_JOINT,
         "help": "Joint to test (default: L_Knee)"},
        {"name": "--duration", "type": float, "default": DURATION,
         "help": "Test duration in seconds"},
    ]
    args = gymutil.parse_arguments(
        description="Knee Pendulum Test: Patient Profile Comparison",
        custom_parameters=custom_params,
    )

    gym = gymapi.acquire_gym()

    # ---- Sim params ----
    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0

    # Device
    compute_device = args.compute_device_id
    graphics_device = args.graphics_device_id
    if args.headless:
        graphics_device = -1

    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("ERROR: Failed to create sim")
        return

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ---- Load SMPL humanoid asset (fixed root — no freejoint) ----
    asset_root = os.path.dirname(os.path.abspath(__file__))
    asset_file = "smpl_humanoid_fixed.xml"

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset_options.fix_base_link = True

    humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if humanoid_asset is None:
        print("ERROR: Failed to load asset")
        return

    num_dof = gym.get_asset_dof_count(humanoid_asset)
    num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)
    print(f"Asset loaded: {num_dof} DOFs, {num_bodies} bodies")

    # ---- Create environments (one per profile) ----
    num_envs = len(PROFILES)
    spacing = 3.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing * 2)

    envs = []
    actor_handles = []

    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_envs)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 2.0)  # Suspended in air
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        handle = gym.create_actor(
            env, humanoid_asset, start_pose,
            f"humanoid_{i}", i, 0, 0,
        )

        # Non-test DOFs: POS mode (PhysX internal PD holds at neutral)
        # Test DOF: EFFORT mode (receives bio-torques)
        dof_props = gym.get_actor_dof_properties(env, handle)
        for j in range(num_dof):
            if j == TEST_DOF_IDX:
                dof_props["driveMode"][j] = int(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"][j] = 0.0
                dof_props["damping"][j] = 0.0
            else:
                dof_props["driveMode"][j] = int(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][j] = HOLD_KP
                dof_props["damping"][j] = HOLD_KD
        gym.set_actor_dof_properties(env, handle, dof_props)

        envs.append(env)
        actor_handles.append(handle)

    # ---- Create bio-model instances ----
    bodies = []
    for profile in PROFILES:
        body = create_human_body(profile["modifications"])
        bodies.append(body)
        print(f"  Bio-model created: {profile['name']}")

    # ---- Prepare simulation ----
    gym.prepare_sim(sim)

    # Acquire state tensors (no root tensor needed — root is fixed)
    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_state_tensor)   # (num_envs * num_dof, 2)

    # Reshape DOF states
    dof_pos_all = dof_states[:, 0].view(num_envs, num_dof)
    dof_vel_all = dof_states[:, 1].view(num_envs, num_dof)

    # Set initial knee angle + kick velocity for all envs
    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_pos_all[i, TEST_DOF_IDX] = INITIAL_KNEE_ANGLE
        dof_vel_all[i, :] = 0.0
        dof_vel_all[i, TEST_DOF_IDX] = INITIAL_KICK_VEL

    env_ids = torch.arange(num_envs, dtype=torch.int32, device=dof_states.device)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    # Torque tensor
    torques = torch.zeros(num_envs * num_dof, dtype=torch.float32,
                          device=dof_states.device)

    # ---- Viewer setup ----
    viewer = None
    if not args.headless:
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
        viewer = gym.create_viewer(sim, camera_props)

        # Camera looking at all 3 humanoids from the side
        cam_pos = gymapi.Vec3(8.0, -2.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 4.0, 1.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ---- Recording ----
    max_steps = int(args.duration / DT)
    knee_angle_history = np.zeros((num_envs, max_steps))
    knee_torque_history = np.zeros((num_envs, max_steps))
    time_array = np.arange(max_steps) * DT

    print(f"\n{'='*60}")
    print(f"Knee Pendulum Test")
    print(f"  Test joint: {TEST_JOINT} (DOF {TEST_DOF_IDX})")
    print(f"  Initial angle: {np.degrees(INITIAL_KNEE_ANGLE):.1f} deg")
    print(f"  Duration: {args.duration}s ({max_steps} steps @ {1/DT:.0f} Hz)")
    print(f"  Profiles: {', '.join(p['name'] for p in PROFILES)}")
    print(f"{'='*60}\n")

    # Reset bio-model state once
    for body in bodies:
        body.activation_dyn.reset()
        body.reflex.reset()

    # ---- Simulation loop ----
    for step in range(max_steps):
        # Refresh state tensors
        gym.refresh_dof_state_tensor(sim)

        # Compute bio-torques for test joint only
        # (non-test joints held by PhysX internal PD in DOF_MODE_POS)
        torques.zero_()
        torques_2d = torques.view(num_envs, num_dof)

        for i in range(num_envs):
            pos_i = dof_pos_all[i]
            vel_i = dof_vel_all[i]

            pos_cpu = pos_i.unsqueeze(0).cpu() if pos_i.is_cuda else pos_i.unsqueeze(0)
            vel_cpu = vel_i.unsqueeze(0).cpu() if vel_i.is_cuda else vel_i.unsqueeze(0)
            descending_cmd = torch.zeros(1, bodies[i].num_muscles)

            bio_tau = bodies[i].compute_torques(
                pos_cpu, vel_cpu, descending_cmd, dt=DT,
            )

            # Clamp bio-torque to reasonable range
            bio_torque_val = bio_tau[0, TEST_DOF_IDX].item()
            bio_torque_val = max(-500.0, min(500.0, bio_torque_val))
            torques_2d[i, TEST_DOF_IDX] = bio_torque_val

            # Record
            knee_angle_history[i, step] = pos_i[TEST_DOF_IDX].item()
            knee_torque_history[i, step] = bio_torque_val

        # Apply torques
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))

        # Physics step
        gym.simulate(sim)
        if dof_states.device.type == "cpu":
            gym.fetch_results(sim, True)

        # Render
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

            if gym.query_viewer_has_closed(viewer):
                break

        # Progress
        if (step + 1) % (max_steps // 5) == 0:
            pct = (step + 1) / max_steps * 100
            angles = [f"{np.degrees(knee_angle_history[i, step]):.1f}" for i in range(num_envs)]
            print(f"  [{pct:3.0f}%] Knee angles: {' | '.join(angles)} deg")

    # ---- Cleanup viewer ----
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # ============================================================
    # Plot results
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Knee Pendulum Test: Patient Profile Comparison (IsaacGym)",
        fontsize=14, fontweight="bold",
    )
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    # --- Subplot 1: Knee angle ---
    ax1 = axes[0]
    for i, profile in enumerate(PROFILES):
        ax1.plot(
            time_array, np.degrees(knee_angle_history[i]),
            color=colors[i], linewidth=2, label=profile["name"],
        )
    ax1.set_ylabel("Knee Flexion Angle (deg)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linewidth=0.5)
    ax1.set_ylim(-10, 100)

    # Annotation box
    ax1.text(
        0.02, 0.95,
        f"Initial angle: {np.degrees(INITIAL_KNEE_ANGLE):.0f} deg\n"
        f"Initial kick: {INITIAL_KICK_VEL} rad/s\n"
        f"Gravity: 9.81 m/s²\n"
        f"All joints locked except {TEST_JOINT}\n"
        f"Only bio-torques (no PD) on test joint",
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Subplot 2: Bio-torque ---
    ax2 = axes[1]
    for i, profile in enumerate(PROFILES):
        ax2.plot(
            time_array, knee_torque_history[i],
            color=colors[i], linewidth=1.5, alpha=0.8, label=profile["name"],
        )
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Bio-Torque on Knee (Nm)", fontsize=12)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "demo_knee_pendulum.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Summary:")
    for i, profile in enumerate(PROFILES):
        final_angle = np.degrees(knee_angle_history[i, -1])
        max_torque = np.max(np.abs(knee_torque_history[i]))
        # Count zero-crossings (oscillation measure)
        angles = knee_angle_history[i]
        mean_angle = np.mean(angles[len(angles)//2:])
        centered = angles - mean_angle
        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        print(
            f"  {profile['name']:<20}: "
            f"final={final_angle:>6.1f} deg, "
            f"max_torque={max_torque:>7.2f} Nm, "
            f"oscillations~{crossings//2}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

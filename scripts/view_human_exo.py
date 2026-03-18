"""
View SMPL Human + WalkON Exoskeleton (VSD Coupling)
----------------------------------------------------
Interactive viewer for human-exoskeleton integration in IsaacGym.
Two separate actors coupled via Virtual Spring-Damper (VSD) forces.

VSD Coupling Points:
  1. Pelvis: SMPL Torso <-> WalkON LINK_BASE
  2. L Shank: SMPL L_Knee <-> WalkON LINK_L_SHANK
  3. R Shank: SMPL R_Knee <-> WalkON LINK_R_SHANK
  4. L Foot:  SMPL L_Ankle <-> WalkON LINK_L_FOOT
  5. R Foot:  SMPL R_Ankle <-> WalkON LINK_R_FOOT

Controls:
  V         : Toggle VSD on/off
  +/-       : Increase/Decrease VSD stiffness
  UP/DOWN   : Select joint (human)
  LEFT/RIGHT: Adjust selected joint angle
  0         : Reset human pose
  F/B       : Push human forward/backward
  SPACE     : Push human upward
  T         : Toggle fix_base_link (drop test)

Usage:
    conda activate phc
    python scripts/view_human_exo.py
"""

import os
import re
import glob
import math
import numpy as np
from isaacgym import gymapi, gymutil


# ── SMPL XML generation (reuse from view_smpl_model.py) ──
def find_valid_mesh_xml():
    candidates = sorted(glob.glob("/tmp/smpl/smpl_humanoid_*.xml"),
                        key=os.path.getmtime, reverse=True)
    for c in candidates:
        with open(c, 'r') as f:
            match = re.search(r'file="([^/]+)/geom/', f.read())
        if match and os.path.isdir(os.path.join("/tmp/smpl", match.group(1), "geom")):
            return c
    return None


def generate_smpl_xml(beta0=0.0):
    """Generate SMPL XML with mesh. beta0 controls height (0=avg, 2=tall)."""
    # Use different filename for different beta values
    # Avoid dots in filename (IsaacGym parses extension from last dot)
    suffix = f"_b{str(beta0).replace('.', 'p')}" if beta0 != 0.0 else ""
    path = f"/tmp/smpl/view_exo{suffix}.xml"
    if os.path.exists(path):
        with open(path, 'r') as f:
            match = re.search(r'file="([^/]+)/geom/', f.read())
        if match and os.path.isdir(os.path.join("/tmp/smpl", match.group(1), "geom")):
            return path

    try:
        import torch as th
        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
        robot_cfg = {
            'mesh': True, 'replace_feet': True, 'rel_joint_lm': True,
            'upright_start': True, 'remove_toe': False, 'freeze_hand': True,
            'real_weight_porpotion_capsules': True, 'real_weight_porpotion_boxes': True,
            'real_weight': True, 'masterfoot': False, 'master_range': 30,
            'big_ankle': True, 'box_body': True,
            'body_params': {}, 'joint_params': {}, 'geom_params': {},
            'actuator_params': {}, 'model': 'smpl', 'sim': 'isaacgym'
        }
        robot = SMPL_Robot(robot_cfg, data_dir='data/smpl')
        betas = th.zeros(1, 10)
        betas[0, 0] = beta0  # height scaling
        robot.load_from_skeleton(betas=betas, gender=[0])
        robot.write_xml(path)
        print(f"  Generated SMPL XML: {path} (beta[0]={beta0})")
    except Exception as e:
        print(f"Failed to generate SMPL XML: {e}")
        return find_valid_mesh_xml()
    return path


# ── Paths ──
WALKON_URDF_ROOT = "/home/gunhee/workspace/walkon_model_normalization/human_robot_integrated_model/robots/walkonsuit/urdf"
WALKON_URDF_FILE = "walkonsuit.urdf"

# ── Parse arguments ──
args = gymutil.parse_arguments(
    description="View SMPL Human + WalkON Exoskeleton with VSD coupling",
    custom_parameters=[{
        "name": "--human_xml",
        "type": str,
        "default": "",
        "help": "Path to SMPL humanoid XML. If empty, auto-generates."
    }, {
        "name": "--no_vsd",
        "action": "store_true",
        "help": "Start with VSD disabled"
    }, {
        "name": "--beta0",
        "type": float,
        "default": 0.0,
        "help": "SMPL beta[0] for height scaling (0=average, 2=tall, -2=short)"
    }])

# ── Generate SMPL XML ──
print("Preparing SMPL human model...")
smpl_xml = args.human_xml if args.human_xml and os.path.exists(args.human_xml) else generate_smpl_xml(beta0=args.beta0)
if smpl_xml is None:
    print("No valid SMPL mesh XML found. Run training/eval first.")
    quit()
print(f"  SMPL XML: {smpl_xml}")

# ── Initialize gym ──
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# ── Load SMPL human asset ──
smpl_asset_root = os.path.dirname(smpl_xml)
smpl_asset_file = os.path.basename(smpl_xml)

smpl_opts = gymapi.AssetOptions()
smpl_opts.fix_base_link = True
smpl_opts.angular_damping = 0.01
smpl_opts.max_angular_velocity = 100.0
smpl_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)

smpl_asset = gym.load_asset(sim, smpl_asset_root, smpl_asset_file, smpl_opts)
smpl_num_dofs = gym.get_asset_dof_count(smpl_asset)
smpl_num_bodies = gym.get_asset_rigid_body_count(smpl_asset)
smpl_dof_names = list(gym.get_asset_dof_names(smpl_asset))
smpl_body_names = list(gym.get_asset_rigid_body_names(smpl_asset))
print(f"\n=== SMPL Human: {smpl_num_bodies} bodies, {smpl_num_dofs} DOFs ===")

# ── Load WalkON exo asset ──
exo_opts = gymapi.AssetOptions()
exo_opts.fix_base_link = True
exo_opts.angular_damping = 0.01
exo_opts.max_angular_velocity = 100.0
exo_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

exo_asset = gym.load_asset(sim, WALKON_URDF_ROOT, WALKON_URDF_FILE, exo_opts)
exo_num_dofs = gym.get_asset_dof_count(exo_asset)
exo_num_bodies = gym.get_asset_rigid_body_count(exo_asset)
exo_dof_names = list(gym.get_asset_dof_names(exo_asset))
exo_body_names = list(gym.get_asset_rigid_body_names(exo_asset))
print(f"=== WalkON Exo: {exo_num_bodies} bodies, {exo_num_dofs} DOFs ===")

# ── Body index mapping for VSD coupling ──
# SMPL body indices
smpl_torso_idx = smpl_body_names.index("Torso")
smpl_l_knee_idx = smpl_body_names.index("L_Knee")
smpl_r_knee_idx = smpl_body_names.index("R_Knee")
smpl_l_ankle_idx = smpl_body_names.index("L_Ankle")
smpl_r_ankle_idx = smpl_body_names.index("R_Ankle")

# WalkON body indices
exo_base_idx = exo_body_names.index("LINK_BASE")
exo_l_shank_idx = exo_body_names.index("LINK_L_SHANK")
exo_r_shank_idx = exo_body_names.index("LINK_R_SHANK")
exo_l_foot_idx = exo_body_names.index("LINK_L_FOOT")
exo_r_foot_idx = exo_body_names.index("LINK_R_FOOT")

print(f"\n=== VSD Coupling Points ===")
print(f"  Pelvis:  SMPL Torso[{smpl_torso_idx}] <-> Exo BASE[{exo_base_idx}]")
print(f"  L Shank: SMPL L_Knee[{smpl_l_knee_idx}] <-> Exo L_SHANK[{exo_l_shank_idx}]")
print(f"  R Shank: SMPL R_Knee[{smpl_r_knee_idx}] <-> Exo R_SHANK[{exo_r_shank_idx}]")
print(f"  L Foot:  SMPL L_Ankle[{smpl_l_ankle_idx}] <-> Exo L_FOOT[{exo_l_foot_idx}]")
print(f"  R Foot:  SMPL R_Ankle[{smpl_r_ankle_idx}] <-> Exo R_FOOT[{exo_r_foot_idx}]")

# ── Create environment with both actors ──
spacing = 3.0
env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0.0),
                     gymapi.Vec3(spacing, spacing, spacing), 1)

# Spawn WalkON exo first (reference position)
exo_pose = gymapi.Transform()
exo_pose.p = gymapi.Vec3(0.0, 0.0, 1.2)  # WalkON BASE at hip height
# collision_group=1: separate group from human so they never collide
exo_actor = gym.create_actor(env, exo_asset, exo_pose, "exo", 1, 0)

# Spawn SMPL human — pelvis behind exo base (exo wraps around belly)
human_pose = gymapi.Transform()
human_pose.p = gymapi.Vec3(-0.25, 0.0, 1.05)  # further back (-0.25X), lower (-0.15Z)
# collision_group=0: different group from exo, no inter-actor collision
human_actor = gym.create_actor(env, smpl_asset, human_pose, "human", 0, 0)

# ── Color human model orange ──
for i in range(smpl_num_bodies):
    gym.set_rigid_body_color(env, human_actor, i, gymapi.MESH_VISUAL,
                             gymapi.Vec3(1.0, 0.5, 0.0))  # orange

# ── Set DOF properties ──
# Human DOFs: paralyzed lower body + compliant trunk + free arms
# Lower body joints (Hip, Knee, Ankle, Toe): NO control (paralyzed)
# Trunk (Torso, Spine, Chest): compliant PD to maintain posture
# Arms, Neck, Head: NO control (gravity pulls arms down from T-pose)
# Joint groups for DOF control
SPINE_JOINTS = {"Torso", "Spine", "Chest", "Neck", "Head"}  # posture control
LOWER_BODY_JOINTS = {"L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"}

human_dof_props = gym.get_actor_dof_properties(env, human_actor)
for i, dof_name in enumerate(smpl_dof_names):
    joint_name = dof_name.rsplit('_', 1)[0]  # "L_Hip_x" -> "L_Hip"
    if joint_name in SPINE_JOINTS:
        # Spine + neck + head: hold upright posture
        human_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
        human_dof_props['stiffness'][i] = 400.0
        human_dof_props['damping'][i] = 40.0
    elif joint_name in LOWER_BODY_JOINTS:
        # Paralyzed: no active control, only passive tissue damping
        human_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_NONE)
        human_dof_props['stiffness'][i] = 0.0
        human_dof_props['damping'][i] = 5.0   # passive tissue resistance
    else:
        # Shoulder, Elbow, Hand: NO control, gravity pulls arms down
        human_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_NONE)
        human_dof_props['stiffness'][i] = 0.0
        human_dof_props['damping'][i] = 0.1   # near-zero damping for free fall
gym.set_actor_dof_properties(env, human_actor, human_dof_props)

# Exo: position control (exo drives the motion)
exo_dof_props = gym.get_actor_dof_properties(env, exo_actor)
for i in range(exo_num_dofs):
    exo_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
    exo_dof_props['stiffness'][i] = 200.0
    exo_dof_props['damping'][i] = 20.0
gym.set_actor_dof_properties(env, exo_actor, exo_dof_props)

# ── Set initial pose ──
human_targets = np.zeros(smpl_num_dofs, dtype=np.float32)

# T-pose spawn: shoulders at 0 (arms horizontal), gravity will pull them down
# Only set hip adduction to align feet with exo
for i, dof_name in enumerate(smpl_dof_names):
    pass  # No hip adduction — spawn with legs straight

gym.set_actor_dof_position_targets(env, human_actor, human_targets)

human_dof_states = np.zeros(smpl_num_dofs, dtype=gymapi.DofState.dtype)
for i in range(smpl_num_dofs):
    human_dof_states['pos'][i] = human_targets[i]
gym.set_actor_dof_states(env, human_actor, human_dof_states, gymapi.STATE_ALL)

# Exo: default pose (all zeros)
exo_targets = np.zeros(exo_num_dofs, dtype=np.float32)
gym.set_actor_dof_position_targets(env, exo_actor, exo_targets)

# ── VSD parameters ──
vsd_enabled = not args.no_vsd
vsd_stiffness_base = 200.0      # N/m for pelvis (translational)
vsd_damping_base = 80.0         # N·s/m for pelvis (high damping ratio)
vsd_stiffness_limb = 80.0       # N/m for shank/foot (reduced)
vsd_damping_limb = 40.0         # N·s/m for shank/foot (overdamped)
vsd_max_force = 100.0           # N, clamp per coupling point
vsd_scale = 1.0                 # user-adjustable multiplier
vsd_ramp_frames = 120           # 2 seconds ramp-up to avoid initial shock

# Coupling pairs: (smpl_body_idx, exo_body_idx, K, D, name)
vsd_pairs = [
    (smpl_torso_idx,   exo_base_idx,    vsd_stiffness_base, vsd_damping_base, "Pelvis"),
    (smpl_l_knee_idx,  exo_l_shank_idx, vsd_stiffness_limb, vsd_damping_limb, "L_Shank"),
    (smpl_r_knee_idx,  exo_r_shank_idx, vsd_stiffness_limb, vsd_damping_limb, "R_Shank"),
    (smpl_l_ankle_idx, exo_l_foot_idx,  vsd_stiffness_limb, vsd_damping_limb, "L_Foot"),
    (smpl_r_ankle_idx, exo_r_foot_idx,  vsd_stiffness_limb, vsd_damping_limb, "R_Foot"),
]

# ── Keyboard events ──
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_vsd")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_EQUAL, "vsd_up")      # +
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_MINUS, "vsd_down")    # -
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "prev_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "next_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "decrease")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "increase")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "reset_pose")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "push_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "push_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "push_up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "toggle_fixed")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "toggle_contacts")

# Camera
cam_pos = gymapi.Vec3(3.0, -2.0, 1.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.9)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# State
selected_joint = 0
joint_step = math.radians(5)
force_magnitude = 200.0
is_fixed = True  # base link fixed
show_contacts = False  # contact visualization toggle

print(f"\n=== Controls ===")
print(f"  V         : Toggle VSD ({'ON' if vsd_enabled else 'OFF'})")
print(f"  +/-       : VSD stiffness scale (current: {vsd_scale:.1f}x)")
print(f"  C         : Toggle contact point visualization")
print(f"  UP/DOWN   : Select human joint")
print(f"  LEFT/RIGHT: Adjust joint angle")
print(f"  0         : Reset human pose")
print(f"  F/B/SPACE : Push human")
print(f"  T         : Toggle fix_base (drop test)")
print(f"\n  Selected joint: [{selected_joint}] {smpl_dof_names[selected_joint]}")


def get_body_pos_vel(env_handle, actor_handle, body_idx):
    """Get position and linear velocity of a rigid body."""
    body_states = gym.get_actor_rigid_body_states(env_handle, actor_handle, gymapi.STATE_ALL)
    pos = body_states['pose']['p'][body_idx]
    vel = body_states['vel']['linear'][body_idx]
    pos_np = np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
    vel_np = np.array([vel[0], vel[1], vel[2]], dtype=np.float32)
    return pos_np, vel_np


def get_rigid_handle(env_handle, actor_handle, body_idx):
    """Get rigid body handle for force application."""
    return gym.get_actor_rigid_body_handle(env_handle, actor_handle, body_idx)


def apply_force_to_body(env_handle, actor_handle, body_idx, force_vec):
    """Apply force to a specific rigid body using correct API."""
    rh = get_rigid_handle(env_handle, actor_handle, body_idx)
    gym.apply_body_forces(env_handle, rh, force_vec, None, gymapi.ENV_SPACE)


def get_body_orientation(env_handle, actor_handle, body_idx):
    """Get orientation quaternion [x,y,z,w] and angular velocity of a rigid body."""
    body_states = gym.get_actor_rigid_body_states(env_handle, actor_handle, gymapi.STATE_ALL)
    rot = body_states['pose']['r'][body_idx]
    ang_vel = body_states['vel']['angular'][body_idx]
    rot_np = np.array([rot[0], rot[1], rot[2], rot[3]], dtype=np.float32)  # x,y,z,w
    ang_np = np.array([ang_vel[0], ang_vel[1], ang_vel[2]], dtype=np.float32)
    return rot_np, ang_np


def quat_diff_angular(q1, q2):
    """Compute approximate angular error between two quaternions (small angle)."""
    # q_err = q2 * q1_inv  →  angular_error ≈ 2 * vec(q_err) for small angles
    # q1_inv = conjugate(q1) since unit quat
    q1_inv = np.array([-q1[0], -q1[1], -q1[2], q1[3]], dtype=np.float32)
    # Hamilton product: q_err = q2 * q1_inv
    x1, y1, z1, w1 = q1_inv
    x2, y2, z2, w2 = q2
    qe = np.array([
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
    ], dtype=np.float32)
    # Ensure positive w (shortest path)
    if qe[3] < 0:
        qe = -qe
    # Angular error vector ≈ 2 * [qx, qy, qz]
    return 2.0 * qe[:3]


# Rotational VSD parameters (for foot/shank orientation coupling)
vsd_rot_stiffness = 10.0    # N·m/rad (gentle)
vsd_rot_damping = 8.0       # N·m·s/rad (overdamped to avoid oscillation)
vsd_max_torque = 10.0       # N·m clamp


def apply_vsd_forces():
    """Compute and apply VSD coupling forces (translational + rotational)."""
    if not vsd_enabled:
        return

    # Ramp-up factor: 0→1 over vsd_ramp_frames to avoid initial shock
    ramp = min(1.0, frame / max(1, vsd_ramp_frames))

    for smpl_bidx, exo_bidx, K, D, _name in vsd_pairs:
        # --- Translational VSD ---
        h_pos, h_vel = get_body_pos_vel(env, human_actor, smpl_bidx)
        e_pos, e_vel = get_body_pos_vel(env, exo_actor, exo_bidx)

        delta_pos = e_pos - h_pos
        delta_vel = e_vel - h_vel

        force = ramp * vsd_scale * (K * delta_pos + D * delta_vel)

        # Clamp force magnitude
        force_mag = np.linalg.norm(force)
        if force_mag > vsd_max_force * vsd_scale:
            force = force * (vsd_max_force * vsd_scale / force_mag)

        human_force = gymapi.Vec3(float(force[0]), float(force[1]), float(force[2]))
        apply_force_to_body(env, human_actor, smpl_bidx, human_force)

        exo_force = gymapi.Vec3(float(-force[0]), float(-force[1]), float(-force[2]))
        apply_force_to_body(env, exo_actor, exo_bidx, exo_force)

        # --- Rotational VSD (for shank/foot to prevent drooping/rotation) ---
        if _name != "Pelvis":
            h_rot, h_ang = get_body_orientation(env, human_actor, smpl_bidx)
            e_rot, e_ang = get_body_orientation(env, exo_actor, exo_bidx)

            ang_err = quat_diff_angular(h_rot, e_rot)
            ang_vel_err = e_ang - h_ang

            torque = ramp * vsd_scale * (vsd_rot_stiffness * ang_err + vsd_rot_damping * ang_vel_err)

            # Clamp torque
            tmag = np.linalg.norm(torque)
            if tmag > vsd_max_torque * vsd_scale:
                torque = torque * (vsd_max_torque * vsd_scale / tmag)

            # Apply torques via force API (torque parameter)
            h_rh = gym.get_actor_rigid_body_handle(env, human_actor, smpl_bidx)
            e_rh = gym.get_actor_rigid_body_handle(env, exo_actor, exo_bidx)
            h_torque = gymapi.Vec3(float(torque[0]), float(torque[1]), float(torque[2]))
            e_torque = gymapi.Vec3(float(-torque[0]), float(-torque[1]), float(-torque[2]))
            gym.apply_body_forces(env, h_rh, None, h_torque, gymapi.ENV_SPACE)
            gym.apply_body_forces(env, e_rh, None, e_torque, gymapi.ENV_SPACE)


# ── Print initial coupling distances ──
gym.simulate(sim)
gym.fetch_results(sim, True)
print(f"\n=== Initial VSD Coupling Distances ===")
for smpl_bidx, exo_bidx, K, D, name in vsd_pairs:
    h_pos, _ = get_body_pos_vel(env, human_actor, smpl_bidx)
    e_pos, _ = get_body_pos_vel(env, exo_actor, exo_bidx)
    dist = np.linalg.norm(e_pos - h_pos)
    raw_force = K * dist
    clamped = min(raw_force, vsd_max_force)
    print(f"  {name:10s}: dist={dist:.3f}m  raw_F={raw_force:.0f}N  clamped={clamped:.0f}N")

# ── Main loop ──
frame = 0
while not gym.query_viewer_has_closed(viewer):
    # Handle keyboard
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "toggle_vsd" and evt.value > 0:
            vsd_enabled = not vsd_enabled
            print(f"  VSD: {'ON' if vsd_enabled else 'OFF'} (scale={vsd_scale:.1f}x)")

        elif evt.action == "vsd_up" and evt.value > 0:
            vsd_scale *= 1.5
            print(f"  VSD scale: {vsd_scale:.2f}x")

        elif evt.action == "vsd_down" and evt.value > 0:
            vsd_scale /= 1.5
            print(f"  VSD scale: {vsd_scale:.2f}x")

        elif evt.action == "prev_joint" and evt.value > 0:
            selected_joint = (selected_joint - 1) % smpl_num_dofs
            print(f"  Joint: [{selected_joint}] {smpl_dof_names[selected_joint]}  "
                  f"({math.degrees(human_targets[selected_joint]):.1f} deg)")

        elif evt.action == "next_joint" and evt.value > 0:
            selected_joint = (selected_joint + 1) % smpl_num_dofs
            print(f"  Joint: [{selected_joint}] {smpl_dof_names[selected_joint]}  "
                  f"({math.degrees(human_targets[selected_joint]):.1f} deg)")

        elif evt.action == "decrease" and evt.value > 0:
            human_targets[selected_joint] -= joint_step
            gym.set_actor_dof_position_targets(env, human_actor, human_targets)
            print(f"    {smpl_dof_names[selected_joint]} = "
                  f"{math.degrees(human_targets[selected_joint]):.1f} deg")

        elif evt.action == "increase" and evt.value > 0:
            human_targets[selected_joint] += joint_step
            gym.set_actor_dof_position_targets(env, human_actor, human_targets)
            print(f"    {smpl_dof_names[selected_joint]} = "
                  f"{math.degrees(human_targets[selected_joint]):.1f} deg")

        elif evt.action == "reset_pose" and evt.value > 0:
            human_targets = np.zeros(smpl_num_dofs, dtype=np.float32)
            gym.set_actor_dof_position_targets(env, human_actor, human_targets)
            print("  Human pose reset (T-pose, arms free)")

        elif evt.action == "push_forward" and evt.value > 0:
            f = gymapi.Vec3(force_magnitude, 0.0, 0.0)
            apply_force_to_body(env, human_actor, 0, f)
            print(f"  Push forward: {force_magnitude}N")

        elif evt.action == "push_backward" and evt.value > 0:
            f = gymapi.Vec3(-force_magnitude, 0.0, 0.0)
            apply_force_to_body(env, human_actor, 0, f)
            print(f"  Push backward: {force_magnitude}N")

        elif evt.action == "push_up" and evt.value > 0:
            f = gymapi.Vec3(0.0, 0.0, force_magnitude * 3)
            apply_force_to_body(env, human_actor, 0, f)
            print(f"  Push up: {force_magnitude * 3}N")

        elif evt.action == "toggle_fixed" and evt.value > 0:
            is_fixed = not is_fixed
            print(f"  fix_base_link: {is_fixed} (restart required to take effect)")
            print(f"  NOTE: To test free-fall, restart with --no_vsd or press V to toggle")

        elif evt.action == "toggle_contacts" and evt.value > 0:
            show_contacts = not show_contacts
            print(f"  Contact visualization: {'ON' if show_contacts else 'OFF'}")

    # Apply VSD coupling forces
    apply_vsd_forces()

    # ── Draw VSD coupling lines ──
    gym.clear_lines(viewer)
    if vsd_enabled:
        vsd_colors = {
            "Pelvis": gymapi.Vec3(1.0, 0.0, 0.0),   # red
            "L_Shank": gymapi.Vec3(0.0, 1.0, 0.0),   # green
            "R_Shank": gymapi.Vec3(0.0, 0.8, 0.0),   # dark green
            "L_Foot": gymapi.Vec3(0.0, 0.5, 1.0),    # blue
            "R_Foot": gymapi.Vec3(0.0, 0.3, 0.8),    # dark blue
        }
        for smpl_bidx, exo_bidx, K, D, name in vsd_pairs:
            h_pos, h_vel = get_body_pos_vel(env, human_actor, smpl_bidx)
            e_pos, e_vel = get_body_pos_vel(env, exo_actor, exo_bidx)

            # Draw line between coupling points
            p1 = gymapi.Vec3(float(h_pos[0]), float(h_pos[1]), float(h_pos[2]))
            p2 = gymapi.Vec3(float(e_pos[0]), float(e_pos[1]), float(e_pos[2]))
            color = vsd_colors.get(name, gymapi.Vec3(1.0, 1.0, 0.0))
            gymutil.draw_line(p1, p2, color, gym, viewer, env)

            # Draw small cross at each coupling point for visibility
            cross_size = 0.03
            for axis in range(3):
                offset = [0.0, 0.0, 0.0]
                offset[axis] = cross_size
                pa = gymapi.Vec3(float(h_pos[0]-offset[0]), float(h_pos[1]-offset[1]), float(h_pos[2]-offset[2]))
                pb = gymapi.Vec3(float(h_pos[0]+offset[0]), float(h_pos[1]+offset[1]), float(h_pos[2]+offset[2]))
                gymutil.draw_line(pa, pb, gymapi.Vec3(1.0, 0.5, 0.0), gym, viewer, env)  # orange cross on human
                pa2 = gymapi.Vec3(float(e_pos[0]-offset[0]), float(e_pos[1]-offset[1]), float(e_pos[2]-offset[2]))
                pb2 = gymapi.Vec3(float(e_pos[0]+offset[0]), float(e_pos[1]+offset[1]), float(e_pos[2]+offset[2]))
                gymutil.draw_line(pa2, pb2, gymapi.Vec3(0.5, 0.5, 0.5), gym, viewer, env)  # gray cross on exo

        # Print VSD details to console periodically (every 2 seconds)
        if frame % 120 == 0:
            lines = ["  [VSD]"]
            total_f = 0.0
            for smpl_bidx, exo_bidx, K, D, name in vsd_pairs:
                h_pos, h_vel = get_body_pos_vel(env, human_actor, smpl_bidx)
                e_pos, e_vel = get_body_pos_vel(env, exo_actor, exo_bidx)
                delta = e_pos - h_pos
                dist = np.linalg.norm(delta)
                force = vsd_scale * (K * (e_pos - h_pos) + D * (e_vel - h_vel))
                fmag = np.linalg.norm(force)
                clamped = min(fmag, vsd_max_force * vsd_scale)
                total_f += clamped
                lines.append(f"    {name:8s} d={dist:.3f}m  F={fmag:.0f}N"
                             f"{'(clamp)' if fmag > vsd_max_force * vsd_scale else ''}"
                             f"  dx=[{delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}]")
            lines.append(f"    TOTAL ~{total_f:.0f}N  (scale={vsd_scale:.2f}x)")
            print("\n".join(lines))

    # ── Collision body visualization ──
    if show_contacts:
        axis_len = 0.06  # length of axis lines

        def draw_body_axes(actor_handle, num_bodies, color_base):
            """Draw local coordinate axes at each rigid body center."""
            body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
            for bi in range(num_bodies):
                pos = body_states['pose']['p'][bi]
                rot = body_states['pose']['r'][bi]
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                # Quaternion to rotation matrix columns (x,y,z,w order)
                qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
                # Local X axis (red)
                xx = 1 - 2*(qy*qy + qz*qz)
                xy = 2*(qx*qy + qw*qz)
                xz = 2*(qx*qz - qw*qy)
                # Local Y axis (green)
                yx = 2*(qx*qy - qw*qz)
                yy = 1 - 2*(qx*qx + qz*qz)
                yz = 2*(qy*qz + qw*qx)
                # Local Z axis (blue)
                zx = 2*(qx*qz + qw*qy)
                zy = 2*(qy*qz - qw*qx)
                zz = 1 - 2*(qx*qx + qy*qy)

                o = gymapi.Vec3(px, py, pz)
                gymutil.draw_line(o, gymapi.Vec3(px + xx*axis_len, py + xy*axis_len, pz + xz*axis_len),
                                  gymapi.Vec3(1.0, 0.2, 0.2), gym, viewer, env)  # X=red
                gymutil.draw_line(o, gymapi.Vec3(px + yx*axis_len, py + yy*axis_len, pz + yz*axis_len),
                                  gymapi.Vec3(0.2, 1.0, 0.2), gym, viewer, env)  # Y=green
                gymutil.draw_line(o, gymapi.Vec3(px + zx*axis_len, py + zy*axis_len, pz + zz*axis_len),
                                  gymapi.Vec3(0.2, 0.2, 1.0), gym, viewer, env)  # Z=blue

                # Small diamond marker at body center
                sz = 0.015
                gymutil.draw_line(gymapi.Vec3(px-sz, py, pz), gymapi.Vec3(px+sz, py, pz),
                                  color_base, gym, viewer, env)
                gymutil.draw_line(gymapi.Vec3(px, py-sz, pz), gymapi.Vec3(px, py+sz, pz),
                                  color_base, gym, viewer, env)
                gymutil.draw_line(gymapi.Vec3(px, py, pz-sz), gymapi.Vec3(px, py, pz+sz),
                                  color_base, gym, viewer, env)

        # Draw axes for both actors
        draw_body_axes(human_actor, smpl_num_bodies, gymapi.Vec3(1.0, 0.5, 0.0))  # orange markers
        draw_body_axes(exo_actor, exo_num_bodies, gymapi.Vec3(0.5, 0.5, 0.5))     # gray markers

        # Print body positions periodically
        if frame % 240 == 0:
            print(f"\n  [COLLISION BODIES] frame={frame}")
            h_states = gym.get_actor_rigid_body_states(env, human_actor, gymapi.STATE_POS)
            e_states = gym.get_actor_rigid_body_states(env, exo_actor, gymapi.STATE_POS)
            print(f"  Human ({smpl_num_bodies} bodies):")
            for bi in range(smpl_num_bodies):
                p = h_states['pose']['p'][bi]
                print(f"    [{bi:2d}] {smpl_body_names[bi]:15s} pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
            print(f"  Exo ({exo_num_bodies} bodies):")
            for bi in range(exo_num_bodies):
                p = e_states['pose']['p'][bi]
                print(f"    [{bi:2d}] {exo_body_names[bi]:20s} pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    frame += 1

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

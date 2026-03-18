"""
View SMPL Humanoid Model (with mesh)
-------------------------------------
Interactive SMPL humanoid viewer in IsaacGym.

Controls:
  1/2/3     : Switch body type (Average / Thin / Heavy)
  UP/DOWN   : Select joint (prev/next)
  LEFT/RIGHT: Decrease/Increase selected joint angle
  0 (zero)  : Reset all joints to default pose
  F         : Apply forward push force to pelvis
  B         : Apply backward push force to pelvis
  SPACE     : Apply upward impulse

Usage:
    conda activate phc
    python scripts/view_smpl_model.py
    python scripts/view_smpl_model.py --xml /tmp/smpl/smpl_humanoid_XXXX.xml
"""

import os
import re
import glob
import math
import numpy as np
from isaacgym import gymapi, gymutil


def find_valid_mesh_xml():
    """Find the most recent XML in /tmp/smpl/ whose mesh directory still exists."""
    candidates = sorted(glob.glob("/tmp/smpl/smpl_humanoid_*.xml"),
                        key=os.path.getmtime, reverse=True)
    for c in candidates:
        with open(c, 'r') as f:
            match = re.search(r'file="([^/]+)/geom/', f.read())
        if match and os.path.isdir(os.path.join("/tmp/smpl", match.group(1), "geom")):
            return c
    return None


def generate_body_xmls():
    """Generate thin/average/heavy SMPL XMLs using SMPL_Robot."""
    paths = {
        "average": "/tmp/smpl/view_average.xml",
        "thin": "/tmp/smpl/view_thin.xml",
        "heavy": "/tmp/smpl/view_heavy.xml",
    }
    # check if all already exist with valid mesh dirs
    all_valid = True
    for p in paths.values():
        if not os.path.exists(p):
            all_valid = False
            break
        with open(p, 'r') as f:
            match = re.search(r'file="([^/]+)/geom/', f.read())
        if not match or not os.path.isdir(os.path.join("/tmp/smpl", match.group(1), "geom")):
            all_valid = False
            break
    if all_valid:
        return paths

    try:
        import torch
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

        for name, beta1_val in [("average", 0.0), ("thin", -2.0), ("heavy", 2.0)]:
            betas = torch.zeros(1, 10)
            betas[0, 1] = beta1_val
            robot.load_from_skeleton(betas=betas, gender=[0])
            robot.write_xml(paths[name])
            print(f"  Generated: {name} (beta[1]={beta1_val})")

        # NOTE: do NOT call robot.remove_geoms() — it deletes the STL mesh directories
    except Exception as e:
        print(f"Failed to generate body XMLs: {e}")
        print("Falling back to auto-detected XML")
        return None

    return paths


# ── parse arguments ──
args = gymutil.parse_arguments(
    description="View SMPL humanoid model in IsaacGym",
    custom_parameters=[{
        "name": "--xml",
        "type": str,
        "default": "",
        "help": "Path to SMPL humanoid XML (with mesh). If empty, generates average/thin/heavy."
    }])

# ── generate / find XMLs ──
if args.xml and os.path.exists(args.xml):
    body_xmls = {"custom": args.xml}
else:
    print("Generating body type XMLs...")
    body_xmls = generate_body_xmls()
    if body_xmls is None:
        fallback = find_valid_mesh_xml()
        if fallback is None:
            print("No valid mesh XML found. Run training/eval first, or specify --xml.")
            quit()
        body_xmls = {"default": fallback}

body_names_list = list(body_xmls.keys())
current_body_idx = 0
if "average" in body_xmls:
    current_body_idx = body_names_list.index("average")

# ── initialize gym ──
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
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


# ── helper: create humanoid in env ──
def load_humanoid(xml_path):
    """Load asset and create actor, returns (env, actor, num_dofs, dof_names)."""
    asset_root = os.path.dirname(xml_path)
    asset_file = os.path.basename(xml_path)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    num_dofs = gym.get_asset_dof_count(asset)
    num_bodies = gym.get_asset_rigid_body_count(asset)
    dof_names = list(gym.get_asset_dof_names(asset))
    bnames = list(gym.get_asset_rigid_body_names(asset))

    spacing = 2.0
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0.0),
                         gymapi.Vec3(spacing, spacing, spacing), 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.1)
    actor = gym.create_actor(env, asset, pose, "smpl_humanoid", 0, 1)

    # position control for all DOFs
    dof_props = gym.get_actor_dof_properties(env, actor)
    for i in range(num_dofs):
        dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
        dof_props['stiffness'][i] = 300.0
        dof_props['damping'][i] = 30.0
    gym.set_actor_dof_properties(env, actor, dof_props)

    # default arm-down pose
    targets = np.zeros(num_dofs, dtype=np.float32)

    # Set arms down: rotate shoulders around X axis
    # L_Shoulder extends in +Y, so negative X rotation brings it down toward -Z
    # R_Shoulder extends in -Y, so positive X rotation brings it down toward -Z
    for i, name in enumerate(dof_names):
        if name == "L_Shoulder_x":
            targets[i] = math.radians(-80)
        elif name == "R_Shoulder_x":
            targets[i] = math.radians(80)

    gym.set_actor_dof_position_targets(env, actor, targets)

    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    for i in range(num_dofs):
        dof_states['pos'][i] = targets[i]
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

    print(f"\n=== Loaded: {os.path.basename(xml_path)} ===")
    print(f"  Bodies: {num_bodies}, DOFs: {num_dofs}")

    return env, actor, num_dofs, dof_names, targets


# ── initial load ──
xml_path = body_xmls[body_names_list[current_body_idx]]
env, actor, num_dofs, dof_names, dof_targets = load_humanoid(xml_path)

# ── keyboard events ──
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "prev_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "next_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "decrease")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "increase")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "reset_pose")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "push_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "push_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "push_up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "body_0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "body_1")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "body_2")

# camera
cam_pos = gymapi.Vec3(3.0, -1.5, 1.5)
cam_target = gymapi.Vec3(0.0, 0.0, 0.9)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# joint control state
selected_joint = 0
joint_step = math.radians(5)  # 5 degrees per press
force_magnitude = 200.0       # N

print(f"\n=== Controls ===")
print(f"  UP/DOWN   : Select joint")
print(f"  LEFT/RIGHT: Change joint angle (+/-5 deg)")
print(f"  0         : Reset pose (arms down)")
print(f"  F/B       : Push forward/backward")
print(f"  SPACE     : Push upward")
if len(body_names_list) > 1:
    for i, name in enumerate(body_names_list):
        print(f"  {i+1}         : Switch to {name}")
print(f"\n  Selected joint: [{selected_joint}] {dof_names[selected_joint]}")

needs_reload = False
reload_body_idx = 0

# ── main loop ──
while not gym.query_viewer_has_closed(viewer):
    # handle keyboard
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "prev_joint" and evt.value > 0:
            selected_joint = (selected_joint - 1) % num_dofs
            print(f"  Selected joint: [{selected_joint}] {dof_names[selected_joint]}  "
                  f"(current: {math.degrees(dof_targets[selected_joint]):.1f} deg)")

        elif evt.action == "next_joint" and evt.value > 0:
            selected_joint = (selected_joint + 1) % num_dofs
            print(f"  Selected joint: [{selected_joint}] {dof_names[selected_joint]}  "
                  f"(current: {math.degrees(dof_targets[selected_joint]):.1f} deg)")

        elif evt.action == "decrease" and evt.value > 0:
            dof_targets[selected_joint] -= joint_step
            gym.set_actor_dof_position_targets(env, actor, dof_targets)
            print(f"    {dof_names[selected_joint]} = {math.degrees(dof_targets[selected_joint]):.1f} deg")

        elif evt.action == "increase" and evt.value > 0:
            dof_targets[selected_joint] += joint_step
            gym.set_actor_dof_position_targets(env, actor, dof_targets)
            print(f"    {dof_names[selected_joint]} = {math.degrees(dof_targets[selected_joint]):.1f} deg")

        elif evt.action == "reset_pose" and evt.value > 0:
            dof_targets = np.zeros(num_dofs, dtype=np.float32)
            gym.set_actor_dof_position_targets(env, actor, dof_targets)
            print("  Pose reset (arms down)")

        elif evt.action == "push_forward" and evt.value > 0:
            forces = gymapi.Vec3(force_magnitude, 0.0, 0.0)
            rh = gym.get_actor_rigid_body_handle(env, actor, 0)
            gym.apply_body_forces(env, rh, forces, None, gymapi.ENV_SPACE)
            print(f"  Push forward: {force_magnitude}N")

        elif evt.action == "push_backward" and evt.value > 0:
            forces = gymapi.Vec3(-force_magnitude, 0.0, 0.0)
            rh = gym.get_actor_rigid_body_handle(env, actor, 0)
            gym.apply_body_forces(env, rh, forces, None, gymapi.ENV_SPACE)
            print(f"  Push backward: {force_magnitude}N")

        elif evt.action == "push_up" and evt.value > 0:
            forces = gymapi.Vec3(0.0, 0.0, force_magnitude * 3)
            rh = gym.get_actor_rigid_body_handle(env, actor, 0)
            gym.apply_body_forces(env, rh, forces, None, gymapi.ENV_SPACE)
            print(f"  Push up: {force_magnitude * 3}N")

        elif evt.action.startswith("body_") and evt.value > 0:
            idx = int(evt.action.split("_")[1])
            if idx < len(body_names_list) and idx != current_body_idx:
                needs_reload = True
                reload_body_idx = idx

    # reload body if requested (must be done outside event loop)
    if needs_reload:
        needs_reload = False
        current_body_idx = reload_body_idx
        gym.destroy_env(env)
        xml_path = body_xmls[body_names_list[current_body_idx]]
        env, actor, num_dofs, dof_names, dof_targets = load_humanoid(xml_path)
        selected_joint = min(selected_joint, num_dofs - 1)
        print(f"  Switched to: {body_names_list[current_body_idx]}")

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

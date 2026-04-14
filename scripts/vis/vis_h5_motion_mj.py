"""
H5 Motion Library 시각화 (MuJoCo viewer)
- FK-변환된 H5 motion library를 SMPL humanoid에서 재생
- 물리 시뮬레이션 없이 reference pose만 표시

Usage:
    python scripts/vis/vis_h5_motion_mj.py
    python scripts/vis/vis_h5_motion_mj.py --motion_file sample_data/h5_motion_library.pkl
    python scripts/vis/vis_h5_motion_mj.py --clip_idx 5

Controls:
    T: next clip
    R: restart current clip
    Space: pause/resume
"""
import os
import sys
import time
import argparse

sys.path.append(os.getcwd())

from phc.utils.flags import flags
flags.im_eval = True  # disable random heading (which assumes Z-up pose_aa)

from phc.utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
import torch
import numpy as np
from copy import deepcopy
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


def add_visual_capsule(scene, point1, point2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, time_step, paused
    if chr(keycode) == "T":
        curr_start += num_motions
        print(f"Next clip (start_idx={curr_start})")
        motion_lib.load_motions(
            skeleton_trees=[sk_tree] * num_motions,
            gender_betas=[torch.zeros(17)] * num_motions,
            limb_weights=[np.zeros(10)] * num_motions,
            random_sample=False, start_idx=curr_start
        )
        time_step = 0
    elif chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        paused = not paused
        print("Paused" if paused else "Playing")
    else:
        print(f"Key not mapped: {chr(keycode)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", default="sample_data/h5_motion_library.pkl")
    parser.add_argument("--clip_idx", type=int, default=0, help="Starting clip index")
    args = parser.parse_args()

    device = torch.device("cpu")
    curr_start = args.clip_idx
    num_motions = 1
    motion_id = 0
    time_step = 0
    dt = 1/30
    paused = False

    motion_lib_cfg = EasyDict({
        "motion_file": args.motion_file,
        "device": device,
        "fix_height": FixHeightMode.no_fix,  # our pose_aa is bone-order Y-up SMPL, fix_height mis-computes min Z otherwise
        "min_length": -1,
        "max_length": -1,
        "im_eval": False,
        "multi_thread": False,
        "smpl_type": 'smpl',
        "randomrize_heading": False,
    })

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": True,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smpl",
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")

    gender_beta = np.zeros(17)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(gender_beta[None, 1:]),
        gender=gender_beta[0:1],
        objs_info=None
    )
    test_xml = "/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_xml)
    sk_tree = SkeletonTree.from_mjcf(test_xml)

    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[sk_tree] * num_motions,
        gender_betas=[torch.zeros(17)] * num_motions,
        limb_weights=[np.zeros(10)] * num_motions,
        random_sample=False, start_idx=curr_start
    )

    print(f"Loaded motion library: {args.motion_file}")
    print(f"Starting at clip index: {curr_start}")
    print(f"Controls: T=next clip, R=restart, Space=pause")

    mj_model = mujoco.MjModel.from_xml_path(test_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(len(sk_tree._node_indices)):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))

        # Orbiting third-person camera that tracks the pelvis
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0

        while viewer.is_running():
            step_start = time.time()
            motion_len = motion_lib.get_motion_length(motion_id).item()
            motion_time = time_step % motion_len

            motion_res = motion_lib.get_motion_state(
                torch.tensor([motion_id]).to(device),
                torch.tensor([motion_time]).to(device)
            )

            root_pos = motion_res["root_pos"]
            root_rot = motion_res["root_rot"]
            dof_pos = motion_res["dof_pos"]
            rb_pos = motion_res["rg_pos"]

            # Set MuJoCo state directly (no physics, just visualization)
            mj_data.qpos[:3] = root_pos[0].cpu().numpy()
            mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
            mj_data.qpos[7:] = sRot.from_rotvec(
                dof_pos[0].cpu().numpy().reshape(-1, 3)
            ).as_euler("XYZ").flatten()

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += dt

            for i in range(rb_pos.shape[1]):
                viewer.user_scn.geoms[i].pos = rb_pos[0, i]

            # Camera follows pelvis
            viewer.cam.lookat[0] = root_pos[0, 0].item()
            viewer.cam.lookat[1] = root_pos[0, 1].item()
            viewer.cam.lookat[2] = 0.9

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

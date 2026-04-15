"""Render H5 motion library to mp4 using MuJoCo offscreen renderer (EGL).

Produces a realistic 3D view of the SMPL humanoid playing back H5 motion clips,
suitable for headless verification before launching VIC training.

Usage:
    MUJOCO_GL=egl python scripts/vis/render_h5_mp4_mj.py --num-clips 3

Output: output/h5_motion_preview_mj.mp4
"""
import argparse
import os
import sys

# Ensure EGL backend for MuJoCo offscreen rendering on headless server.
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.append(os.getcwd())

import numpy as np
import torch
import imageio
import mujoco
from easydict import EasyDict
from scipy.spatial.transform import Rotation as sRot

from phc.utils.flags import flags
flags.im_eval = True  # avoid random heading during motion load

from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


def build_smpl_mjcf():
    robot_cfg = {
        "mesh": False, "rel_joint_lm": False, "upright_start": True,
        "remove_toe": False, "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, "model": "smpl",
        "big_ankle": True, "freeze_hand": False, "box_body": True,
        "body_params": {}, "joint_params": {}, "geom_params": {},
        "actuator_params": {},
    }
    robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    beta = np.zeros(17)
    robot.load_from_skeleton(
        betas=torch.from_numpy(beta[None, 1:]),
        gender=beta[0:1], objs_info=None,
    )
    xml = "/tmp/smpl/render_h5_mp4_mj.xml"
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    robot.write_xml(xml)
    return xml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", default="sample_data/h5_motion_library.pkl")
    parser.add_argument("--out", default="output/h5_motion_preview_mj.mp4")
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")  # motion lib on cpu is plenty fast for preview

    xml_path = build_smpl_mjcf()
    sk_tree = SkeletonTree.from_mjcf(xml_path)

    mlib_cfg = EasyDict({
        "motion_file": args.motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": "smpl", "randomrize_heading": False,
    })
    mlib = MotionLibSMPL(mlib_cfg)
    mlib.load_motions(
        skeleton_trees=[sk_tree] * args.num_clips,
        gender_betas=[torch.zeros(17)] * args.num_clips,
        limb_weights=[np.zeros(10)] * args.num_clips,
        random_sample=False, start_idx=args.start_idx,
    )

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    cam = mujoco.MjvCamera()
    cam.distance = 3.5
    cam.azimuth = 135.0
    cam.elevation = -15.0
    cam.lookat[:] = [0.0, 0.0, 0.9]

    opt = mujoco.MjvOption()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)

    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)

    total = 0
    try:
        for clip_i in range(args.num_clips):
            motion_len = float(mlib.get_motion_length(clip_i).item())
            n_frames = max(1, int(motion_len * args.fps))
            print(f"[mj] clip {clip_i}: {motion_len:.2f}s -> {n_frames} frames")
            times = torch.linspace(0.0, max(motion_len - 0.01, 0.0), steps=n_frames)
            for f in range(n_frames):
                t = times[f].item()
                res = mlib.get_motion_state(
                    torch.tensor([clip_i]), torch.tensor([t])
                )
                root_pos = res["root_pos"][0].cpu().numpy()
                root_rot = res["root_rot"][0].cpu().numpy()  # xyzw
                dof_pos = res["dof_pos"][0].cpu().numpy()    # axis-angle

                mj_data.qpos[:3] = root_pos
                mj_data.qpos[3:7] = root_rot[[3, 0, 1, 2]]   # wxyz
                mj_data.qpos[7:] = sRot.from_rotvec(
                    dof_pos.reshape(-1, 3)
                ).as_euler("XYZ").flatten()
                mujoco.mj_forward(mj_model, mj_data)

                # Track pelvis x,y (z fixed at 0.9 for stable camera)
                cam.lookat[0] = float(root_pos[0])
                cam.lookat[1] = float(root_pos[1])

                renderer.update_scene(mj_data, camera=cam, scene_option=opt)
                frame = renderer.render()
                writer.append_data(frame)
                total += 1
    finally:
        writer.close()

    print(f"[mj] Done -> {args.out} ({total} frames, {total/args.fps:.2f}s)")


if __name__ == "__main__":
    main()

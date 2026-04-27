"""Offscreen-render a clip from a pkl and save as mp4 (no display required).

Usage:
  conda activate phc
  python scripts/render_clip_video.py <pkl> <motion_idx> [--seconds N] [--out path]
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import mujoco
import imageio
from scipy.spatial.transform import Rotation as sRot
from easydict import EasyDict

from phc.utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('motion_idx', type=int)
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    device = torch.device('cpu')
    motion_lib_cfg = EasyDict({
        "motion_file": args.pkl,
        "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1, "im_eval": False,
        "multi_thread": False, "smpl_type": 'smpl', "randomrize_heading": False,
    })
    robot_cfg = {
        "mesh": False, "rel_joint_lm": False, "upright_start": True,
        "remove_toe": False, "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, "model": "smpl",
        "big_ankle": True, "freeze_hand": False, "box_body": True,
        "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gender_beta = np.zeros((17))
    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]),
                                  gender=gender_beta[0:1], objs_info=None)
    test_xml = "/tmp/smpl/render_clip.xml"
    os.makedirs(os.path.dirname(test_xml), exist_ok=True)
    smpl_robot.write_xml(test_xml)
    sk_tree = SkeletonTree.from_mjcf(test_xml)

    import joblib
    n_total = len(joblib.load(args.pkl))
    print(f"[render] pkl has {n_total} clip(s), motion_idx={args.motion_idx}")

    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[sk_tree] * n_total,
        gender_betas=[torch.zeros(17)] * n_total,
        limb_weights=[np.zeros(10)] * n_total,
        random_sample=False, start_idx=0,
    )
    motion_id = args.motion_idx
    motion_len = motion_lib.get_motion_length(motion_id).item()
    print(f"[render] motion {motion_id} length: {motion_len:.2f}s, capturing {args.seconds:.1f}s")

    mj_model = mujoco.MjModel.from_xml_path(test_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 1 / args.fps

    out_path = args.out or f'videos/clip_{os.path.splitext(os.path.basename(args.pkl))[0]}_idx{args.motion_idx}.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_frames = int(args.seconds * args.fps)
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 1.0]
    cam.distance = 4.0
    cam.azimuth = 90
    cam.elevation = -10

    # Use raw png frames -> ffmpeg, since imageio with libopenh264 fallback flaky
    frames_dir = '/tmp/render_clip_frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    dt = 1 / args.fps
    for i in range(n_frames):
        motion_time = (i * dt) % motion_len
        mres = motion_lib.get_motion_state(
            torch.tensor([motion_id]).to(device),
            torch.tensor([motion_time]).to(device))
        root_pos = mres["root_pos"][0].cpu().numpy()
        root_rot = mres["root_rot"][0].cpu().numpy()
        dof_pos = mres["dof_pos"][0].cpu().numpy()
        mj_data.qpos[:3] = root_pos
        mj_data.qpos[3:7] = root_rot[[3, 0, 1, 2]]
        mj_data.qpos[7:] = sRot.from_rotvec(dof_pos.reshape(-1, 3)).as_euler("XYZ").flatten()
        mujoco.mj_forward(mj_model, mj_data)

        # Camera follow
        cam.lookat[0] = root_pos[0]
        cam.lookat[1] = root_pos[1]
        cam.lookat[2] = root_pos[2] + 0.3

        renderer.update_scene(mj_data, camera=cam)
        img = renderer.render()
        imageio.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), img)
        if i % 30 == 0:
            print(f"[render] frame {i}/{n_frames}")

    ffmpeg_bin = shutil.which('ffmpeg') or '/home/exolab/miniconda3/bin/ffmpeg'
    cmd = [ffmpeg_bin, '-y', '-framerate', str(args.fps),
           '-i', os.path.join(frames_dir, 'frame_%06d.png'),
           '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', out_path]
    print(f"[render] combining: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[render] saved {out_path} ({size_mb:.1f} MB)")
        shutil.rmtree(frames_dir)


if __name__ == '__main__':
    main()

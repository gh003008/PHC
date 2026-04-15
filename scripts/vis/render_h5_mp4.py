"""Render first K clips of an H5-derived motion library pkl to a single mp4.

Headless-safe (matplotlib Agg + imageio-ffmpeg). Intended for quick visual
verification of H5 motion data before launching VIC training.

Usage:
    python scripts/vis/render_h5_mp4.py
    python scripts/vis/render_h5_mp4.py --motion-file sample_data/h5_motion_library.pkl --num-clips 3

Output: output/h5_motion_preview.mp4
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
from easydict import EasyDict

from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


SMPL_BONES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21), (20, 22), (21, 23),
]


def setup_skeleton_tree():
    robot_cfg = {
        "mesh": False, "rel_joint_lm": False, "upright_start": True,
        "remove_toe": False, "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, "model": "smpl",
        "big_ankle": True, "freeze_hand": False, "box_body": True,
        "body_params": {}, "joint_params": {}, "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gender_beta = np.zeros(17)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(gender_beta[None, 1:]),
        gender=gender_beta[0:1], objs_info=None,
    )
    xml_path = "/tmp/smpl/render_h5_mp4_tree.xml"
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    smpl_robot.write_xml(xml_path)
    return SkeletonTree.from_mjcf(xml_path)


def render_frame(rb_pos, title, bounds):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rb_pos[:, 0], rb_pos[:, 1], rb_pos[:, 2], c="tab:blue", s=24)
    for i, j in SMPL_BONES:
        if i < len(rb_pos) and j < len(rb_pos):
            ax.plot(
                [rb_pos[i, 0], rb_pos[j, 0]],
                [rb_pos[i, 1], rb_pos[j, 1]],
                [rb_pos[i, 2], rb_pos[j, 2]],
                c="tab:blue", linewidth=1.6,
            )
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=18, azim=45)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba())  # H, W, 4
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", default="sample_data/h5_motion_library.pkl")
    parser.add_argument("--out", default="output/h5_motion_preview.mp4")
    parser.add_argument("--num-clips", type=int, default=3,
                        help="Number of clips from start of library to render.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1,
                        help="Render every Nth frame (1 = every frame).")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[render_h5_mp4] device={device}  motion_file={args.motion_file}")

    sk_tree = setup_skeleton_tree()

    cfg = EasyDict({
        "motion_file": args.motion_file, "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False,
        "smpl_type": "smpl", "randomrize_heading": False,
    })
    mlib = MotionLibSMPL(cfg)
    mlib.load_motions(
        skeleton_trees=[sk_tree] * args.num_clips,
        gender_betas=[torch.zeros(17)] * args.num_clips,
        limb_weights=[np.zeros(10)] * args.num_clips,
        random_sample=False, start_idx=0,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)

    total_frames = 0
    for clip_i in range(args.num_clips):
        motion_len = float(mlib.get_motion_length(clip_i).item())
        n_frames = max(1, int(motion_len * args.fps))
        print(f"[render_h5_mp4] clip {clip_i}: {motion_len:.2f}s -> {n_frames} frames")
        times = torch.linspace(0.0, max(motion_len - 0.01, 0.0), steps=n_frames).to(device)
        motion_ids = torch.full((n_frames,), clip_i, device=device, dtype=torch.long)
        res = mlib.get_motion_state(motion_ids, times)
        rb_pos_all = res["rg_pos"].cpu().numpy()  # (n_frames, J, 3)

        flat = rb_pos_all.reshape(-1, 3)
        bounds = (
            float(flat[:, 0].min()) - 0.3, float(flat[:, 0].max()) + 0.3,
            float(flat[:, 1].min()) - 0.3, float(flat[:, 1].max()) + 0.3,
            0.0, 2.2,
        )
        for f in range(0, n_frames, args.stride):
            title = f"clip {clip_i}  t={f/args.fps:.2f}s  frame {total_frames}"
            frame_rgb = render_frame(rb_pos_all[f], title, bounds)
            writer.append_data(frame_rgb)
            total_frames += 1

    writer.close()
    print(f"[render_h5_mp4] Done -> {args.out} ({total_frames} frames)")


if __name__ == "__main__":
    main()

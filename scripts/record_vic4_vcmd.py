"""Record IsaacGym viewer frames for VIC4+v_cmd slots and combine into mp4.

Hooks Humanoid.render to call gym.write_viewer_image_to_file every step into
/tmp/record_frames_<slot>/, runs for --record_seconds, then exits and combines
PNGs into mp4 via ffmpeg.

Usage:
  conda activate phc
  python scripts/record_vic4_vcmd.py --slot S8 --epoch 12700 --record_seconds 20
  python scripts/record_vic4_vcmd.py --slot S5 --record_seconds 20  # uses S5.pth
"""
from __future__ import annotations
import argparse
import glob
import os
import shutil
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for p in (_PHC_PKG, _PHC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np
import torch


V_LO, V_HI = 0.32, 0.75
ARROW_LEN_AT_VHI = 1.5


def install_patches(record_dir, target_frames):
    from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
    from phc.env.tasks.humanoid import Humanoid

    state = {'frame_count': 0, 'done': False}

    orig_reset = HumanoidImVIC._compute_reset

    def patched_reset(self):
        if not getattr(self, '_term_dist_restored', False):
            cfg_dist = float(self.cfg["env"].get("terminationDistance", 0.5))
            self._termination_distances[:] = cfg_dist
            self._term_dist_restored = True
            print(f"[rec] _termination_distances overridden to {cfg_dist}")
        return orig_reset(self)

    HumanoidImVIC._compute_reset = patched_reset

    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        ret = orig_render(self, sync_frame_time)
        if self.viewer is None or not hasattr(self, '_current_cmd'):
            return ret
        if state['done']:
            return ret

        # v_cmd arrows
        self.gym.clear_lines(self.viewer)
        root_states = self._humanoid_root_states
        n = root_states.shape[0]
        v_cmd = self._current_cmd[:, 0].detach().cpu().numpy()
        root_pos = root_states[:, :3].detach().cpu().numpy()
        root_rot = root_states[:, 3:7].detach().cpu().numpy()
        x, y, z, w = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        fwd_x = 1 - 2 * (y * y + z * z)
        fwd_y = 2 * (x * y + w * z)
        norm = np.sqrt(fwd_x ** 2 + fwd_y ** 2) + 1e-8
        fwd_x /= norm
        fwd_y /= norm
        head_z = root_pos[:, 2] + 0.6
        scale = (v_cmd / V_HI) * ARROW_LEN_AT_VHI
        t_color = np.clip((v_cmd - V_LO) / (V_HI - V_LO), 0.0, 1.0)
        r = t_color
        g = np.zeros_like(t_color)
        b = 1.0 - t_color
        for env_i in range(n):
            sx, sy, sz = root_pos[env_i, 0], root_pos[env_i, 1], head_z[env_i]
            ex = sx + fwd_x[env_i] * scale[env_i]
            ey = sy + fwd_y[env_i] * scale[env_i]
            verts = np.array([sx, sy, sz, ex, ey, sz], dtype=np.float32)
            colors = np.array([r[env_i], g[env_i], b[env_i]], dtype=np.float32)
            self.gym.add_lines(self.viewer, self.envs[env_i], 1, verts, colors)
            head_size = 0.1
            lx = ex - fwd_x[env_i] * head_size + fwd_y[env_i] * head_size * 0.5
            ly = ey - fwd_y[env_i] * head_size - fwd_x[env_i] * head_size * 0.5
            verts2 = np.array([ex, ey, sz, lx, ly, sz], dtype=np.float32)
            self.gym.add_lines(self.viewer, self.envs[env_i], 1, verts2, colors)
            rx = ex - fwd_x[env_i] * head_size - fwd_y[env_i] * head_size * 0.5
            ry = ey - fwd_y[env_i] * head_size + fwd_x[env_i] * head_size * 0.5
            verts3 = np.array([ex, ey, sz, rx, ry, sz], dtype=np.float32)
            self.gym.add_lines(self.viewer, self.envs[env_i], 1, verts3, colors)

        # Frame capture
        img_path = os.path.join(record_dir, f"frame_{state['frame_count']:06d}.png")
        try:
            self.gym.write_viewer_image_to_file(self.viewer, img_path)
        except Exception as e:
            print(f"[rec] capture failed at frame {state['frame_count']}: {e}")
        state['frame_count'] += 1

        if state['frame_count'] % 30 == 0:
            print(f"[rec] frame {state['frame_count']}/{target_frames} captured")

        if state['frame_count'] >= target_frames:
            print(f"[rec] target {target_frames} frames reached, exiting")
            state['done'] = True
            sys.exit(0)

        return ret

    Humanoid.render = patched_render


def find_checkpoint(slot, epoch):
    if epoch < 0:
        # Look for SX.pth (final) or latest SX_NNNNNNNN.pth
        final = f"output/VIC4_VCMD_{slot}.pth"
        if os.path.exists(final):
            return final, -1
        candidates = sorted(glob.glob(f"output/VIC4_VCMD_{slot}_*.pth"))
        if candidates:
            return candidates[-1], int(os.path.basename(candidates[-1]).split('_')[-1].replace('.pth', ''))
        raise FileNotFoundError(f"no checkpoint for {slot}")
    return f"output/VIC4_VCMD_{slot}_{epoch:08d}.pth", epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot', choices=['S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S19B'], required=True)
    ap.add_argument('--epoch', type=int, default=-1)
    ap.add_argument('--num_envs', type=int, default=4)
    ap.add_argument('--record_seconds', type=int, default=20)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--out_dir', default='videos')
    args = ap.parse_args()

    ckpt_path, resolved_epoch = find_checkpoint(args.slot, args.epoch)
    print(f"[rec] using checkpoint: {ckpt_path}")

    target_frames = args.record_seconds * args.fps
    record_dir = f"/tmp/record_frames_{args.slot}_{resolved_epoch}"
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.makedirs(record_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.slot in ('S18', 'S19', 'S19B'):
        exp_dir = 'exp_config/forward_walking/260501_VIC4_VCMD_S18_S19'
    elif args.slot in ('S16', 'S17'):
        exp_dir = 'exp_config/forward_walking/260429_VIC4_VCMD_S16_S17'
    elif args.slot in ('S12', 'S13', 'S14', 'S15'):
        exp_dir = 'exp_config/forward_walking/260428_VIC4_VCMD_v9'
    elif args.slot in ('S9', 'S10', 'S11'):
        exp_dir = 'exp_config/forward_walking/260428_VIC4_VCMD_v8'
    else:
        exp_dir = 'exp_config/forward_walking/260427_VIC4_VCMD'
    src_env = f'{exp_dir}/env_im_walk_vic_{args.slot}.yaml'
    cfg_env = f'/tmp/env_vic4_vcmd_{args.slot}_rec.yaml'
    with open(src_env) as f:
        env_cfg = f.read()
    import re
    env_cfg = re.sub(r'^(\s*num_envs:\s*)\d+', rf'\g<1>{args.num_envs}', env_cfg, flags=re.MULTILINE)
    env_cfg = re.sub(r'^(\s*numEnvs:\s*)\d+', rf'\g<1>{args.num_envs}', env_cfg, flags=re.MULTILINE)
    with open(cfg_env, 'w') as f:
        f.write(env_cfg)

    cfg_train = f'{exp_dir}/im_walk_vic.yaml'
    task_name = 'HumanoidImVICCmdRetime' if args.slot == 'S4' else 'HumanoidImVICCmdMultiClip'

    sys.argv = [
        'run.py',
        '--task', task_name,
        '--cfg_env', cfg_env,
        '--cfg_train', cfg_train,
        '--num_envs', str(args.num_envs),
        '--test', '--epoch', str(resolved_epoch),
        '--experiment', f'VIC4_VCMD_{args.slot}',
    ]

    install_patches(record_dir, target_frames)

    print(f"[rec] slot={args.slot} resolved_epoch={resolved_epoch} num_envs={args.num_envs}")
    print(f"[rec] target_frames={target_frames} ({args.record_seconds}s @ {args.fps}fps)")
    print(f"[rec] frames -> {record_dir}")

    try:
        from phc import run as phc_run
        phc_run.main()
    except SystemExit:
        pass

    # Combine PNGs to mp4
    out_path = os.path.join(args.out_dir, f"VIC4_VCMD_{args.slot}_ep{resolved_epoch}.mp4")
    ffmpeg_bin = shutil.which('ffmpeg') or '/home/exolab/miniconda3/bin/ffmpeg'
    cmd = [
        ffmpeg_bin, '-y',
        '-framerate', str(args.fps),
        '-i', os.path.join(record_dir, 'frame_%06d.png'),
        '-c:v', 'libopenh264',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    print(f"[rec] combining frames: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[rec] saved {out_path} ({size_mb:.1f} MB)")
        # Cleanup PNGs
        shutil.rmtree(record_dir)
    else:
        print(f"[rec] WARN: ffmpeg did not produce {out_path}; PNGs kept at {record_dir}")


if __name__ == '__main__':
    main()

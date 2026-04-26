"""IsaacGym viewer for VIC4+v_cmd slots (S4/S5/S6) with v_cmd visualization.

Wraps `phc/run.py --test` with:
  1. Restores env yaml's terminationDistance (0.4 or 0.6) — overrides the
     hardcoded 0.5m in `phc/learning/im_amp_players.py:41`.
  2. v_cmd visualization:
       - Forward arrow above each humanoid head, length proportional to v_cmd.
         Color: blue (slow) -> red (fast).
       - Terminal log of v_cmd at periodic intervals.

Usage:
  conda activate phc
  python scripts/vis_vic4_vcmd.py --slot S4
  python scripts/vis_vic4_vcmd.py --slot S5 --epoch 10000
  python scripts/vis_vic4_vcmd.py --slot S6 --no_virtual_display

Notes:
  - num_envs=1 NOT supported by HumanoidImVICCmdMultiClip
  - Default num_envs=8
"""
from __future__ import annotations
import argparse
import os
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


# v_cmd range for arrow color/length (user option c: S5/S6 multiclip_v_cmd_range)
V_LO, V_HI = 0.32, 0.75
ARROW_LEN_AT_VHI = 1.5


def install_patches():
    from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
    from phc.env.tasks.humanoid import Humanoid

    # Restore env yaml's terminationDistance (overrides player 0.5m hardcode)
    orig_reset = HumanoidImVIC._compute_reset

    def patched_reset(self):
        if not getattr(self, '_term_dist_restored', False):
            cfg_dist = float(self.cfg["env"].get("terminationDistance", 0.5))
            self._termination_distances[:] = cfg_dist
            self._term_dist_restored = True
            self._vis_step = 0
            print(f"[vis] _termination_distances overridden to {cfg_dist} (env yaml)")
        return orig_reset(self)

    HumanoidImVIC._compute_reset = patched_reset

    # Draw forward arrows + log v_cmd
    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        ret = orig_render(self, sync_frame_time)
        if self.viewer is None or not hasattr(self, '_current_cmd'):
            return ret
        self._vis_step = getattr(self, '_vis_step', 0) + 1

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

        if self._vis_step % 60 == 0:
            v_str = "  ".join(f"env{e:02d}={v_cmd[e]:.2f}" for e in range(min(3, n)))
            print(f"[vis] step={self._vis_step:6d}  {v_str}")

        return ret

    Humanoid.render = patched_render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot', choices=['S4', 'S5', 'S6'], required=True)
    ap.add_argument('--epoch', type=int, default=-1)
    ap.add_argument('--num_envs', type=int, default=8)
    ap.add_argument('--no_virtual_display', action='store_true')
    args = ap.parse_args()

    src_env = f'exp_config/forward_walking/260427_VIC4_VCMD/env_im_walk_vic_{args.slot}.yaml'
    cfg_env = f'/tmp/env_vic4_vcmd_{args.slot}_vis.yaml'
    with open(src_env) as f:
        env_cfg = f.read()
    env_cfg = env_cfg.replace('num_envs: 512', f'num_envs: {args.num_envs}')
    env_cfg = env_cfg.replace('numEnvs: 512', f'numEnvs: {args.num_envs}')
    with open(cfg_env, 'w') as f:
        f.write(env_cfg)

    cfg_train = 'exp_config/forward_walking/260427_VIC4_VCMD/im_walk_vic.yaml'
    task_name = 'HumanoidImVICCmdRetime' if args.slot == 'S4' else 'HumanoidImVICCmdMultiClip'

    sys.argv = [
        'run.py',
        '--task', task_name,
        '--cfg_env', cfg_env,
        '--cfg_train', cfg_train,
        '--num_envs', str(args.num_envs),
        '--test', '--epoch', str(args.epoch),
        '--experiment', f'VIC4_VCMD_{args.slot}',
    ]
    if args.no_virtual_display:
        sys.argv.append('--no_virtual_display')

    install_patches()

    print(f"[vis] slot={args.slot} epoch={args.epoch} num_envs={args.num_envs}")
    print(f"[vis] task={task_name}")
    print(f"[vis] arrow color: blue (v_cmd={V_LO} m/s) -> red (v_cmd={V_HI} m/s)")
    from phc import run as phc_run
    phc_run.main()


if __name__ == '__main__':
    main()

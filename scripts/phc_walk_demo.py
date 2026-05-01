"""IsaacGym interactive demo: PHC pretrained phc_3 + continuous v_cmd walking.

No training. Multi-clip + per-step retime via _motion_start_times_offset
accumulation. The pretrained phc_3 imitates the active retimed walking clip;
v_cmd controls the playback rate.

v2 fixes (vs scripts/phc_walk_demo_v1.py):
- Clip switching now polls every CLIP_SWITCH_INTERVAL render steps (~2 s)
  instead of waiting for cycle_count change, which was stale across episode
  resets and tied to the 60 s clip length.
- episode_length raised from 300 (5 s) to 99999 so the reference motion
  doesn't get re-sampled to a random start every 5 s, eliminating the
  visible jump at episode boundaries.
- On clip switch, force env reset (reset_buf=1) AND zero out
  _motion_start_times_offset before the next render. Without this, the
  stale _global_offset / accumulated retime offset get applied to the new
  motion_id, which crashes motion_lib slerp with a CUDA device-side assert
  the moment v_cmd is pushed high enough to trigger a switch.

Run:
  conda activate phc
  python scripts/phc_walk_demo.py
  # optional 30s recording:
  python scripts/phc_walk_demo.py --record_seconds 30

Controls (in the IsaacGym window):
  Up   / Down  : v_cmd ±0.02 (clamped to [0.76, 1.23])
  1..9         : v_cmd snap (linear interp between V_CMD_MIN and V_CMD_MAX)
  R            : episode reset
  P            : pause / resume
  Q            : quit

The Tkinter panel (auto-spawned) supports numeric input, slider, and
Apply/Reset/Pause buttons. v_cmd is smoothed by max_accel=0.5 m/s².
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys

EXP_NAME = "phc_3"
LEARNING = "im_pnn_big"
ENV_CFG_NAME = "env_im_pnn"
MOTION_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9.pkl"
DIRMETA_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json"

V_NATURAL = (0.897, 0.975, 1.068)
V_CMD_INIT = 1.0
V_CMD_MIN = 0.76
V_CMD_MAX = 1.23
V_CMD_KEY_STEP = 0.02
V_CMD_MAX_ACCEL = 0.5
HYSTERESIS = 0.02
MIDPOINT_AB = (V_NATURAL[0] + V_NATURAL[1]) / 2.0
MIDPOINT_BC = (V_NATURAL[1] + V_NATURAL[2]) / 2.0

NUM_ENVS = 3            # MUST be >= len(V_NATURAL): motion_lib subsets to num_envs
ENV_SPACING = 50        # extras hidden far away; env 0 is the camera-followed one

PANEL_STATE_PATH = "/tmp/phc_walk_state.json"
PANEL_INPUT_PATH = "/tmp/phc_walk_input.json"
ARROW_LEN_AT_VHI = 1.5
LOG_STEP_INTERVAL = 60
CLIP_SWITCH_INTERVAL = 60   # render steps between clip-switch polls (~2 s @ 30 fps)
EPISODE_LENGTH = 99999      # effectively never auto-reset; phc_3 walks continuously

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, "phc")
for _p in (_PHC_PKG, _PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np


# Module-level mutable state shared between monkey-patches and main thread.
_STATE = {
    "v_cmd_target": V_CMD_INIT,
    "v_cmd_ramped": V_CMD_INIT,
    "active_clip": 1,                 # index into V_NATURAL
    "v_natural": V_NATURAL[1],
    "retime_ratio": 1.0,
    "v_actual": 0.0,
    "step": 0,
    "paused": False,
    "subs_ready": False,
    "pending_clip": None,             # set in render, applied in pre_physics_step
}

_REC = {
    "enabled": False,
    "target_frames": 0,
    "frame_count": 0,
    "record_dir": "",
    "fps": 30,
    "out_dir": "videos",
    "out_name": "phc_walk_demo",
    "done": False,
}


def _write_panel_state():
    """Dump _STATE to /tmp/phc_walk_state.json for the Tk panel to read."""
    try:
        with open(PANEL_STATE_PATH, "w") as f:
            json.dump({
                "v_cmd_target": _STATE["v_cmd_target"],
                "v_cmd_ramped": _STATE["v_cmd_ramped"],
                "active_clip": _STATE["active_clip"],
                "v_natural": _STATE["v_natural"],
                "retime_ratio": _STATE["retime_ratio"],
                "v_actual": _STATE["v_actual"],
                "step": _STATE["step"],
                "paused": _STATE["paused"],
                "v_min": V_CMD_MIN,
                "v_max": V_CMD_MAX,
            }, f)
    except (OSError, IOError, TypeError):
        pass


def _read_panel_input():
    """If panel wrote a new v_cmd_target, consume it.
    Panel writes {'v_cmd_target': float, 'reset': bool, 'pause_toggle': bool}
    and we delete the file after reading so each input is single-shot."""
    if not os.path.exists(PANEL_INPUT_PATH):
        return None
    try:
        with open(PANEL_INPUT_PATH) as f:
            inp = json.load(f)
        os.remove(PANEL_INPUT_PATH)
        return inp
    except (OSError, json.JSONDecodeError):
        return None


def _spawn_panel(host_display: str | None) -> subprocess.Popen | None:
    """Launch phc_walk_demo_panel.py in a separate process so it stays bound to
    the user's real DISPLAY even if PHC swaps to a virtual display. Returns the
    Popen handle or None if spawn failed."""
    panel_script = os.path.join(_THIS_DIR, "phc_walk_demo_panel.py")
    env = os.environ.copy()
    if host_display:
        env["DISPLAY"] = host_display
    try:
        return subprocess.Popen([sys.executable, panel_script],
                                env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True)
    except Exception as e:
        print(f"[demo] panel spawn failed: {e}")
        return None


_KEY_BINDINGS = [
    ("v_up", "KEY_UP"),
    ("v_down", "KEY_DOWN"),
    ("v_set_1", "KEY_1"),
    ("v_set_2", "KEY_2"),
    ("v_set_3", "KEY_3"),
    ("v_set_4", "KEY_4"),
    ("v_set_5", "KEY_5"),
    ("v_set_6", "KEY_6"),
    ("v_set_7", "KEY_7"),
    ("v_set_8", "KEY_8"),
    ("v_set_9", "KEY_9"),
    ("demo_reset", "KEY_R"),
    ("demo_pause", "KEY_P"),
    ("demo_quit", "KEY_Q"),
]


def _maybe_subscribe_keys(env):
    if _STATE["subs_ready"] or env.viewer is None:
        return
    for action_name, key_attr in _KEY_BINDINGS:
        key_code = getattr(gymapi, key_attr)
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key_code, action_name)
    _STATE["subs_ready"] = True
    print("[demo] keyboard subscriptions registered")


def _handle_events(env):
    """Poll keyboard + panel input; update _STATE['v_cmd_target'] and trigger reset/pause/quit."""
    if env.viewer is None:
        return
    # Keyboard events
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue
        a = evt.action
        if a == "v_up":
            _STATE["v_cmd_target"] = float(np.clip(
                _STATE["v_cmd_target"] + V_CMD_KEY_STEP, V_CMD_MIN, V_CMD_MAX))
        elif a == "v_down":
            _STATE["v_cmd_target"] = float(np.clip(
                _STATE["v_cmd_target"] - V_CMD_KEY_STEP, V_CMD_MIN, V_CMD_MAX))
        elif a.startswith("v_set_"):
            k = int(a.split("_")[-1])  # 1..9
            _STATE["v_cmd_target"] = V_CMD_MIN + (k - 1) / 8.0 * (V_CMD_MAX - V_CMD_MIN)
        elif a == "demo_reset":
            env.reset_buf[0] = 1
            print(f"[demo] reset (R) — v_cmd target stays at {_STATE['v_cmd_target']:.3f}")
        elif a == "demo_pause":
            _STATE["paused"] = not _STATE["paused"]
            env.paused = _STATE["paused"]
            print(f"[demo] {'PAUSED' if _STATE['paused'] else 'RESUMED'}")
        elif a == "demo_quit":
            print("[demo] quit (Q)")
            sys.exit(0)
    # Panel input (single-shot)
    inp = _read_panel_input()
    if inp is not None:
        if "v_cmd_target" in inp:
            v = float(inp["v_cmd_target"])
            _STATE["v_cmd_target"] = float(np.clip(v, V_CMD_MIN, V_CMD_MAX))
        if inp.get("reset"):
            env.reset_buf[0] = 1
        if inp.get("pause_toggle"):
            _STATE["paused"] = not _STATE["paused"]
            env.paused = _STATE["paused"]
        if inp.get("quit"):
            sys.exit(0)


def _draw_forward_arrow(env):
    """Draw a v_cmd-scaled forward arrow above env 0's head.
    Color: blue (slow) → red (fast) on (v_cmd - V_CMD_MIN) / (V_CMD_MAX - V_CMD_MIN)."""
    if env.viewer is None:
        return
    env.gym.clear_lines(env.viewer)
    root = env._humanoid_root_states[0].detach().cpu().numpy()
    pos = root[:3]
    quat = root[3:7]
    x, y, z, w = quat
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    norm = float(np.sqrt(fwd_x ** 2 + fwd_y ** 2)) + 1e-8
    fwd_x /= norm
    fwd_y /= norm
    head_z = pos[2] + 0.6
    v = _STATE["v_cmd_ramped"]
    scale = (v / V_CMD_MAX) * ARROW_LEN_AT_VHI
    t_color = float(np.clip(
        (v - V_CMD_MIN) / max(V_CMD_MAX - V_CMD_MIN, 1e-6), 0.0, 1.0))
    color = np.array([t_color, 0.0, 1.0 - t_color], dtype=np.float32)
    sx, sy = pos[0], pos[1]
    ex, ey = sx + fwd_x * scale, sy + fwd_y * scale
    main = np.array([sx, sy, head_z, ex, ey, head_z], dtype=np.float32)
    env.gym.add_lines(env.viewer, env.envs[0], 1, main, color)
    head_size = 0.12
    lx = ex - fwd_x * head_size + fwd_y * head_size * 0.5
    ly = ey - fwd_y * head_size - fwd_x * head_size * 0.5
    rx = ex - fwd_x * head_size - fwd_y * head_size * 0.5
    ry = ey - fwd_y * head_size + fwd_x * head_size * 0.5
    env.gym.add_lines(env.viewer, env.envs[0], 1,
                      np.array([ex, ey, head_z, lx, ly, head_z], dtype=np.float32), color)
    env.gym.add_lines(env.viewer, env.envs[0], 1,
                      np.array([ex, ey, head_z, rx, ry, head_z], dtype=np.float32), color)


def _measure_v_actual(env):
    """Read env 0's planar root velocity magnitude → _STATE['v_actual']."""
    import torch
    root_vel = env._humanoid_root_states[0, 7:10]
    _STATE["v_actual"] = float(torch.linalg.norm(root_vel[:2]).item())


def _lock_camera(env):
    """Quartering follow-shot of env 0's pelvis."""
    if env.viewer is None:
        return
    pos = env._humanoid_root_states[0, :3].detach().cpu().numpy()
    cam_pos = gymapi.Vec3(float(pos[0] - 3.0), float(pos[1] - 3.0), float(pos[2] + 1.5))
    cam_target = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)


def _install_render_hook():
    from phc.env.tasks.humanoid import Humanoid
    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        _maybe_subscribe_keys(self)
        _handle_events(self)
        _maybe_switch_clip(self)
        _draw_forward_arrow(self)
        _measure_v_actual(self)
        _lock_camera(self)
        _STATE["step"] += 1
        if _STATE["step"] % LOG_STEP_INTERVAL == 0:
            print(f"[demo] step={_STATE['step']:6d}  "
                  f"v_target={_STATE['v_cmd_target']:.3f}  "
                  f"v_ramped={_STATE['v_cmd_ramped']:.3f}  "
                  f"v_actual={_STATE['v_actual']:.3f}  "
                  f"clip={_STATE['active_clip']}  "
                  f"ratio={_STATE['retime_ratio']:.3f}")
        if _STATE["step"] % 3 == 0:
            _write_panel_state()
        if _STATE["paused"]:
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            return None
        ret = orig_render(self, sync_frame_time)
        # Recording (after rendering so the frame includes everything)
        if _REC["enabled"] and not _REC["done"] and self.viewer is not None:
            img_path = os.path.join(_REC["record_dir"],
                                    f"frame_{_REC['frame_count']:06d}.png")
            try:
                self.gym.write_viewer_image_to_file(self.viewer, img_path)
            except Exception as e:
                print(f"[rec] capture failed at frame {_REC['frame_count']}: {e}")
            _REC["frame_count"] += 1
            if _REC["frame_count"] % 30 == 0:
                print(f"[rec] frame {_REC['frame_count']}/{_REC['target_frames']}", flush=True)
            if _REC["frame_count"] >= _REC["target_frames"]:
                print(f"[rec] target reached, exiting", flush=True)
                _REC["done"] = True
                sys.exit(0)
        return ret

    Humanoid.render = patched_render


def _ramp_v_cmd(env_dt):
    """Smooth v_cmd_ramped toward v_cmd_target with max_accel rate limit.
    Called once per pre_physics_step (so dt = sim physics dt, not render dt)."""
    delta_max = V_CMD_MAX_ACCEL * env_dt
    diff = _STATE["v_cmd_target"] - _STATE["v_cmd_ramped"]
    step = max(-delta_max, min(delta_max, diff))
    _STATE["v_cmd_ramped"] += step


def _select_clip(v_cmd, current):
    """Hysteresis: only switch when v_cmd crosses midpoint by HYSTERESIS margin.
    Returns the new active_clip index (0/1/2)."""
    if current == 0:
        if v_cmd > MIDPOINT_AB + HYSTERESIS:
            return 1
        return 0
    if current == 1:
        if v_cmd < MIDPOINT_AB - HYSTERESIS:
            return 0
        if v_cmd > MIDPOINT_BC + HYSTERESIS:
            return 2
        return 1
    if current == 2:
        if v_cmd < MIDPOINT_BC - HYSTERESIS:
            return 1
        return 2
    return current


def _maybe_switch_clip(env):
    """Decide whether to switch clips and queue it.

    The actual switch must happen in pre_physics_step, NOT here in render:
    if we change motion_id mid-render, the same render's _update_marker
    queries motion_lib with the new motion_id but the still-stale
    _global_offset, which crashes slerp with a CUDA device-side assert.

    Polls every CLIP_SWITCH_INTERVAL render steps (~2 s)."""
    if not hasattr(env, "_sampled_motion_ids"):
        return
    if _STATE["step"] == 0 or _STATE["step"] % CLIP_SWITCH_INTERVAL != 0:
        return
    desired = _select_clip(_STATE["v_cmd_ramped"], _STATE["active_clip"])
    if desired != _STATE["active_clip"]:
        _STATE["pending_clip"] = desired


def _apply_pending_clip_switch(env):
    """Apply queued clip switch atomically: motion_id, motion times, AND
    _global_offset all updated together so the next motion_lib query in this
    same physics step is consistent. Called from the patched pre_physics_step
    BEFORE the base class runs."""
    desired = _STATE["pending_clip"]
    if desired is None:
        return
    if not hasattr(env, "_sampled_motion_ids") or not hasattr(env, "_global_offset"):
        _STATE["pending_clip"] = None
        return
    import torch
    env._sampled_motion_ids[:] = desired
    env._motion_start_times[:] = 0.0
    env._motion_start_times_offset[:] = 0.0
    if hasattr(env, "progress_buf"):
        env.progress_buf[:] = 0
    # Re-sync _global_offset to new motion's root pos at t=0 so the existing
    # humanoid_root_states stays aligned with the new reference.
    times = torch.zeros_like(env._motion_start_times)
    root_res = env._motion_lib.get_root_pos_smpl(env._sampled_motion_ids, times)
    new_root = root_res["root_pos"] if isinstance(root_res, dict) else root_res
    env._global_offset[:, :2] = env._humanoid_root_states[:, :2] - new_root[:, :2]
    if env._global_offset.shape[1] >= 3:
        env._global_offset[:, 2] = 0.0
    if hasattr(env, "ref_motion_cache"):
        env.ref_motion_cache.clear()
    _STATE["active_clip"] = desired
    _STATE["v_natural"] = V_NATURAL[desired]
    _STATE["pending_clip"] = None
    print(f"[demo] clip switch → {desired} (v_natural={V_NATURAL[desired]:.3f}, "
          f"global_offset re-synced)")


def _install_pre_physics_patch():
    """Patch Humanoid.pre_physics_step to:
      1. Ramp v_cmd_ramped toward v_cmd_target.
      2. Accumulate dt × (v_cmd_ramped/v_natural - 1.0) into _motion_start_times_offset
         so reference motion advances at the v_cmd-scaled rate.
    The base class's pre_physics_step is preserved by calling orig at the end."""
    from phc.env.tasks.humanoid import Humanoid
    orig_pre = Humanoid.pre_physics_step

    def patched_pre(self, actions):
        # Apply queued clip switch BEFORE the base step / motion-lib reads.
        # Doing it here (not in render) guarantees motion_id and _global_offset
        # are updated together, so this step's render won't query motion_lib
        # with a stale offset and crash.
        _apply_pending_clip_switch(self)
        if not _STATE["paused"]:
            _ramp_v_cmd(self.dt)
            v_natural = _STATE["v_natural"]
            ratio = _STATE["v_cmd_ramped"] / v_natural
            _STATE["retime_ratio"] = ratio
            # Broadcast to all envs (we use NUM_ENVS=2 with the same v_cmd).
            self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return orig_pre(self, actions)

    Humanoid.pre_physics_step = patched_pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record_seconds", type=int, default=0,
                    help="If > 0, capture frames and save mp4 then exit.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", default="videos")
    ap.add_argument("--out_name", default="phc_walk_demo")
    args = ap.parse_args()

    if args.record_seconds > 0:
        _REC["enabled"] = True
        _REC["target_frames"] = args.record_seconds * args.fps
        _REC["fps"] = args.fps
        _REC["out_dir"] = args.out_dir
        _REC["out_name"] = args.out_name
        _REC["record_dir"] = f"/tmp/record_frames_{args.out_name}"
        if os.path.exists(_REC["record_dir"]):
            shutil.rmtree(_REC["record_dir"])
        os.makedirs(_REC["record_dir"])
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[rec] enabled — target {_REC['target_frames']} frames "
              f"({args.record_seconds}s @ {args.fps}fps) → {_REC['record_dir']}")

    sys.argv = [
        "run_hydra.py",
        f"learning={LEARNING}",
        f"env={ENV_CFG_NAME}",
        "robot=smpl_humanoid",
        f"exp_name={EXP_NAME}",
        "epoch=-1",
        "test=True",
        "headless=False",
        "no_virtual_display=True",
        f"env.num_envs={NUM_ENVS}",
        f"env.env_spacing={ENV_SPACING}",
        f"env.motion_file={MOTION_FILE}",
        "env.cycle_motion=True",
        f"env.episode_length={EPISODE_LENGTH}",
    ]

    _install_render_hook()
    _install_pre_physics_patch()
    host_display = os.environ.get("DISPLAY", "")
    _write_panel_state()
    panel_proc = _spawn_panel(host_display)
    if panel_proc is not None:
        print(f"[demo] panel spawned (pid={panel_proc.pid}) on DISPLAY={host_display!r}")

    print("=" * 70)
    print(f"[demo] EXP={EXP_NAME} LEARNING={LEARNING}")
    print(f"[demo] motion={MOTION_FILE}")
    print(f"[demo] v_cmd range: [{V_CMD_MIN:.2f}, {V_CMD_MAX:.2f}]  init={V_CMD_INIT}")
    print(f"[demo] keys:  ↑/↓ ±{V_CMD_KEY_STEP}  |  1..9 snap  |  R reset  |  P pause  |  Q quit")
    print("=" * 70)

    import runpy
    try:
        runpy.run_path("phc/run_hydra.py", run_name="__main__")
    except SystemExit:
        pass

    # Combine PNGs to mp4 if recording was on
    if _REC["enabled"] and _REC["frame_count"] > 0:
        out_path = os.path.join(_REC["out_dir"], f"{_REC['out_name']}.mp4")
        ffmpeg_bin = shutil.which("ffmpeg") or "/home/exolab/miniconda3/bin/ffmpeg"
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(_REC["fps"]),
            "-i", os.path.join(_REC["record_dir"], "frame_%06d.png"),
            "-c:v", "libopenh264",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        print(f"[rec] combining frames: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"[rec] saved {out_path} ({size_mb:.1f} MB)")
            shutil.rmtree(_REC["record_dir"])
        else:
            print(f"[rec] WARN: ffmpeg did not produce {out_path}; PNGs kept at {_REC['record_dir']}")


if __name__ == "__main__":
    main()

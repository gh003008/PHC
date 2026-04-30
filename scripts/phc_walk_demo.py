"""IsaacGym interactive demo: PHC pretrained phc_3 + continuous v_cmd walking.

No training. Multi-clip + per-step retime via _motion_start_times_offset
accumulation. The pretrained phc_3 imitates the active retimed walking clip;
v_cmd controls the playback rate.

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

NUM_ENVS = 2
ENV_SPACING = 50

PANEL_STATE_PATH = "/tmp/phc_walk_state.json"
PANEL_INPUT_PATH = "/tmp/phc_walk_input.json"
ARROW_LEN_AT_VHI = 1.5
LOG_STEP_INTERVAL = 60

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
    "fall_count": 0,
    "term_height": 0.15,
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
    except Exception:
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
    except Exception:
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
            _STATE["fall_count"] = 0
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
            _STATE["fall_count"] = 0
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


def _install_render_hook():
    from phc.env.tasks.humanoid import Humanoid
    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        _maybe_subscribe_keys(self)
        _handle_events(self)
        _maybe_switch_clip(self)
        _draw_forward_arrow(self)
        _STATE["step"] += 1
        if _STATE["step"] % 3 == 0:
            _write_panel_state()
        if _STATE["paused"]:
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            return None
        return orig_render(self, sync_frame_time)

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
    """If v_cmd_ramped suggests a different clip than active, switch the env's
    motion id at cycle boundary. We detect cycle boundary via cycle_count change."""
    if not hasattr(env, "_sampled_motion_ids"):
        return
    # Compute current motion_time for env 0 to detect cycle boundary
    if not hasattr(env, "_motion_lib"):
        return
    motion_id_t = env._sampled_motion_ids[0].item()
    motion_len = env._motion_lib.get_motion_length(
        env._sampled_motion_ids[:1]).item()
    if motion_len <= 0:
        return
    motion_time = (env.progress_buf[0].item() * env.dt
                   + env._motion_start_times[0].item()
                   + env._motion_start_times_offset[0].item())
    cycle_count = int(motion_time // motion_len)
    last_cycle = _STATE.get("last_cycle_count", -1)
    _STATE["last_cycle_count"] = cycle_count
    if cycle_count <= last_cycle:
        return  # not a new cycle yet (or first call)
    # New cycle started — re-evaluate active clip
    desired = _select_clip(_STATE["v_cmd_ramped"], _STATE["active_clip"])
    if desired != _STATE["active_clip"]:
        _STATE["active_clip"] = desired
        _STATE["v_natural"] = V_NATURAL[desired]
        # Set every env to the new motion id; motion_lib will sample from it.
        env._sampled_motion_ids[:] = desired
        print(f"[demo] cycle boundary — clip switch → {desired} "
              f"(v_natural={V_NATURAL[desired]:.3f})")


def _install_pre_physics_patch():
    """Patch Humanoid.pre_physics_step to:
      1. Ramp v_cmd_ramped toward v_cmd_target.
      2. Accumulate dt × (v_cmd_ramped/v_natural - 1.0) into _motion_start_times_offset
         so reference motion advances at the v_cmd-scaled rate.
    The base class's pre_physics_step is preserved by calling orig at the end."""
    from phc.env.tasks.humanoid import Humanoid
    orig_pre = Humanoid.pre_physics_step

    def patched_pre(self, actions):
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
    args = ap.parse_args()

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
        "env.episode_length=300",
    ]

    print("=" * 70)
    print(f"[demo] EXP={EXP_NAME} LEARNING={LEARNING}")
    print(f"[demo] motion={MOTION_FILE}")
    print(f"[demo] v_cmd range: [{V_CMD_MIN:.2f}, {V_CMD_MAX:.2f}]  init={V_CMD_INIT}")
    print(f"[demo] keys:  ↑/↓ ±{V_CMD_KEY_STEP}  |  1..9 snap  |  R reset  |  P pause  |  Q quit")
    print("=" * 70)

    _install_render_hook()
    _install_pre_physics_patch()
    _write_panel_state()  # initial state for panel to display at boot

    import runpy
    try:
        runpy.run_path("phc/run_hydra.py", run_name="__main__")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()

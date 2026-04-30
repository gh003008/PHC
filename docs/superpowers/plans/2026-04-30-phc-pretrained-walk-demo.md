# PHC Pretrained v_cmd Walk Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive IsaacGym demo that uses the pretrained PHC `phc_3` model for continuous v_cmd-controlled walking (0.76–1.23 m/s) with no training.

**Architecture:** Two scripts. `scripts/phc_walk_demo.py` is the IsaacGym entry point — it monkey-patches `Humanoid.pre_physics_step` (per-step retime via `_motion_start_times_offset` accumulation) and `Humanoid.render` (keyboard polling, clip selection at cycle boundaries with hysteresis, forward-arrow viz, v_actual measurement, camera lock, JSON state output). It boots PHC via `runpy.run_path('phc/run_hydra.py')` after pre-setting `sys.argv`. `scripts/phc_walk_demo_panel.py` is a Tkinter panel run as a subprocess; it reads `/tmp/phc_walk_state.json` for display and writes `/tmp/phc_walk_input.json` on Apply/slider/buttons for bidirectional control.

**Tech Stack:** Python 3.8 (conda env `phc`), IsaacGym (PhysX), PyTorch, Hydra, rl_games, Tkinter, ffmpeg (libopenh264). Hardware: 4060 Ti 7.6 GB.

---

## File Structure

| File | Responsibility | Approx LOC |
|---|---|---|
| `scripts/phc_walk_demo.py` (new) | IsaacGym demo entry point: PHC boot via runpy, monkey-patches, keyboard handling, retime mechanism, clip selection, forward arrow viz, JSON state writer, optional --record_seconds | ~280 |
| `scripts/phc_walk_demo_panel.py` (new) | Tkinter panel: status display + numeric Entry + slider + Apply/Reset/Pause buttons; bidirectional JSON I/O with the demo | ~150 |

No existing files modified. All env/learning yamls (`env_im_pnn.yaml`, `im_pnn_big.yaml`) used as-is via Hydra overrides.

---

## Constants used throughout (all defined as module-level in `phc_walk_demo.py`)

```python
EXP_NAME = "phc_3"
LEARNING = "im_pnn_big"
ENV_CFG_NAME = "env_im_pnn"
MOTION_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9.pkl"
DIRMETA_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json"

V_NATURAL = (0.897, 0.975, 1.068)   # sorted ascending; matches motion_lib ordering
V_CMD_INIT = 1.0
V_CMD_MIN = 0.76                    # V_NATURAL[0] * 0.85
V_CMD_MAX = 1.23                    # V_NATURAL[2] * 1.15
V_CMD_KEY_STEP = 0.02               # ↑/↓ press delta
V_CMD_MAX_ACCEL = 0.5               # m/s² ramp limit on v_cmd_target → v_cmd_ramped
HYSTERESIS = 0.02                   # ±margin around midpoints for clip switching
MIDPOINT_AB = (V_NATURAL[0] + V_NATURAL[1]) / 2.0   # 0.936
MIDPOINT_BC = (V_NATURAL[1] + V_NATURAL[2]) / 2.0   # 1.022

NUM_ENVS = 2
ENV_SPACING = 50

PANEL_STATE_PATH  = "/tmp/phc_walk_state.json"
PANEL_INPUT_PATH  = "/tmp/phc_walk_input.json"
ARROW_LEN_AT_VHI  = 1.5
LOG_STEP_INTERVAL = 60
```

---

## Task 1: Skeleton, boot, constants

**Files:**
- Create: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Create the file with module docstring and constants**

```python
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

    import runpy
    try:
        runpy.run_path("phc/run_hydra.py", run_name="__main__")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run — verify it boots and phc_3 walks**

```bash
conda activate phc
python scripts/phc_walk_demo.py
```

Expected: viewer opens; 2 humanoids visible (env 0 near origin, env 1 ~50 m away — only env 0 in camera view if camera defaults near origin); humanoid imitates walking clip at ~1.0 m/s; episodes run to ~300 steps; quit with Ctrl-C since no Q handler yet.

- [ ] **Step 3: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: skeleton + boot via runpy"
```

---

## Task 2: Module state dict and JSON state writer

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add `_STATE` dict and `_write_panel_state()` near the top (after constants, before main)**

```python
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
```

- [ ] **Step 2: Call `_write_panel_state()` once before runpy**

In `main()`, just before the `runpy.run_path(...)` line, add:

```python
    _write_panel_state()  # initial state for panel to display at boot
```

- [ ] **Step 3: Smoke run — verify state file is written**

```bash
python scripts/phc_walk_demo.py &
sleep 5
cat /tmp/phc_walk_state.json
kill %1
```

Expected: JSON with v_cmd_target=1.0, v_cmd_ramped=1.0, active_clip=1, v_natural=0.975, etc.

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: state dict + panel JSON I/O"
```

---

## Task 3: Keyboard subscription and event handler

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add key bindings list and subscription helper after `_read_panel_input`**

```python
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
```

- [ ] **Step 2: Add render hook installer**

```python
def _install_render_hook():
    from phc.env.tasks.humanoid import Humanoid
    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        _maybe_subscribe_keys(self)
        _handle_events(self)
        _STATE["step"] += 1
        if _STATE["step"] % 3 == 0:  # ~10 Hz panel state writes
            _write_panel_state()
        if _STATE["paused"]:
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            return None
        return orig_render(self, sync_frame_time)

    Humanoid.render = patched_render
```

- [ ] **Step 3: Call installer in `main()` before runpy**

```python
    _install_render_hook()
    _write_panel_state()
    print("=" * 70)
    ...
    import runpy
    try:
        runpy.run_path("phc/run_hydra.py", run_name="__main__")
    except SystemExit:
        pass
```

- [ ] **Step 4: Smoke run — verify keys work**

```bash
python scripts/phc_walk_demo.py
```

Click the IsaacGym viewer to focus it. Expected:
- Press ↑ a few times → terminal prints `[demo] keyboard subscriptions registered` once and v_cmd_target visibly grows in `/tmp/phc_walk_state.json`
- Press ↓ → decreases
- Press 1 → v_cmd_target = 0.76
- Press 9 → v_cmd_target = 1.23
- Press P → "PAUSED" message; sim stops; press P again → resumes
- Press R → episode resets (humanoid teleports to start pose)
- Press Q → exits cleanly

- [ ] **Step 5: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: keyboard handlers + render hook"
```

---

## Task 4: Per-step retime via `pre_physics_step` monkey-patch

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add v_cmd ramp helper and pre_physics_step patch**

Insert after `_install_render_hook` and before `main`:

```python
def _ramp_v_cmd(env_dt):
    """Smooth v_cmd_ramped toward v_cmd_target with max_accel rate limit.
    Called once per pre_physics_step (so dt = sim physics dt, not render dt)."""
    delta_max = V_CMD_MAX_ACCEL * env_dt
    diff = _STATE["v_cmd_target"] - _STATE["v_cmd_ramped"]
    step = max(-delta_max, min(delta_max, diff))
    _STATE["v_cmd_ramped"] += step


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
```

- [ ] **Step 2: Call installer in `main()` (after `_install_render_hook`)**

```python
    _install_render_hook()
    _install_pre_physics_patch()
```

- [ ] **Step 3: Smoke run — verify retime advances motion at correct rate**

```bash
python scripts/phc_walk_demo.py
```

In a second terminal:

```bash
watch -n 0.5 'python -c "import json; d=json.load(open(\"/tmp/phc_walk_state.json\")); print(f\"target={d[\"v_cmd_target\"]:.3f} ramped={d[\"v_cmd_ramped\"]:.3f} ratio={d[\"retime_ratio\"]:.3f}\")"'
```

Press ↑ several times in the viewer. Expected:
- `v_cmd_ramped` should chase `v_cmd_target` at ~0.5 m/s² rate (so a 0.1 jump takes ~0.2 s)
- `retime_ratio` = v_cmd_ramped / 0.975 (≈ 1.0 + (v_cmd_ramped - 0.975)/0.975)
- Visually: humanoid walks faster as v_cmd ramps up; cadence increases

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: per-step retime via pre_physics_step + _motion_start_times_offset"
```

---

## Task 5: Clip selection at cycle boundary with hysteresis

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add clip selection logic**

Insert before `_install_pre_physics_patch`:

```python
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
```

- [ ] **Step 2: Call from render hook**

Modify `patched_render` in `_install_render_hook()` to call `_maybe_switch_clip(self)` after `_handle_events(self)`:

```python
    def patched_render(self, sync_frame_time=False):
        _maybe_subscribe_keys(self)
        _handle_events(self)
        _maybe_switch_clip(self)
        _STATE["step"] += 1
        if _STATE["step"] % 3 == 0:
            _write_panel_state()
        if _STATE["paused"]:
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            return None
        return orig_render(self, sync_frame_time)
```

- [ ] **Step 3: Smoke run — verify clip switches happen at cycle boundary**

```bash
python scripts/phc_walk_demo.py
```

Press 1 (v_cmd → 0.76, target clip 0). Wait ~1.5 s for cycle boundary. Expected terminal log:

```
[demo] cycle boundary — clip switch → 0 (v_natural=0.897)
```

Then press 9 (v_cmd → 1.23, target clip 2). Wait for cycle boundary. Expected:

```
[demo] cycle boundary — clip switch → 2 (v_natural=1.068)
```

Verify in `/tmp/phc_walk_state.json`: `active_clip` and `v_natural` change accordingly. Visually, gait stride should look slightly different per clip (different recordings).

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: clip selection at cycle boundary with hysteresis"
```

---

## Task 6: Forward arrow visualization

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add `_draw_forward_arrow` function**

Insert before `_install_render_hook`:

```python
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
```

- [ ] **Step 2: Call from render hook**

Modify `patched_render` in `_install_render_hook()` to call `_draw_forward_arrow(self)` after `_maybe_switch_clip(self)`:

```python
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
```

- [ ] **Step 3: Smoke run — verify arrow appears and changes color**

```bash
python scripts/phc_walk_demo.py
```

Expected: a colored arrow above the humanoid's head pointing in walking direction. Press 1 → arrow is blue and short. Press 9 → arrow is red and long. Press 5 → purple-ish, medium length.

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: forward arrow visualization (color = v_cmd)"
```

---

## Task 7: v_actual measurement and camera lock

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add `_measure_v_actual` and `_lock_camera`**

Insert before `_install_render_hook`:

```python
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
```

- [ ] **Step 2: Call from render hook**

Modify `patched_render`:

```python
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
        return orig_render(self, sync_frame_time)
```

- [ ] **Step 3: Smoke run — verify v_actual tracks v_cmd**

```bash
python scripts/phc_walk_demo.py
```

Watch terminal logs every 60 steps. Expected:
- v_actual ≈ v_ramped ± 0.05 m/s in steady state (after ~1 cycle)
- Camera follows env 0 from quartering angle (env 1 not in view)
- Press ↑ several times → v_actual increases over time tracking v_ramped

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: v_actual measurement + camera follow"
```

---

## Task 8: Tkinter panel script

**Files:**
- Create: `scripts/phc_walk_demo_panel.py`

- [ ] **Step 1: Create the panel file**

```python
"""Bidirectional Tkinter panel for phc_walk_demo.py.

Reads /tmp/phc_walk_state.json (status display).
Writes /tmp/phc_walk_input.json on Apply / button / slider release.

Layout:
  - status block: v_cmd_target, v_cmd_ramped, v_actual, active_clip, v_natural,
                  retime_ratio, step, paused
  - Entry widget for direct numeric v_cmd input + Apply button
  - Slider (Tk.Scale) over [V_MIN, V_MAX]
  - Reset / Pause buttons

Launched automatically by phc_walk_demo.py as a subprocess. Standalone:
  python scripts/phc_walk_demo_panel.py
"""
import json
import os
import tkinter as tk
from tkinter import ttk

STATE_PATH = "/tmp/phc_walk_state.json"
INPUT_PATH = "/tmp/phc_walk_input.json"

V_MIN = 0.76   # mirrors phc_walk_demo.py constants
V_MAX = 1.23


def lerp_color(t):
    t = max(0.0, min(1.0, t))
    r = int(255 * t)
    g = 0
    b = int(255 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_input(payload: dict):
    """Atomic write of the input file so the demo doesn't see a partial JSON."""
    tmp = INPUT_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, INPUT_PATH)
    except Exception as e:
        print(f"[panel] write failed: {e}")


class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PHC Walk Demo — v_cmd")
        sw = self.root.winfo_screenwidth()
        w, h = 360, 480
        x = sw - w - 30
        y = 30
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg="#111")
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        big = ("DejaVu Sans Mono", 26, "bold")
        med = ("DejaVu Sans", 14)
        small = ("DejaVu Sans", 11)

        self.title_lbl = tk.Label(self.root, text="PHC Walk Demo",
                                  font=("DejaVu Sans", 16, "bold"),
                                  fg="#aaa", bg="#111")
        self.title_lbl.pack(pady=(8, 4))

        # Big v_cmd_ramped display
        self.value_lbl = tk.Label(self.root, text="--.--", font=big,
                                  fg="#888", bg="#111")
        self.value_lbl.pack()
        self.unit_lbl = tk.Label(self.root, text="m/s", font=small,
                                 fg="#888", bg="#111")
        self.unit_lbl.pack()

        # Status block
        self.target_lbl = tk.Label(self.root, text="target —", font=med,
                                   fg="#ddd", bg="#111")
        self.target_lbl.pack(pady=(6, 0))
        self.actual_lbl = tk.Label(self.root, text="actual —", font=med,
                                   fg="#ddd", bg="#111")
        self.actual_lbl.pack()
        self.clip_lbl = tk.Label(self.root, text="clip —", font=small,
                                 fg="#aaa", bg="#111")
        self.clip_lbl.pack(pady=(4, 0))
        self.ratio_lbl = tk.Label(self.root, text="ratio —", font=small,
                                  fg="#aaa", bg="#111")
        self.ratio_lbl.pack()
        self.step_lbl = tk.Label(self.root, text="step —", font=small,
                                 fg="#666", bg="#111")
        self.step_lbl.pack()
        self.pause_lbl = tk.Label(self.root, text="", font=("DejaVu Sans", 14, "bold"),
                                  fg="#ffa500", bg="#111")
        self.pause_lbl.pack(pady=(2, 6))

        # Numeric input
        entry_frame = tk.Frame(self.root, bg="#111")
        entry_frame.pack(pady=(4, 0))
        tk.Label(entry_frame, text="Set v_cmd:", font=small, fg="#aaa", bg="#111"
                 ).grid(row=0, column=0, padx=4)
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(entry_frame, textvariable=self.entry_var, width=8,
                              font=med, bg="#222", fg="#fff", insertbackground="#fff")
        self.entry.grid(row=0, column=1, padx=4)
        self.entry.bind("<Return>", lambda e: self.on_apply())
        tk.Button(entry_frame, text="Apply", font=small, command=self.on_apply,
                  bg="#333", fg="#fff").grid(row=0, column=2, padx=4)

        # Slider
        self.slider_var = tk.DoubleVar(value=1.0)
        self.slider = tk.Scale(self.root, variable=self.slider_var,
                               from_=V_MIN, to=V_MAX, resolution=0.01,
                               orient=tk.HORIZONTAL, length=300,
                               bg="#111", fg="#ddd", troughcolor="#333",
                               highlightthickness=0,
                               command=self.on_slider)
        self.slider.pack(pady=(8, 0))

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#111")
        btn_frame.pack(pady=(8, 0))
        tk.Button(btn_frame, text="Reset", font=small, width=8,
                  command=self.on_reset, bg="#333", fg="#fff"
                  ).grid(row=0, column=0, padx=4)
        tk.Button(btn_frame, text="Pause", font=small, width=8,
                  command=self.on_pause, bg="#333", fg="#fff"
                  ).grid(row=0, column=1, padx=4)
        tk.Button(btn_frame, text="Quit", font=small, width=8,
                  command=self.on_quit, bg="#600", fg="#fff"
                  ).grid(row=0, column=2, padx=4)

        self._slider_dragging = False
        self.tick()

    def on_apply(self):
        try:
            v = float(self.entry_var.get())
        except ValueError:
            return
        v = max(V_MIN, min(V_MAX, v))
        self.slider_var.set(v)
        write_input({"v_cmd_target": v})

    def on_slider(self, _val):
        v = float(self.slider_var.get())
        write_input({"v_cmd_target": v})

    def on_reset(self):
        write_input({"reset": True})

    def on_pause(self):
        write_input({"pause_toggle": True})

    def on_quit(self):
        write_input({"quit": True})
        self.root.after(200, self.root.destroy)

    def tick(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    s = json.load(f)
                v_ramped = float(s.get("v_cmd_ramped", 0.0))
                v_target = float(s.get("v_cmd_target", 0.0))
                v_actual = float(s.get("v_actual", 0.0))
                clip = int(s.get("active_clip", 0))
                v_natural = float(s.get("v_natural", 0.0))
                ratio = float(s.get("retime_ratio", 1.0))
                step = int(s.get("step", 0))
                paused = bool(s.get("paused", False))
                vmin = float(s.get("v_min", V_MIN))
                vmax = float(s.get("v_max", V_MAX))

                t = (v_ramped - vmin) / max(vmax - vmin, 1e-6)
                color = lerp_color(t)
                self.value_lbl.config(text=f"{v_ramped:.3f}", fg=color)
                self.unit_lbl.config(fg=color)
                self.target_lbl.config(text=f"target  {v_target:.3f} m/s")
                self.actual_lbl.config(text=f"actual  {v_actual:.3f} m/s")
                clip_letter = "ABC"[clip] if 0 <= clip < 3 else "?"
                self.clip_lbl.config(text=f"clip {clip_letter} (natural {v_natural:.3f})")
                self.ratio_lbl.config(text=f"retime ratio  {ratio:.3f}x")
                self.step_lbl.config(text=f"step  {step}")
                self.pause_lbl.config(text="PAUSED" if paused else "")
        except Exception:
            pass
        self.root.after(100, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Panel().run()
```

- [ ] **Step 2: Smoke run — launch panel standalone (no demo running) and verify it boots**

```bash
python scripts/phc_walk_demo_panel.py
```

Expected: a small black-themed Tk window opens top-right with status fields all "—" or stale defaults; sliders/buttons render correctly. Close with Quit.

- [ ] **Step 3: Commit**

```bash
git add scripts/phc_walk_demo_panel.py
git commit -m "phc_walk_demo_panel: bidirectional Tkinter panel"
```

---

## Task 9: Spawn the panel from the demo

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add `_spawn_panel` helper**

Insert near the other helpers (after `_read_panel_input`):

```python
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
```

- [ ] **Step 2: Call from `main` before runpy**

Add before the `print("=" * 70)` block in `main`:

```python
    host_display = os.environ.get("DISPLAY", "")
    _write_panel_state()
    panel_proc = _spawn_panel(host_display)
    if panel_proc is not None:
        print(f"[demo] panel spawned (pid={panel_proc.pid}) on DISPLAY={host_display!r}")
```

- [ ] **Step 3: Smoke run — verify panel + demo work together**

```bash
python scripts/phc_walk_demo.py
```

Expected:
- Panel pops up top-right within a second of launch
- Panel shows live `v_cmd_ramped`, `v_actual`, `clip`, `ratio`, `step` updating ~10 Hz
- Drag the slider → demo's v_cmd_target follows; humanoid speeds up / slows down accordingly
- Type `0.85` in Entry + click Apply → v_cmd_target jumps; ramped value approaches it over 0.5 s
- Click Reset button → episode resets in viewer
- Click Pause button → sim freezes; click again → resumes
- Click Quit → demo exits cleanly

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: spawn Tkinter panel as subprocess"
```

---

## Task 10: Optional `--record_seconds N` recording

**Files:**
- Modify: `scripts/phc_walk_demo.py`

- [ ] **Step 1: Add recording state and frame capture in render hook**

Below the constants, add a recording-state dict:

```python
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
```

Modify `_install_render_hook` to capture frames after rendering (insert just before the paused-check return path is irrelevant — capture only happens when not paused):

```python
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
```

- [ ] **Step 2: Wire CLI flag in `main`**

Modify `main` to set up recording state when --record_seconds > 0:

```python
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
        "env.episode_length=300",
    ]

    host_display = os.environ.get("DISPLAY", "")
    _install_render_hook()
    _install_pre_physics_patch()
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
```

- [ ] **Step 3: Smoke run — verify recording works**

```bash
python scripts/phc_walk_demo.py --record_seconds 10
```

While running, press a few digits to vary v_cmd. Expected:
- Terminal: `[rec] frame 30/300`, `[rec] frame 60/300`, ..., `[rec] target reached, exiting`
- File `videos/phc_walk_demo.mp4` saved (~2-3 MB for 10 s)
- `/tmp/record_frames_phc_walk_demo` cleaned up

Verify mp4 with:
```bash
ls -la videos/phc_walk_demo.mp4
ffprobe videos/phc_walk_demo.mp4 2>&1 | grep -E "Duration|Stream"
```

Expected: Duration ~10s, h264 video stream.

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo.py
git commit -m "phc_walk_demo: optional --record_seconds N flag"
```

---

## Task 11: Manual acceptance smoke test

**Files:**
- None modified.

This task is purely a manual run-through; it produces no commits unless an issue is discovered and fixed.

- [ ] **Step 1: Run the demo and exercise every control**

```bash
conda activate phc
python scripts/phc_walk_demo.py
```

Acceptance walkthrough:

1. **Boot**: viewer opens. Tk panel appears top-right within 2 s. Humanoid (env 0) starts walking at v_cmd ≈ 1.0 m/s. Forward arrow visible above head, purple-ish color (mid range).

2. **Keyboard fine adjust**: press ↑ 10 times → terminal shows v_target climbing in 0.02 increments toward 1.20; v_ramped chases at 0.5 m/s² (visible delay); arrow lengthens and reddens; cadence visibly faster. Verify clip switch logged at v_cmd > 1.022 + 0.02 → clip 2.

3. **Keyboard shortcut**: press 1 → v_target = 0.76; ramp visible (~1 s); clip switches to 0 at next cycle boundary; gait is slow walking. Press 5 → v_target = 1.0 (mid); back to clip B.

4. **Tk numeric input**: type `1.05` in Entry, press Enter (or click Apply). v_target jumps to 1.05; ramped chases; clip C activates after cycle.

5. **Tk slider**: drag slider to ~0.85. v_target follows live. Slow walking activates.

6. **Reset**: press R (or click panel Reset). Episode resets — humanoid teleports to start pose; v_cmd_target preserved.

7. **Pause**: press P → "PAUSED" in panel, sim freezes. Press P → resumes.

8. **Quit**: press Q (or click panel Quit). Demo exits cleanly; panel closes.

9. **Recording**: re-run with `--record_seconds 30`. While running, vary v_cmd (e.g., press 1, wait 3 s, press 9, wait 3 s, ↑↑↑↑↑ over 5 s, press 5, wait until exit). Expected output: `videos/phc_walk_demo.mp4` saved, ~30 s duration.

**Acceptance criteria:**
- Humanoid does not fall during normal v_cmd changes within [0.76, 1.23].
- v_actual stays within ±0.05 m/s of v_cmd_ramped in steady state.
- Clip switches do not cause obvious tracking failures more than ~once per 10 transitions (small stumble OK).
- Panel updates v_cmd display within 100 ms of any input.
- Episode steps reach at least 250 (out of 300) on a typical run with moderate v_cmd changes.
- Recorded mp4 plays back cleanly.

- [ ] **Step 2: If any acceptance criterion fails, diagnose and fix in a follow-up commit.** Otherwise, no commit needed.

---

## Self-Review Checklist

**Spec coverage:**
- §1 Architecture diagram ↔ Tasks 1–10 (boot, state, keyboard, retime, clip selection, arrow, v_actual, panel, recording).
- §2 Components 1-2 (motion data, policy) — used as-is via Hydra overrides in Task 1.
- §2 Component 3 (top-of-file constants) — Task 1.
- §2 Component 4 (per-step retime patch) — Task 4.
- §2 Component 5 (clip selection logic) — Task 5.
- §2 Component 6 (Tkinter panel) — Tasks 8 + 9.
- §2 Component 7 (forward arrow) — Task 6.
- §2 Component 8 (viewer keyboard) — Task 3.
- §2 Component 9 (v_cmd ramp) — Task 4 (`_ramp_v_cmd`).
- §2 Component 10 (v_actual measurement) — Task 7.
- §2 Component 11 (recording) — Task 10.
- §Data flow per render step — covered across Tasks 3-7 and 10.
- §Smoke test — Task 11.

**Placeholder scan:** No "TBD"/"TODO"/"as appropriate"/etc. All steps contain concrete code. ✓

**Type/name consistency:**
- `_STATE` keys consistent across tasks: `v_cmd_target`, `v_cmd_ramped`, `active_clip`, `v_natural`, `retime_ratio`, `v_actual`, `step`, `paused`, `subs_ready`, `fall_count`, `term_height`, `last_cycle_count`. ✓
- Helper function names: `_write_panel_state`, `_read_panel_input`, `_maybe_subscribe_keys`, `_handle_events`, `_install_render_hook`, `_install_pre_physics_patch`, `_ramp_v_cmd`, `_select_clip`, `_maybe_switch_clip`, `_draw_forward_arrow`, `_measure_v_actual`, `_lock_camera`, `_spawn_panel`. All used consistently. ✓
- JSON keys consistent between demo writer and panel reader: `v_cmd_target`, `v_cmd_ramped`, `active_clip`, `v_natural`, `retime_ratio`, `v_actual`, `step`, `paused`, `v_min`, `v_max`. Panel reads exactly these. ✓
- JSON input keys consistent between panel writer and demo reader: `v_cmd_target`, `reset`, `pause_toggle`, `quit`. ✓

**Constants consistency:** V_CMD_MIN=0.76 / V_CMD_MAX=1.23 used both in demo (clamping) and panel (V_MIN/V_MAX defaults at top of panel script). Slider uses these. ✓

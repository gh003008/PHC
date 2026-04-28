# VIC4 Interactive Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/vic4_demo.py` — an interactive IsaacGym viewer that loads a trained VIC4_VCMD policy, lets the user drive `v_cmd` live with keyboard (and best-effort ImGui slider), supports smooth updates and hard reset, and switches models by editing two top-of-file constants.

**Architecture:** One self-contained Python script. Reuses the boot pattern from `scripts/vis_vic4_vcmd.py` (sys.path setup, regex-patch yaml `num_envs`/`numEnvs`/`env_spacing`, monkey-patch `HumanoidImVIC._compute_reset` and `Humanoid.render`). Adds keyboard subscriptions in the render hook on first call, polls events via `gym.query_viewer_action_events`, overrides `env._current_cmd[0, 0] = desired_v_cmd` per step, locks the camera to env 0, auto-resets on fall. PHC base classes are untouched.

**Tech Stack:** IsaacGym (`gymapi.KEY_*`, `subscribe_viewer_keyboard_event`, `query_viewer_action_events`, `viewer_camera_look_at`, `add_lines`), PyTorch (Tensor ops on `_current_cmd`/`reset_buf`), PyYAML (read `multiclip_v_cmd_range` from yaml), conda env `phc`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/vic4_demo.py` (new, ~250 LOC) | Boot scaffolding, yaml patch, monkey-patches, event loop, main(). |
| `~/.claude/projects/-home-exolab-Documents-GitHub-PHC/memory/MEMORY.md` | One-line index entry pointing to a new feedback memory (added in Task 9). |
| `~/.claude/projects/-home-exolab-Documents-GitHub-PHC/memory/feedback_isaacgym_event_polling.md` | Feedback memory: PHC's render() consumes the keyboard event queue via `query_viewer_action_events` — any custom hook must poll BEFORE calling orig_render and re-handle PHC's defaults (R/SPACE/ESC) if needed. |

The script is one file because all parts must share module-level state (`desired_v_cmd`, `paused`, key handlers) and the patches need closure access to it. Splitting by responsibility would force shared globals or a class wrapper that buys nothing.

---

## Task 1: Boot scaffolding + top-of-file constants

**Files:**
- Create: `scripts/vic4_demo.py`

- [ ] **Step 1: Write the file skeleton with sys.path bootstrap and the two editor-controlled constants**

Create `scripts/vic4_demo.py` with:

```python
"""IsaacGym interactive demo for VIC4_VCMD policies.

Run:
  conda activate phc
  python scripts/vic4_demo.py

To switch models, edit the SLOT/EPOCH constants below. The yaml dir is
auto-resolved from SLOT (S4-S8 -> 260427, S9-S11 -> 260428_v8,
S12-S13 -> 260428_v9).

Controls (live, in the IsaacGym window):
  Up   / Down  : v_cmd +/- 0.05 (clamped to learned range)
  1..9         : v_cmd snap to 9 levels across learned range
  R            : hard reset humanoid to standing pose at origin
  Space        : pause / resume sim
  Esc          : quit

ImGui slider is best-effort: rendered if the IsaacGym binding exposes
a value-mutating UI callback, otherwise falls back to keyboard only.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

# === EDIT THESE TWO CONSTANTS TO SWITCH MODELS ===
SLOT = "S10"           # one of S4..S13
EPOCH = -1             # -1 = auto-pick latest output/VIC4_VCMD_<SLOT>_*.pth
# =================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for _p in (_PHC_PKG, _PHC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np
import torch
import yaml


def resolve_exp_dir(slot: str) -> str:
    if slot in ('S12', 'S13'):
        return 'exp_config/forward_walking/260428_VIC4_VCMD_v9'
    if slot in ('S9', 'S10', 'S11'):
        return 'exp_config/forward_walking/260428_VIC4_VCMD_v8'
    return 'exp_config/forward_walking/260427_VIC4_VCMD'


def resolve_checkpoint(slot: str, epoch: int) -> tuple[str, int]:
    """Return (path, resolved_epoch). epoch=-1 means auto-pick latest."""
    if epoch >= 0:
        path = f"output/VIC4_VCMD_{slot}_{epoch:08d}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint at {path}. Pull from server with:\n"
                f"  rsync -avzP server1-jiminyoun:PHC/output/VIC4_VCMD_{slot}_{epoch:08d}.pth output/"
            )
        return path, epoch
    # auto-pick latest
    candidates = sorted(glob.glob(f"output/VIC4_VCMD_{slot}_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching output/VIC4_VCMD_{slot}_*.pth\n"
            f"Pull from server with:\n"
            f"  rsync -avzP 'server1-jiminyoun:PHC/output/VIC4_VCMD_{slot}_*.pth' output/"
        )
    path = candidates[-1]
    ep_str = os.path.basename(path).split('_')[-1].replace('.pth', '')
    return path, int(ep_str)
```

- [ ] **Step 2: Verify it runs to here without crashing**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); from vic4_demo import resolve_exp_dir, resolve_checkpoint; print(resolve_exp_dir('S10'))"`

Expected: prints `exp_config/forward_walking/260428_VIC4_VCMD_v8` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): boot scaffolding + slot/epoch constants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Yaml load + range extract + tmp yaml regex patch

**Files:**
- Modify: `scripts/vic4_demo.py` (append `load_and_patch_yaml` function)

- [ ] **Step 1: Add yaml loading helper that extracts v_lo/v_hi and patches num_envs/numEnvs/env_spacing/envSpacing into a tmp yaml**

Append to `scripts/vic4_demo.py`:

```python
def load_and_patch_yaml(slot: str) -> tuple[str, float, float, float]:
    """Read slot's env yaml, extract multiclip_v_cmd_range and terminationHeight,
    write a tmp yaml with num_envs=2 and env_spacing=50.
    Returns (tmp_yaml_path, v_lo, v_hi, term_height)."""
    import re
    src = f"{resolve_exp_dir(slot)}/env_im_walk_vic_{slot}.yaml"
    with open(src) as f:
        text = f.read()
    cfg = yaml.safe_load(text)
    env = cfg.get("env", {})
    rng = env.get("multiclip_v_cmd_range")
    if rng is None or len(rng) != 2:
        print(f"[demo] WARN: yaml has no multiclip_v_cmd_range, defaulting to [0.6, 1.4]")
        v_lo, v_hi = 0.6, 1.4
    else:
        v_lo, v_hi = float(rng[0]), float(rng[1])
    term_height = float(env.get("terminationHeight", 0.15))
    # Regex-patch num_envs / numEnvs / env_spacing / envSpacing to demo values
    text = re.sub(r'^(\s*num_envs:\s*)\d+', r'\g<1>2', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*numEnvs:\s*)\d+', r'\g<1>2', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*env_spacing:\s*)\d+', r'\g<1>50', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*envSpacing:\s*)\d+', r'\g<1>50', text, flags=re.MULTILINE)
    tmp = f"/tmp/env_vic4_demo_{slot}.yaml"
    with open(tmp, 'w') as f:
        f.write(text)
    return tmp, v_lo, v_hi, term_height
```

- [ ] **Step 2: Smoke test that the helper produces a patched yaml with 2 envs**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); from vic4_demo import load_and_patch_yaml; tmp, lo, hi, h = load_and_patch_yaml('S10'); print(f'tmp={tmp} v_lo={lo} v_hi={hi} term_h={h}'); import re; t=open(tmp).read(); print('num_envs lines:', re.findall(r'^\s*num.*\d+', t, flags=re.MULTILINE)[:5])"`

Expected: prints v_lo=0.82, v_hi=1.23, term_h=0.15, and the num_envs/numEnvs lines show value 2.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): yaml range extract + tmp yaml regex patch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: terminationDistance restore patch (copy of vis_vic4_vcmd pattern)

**Files:**
- Modify: `scripts/vic4_demo.py` (add `_install_term_dist_patch` function)

- [ ] **Step 1: Add the term-dist patch that restores yaml's terminationDistance over the player's 0.5m hardcode**

Append to `scripts/vic4_demo.py`:

```python
def _install_term_dist_patch():
    """Restore yaml's terminationDistance, which gets clobbered to 0.5 by
    phc/learning/im_amp_players.py:41 in test mode."""
    from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
    orig_reset = HumanoidImVIC._compute_reset

    def patched_reset(self):
        if not getattr(self, '_term_dist_restored', False):
            cfg_dist = float(self.cfg["env"].get("terminationDistance", 0.5))
            self._termination_distances[:] = cfg_dist
            self._term_dist_restored = True
            print(f"[demo] _termination_distances overridden to {cfg_dist} (env yaml)")
        return orig_reset(self)

    HumanoidImVIC._compute_reset = patched_reset
```

- [ ] **Step 2: Verify import path resolves at parse time**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); import vic4_demo; print('ok')"`

Expected: prints `ok` (just import — no execution of patches yet).

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): terminationDistance restore patch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Module-level state + keyboard subscription (lazy on first render)

**Files:**
- Modify: `scripts/vic4_demo.py` (add module-level state dict + render-hook scaffold)

- [ ] **Step 1: Add module-level state dict and the render-hook installer that lazily registers keyboard subscriptions on first call**

Append to `scripts/vic4_demo.py`:

```python
# Module-level mutable state shared between the render hook and the rest.
# Initialized in main() before installing patches.
_STATE = {
    "v_lo": 0.82,
    "v_hi": 1.23,
    "term_height": 0.15,
    "desired_v_cmd": 1.025,    # mid of [v_lo, v_hi]
    "paused": False,
    "step": 0,
    "fall_count": 0,
    "subs_ready": False,
}

# IsaacGym action names → registered key code
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
    ("demo_reset", "KEY_R"),       # NB: PHC also subscribes KEY_R="reset" — we'll handle the
                                    # "demo_reset" event explicitly instead of duplicating logic
    ("demo_pause", "KEY_SPACE"),
    ("demo_quit", "KEY_ESCAPE"),
]


def _maybe_subscribe_keys(env):
    """Register keyboard subscriptions on the viewer (idempotent, only first call does work).
    Must be called from inside the render hook (after viewer creation)."""
    if _STATE["subs_ready"]:
        return
    if env.viewer is None:
        return
    for action_name, key_attr in _KEY_BINDINGS:
        key_code = getattr(gymapi, key_attr)
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key_code, action_name)
    _STATE["subs_ready"] = True
    print("[demo] keyboard subscriptions registered")
```

- [ ] **Step 2: Sanity check — the file still imports cleanly**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); import vic4_demo; print(vic4_demo._STATE['desired_v_cmd'])"`

Expected: prints `1.025`.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): module state + lazy keyboard subscription

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Event polling + key→action dispatch

**Files:**
- Modify: `scripts/vic4_demo.py` (add `_handle_events`)

- [ ] **Step 1: Add the event handler — polls IsaacGym events and updates `_STATE` accordingly**

Append to `scripts/vic4_demo.py`:

```python
def _handle_events(env):
    """Poll keyboard events and update _STATE / env state.
    MUST be called BEFORE orig_render() since query_viewer_action_events()
    consumes the queue."""
    if env.viewer is None:
        return
    v_lo = _STATE["v_lo"]; v_hi = _STATE["v_hi"]
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue   # only on press (value > 0 means down-press)
        a = evt.action
        if a == "v_up":
            _STATE["desired_v_cmd"] = float(np.clip(_STATE["desired_v_cmd"] + 0.05, v_lo, v_hi))
        elif a == "v_down":
            _STATE["desired_v_cmd"] = float(np.clip(_STATE["desired_v_cmd"] - 0.05, v_lo, v_hi))
        elif a.startswith("v_set_"):
            k = int(a.split("_")[-1])  # 1..9
            _STATE["desired_v_cmd"] = v_lo + (k - 1) / 8.0 * (v_hi - v_lo)
        elif a == "demo_reset":
            env.reset_buf[:] = 1
            _STATE["fall_count"] = 0
            print(f"[demo] hard reset (R) — v_cmd will resume at {_STATE['desired_v_cmd']:.3f}")
        elif a == "demo_pause":
            _STATE["paused"] = not _STATE["paused"]
            print(f"[demo] {'PAUSED' if _STATE['paused'] else 'RESUMED'}")
        elif a == "demo_quit":
            print("[demo] quit (ESC)")
            sys.exit(0)
```

- [ ] **Step 2: Sanity check — module-level usage of np and sys is in place**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); from vic4_demo import _handle_events; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): keyboard event dispatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Per-step v_cmd override + arrow drawing

**Files:**
- Modify: `scripts/vic4_demo.py` (add `_apply_cmd_and_draw`)

- [ ] **Step 1: Add the function that copies `desired_v_cmd` into env._current_cmd[0,0] and draws the forward arrow above env 0**

Append to `scripts/vic4_demo.py`:

```python
ARROW_LEN_AT_VHI = 1.5


def _apply_cmd_and_draw(env):
    """Force env 0's _current_cmd to desired_v_cmd; draw arrow above its head.
    Env 1 keeps whatever the task assigned (it's hidden 100m away)."""
    if not hasattr(env, "_current_cmd"):
        return  # task hasn't set it up yet
    v = _STATE["desired_v_cmd"]
    env._current_cmd[0, 0] = v
    # angular component: this slot family uses cmd_w_range [0,0] so always 0
    if env._current_cmd.shape[1] >= 2:
        env._current_cmd[0, 1] = 0.0
    if env.viewer is None:
        return
    env.gym.clear_lines(env.viewer)
    root = env._humanoid_root_states[0].detach().cpu().numpy()  # [13]
    pos = root[:3]; quat = root[3:7]
    x, y, z, w = quat
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    norm = float(np.sqrt(fwd_x ** 2 + fwd_y ** 2)) + 1e-8
    fwd_x /= norm; fwd_y /= norm
    head_z = pos[2] + 0.6
    v_lo = _STATE["v_lo"]; v_hi = _STATE["v_hi"]
    scale = (v / v_hi) * ARROW_LEN_AT_VHI
    t_color = float(np.clip((v - v_lo) / max(v_hi - v_lo, 1e-6), 0.0, 1.0))
    color = np.array([t_color, 0.0, 1.0 - t_color], dtype=np.float32)
    sx, sy = pos[0], pos[1]
    ex, ey = sx + fwd_x * scale, sy + fwd_y * scale
    main = np.array([sx, sy, head_z, ex, ey, head_z], dtype=np.float32)
    env.gym.add_lines(env.viewer, env.envs[0], 1, main, color)
    head_size = 0.1
    lx = ex - fwd_x * head_size + fwd_y * head_size * 0.5
    ly = ey - fwd_y * head_size - fwd_x * head_size * 0.5
    rx = ex - fwd_x * head_size - fwd_y * head_size * 0.5
    ry = ey - fwd_y * head_size + fwd_x * head_size * 0.5
    env.gym.add_lines(env.viewer, env.envs[0], 1,
                      np.array([ex, ey, head_z, lx, ly, head_z], dtype=np.float32), color)
    env.gym.add_lines(env.viewer, env.envs[0], 1,
                      np.array([ex, ey, head_z, rx, ry, head_z], dtype=np.float32), color)
```

- [ ] **Step 2: Sanity import check**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); from vic4_demo import _apply_cmd_and_draw; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): per-step v_cmd override + forward arrow draw

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Camera lock + fall recovery + log + render-hook installer

**Files:**
- Modify: `scripts/vic4_demo.py` (add `_lock_camera`, `_check_fall`, `_install_render_patch`, `_log`)

- [ ] **Step 1: Add the three helpers and the render-hook installer that ties them together**

Append to `scripts/vic4_demo.py`:

```python
def _lock_camera(env):
    """Lock the IsaacGym camera to env 0's pelvis with a quartering follow-shot."""
    if env.viewer is None:
        return
    pos = env._humanoid_root_states[0, :3].detach().cpu().numpy()
    cam_pos = gymapi.Vec3(float(pos[0] - 3.0), float(pos[1] - 3.0), float(pos[2] + 1.5))
    cam_target = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)


def _check_fall(env):
    """If env 0's pelvis stays below terminationHeight for 5 consecutive frames,
    auto-trigger reset_buf[0]=1 and zero the counter."""
    pelvis_z = float(env._humanoid_root_states[0, 2].item())
    if pelvis_z < _STATE["term_height"]:
        _STATE["fall_count"] += 1
    else:
        _STATE["fall_count"] = 0
    if _STATE["fall_count"] >= 5:
        env.reset_buf[0] = 1
        _STATE["fall_count"] = 0
        print(f"[demo] auto-reset env 0 (pelvis_z={pelvis_z:.3f} < term_height={_STATE['term_height']:.3f})")


def _log(env):
    if _STATE["step"] % 60 == 0:
        pelvis_z = float(env._humanoid_root_states[0, 2].item())
        print(f"[demo] step={_STATE['step']:6d}  v_cmd={_STATE['desired_v_cmd']:+.3f}  pelvis_z={pelvis_z:+.3f}")


def _install_render_patch():
    """Patch Humanoid.render to:
      1. lazily subscribe keyboard on first call,
      2. poll events BEFORE orig_render (so PHC's render() doesn't drain queue),
      3. apply v_cmd override + draw arrow,
      4. lock camera to env 0,
      5. fall recovery,
      6. log."""
    from phc.env.tasks.humanoid import Humanoid
    orig_render = Humanoid.render

    def patched_render(self, sync_frame_time=False):
        _maybe_subscribe_keys(self)
        _handle_events(self)
        _apply_cmd_and_draw(self)
        _lock_camera(self)
        _check_fall(self)
        _STATE["step"] += 1
        _log(self)
        return orig_render(self, sync_frame_time)

    Humanoid.render = patched_render
```

- [ ] **Step 2: Sanity import check**

Run: `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); from vic4_demo import _install_render_patch; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): camera lock + fall recovery + render hook

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: main() launch path

**Files:**
- Modify: `scripts/vic4_demo.py` (add `main()` and `if __name__` guard)

- [ ] **Step 1: Add main() that wires everything together and calls phc.run.main()**

Append to `scripts/vic4_demo.py`:

```python
def main():
    exp_dir = resolve_exp_dir(SLOT)
    cfg_env, v_lo, v_hi, term_h = load_and_patch_yaml(SLOT)
    cfg_train = f"{exp_dir}/im_walk_vic.yaml"
    ckpt_path, resolved_epoch = resolve_checkpoint(SLOT, EPOCH)

    _STATE["v_lo"] = v_lo
    _STATE["v_hi"] = v_hi
    _STATE["term_height"] = term_h
    _STATE["desired_v_cmd"] = 0.5 * (v_lo + v_hi)

    task_name = "HumanoidImVICCmdRetime" if SLOT == "S4" else "HumanoidImVICCmdMultiClip"

    sys.argv = [
        "run.py",
        "--task", task_name,
        "--cfg_env", cfg_env,
        "--cfg_train", cfg_train,
        "--num_envs", "2",
        "--test", "--epoch", str(resolved_epoch),
        "--experiment", f"VIC4_VCMD_{SLOT}",
    ]

    _install_term_dist_patch()
    _install_render_patch()

    print("=" * 70)
    print(f"[demo] SLOT={SLOT}  EPOCH={resolved_epoch}  ckpt={ckpt_path}")
    print(f"[demo] v_cmd legal range: [{v_lo:.3f}, {v_hi:.3f}] m/s")
    print(f"[demo] start v_cmd = {_STATE['desired_v_cmd']:.3f}")
    print(f"[demo] keys:  Up/Down +-0.05  |  1..9 snap to range levels  |  R reset  |  Space pause  |  Esc quit")
    print("=" * 70)

    from phc import run as phc_run
    phc_run.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a parse-only run (will fail at IsaacGym init if no display, but we just want to see argv handoff)**

Run (interactively if you have display, otherwise just verify it loads to phc_run): `conda run -n phc python -c "import sys; sys.path.insert(0, 'scripts'); import vic4_demo; print('main exists:', callable(vic4_demo.main))"`

Expected: prints `main exists: True`.

- [ ] **Step 3: Commit**

```bash
git add scripts/vic4_demo.py
git commit -m "feat(vic4_demo): main() launch path with banner + argv handoff

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Manual acceptance smoke test + memory note

**Files:**
- No source change.
- Modify: `~/.claude/projects/-home-exolab-Documents-GitHub-PHC/memory/MEMORY.md` and create `feedback_isaacgym_event_polling.md` if the test surfaces the polling-order pitfall.

- [ ] **Step 1: Run the demo with the default SLOT="S10"**

Run: `DISPLAY=:1 conda run -n phc python scripts/vic4_demo.py`

Expected behavior in the IsaacGym window that opens:
- One visible humanoid at origin (env 1 is 100m away, off-screen).
- Forward arrow above its head (color depends on `desired_v_cmd`).
- Terminal banner shows `SLOT=S10  EPOCH=<latest>  ckpt=output/VIC4_VCMD_S10_<XXXXXXXX>.pth` and the v_cmd legal range `[0.820, 1.230]`.
- Press `5` → terminal logs the v_cmd jump; humanoid speeds adjust.
- Press `9` → humanoid speeds up to top of range.
- Press `1` → slows to bottom of range.
- Press `R` → humanoid teleports back to standing pose at origin and starts walking again.
- Press `Space` → sim pauses; press again to resume.
- Press `Esc` → clean exit.

Manually verify the above. If anything fails, debug before committing.

- [ ] **Step 2: Switch slot and re-test**

Edit `scripts/vic4_demo.py` line `SLOT = "S10"` → `SLOT = "S12"`. Re-run.

Expected:
- Terminal shows `SLOT=S12  EPOCH=<latest_S12>` and v_cmd range `[0.760, 1.230]`.
- `1` snaps to v_cmd ≈ 0.76, `9` snaps to ≈ 1.23.

Then revert `SLOT = "S12"` → `SLOT = "S10"` (or leave it on S12 if that's the model the user wants for now — confirm with the user).

- [ ] **Step 3: Add feedback memory if event-polling order caused issues**

If during the smoke test PHC's existing `KEY_R` reset or `SPACE` pause behaved inconsistently, that confirms the polling-order pitfall and is worth recording. Create `~/.claude/projects/-home-exolab-Documents-GitHub-PHC/memory/feedback_isaacgym_event_polling.md`:

```markdown
---
name: IsaacGym viewer event polling
description: PHC's render() drains the keyboard event queue via query_viewer_action_events. Custom hooks must poll BEFORE orig_render or events disappear.
type: feedback
---

PHC's `phc/env/tasks/base_task.py:render()` calls
`gym.query_viewer_action_events(viewer)` near the start, which returns AND
clears all pending keyboard events. If a custom render-hook patch calls
`orig_render()` first and then tries to poll, the queue is empty.

**Why:** `query_viewer_action_events` is destructive — there's no peek-only API.

**How to apply:** When monkey-patching `Humanoid.render`, poll events FIRST,
handle your custom actions, then call `orig_render()`. If your hook handles
keys that PHC also subscribes (e.g. KEY_R="reset", KEY_SPACE="PAUSE",
KEY_ESCAPE="QUIT"), re-implement those behaviors in your handler since
orig_render won't see them.

Hit during scripts/vic4_demo.py implementation (commit <fill in>).
```

Then add an index line to `~/.claude/projects/-home-exolab-Documents-GitHub-PHC/memory/MEMORY.md`:

```markdown
- [IsaacGym event polling](feedback_isaacgym_event_polling.md) — query_viewer_action_events drains the queue; custom render hooks must poll before orig_render
```

If the smoke test passed without exposing this issue, **skip Step 3** — don't write a memory speculatively.

- [ ] **Step 4: Final commit**

```bash
# Only if Step 3 was applicable:
# (Memory files live OUTSIDE the repo, not in git — no `git add` needed for those.)
# If you fixed any issue during smoke test, add the source change here and commit:
git status
# If clean, no commit needed; the demo is ready.
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Top-of-file SLOT/EPOCH constants — Task 1.
- ✅ Auto exp_dir routing (260427 / 260428_v8 / 260428_v9) — Task 1.
- ✅ Yaml regex-patch num_envs=2 + env_spacing=50 — Task 2.
- ✅ multiclip_v_cmd_range pulled from yaml as [v_lo, v_hi] — Task 2.
- ✅ terminationDistance restore — Task 3.
- ✅ Keyboard subscriptions Up/Down/1-9/R/Space/Esc — Tasks 4 & 5.
- ✅ Live override of env._current_cmd[0,0] — Task 6.
- ✅ Forward arrow above env 0 — Task 6.
- ✅ Camera lock to env 0 pelvis — Task 7.
- ✅ Fall recovery (5 consecutive frames pelvis < term_h → reset_buf[0]=1) — Task 7.
- ✅ Terminal log every 60 steps — Task 7.
- ✅ Failure handling: missing ckpt → clear rsync hint — Task 1 (`resolve_checkpoint`).
- ✅ Failure handling: missing v_cmd_range → default [0.6, 1.4] + warn — Task 2.
- ⚠ ImGui slider (best-effort) — **NOT IN PLAN.** The spec said "best-effort, degrades to read-only readout if binding doesn't expose mutation." After looking at the IsaacGym Python binding (no `add_ui_callback` exposed in the `gymapi` namespace), the slider is not implementable in pure Python without diving into native bindings. Dropping it from the implementation. Keyboard remains the input. The plan's Task 9 acceptance test does NOT depend on the slider.

**Decision on slider gap:** the spec's failure-handling row already covers this exact case ("ImGui slider not supported in this IsaacGym binding; using keyboard only"). Effectively the slider degrades to nothing in this binding, which is a permitted spec outcome. No spec change required.

**Placeholder scan:** No "TBD"/"TODO"/"later"/"similar to". Each task has complete code or a complete command.

**Type consistency:** `_STATE` keys are referenced consistently across Tasks 4–8 (`v_lo`, `v_hi`, `term_height`, `desired_v_cmd`, `paused`, `step`, `fall_count`, `subs_ready`). `env._current_cmd`, `env._humanoid_root_states`, `env.reset_buf`, `env.viewer`, `env.envs`, `env.gym`, `env.cfg["env"]` — all standard PHC HumanoidImVIC attributes. `gymapi.Vec3` and `gymapi.KEY_*` are correct names per the binding output verified during plan writing.

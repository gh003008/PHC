# PHC Pretrained v_cmd Walk Demo (D-2)

**Goal:** Interactive IsaacGym demo where the user controls walking velocity (v_cmd, continuous float in 0.7–1.3 m/s) via keyboard or a Tkinter panel. The pretrained PHC `phc_3` model imitates the active reference walking clip, with per-step time scaling so v_cmd changes take effect immediately. **Zero training.**

**Approach:** Multi-clip + per-step retime ("D-2"). Three natural-speed walking clips already exist (`amass_walking_3clips_seamless_60s_v9.pkl`, ~0.90 / 0.97 / 1.07 m/s). At runtime: pick the clip whose natural speed is closest to v_cmd, then advance reference motion time by `dt × (v_cmd / v_natural)` so the clip plays at v_cmd's rate. Clip switches happen at cycle boundaries with hysteresis.

**Non-goals:** No VIC, no impedance control, no policy training. v_cmd is constrained to walking range — no running, no backwards, no turns.

---

## Architecture

```
[keyboard ↑/↓/digit] ─┐
                       │
[Tkinter Entry/slider] ─┼→ v_cmd_target (float)
                       │       │
[Tkinter buttons] ────┘       ▼
                       v_cmd_ramped (smoothed by max_accel=0.5 m/s²)
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
       [clip_selector]         [time_advancer]
       (cycle boundary,         (every render step)
        hysteresis ±0.02)             │
                │              motion_time += dt × (v_cmd / v_natural)
                ▼                     │
        active motion_id              ▼
        ∈ {A, B, C}            [reference state]
                │                     │
                └─────────┬───────────┘
                          ▼
                       [phc_3] → 69 DOF actions → [Humanoid sim]
                          │
                          └─→ [Tkinter panel reads state]
                              [forward arrow drawn in viewer]
                              [v_actual measured & displayed]
```

## Components

### 1. Motion data (existing, no changes)
- File: `sample_data/amass_walking_3clips_seamless_60s_v9.pkl`
- Metadata: `sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json`
- Three clips, natural forward speeds (loaded from dirmeta json):
  - clip A: ~0.90 m/s
  - clip B: ~0.97 m/s
  - clip C: ~1.07 m/s

### 2. Policy (existing, no changes)
- File: `output/HumanoidIm/phc_3/Humanoid.pth` (PHC v3 pretrained, 419 MB, downloaded via `download_data.sh`)
- Architecture: PNN with 3 primitives, hidden `[2048, 1536, 1024, 1024, 512, 512]`, input 934 dim, output 69 DOF
- Task: `HumanoidIm`
- Learning yaml: `phc/data/cfg/learning/im_pnn_big.yaml`
- Env yaml: `phc/data/cfg/env/env_im_pnn.yaml` (with overrides)

### 3. New demo script: `scripts/phc_walk_demo.py` (~250 LOC)

**Boot:** Same pattern as `scripts/record_phc_pretrained.py` (uses `runpy.run_path('phc/run_hydra.py', ...)` after monkey-patching `Humanoid.render`).

**Top-of-file constants** (user-editable):
```python
EXP_NAME = "phc_3"
LEARNING = "im_pnn_big"
MOTION_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9.pkl"
DIRMETA_FILE = "sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json"

V_CMD_INIT = 1.0
V_CMD_MIN = 0.76      # 3-clip × ±15% retime hard floor
V_CMD_MAX = 1.23
V_CMD_KEY_STEP = 0.02
V_CMD_MAX_ACCEL = 0.5  # m/s² for ramp smoothing
HYSTERESIS = 0.02

NUM_ENVS = 2
ENV_SPACING = 50      # hide env 1 (vic4_demo trick)
```

**Demo state (single dict):**
```python
_STATE = {
    "v_cmd_target": V_CMD_INIT,    # what user requested (Tkinter / keyboard)
    "v_cmd_ramped": V_CMD_INIT,    # smoothed actual v_cmd applied
    "active_clip": 1,              # 0/1/2 for A/B/C
    "v_natural": 0.97,             # natural speed of active clip
    "retime_ratio": 1.0,
    "v_actual": 0.0,               # measured root velocity (from sim)
    "steps": 0,
    "paused": False,
    "subs_ready": False,
}
```

### 4. Per-step retime patch (key technical mechanism)

**Mechanism:** PHC's HumanoidIm computes reference motion time fresh each call:
```python
# phc/env/tasks/humanoid_im.py (multiple sites: lines 391, 599, 681, 753, 758, 885)
motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
```

`_motion_start_times_offset` is a `[num_envs]` tensor. We can inject retime by accumulating `dt × (ratio - 1)` into this offset each step. This makes motion_times effectively advance at `dt × ratio` per physics step instead of `dt`.

**Hook point:** `Humanoid.pre_physics_step` (called before physics, so the offset update takes effect on the same step's motion_times computation).

**Patched method:**
```python
orig_pre_physics_step = Humanoid.pre_physics_step

def patched_pre_physics_step(self, actions):
    if not _STATE["paused"]:
        v_natural = _STATE["v_natural"]
        ratio = _STATE["v_cmd_ramped"] / v_natural
        delta = self.dt * (ratio - 1.0)  # 0 when ratio=1.0
        self._motion_start_times_offset += delta  # broadcasts to all envs
        _STATE["retime_ratio"] = ratio
    return orig_pre_physics_step(self, actions)

Humanoid.pre_physics_step = patched_pre_physics_step
```

This advances the reference motion at the v_cmd-scaled rate without touching motion_lib internals.

### 5. Clip selection logic (cycle boundary)

**Triggered:** When the active clip's cycle ends (motion_time wraps modulo cycle_length).

**Logic:**
```python
def select_clip(v_cmd, current_clip):
    # Hysteresis: only switch if v_cmd has crossed a midpoint by HYSTERESIS margin
    midpoint_AB = (V_NATURAL[0] + V_NATURAL[1]) / 2  # ~0.935
    midpoint_BC = (V_NATURAL[1] + V_NATURAL[2]) / 2  # ~1.020

    if current_clip == 0 and v_cmd > midpoint_AB + HYSTERESIS:
        return 1
    if current_clip == 1:
        if v_cmd < midpoint_AB - HYSTERESIS: return 0
        if v_cmd > midpoint_BC + HYSTERESIS: return 2
    if current_clip == 2 and v_cmd < midpoint_BC - HYSTERESIS:
        return 1
    return current_clip  # no change
```

**Switch mechanism:** Set the env's per-env `_motion_ids` tensor to the new clip id. The motion_lib then samples reference state from the new clip on next `get_motion_state` call.

### 6. Tkinter panel: `scripts/phc_walk_demo_panel.py`

**Layout (top-right, always-on-top, ~280×400 px):**
```
┌──────────────────────────────┐
│   PHC Walk Demo              │
├──────────────────────────────┤
│  v_cmd:        1.030 m/s     │
│  v_target:     1.050 m/s     │   (ramp in progress shown)
│  v_actual:     1.018 m/s     │   (measured)
│                              │
│  Active clip:  B (0.97 m/s)  │
│  Retime:       1.062x        │
│                              │
│  Steps:        142 / 300     │
│  Status:       Running       │
├──────────────────────────────┤
│  Set v_cmd:                  │
│  [____1.050____] [Apply]     │
│                              │
│  [↑ +0.02]  [↓ -0.02]        │
│  [Reset]    [Pause]          │
│                              │
│  [─────●─────] 0.76 ── 1.23  │  ← slider
└──────────────────────────────┘
```

**Communication via JSON files (same pattern as vic4_demo):**
- Demo writes `/tmp/phc_walk_state.json` every render step (read-side)
- Panel writes `/tmp/phc_walk_input.json` on Apply / button / slider release (write-side)
- Demo polls input file every render step

**Spawn:** Demo's `_spawn_panel(host_display)` (subprocess) — copy from `vic4_demo.py`.

### 7. Forward-arrow visualization in viewer

Same pattern as `record_vic4_vcmd.py`:
- Per render step, draw a line above each humanoid head pointing in the agent's forward direction (from root quaternion)
- Length = (v_cmd / V_CMD_MAX) × ARROW_LEN (1.5 m at max)
- Color: linear blue (slow) → red (fast) on `(v_cmd - V_CMD_MIN) / (V_CMD_MAX - V_CMD_MIN)`
- Arrowhead: two short lines at the tip

### 8. Viewer keyboard (still works without Tkinter)

| Key | Action |
|-----|--------|
| `↑` | v_cmd_target += 0.02 (clamped to V_CMD_MIN/MAX) |
| `↓` | v_cmd_target -= 0.02 |
| `1`–`9` | Shortcut: maps to V_CMD_MIN + (key-1)/8 × (V_CMD_MAX - V_CMD_MIN) |
| `R` | Episode reset (set `reset_buf[:] = 1`) |
| `P` | Toggle pause (skip env.step in render hook) |
| `Q` | Exit demo |

### 9. v_cmd ramp (smoothing)

Per render step:
```python
delta = V_CMD_MAX_ACCEL * sim_dt
diff = _STATE["v_cmd_target"] - _STATE["v_cmd_ramped"]
step = max(-delta, min(delta, diff))
_STATE["v_cmd_ramped"] += step
```
Effect: v_cmd jump of 0.5 m/s takes 1.0 s; jump of 0.2 m/s takes 0.4 s.

### 10. v_actual measurement

Per render step, read root velocity from sim:
```python
root_vel = env._humanoid_root_states[0, 7:10]  # env 0
v_actual = float(torch.linalg.norm(root_vel[:2]))   # planar magnitude
```
Display in panel.

### 11. Optional recording

If user passes `--record_seconds N`:
- Same Humanoid.render hook adds `gym.write_viewer_image_to_file` per frame to `/tmp/record_frames_phc_walk_demo/`
- After N seconds (or N×fps frames), exit and ffmpeg-combine to `videos/PHC_walk_demo.mp4`
- Pattern reused from `record_phc_pretrained.py`

---

## Data flow per render step

```
1. Poll keyboard events → update _STATE["v_cmd_target"]
2. Read /tmp/phc_walk_input.json → update _STATE["v_cmd_target"] (panel input)
3. Apply ramp: _STATE["v_cmd_ramped"] ← target with max_accel limit
4. If at cycle boundary: select_clip(v_cmd_ramped, active) → set env._motion_ids
5. Compute retime_ratio = v_cmd_ramped / v_natural
6. Advance motion_time by dt × ratio (patched method)
7. orig_render() runs: phc_3 inference + physics step + viewer redraw
8. Draw forward arrow on viewer
9. Measure v_actual from root_vel
10. Write _STATE → /tmp/phc_walk_state.json
11. (Optional) capture frame to /tmp/record_frames_phc_walk_demo/
```

---

## Files

### New (2 files)
- `scripts/phc_walk_demo.py` — main demo (~250 LOC)
- `scripts/phc_walk_demo_panel.py` — Tkinter panel (~120 LOC, copy/adapt from `scripts/vic4_demo_panel.py`)

### Modified
- None (everything via monkey-patches)

### Reused / referenced
- `output/HumanoidIm/phc_3/Humanoid.pth`
- `sample_data/amass_walking_3clips_seamless_60s_v9.pkl`
- `sample_data/amass_walking_3clips_seamless_60s_v9_dirmeta.json`
- `phc/run_hydra.py`
- `phc/data/cfg/learning/im_pnn_big.yaml`
- `phc/data/cfg/env/env_im_pnn.yaml`
- `scripts/vic4_demo.py` (boot + keyboard pattern)
- `scripts/vic4_demo_panel.py` (Tkinter layout)
- `scripts/record_vic4_vcmd.py` (forward arrow + recording pattern)
- `scripts/record_phc_pretrained.py` (PHC hydra boot pattern)

---

## Constraints / Risks

1. **Patch correctness**: We accumulate retime delta into `_motion_start_times_offset`, which is referenced by 6 sites in `humanoid_im.py` (lines 391, 599, 681, 753, 758, 885). All sites use the same formula `progress_buf * dt + start + offset`, so a single offset update propagates to all uses. If any subclass overrides this formula, the patch may be incomplete; subclasses are not used here (we use bare `HumanoidIm`).

2. **Cycle boundary clip jump**: When clip switches, reference pose may not phase-align with previous clip. PHC was trained on continuous AMASS where such transitions are rare. Risk: tracking error spike at switch → potential fall. Mitigation: hysteresis, episode reset on bad fall (auto via `terminationDistance`).

3. **PNN architecture sensitivity**: phc_3 uses 3-primitive PNN. Test mode runs all 3 primitives in some weighted combination — must verify it works for retimed walking (already verified in `record_phc_pretrained.py` smoke run: reward 276 / steps 299/300).

4. **GPU memory**: phc_3 (~580 MB on disk, larger in GPU) + 2 envs simulation. Local GPU (4060 Ti 7.6 GB) — should fit. If not, drop to num_envs=1.

5. **Tkinter on host display**: Panel needs DISPLAY env var. If running over SSH without X forwarding, panel will fail to spawn — keyboard-only mode still works.

6. **v_cmd outside [0.76, 1.23]**: Hard clamp in `_handle_events` — user can request 0.5 but will be clamped to 0.76.

---

## Smoke test (manual acceptance)

1. **Boot:** `conda activate phc; python scripts/phc_walk_demo.py`
   - Expected: IsaacGym viewer opens, Tkinter panel appears top-right, agent boots and starts walking at v_cmd=1.0
2. **Keyboard fine adjust:** Press `↑` 10 times → v_cmd ramps up to 1.20 → walking visibly faster, panel shows clip switching B→C around 1.02
3. **Keyboard shortcut:** Press `1` → v_cmd_target = 0.76 → ramp visible (1 s) → slow walking, clip = A
4. **Tkinter numeric input:** Type `1.05` in Entry, press Apply → v_cmd ramps to 1.05, retime_ratio ≈ 1.082
5. **Slider:** Drag slider to ~0.85 → v_cmd ramps to 0.85, clip switches C/B → A boundary
6. **Reset:** Press R → episode resets, agent reappears in default pose, walks again
7. **Pause:** Press P → physics frozen, viewer still updates
8. **Recording:** Run with `--record_seconds 30` → mp4 saved to `videos/PHC_walk_demo.mp4`, contains 30 s of various v_cmd

**Acceptance criteria:**
- Agent does not fall during normal v_cmd changes within [0.76, 1.23]
- v_actual closely tracks v_cmd_ramped (within ~0.05 m/s in steady state)
- Clip switches do not cause obvious tracking failures more than ~once per 10 transitions
- Panel updates v_cmd display within 100 ms of input
- Eps_steps reaches at least 250 (out of 300) on a typical run with moderate v_cmd changes

---

## Implementation order (preview for plan)

1. Skeleton `phc_walk_demo.py`: boot via runpy + render hook + state dict
2. Keyboard event handlers (↑/↓/digit/R/P/Q)
3. Per-step retime monkey-patch (`pre_physics_step` + `_motion_start_times_offset` accumulation)
4. Clip selection logic + hysteresis at cycle boundary
5. Forward arrow visualization in viewer
6. v_actual measurement from `_humanoid_root_states`
7. Tkinter panel (separate script `phc_walk_demo_panel.py`)
8. JSON state I/O between demo and panel (bidirectional)
9. Recording option (`--record_seconds N`)
10. Manual smoke test

(Plan will detail tasks, files, atomic commits.)

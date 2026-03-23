# IsaacGym Viewer Segfault Debugging Record (260319)

Created: 2026-03-19
Translated from Korean original: `debugging/260319_isaacgym_viewer_segfault_디버깅.md`

## Problem

Running `standard_human_model/validation/03_visualization/run_visualization.py` causes immediate segmentation fault as soon as the viewer window opens.

```
Viewer opened. Press 'Q' or close window to exit.
Segmentation fault (core dumped)
```

## Root Cause Analysis

### Step 1: Pinpoint crash location (faulthandler)
→ Crash at line 176 `gym.step_graphics(sim)`

### Step 2: Reproduction attempt with isolated script
→ Identical logic in /tmp/test_*.py passed all 600 steps. Only the real script crashed at step 0.

Found the only code difference:
```python
# Dead code in run_visualization.py (lines 88-90):
num_bodies = gym.get_asset_rigid_body_count(
    gym.get_actor_asset(envs[0], actor_handles[0])
) if hasattr(gym, "get_actor_asset") else 24
```
This variable was declared but **never used**.

### Step 3: Confirmation
`gym.get_actor_asset()` call corrupts IsaacGym's internal graphics state, causing subsequent `gym.step_graphics(sim)` to segfault.

## Failed Approaches

| Attempt | Result |
|---------|--------|
| CPU pipeline (`use_gpu_pipeline=False`) | `step_graphics` itself crashes on CPU pipeline |
| SmartDisplay (visible=True) + CPU pipeline | "NV-GLX missing on display :1" + segfault |
| SmartDisplay + GPU pipeline | Segfault (different cause) |
| Hardcoded `graphics_device=-1` removal | Viewer window creation fails |
| Conditional `fetch_results` | Necessary fix but not the crash cause |
| `query_viewer_has_closed` position change | Safety measure but not the crash cause |

## Final Fixes

### 1. Remove `gym.get_actor_asset()` (root cause)
### 2. Replace `time.sleep` → `gym.sync_frame_time`
### 3. Move `query_viewer_has_closed` before `step_graphics`
### 4. GPU/CPU pipeline branching in `run_validation.py`

```python
if graphics_device is None:
    graphics_device = -1      # headless: CPU pipeline
    sim_params.use_gpu_pipeline = False
else:
    sim_params.use_gpu_pipeline = True  # viewer: GPU pipeline required
```

## GPU vs CPU Pipeline Behavior

| Mode | Pipeline | `step_graphics` | `fetch_results` |
|------|----------|-----------------|-----------------|
| headless | CPU | Not needed (causes crash) | Required |
| viewer | GPU | Required | Not needed (causes crash) |

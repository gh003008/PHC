# VIC4 Interactive Demo — Design

**Goal:** an IsaacGym viewer script that loads a trained VIC4_VCMD policy, lets the user drive `v_cmd` interactively (keyboard + ImGui slider) within the slot's learned range, and supports both smooth v_cmd updates and hard humanoid resets. Easy to swap to a different trained slot by editing two constants at the top of the script.

**Non-goals:**
- No CLI flags. Slot/epoch selection is editor-only (the user asks the agent to change them).
- No hot-swap between slots at runtime.
- No multi-agent demo. One visible agent.
- No new training behavior — this is purely an inference/eval tool.

## Architecture

One new file `scripts/vic4_demo.py`. Reuses the existing pattern from `scripts/vis_vic4_vcmd.py`: load the slot's env yaml, patch `num_envs` to 2 (multiclip minimum), call `phc.run.main()` in test mode with the latest checkpoint.

Three runtime patches via monkey-patching:
1. `HumanoidImVIC._compute_reset` — restores `terminationDistance` from yaml (overrides the 0.5m hardcode in `im_amp_players.py`).
2. `Humanoid.render` — draws v_cmd forward arrow above env 0, runs the camera tracker, polls keyboard, applies `desired_v_cmd` to env 0's `_current_cmd`.
3. `HumanoidImVIC._compute_observations` (or pre-step hook) — overrides env 0's `_current_cmd` with `desired_v_cmd` so the policy sees the user-set value.

No PHC base classes touched. All patches live in `vic4_demo.py`.

## Components

### Top-of-file config block (the only thing the agent edits when the user asks to change models)

```python
SLOT = "S10"      # one of S4..S13
EPOCH = -1        # -1 = auto-pick latest output/VIC4_VCMD_<SLOT>_*.pth
```

The script derives `exp_dir` from `SLOT` (same routing as `vis_vic4_vcmd.py`: S4–S8 → 260427, S9–S11 → 260428_v8, S12–S13 → 260428_v9). It reads the slot's env yaml to pull `multiclip_v_cmd_range` and uses that as the slider/keyboard clamp.

### Scene setup

- `num_envs = 2` (regex-patched into yaml).
- `env_spacing = 50` (regex-patched into yaml) so env 1 is 100m away from env 0.
- Camera locked to env 0's pelvis: every render call, `gym.viewer_camera_look_at(viewer, None, target=pelvis_xyz, position=pelvis_xyz + offset)` with `offset = (-3, -3, 1.5)` for a quartering follow-shot.
- Forward arrow above env 0's head, length proportional to v_cmd, color interpolated blue (slow) → red (fast). Same shape as `vis_vic4_vcmd.py`'s arrows but only for env 0.

### Input & state

**Single source of truth:** module-level `desired_v_cmd: float`. Initialized to `(v_lo + v_hi) / 2` (mid of legal range). All inputs mutate this; the per-step hook copies it to `env._current_cmd[0, 0]` (and writes 0 for the angular component since this slot family has `cmd_w_range: [0,0]`).

**Keyboard handlers** (registered after `viewer = gym.create_viewer(...)`):

| Key | Action |
|---|---|
| `Up` | `desired_v_cmd = clamp(desired_v_cmd + 0.05, v_lo, v_hi)` |
| `Down` | `desired_v_cmd = clamp(desired_v_cmd - 0.05, v_lo, v_hi)` |
| `1`–`9` | `desired_v_cmd = v_lo + (k-1)/8 * (v_hi - v_lo)` |
| `R` | Hard reset env 0 to standing pose at origin, then continue with current `desired_v_cmd`. Implementation: set `self.reset_buf[0] = 1` so the next env step triggers the standard reset path. |
| `P` | Toggle a `paused` flag; when paused, the render loop calls `gym.draw_viewer` only (no `gym.simulate`). |
| `Q` | Set a `quit` flag → on next render the script calls `sys.exit(0)`. |

**ImGui slider:** uses `gym.add_ui_callback(viewer, callback)` if available in the IsaacGym Python binding, drawing a single horizontal slider labeled `v_cmd [v_lo, v_hi] m/s` at top-left. Slider value bound to `desired_v_cmd`. If the binding does not expose value-mutation callbacks (some IsaacGym versions only support display widgets), the slider degrades to a read-only readout and keyboard remains the input. Detection: try-except on the first `add_ui_callback` call; on failure, log a warning and skip.

### Display feedback

- Forward arrow above env 0's head (length and color from `desired_v_cmd`).
- Terminal print every 60 steps: `[demo] v_cmd = 0.92 m/s | env0_pose_z = 0.89 | mode = smooth`.
- Console-line one-time at startup: legal range, all keybindings, current slot/checkpoint path.

### Auto-recovery

If env 0's pelvis z drops below `terminationHeight` (taken from yaml, e.g. 0.15) for more than 5 consecutive frames, treat as fallen and trigger the same reset path as `R`. Prevents the demo from getting stuck staring at a fallen body.

## Data flow

```
keyboard / slider event ──► desired_v_cmd (module-level float)
                                │
                                ▼
patched Humanoid.render() runs every sim step:
   1. read desired_v_cmd
   2. write env._current_cmd[0, 0] = desired_v_cmd
   3. draw forward arrow above env 0
   4. update camera target = env 0 pelvis xyz
   5. check fall recovery
   6. log v_cmd every 60 steps
```

The policy network is unchanged: it receives `_current_cmd` as part of its observation each step and outputs actions accordingly.

## Failure handling

| Failure | Behavior |
|---|---|
| No checkpoint matching `output/VIC4_VCMD_{SLOT}_*.pth` | Print clear error: `"No checkpoint for SLOT={SLOT}. Run: rsync -avzP server1-jiminyoun:PHC/output/VIC4_VCMD_{SLOT}_*.pth output/"` and exit. |
| Yaml has no `multiclip_v_cmd_range` | Default to `[0.6, 1.4]` and print warning. |
| ImGui slider binding not available in this IsaacGym version | Print `"[demo] ImGui slider not supported in this IsaacGym binding; using keyboard only"` and continue. |
| Humanoid env 1's pose causes physics hang | Not expected since env 1 is just standing in another spot, but if so, kill and restart is the user's path (no special handling). |

## Testing

This is a manual interactive tool. Acceptance test:

1. `python scripts/vic4_demo.py` (with default `SLOT="S10"`).
2. Check terminal prints legal range, keybindings, and checkpoint path.
3. IsaacGym window opens with one visible humanoid walking forward.
4. Press `5` — v_cmd jumps to mid range, humanoid speeds/slows accordingly within ~1s.
5. Press `9` — v_cmd jumps to top of range, humanoid speeds up.
6. Press `1` — bottom of range, humanoid slows.
7. Press `R` — humanoid teleports to standing pose at origin, then begins walking at current v_cmd.
8. Press `Q` — clean exit.

Smoke pass: agent edits `SLOT = "S12"` and re-runs; demo loads S12's checkpoint and uses the v9 yaml's range `[0.76, 1.23]`.

## Out-of-scope

- Recording the session to mp4. (Use `scripts/record_vic4_vcmd.py` for that.)
- Logging v_cmd / pose to a file for plotting later.
- Comparing two slots side-by-side. (Each demo run is one slot.)
- Changing reward/foot weights at runtime — those bake into training only.

# VIC4 + v_cmd — 3-Slot Training Design

**Date**: 2026-04-27
**Goal**: Add v_cmd (random velocity command) to the proven VIC4 baseline, verify reference-like walking is achievable on a v_cmd-conditioned policy. Three slots ablate **clip count** and **terminationDistance** with insurance on the simplest variant.
**Predecessor**: VIC4 (`AMASS_VERIFY_VIC4`, 2026-04-17) — proven single-clip imitation, eps_len 299/300 ceiling, per-frame reward 3.19.
**Failed approaches avoided**: KIT425 PERSONAL B canonical (98% drift termination from 50-clip multiclip); KIT425_CONT S2/S3 (Goodharted with terminationDistance=100m and no recovery).

## 1. Background — Why this design

### 1.1 Working baseline: VIC4
- Single clip (`KIT_11_WalkingStraightForwards05_poses`, 163 frames @ 30fps)
- `cycle_motion: True`, `terminationDistance: 0.25`, `fallInitProb: 0.3`, `hybridInitProb: 0.5`
- `vic_ccf_num_groups: 4`, `task: HumanoidImVIC` (no v_cmd)
- Result: greedy eps_len 299/300, reward 953, per-frame 3.19 → **truly reference-tracking walking**

### 1.2 Recent failures (analyzed)
| Experiment | What failed | Root cause |
|---|---|---|
| KIT425_PERSONAL B (2026-04-24) | 98% drift termination | 50-clip multiclip variance + tight termDist |
| KIT425_CONT S2/S3 (2026-04-26) | Goodhart (flipped over, eps_len=ceiling) | `terminationDistance=100m` removed all constraint |

### 1.3 User's domain insights (anchoring this design)
1. **Reward collision**: v_cmd ≠ reference clip velocity → reward fight. Mitigated by retime (reference vel = v_cmd after rescaling).
2. **Retime kinematics limit**: Real human walking has different joint patterns at different speeds, not just time-scaled kinematics. Therefore retime range MUST be narrow (~±15%, residual compensation only).
3. **Multiclip variance**: 50 clips of varied stride/footing exceed what a single policy can imitate within 0.25m drift bound. Need few hand-picked clips spaced by natural speed.
4. **terminationDistance**: 0.25m may be too tight when episode starts at standing (t₀=0) and v_cmd > 0 — policy needs slack to start moving without instant termination.
5. **No action smoothness penalty** (recorded in feedback memory) — bottleneck is tracking expressiveness, not over-actuation.

## 2. Three-Slot Architecture

| Slot | # of clips | terminationDistance | Hypothesis |
|---|---|---|---|
| **S4 — Insurance** | 1 (KIT_11 single, same as VIC4) | 0.4 | Minimal extension of working baseline. v_cmd + retime work on single-clip foundation? |
| **S5 — Primary** | 3 (slow / med / fast spaced) | 0.4 | User-agreed setup. Multiclip 3 covers wider speed range without breaking? |
| **S6 — Margin** | 3 (same as S5) | 0.6 | Looser termDist gives policy room to recover from drift spikes. Sweet spot? |

### 2.1 Slot rationale

- **S4 (Insurance)**: VIC4 + v_cmd minimal addition. If S4 fails, v_cmd integration itself is broken. If S4 succeeds, we have a reference-tracking v_cmd policy on single clip — already useful for demo.
- **S5 (Primary)**: 3-clip multiclip with same termDist as S4. Tests user's principle: clips spaced by natural speed + narrow retime = continuous v_cmd coverage without 50-clip variance.
- **S6 (Margin)**: Same multiclip as S5 with looser termDist=0.6. Tests whether tighter 0.4 was the binding constraint.

### 2.2 Result-branching plan

| S4 | S5 | S6 | Interpretation | Next milestone |
|---|---|---|---|---|
| ✅ | ✅ | ✅ | All work; multiclip+termDist OK | Keyboard demo wrapper |
| ✅ | ✅ | ✅+ | S6 noticeably better | Adopt 0.6 (or wider sweep next batch) |
| ✅ | ❌ | ✅ | termDist=0.4 too tight for multiclip | 0.6 going forward |
| ✅ | ❌ | ❌ | Multiclip is the issue | Investigate fall_termination / fewer clips |
| ❌ | — | — | v_cmd integration broken | Deep diagnostic batch |

## 3. Common Parameters (all slots)

```yaml
env:
  # Task class & motion
  motion_file: "<varies per slot>"      # see §4
  cycle_motion: True
  episodeLength: 300
  num_envs: 512

  # Init / recovery (VIC4 carryover)
  fallInitProb: 0.3
  hybridInitProb: 0.5
  init_at_clip_start: True              # NEW flag: episode reset → t₀=0 (standing)

  # Termination
  enableEarlyTermination: True
  terminationDistance: <varies per slot>  # 0.4 or 0.6

  # VIC
  vic_enabled: True
  vic_curriculum_stage: 2
  vic_ccf_num_groups: 4
  vic_ccf_sigma_init: -1.0
  vic_ccf_min: -1.0
  vic_ccf_max: 1.0
  vic_metabolic_reward_w: 0.01

  # Reward curriculum (VIC4 carryover, NOT changed)
  reward_curriculum_switch_epoch: 10000
  reward_w_stage1_task: 0.7
  reward_w_stage1_disc: 0.3
  reward_w_stage2_task: 0.3
  reward_w_stage2_disc: 0.7

  # v_cmd (Mode A: sample at episode reset only)
  cmd_v_range: [auto-derived from v_nat × retime_scale_range]
  cmd_w_range: [0.0, 0.0]
  cmd_tracking_w: 0.0                   # retime carries v_cmd signal; no explicit cmd reward
  cmd_track_kv: 2.0
  cmd_track_kw: 1.0

  # Retime (narrow, residual compensation only — user insight)
  retime_scale_range: [0.85, 1.15]      # ±15% only
```

Training hyper-parameters (`im_walk_vic.yaml`):
```yaml
sigma_init: -2.9                  # PD action sigma (~0.055)
max_epochs: 20000
mlp units: [1024, 1024, 512, 512]
learning_rate: 5e-5
```

## 4. Per-Slot Differences

### S4
- `task`: `HumanoidImVICCmdRetime`
- `motion_file`: `sample_data/amass_isaac_walking_forward_single.pkl` (existing, 1 clip KIT_11)
- `terminationDistance`: 0.4
- `retime_v_nat`: ~0.55 (KIT_11 natural; auto-computed)
- Resulting `cmd_v_range`: [0.47, 0.63] m/s

### S5
- `task`: `HumanoidImVICCmdMultiClip`
- `motion_file`: `sample_data/amass_walking_3clips_speedspaced.pkl` (NEW, 3 clips, see §5)
- `terminationDistance`: 0.4
- `multiclip_v_cmd_range`: [0.38, 0.83] (auto-derived from 3 clips × ±15%)

### S6
- `task`: `HumanoidImVICCmdMultiClip`
- `motion_file`: `sample_data/amass_walking_3clips_speedspaced.pkl` (same as S5)
- `terminationDistance`: 0.6
- All else: identical to S5

## 5. Motion Data Preparation

**Source**: `sample_data/amass_isaac_walking_primitive.pkl` (used by AMASS_MULTICLIP_FWD)

**Procedure**:
1. For every clip in source pkl, compute natural forward velocity:
   ```python
   v_x_nat = (root_pos[T-1, 0] - root_pos[0, 0]) / (T / fps)
   ```
2. Filter: forward-only (`v_x > 0.3`), reasonable length (60–250 frames).
3. Hand-pick 3 clips spaced at:
   - Slow: closest to ~0.45 m/s
   - Medium: closest to ~0.58 m/s
   - Fast: closest to ~0.72 m/s
4. Save as `sample_data/amass_walking_3clips_speedspaced.pkl`.
5. Print verification: `clip_name, n_frames, v_x_nat` for each picked clip.

If AMASS pool insufficient, fall back to KIT_425 PERSONAL pool with same selection criterion.

## 6. Code Changes

**Touched files** (project rule: only VIC subclasses; do NOT modify base PHC files):
- `phc/env/tasks/humanoid_im_vic_cmd_retime.py` — add `init_at_clip_start` flag
- `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` — add `init_at_clip_start` flag

**Change**: In `_sample_ref_state` of each class:
```python
# Read flag from yaml (in __init__)
self._init_at_clip_start = env_cfg.get("init_at_clip_start", False)

# In _sample_ref_state, replace _sample_time call:
if self._init_at_clip_start:
    motion_times = torch.zeros(n, device=self.device)
else:
    motion_times = self._sample_time(motion_ids)
self._motion_start_times[env_ids] = motion_times
```

**Cycle reset (mid-episode) unchanged** — still uses random t₀ via `_sample_time`. Only episode reset is forced to t₀=0.

**Affected init paths**:
- fallInit (30% envs): unchanged — fallen pose, recovery curriculum
- hybridInit (50% envs): unchanged — random t₀ mid-trajectory
- normal init (20% envs): forced to t₀=0 (standing)

## 7. Implementation Order

1. **Motion data prep** (§5) — produce `amass_walking_3clips_speedspaced.pkl`. Verify natural speeds.
2. **Code change** (§6) — add `init_at_clip_start` flag to both Retime and MultiClip task classes.
3. **Yaml + sbatch authoring** — 6 files in `exp_config/forward_walking/260427_VIC4_VCMD/`.
4. **Local smoke test** — S4 with `num_envs=4`, 5 minutes, verify no crash and motion lib loads.
5. **Git commit + push** — atomic commits per task per project convention.
6. **Server pull + sbatch ×3** — submit to idx0/idx1/idx2 with motion_lib `file_system` sharing fix already in place.
7. **Monitoring** — first 5 min for crash detection, then check Ep progress periodically.
8. **Side-track during 24h wait** — see §8.

## 8. Side-Track: Long Seamless Clip (during training)

**Goal**: Eliminate cycle reset visual jumps for demo (option D from brainstorm).

**Procedure**:
1. Analyze KIT_11 walk clip (162 frames):
   - Detect heel-strike events on root vertical velocity zero-crossings
   - Identify gait cycle period (~27 frames = 0.9s typical)
2. Extract one full gait cycle from clip → `cycle_segment[t_HS_1 : t_HS_2]`
3. Loop the cycle phase-aligned to produce a 60-second seamless clip
4. Save as `sample_data/amass_walking_kit11_seamless_60s.pkl`
5. Validate visually: render reference motion only (no policy), confirm no visible jumps at loop boundaries.

**Usage**: Demo (post-training) inference uses this long clip. Trained policy (random-cycle) handles long-loop trivially since long-loop is a SUBSET of training distribution (zero phase jump < random phase jump).

**Time budget**: 2–3 hours during 24h training wait.

**Fallback if option D fails**: Implement option C (phase-matched cycle reset, ~30 lines code in `_resample_vcmd_on_cycle`). Defer to next iteration if time-constrained.

## 9. Evaluation Plan

### 9.1 Quantitative
1. `rsync` checkpoints + Ep10000 milestone from server.
2. Run `scripts/test_termination_stats.py --slot S4/S5/S6 --num_envs 32 --max_resets 1500`.
3. Build matrix:
   | Slot | greedy eps_len | drift% | clip_end% | max_episode% | per-frame reward |
   |---|---|---|---|---|---|
4. Pass criteria:
   - eps_len ≥ 290 / 300 (97% ceiling)
   - drift% < 5%
   - per-frame reward ≥ 3.0 (similar to VIC4)

### 9.2 Visual (user judgment)
1. `python scripts/vis_vic4_vcmd.py --slot S{4,5,6} --num_envs 8 --no_virtual_display`
2. v_cmd arrows + cycle hop logging (carry-over from `vis_kit425_cont.py`)
3. Pass criteria:
   - Humanoid walks upright, gait visually matches reference
   - v_cmd arrow color/length variation reflected in actual walking speed
   - Cycle reset visually acceptable (some awkwardness OK; no collapse)

### 9.3 Failure modes to watch for
- Goodhart (flipped/crawling at eps_len=ceiling) — fix: revisit fall_termination
- Drift termination > 10% — fix: revisit termDist or multiclip variance
- Stand-still init failure (humanoid can't start moving) — fix: revisit init_at_clip_start interaction with fallInit

## 10. Post-experiment Analysis (`02_research_dev/260428_vic4_vcmd_results.md`)

**Required sections**:
1. Slot comparison matrix (eps_len, drift, reward)
2. Per-slot training curves (wandb or log parsing)
3. **🔍 cmd_tracking_w review** (user-flagged for analysis):
   - Current: 0.0 (retime carries v_cmd signal)
   - Question: Does humanoid visibly respond to v_cmd in viewer?
   - If response weak → ablate cmd_tracking_w ∈ {0.1, 0.2, 0.3} in next batch
   - If response strong → keep 0.0
4. Multiclip effect (S4 vs S5):
   - Per-frame reward comparison
   - Visual gait quality difference
5. termDist effect (S5 vs S6):
   - Stability vs visual quality trade-off
6. Demo readiness checklist:
   - Press start → walk → reset to standing flow works in viewer
   - Long seamless clip (option D) reduces cycle reset awkwardness
7. Next milestone decision (keyboard demo / different ablation / fall_termination batch)

## 11. Risk Mitigation

| Risk | Mitigation |
|---|---|
| All 3 slots fail | S4 (insurance, single clip) very likely to succeed; even partial success informs next batch |
| Server motion_lib hang | Already fixed (`mp.set_sharing_strategy('file_system')` in commit b95f87a) |
| 3-clip pkl prep fails | Fall back to 2 clips (still tests multiclip vs single) |
| Long seamless clip too complex | Fall back to accept cycle reset awkwardness, defer option D |
| Code change breaks existing tasks | Both Retime and MultiClip changes gated behind yaml flag (default False) — no behavior change for old experiments |

## 12. Out of Scope (this batch)

- Mode B (cycle-boundary v_cmd hop) — deferred until S4/S5/S6 pass
- Mode C (smooth v_cmd ramp) — same
- Keyboard demo wrapper — next milestone after pass
- Stop-and-go walking (v_cmd including 0) — explicitly out per project memory
- fall_termination wiring — held in reserve for next batch if these fail
- Action smoothness penalty — explicitly rejected by user (feedback memory)

## 13. Files Touched (commit summary)

```
sample_data/amass_walking_3clips_speedspaced.pkl       (NEW)
sample_data/amass_walking_kit11_seamless_60s.pkl       (NEW, side-track)
phc/env/tasks/humanoid_im_vic_cmd_retime.py             (modified, init_at_clip_start)
phc/env/tasks/humanoid_im_vic_cmd_multiclip.py          (modified, init_at_clip_start)
exp_config/forward_walking/260427_VIC4_VCMD/
  ├── README.md
  ├── env_im_walk_vic_S4.yaml
  ├── env_im_walk_vic_S5.yaml
  ├── env_im_walk_vic_S6.yaml
  ├── im_walk_vic.yaml
  ├── train_S4_gpu0.sh
  ├── train_S5_gpu1.sh
  └── train_S6_gpu2.sh
scripts/vis_vic4_vcmd.py                                 (NEW, viewer wrapper)
scripts/test_termination_stats.py                        (modified, +S4/S5/S6 slots)
docs/superpowers/specs/2026-04-27-vic4-vcmd-3slot-design.md  (this file)
docs/superpowers/plans/2026-04-27-vic4-vcmd-3slot-plan.md    (next step)
```

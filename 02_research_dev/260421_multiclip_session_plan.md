# Multi-Clip Session Plan (2026-04-21)

**Decision**: skip the Baseline-C (single-clip retime R2) step and jump straight to the multi-clip framework described in `260421_action_plan_and_multiclip_architecture.md` Phase 5a. Reasoning: (1) Baseline A finished and Baseline B data is sufficient to validate the retiming concept, (2) the multi-clip library has materially better speed coverage than the single walk-to-stop clip, (3) running two multi-clip variants in parallel answers "does clip-selection alone work?" vs "is retime on top of clip-selection additive?" in one shot.

Baseline B (job 3852) was cancelled to free idx1. Baseline A remains saved as `output/AMASS_CMD_B.pth` for Phase 4 reference comparison later.

---

## 1. What is running

Both jobs submitted 2026-04-21.

| Job | GPU | Exp name | Variant | Key parameters |
|---|---|---|---|---|
| 3853 | **idx0** | `AMASS_MULTICLIP_NR` | Clip selection only, no retime | `multiclip_retime_enabled=False`, per-env scale fixed at 1.0 |
| 3854 | **idx1** | `AMASS_MULTICLIP_RT` | Clip selection + retime | `multiclip_retime_enabled=True`, s = clamp(v_cmd / v_nat_clip, 0.9, 1.1) |

Shared settings (both jobs):
- Task class: `HumanoidImVICCmdMultiClip` (new; subclass of `HumanoidImVICCmdRetime`)
- Dataset: `sample_data/amass_isaac_walking_primitive.pkl` (**50 AMASS walking clips**, per-clip v_nat range 0.23–0.99 m/s)
- v_cmd sample: U(0.3, 0.9) m/s per env at reset
- Clip selection rule: argmin<sub>i</sub> |v_nat[i] − v_cmd| (single pass, no gating net)
- `cmd_tracking_w = 0.0` — no explicit cmd reward; signal comes from the selected/retimed reference
- `max_epochs = 20000`, `save_frequency = 2500`, `num_envs = 512`
- VIC Stage 2, 4-group CCF, Phase obs, sigma -1.0 (same as Baselines A/B)

Expected ETA: ~16 h → finish around early morning 2026-04-22.

---

## 2. Datasets in play

| File | Role this session |
|---|---|
| `sample_data/amass_isaac_walking_primitive.pkl` | **Main training dataset** (50 clips, total 333 s, per-clip mean speeds 0.21–0.77 m/s, middle-segment means 0.23–0.99). |
| `sample_data/amass_isaac_walking_primitive_v_nat.json` | **New** — per-clip v_nat metadata (duration, fps, v_mean_whole, v_mean_mid, v_p50/p90/max). Loaded at env-task init to drive clip selection. |
| `sample_data/amass_isaac_walking_forward_single.pkl` | **Not in use** this session; retained for Baseline A comparison and potential diagnostic runs later. |
| `sample_data/h5_motion_library*.pkl` | Parked. |
| `sample_data/amass_isaac_gender_betas*.pkl` | Parked (personalization deferred). |

---

## 3. Pre-computation that happened this session

A one-time script (run locally, not a training job) computed per-clip v_nat from `primitive.pkl`:

- Per-frame horizontal velocity magnitude, then whole-clip mean and middle-segment mean (skip first/last 0.5 s).
- Output: `sample_data/amass_isaac_walking_primitive_v_nat.json`.
- Summary of v_mean_mid across 50 clips: min 0.230, p50 0.538, max 0.992 m/s.

This JSON is read by `HumanoidImVICCmdMultiClip.__init__` and converted to a `[num_clips]` tensor aligned with the motion-lib's `curr_motion_keys` ordering, so that `argmin` selection on `v_nat[:]` maps directly to a motion_id.

---

## 4. Code added this session

- `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` — new task class. Overrides:
  - `_sample_ref_state(env_ids)`: samples v_cmd, picks nearest-v_nat clip, writes `_sampled_motion_ids`, `_motion_scale`, `_current_cmd`, then delegates to `super()._sample_ref_state`.
  - `_resample_motion_scale` / `_resample_cmd`: no-op (those are handled inside `_sample_ref_state` for this class).
- `phc/utils/parse_task.py` — registered the new task class alongside existing VIC tasks.

No changes to the policy / value / discriminator networks. One shared network handles all 50 clips — the clip identity is encoded via the reference pose/velocity observed each timestep, not via any architectural branching. (See `260421_action_plan_and_multiclip_architecture.md` §3 for the full explainer.)

---

## 5. What we expect to learn from these two runs

1. **Does clip selection alone (NR) reach VIC4-baseline-quality tracking?** The hypothesis is that with 50 clips covering [0.23, 0.99] m/s, the nearest-clip mismatch is already small (<0.05 m/s worst case for evenly-spaced clips), so a single conditional policy should be able to reach high success rate across the whole command range — without any time-scaling trick.

2. **Does retiming on top (RT) measurably close the residual gap?** If NR already succeeds cleanly, retime becomes redundant. If NR plateaus below the baseline, RT should pick up the missing percent.

3. **How does AMP cope with a mixed 50-clip positive distribution?** With `cmd_tracking_w=0`, the only velocity-style pressure comes from (a) the imitation reward vs the selected clip and (b) the AMP discriminator trained on all 50 clips. Monitoring whether AMP accuracy stays in a healthy range (not collapsing to ~0 or ~1) will indicate whether the mixed-clip positives cause discriminator confusion.

Post-training, all three checkpoints (CMD_B@30k on single clip, MULTICLIP_NR, MULTICLIP_RT) will be evaluated together via fixed-command test-greedy across bins matching their respective valid ranges. The comparison write-up will land in `02_research_dev/260422_multiclip_comparison_analysis.md`.

---

## 6. Risks / things to watch

- `self._motion_lib.curr_motion_keys` alignment. If motion_lib sorts / filters the input pkl's keys, the JSON-derived v_nat tensor must follow that order. The task-class init raises `RuntimeError` on any missing key, so a startup failure will surface immediately.
- Episode length is fixed at 300. The shortest primitive clip is 3.7 s ≈ 110 frames @ 30 fps → policy will hit cycle boundaries mid-episode. Parent classes already handle `cycle_motion=True`, but combined with retiming the cycle-wrap discontinuity (noted in CLAUDE.md "bunny hop") may show up; worth watching in success rate by clip.
- `retime_scale_range` is [1.0, 1.0] for the NR variant. The Retime parent's velocity-multiply code path runs with s=1 so it's a no-op on velocities; correct but the context-manager overhead is still incurred. Acceptable for this session; can be short-circuited later if it matters.
- AMP demo velocity scaling uses `U(0.9, 1.1)` in the parent class for both variants. For NR specifically this is a mild inconsistency (rollout always has s=1, but demos see s'~U(0.9,1.1)). We tolerate this in the first run; if it becomes load-bearing, a follow-up can make demo scaling variant-aware.

---

## 7. Decision gates after this session

- **Both runs reach ≥80 % success across v_cmd bins at test-greedy**: good — proceed to Phase 6 (turning, time-varying commands).
- **NR succeeds, RT does not add much**: retime is redundant for a well-populated library — drop it from future multi-clip work.
- **RT succeeds, NR underperforms**: retime matters even with a dense library — keep it in the multi-clip recipe.
- **Neither succeeds**: fall back to Phase 5b diagnosis — likely suspects are AMP/cycle interactions on the walk-to-stop-like clips in the primitive set.

---

*Authored 2026-04-21 at the start of the multi-clip session. Status to be updated after 3853/3854 finish.*

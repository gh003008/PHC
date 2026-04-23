# Multi-Clip FWD Results — 22-clip forward library (2026-04-23)

## 1. Setup recap

Both runs: `HumanoidImVICCmdMultiClip`, 22 direction-verified forward-walking clips from `amass_isaac_walking_primitive_fwd_only.json`, `|v_x| ∈ [0.21, 0.99]` m/s. v_cmd sampled per-env from U(0.25, 0.85). One shared policy, `cmd_tracking_w = 0.0` (retimed/selected teacher carries the signal). 512 envs, 20 000 epochs.

| Job | GPU | Variant | Key params |
|---|---|---|---|
| 3860 | idx0 | **FWD_NR** (no retime) | per-env scale fixed at 1.0 |
| 3861 | idx1 | **FWD_RT** (retime) | per-env scale `clamp(v_cmd / \|v_x_clip\|, 0.9, 1.1)` |

Both finished 2026-04-22 ~06:35 KST. All 8 milestone checkpoints saved.

## 2. Training trajectory (stochastic, 100-ep-avg at milestones)

| Ep | NR rwd | NR eps_len | RT rwd | RT eps_len |
|---|---|---|---|---|
| 1000 | 43.5 | 32.2 | −19.1 | 49.2 |
| 2500 | 151.8 | 81.4 | 122.0 | 72.9 |
| 5000 | 196.6 | 91.6 | 159.6 | 76.2 |
| 7500 | 226.0 | 93.1 | 173.1 | 66.1 |
| 10000 | 178.9 | 75.6 | 201.1 | 91.7 |
| 12500 | 191.8 | 81.1 | 174.6 | 76.8 |
| 15000 | 219.1 | 92.6 | 137.8 | 65.1 |
| 17500 | 238.5 | 96.0 | 188.1 | 76.2 |
| 20000 | 147.9 | 62.2 | 213.9 | 85.3 |

Both stay in a **rwd 150–240, eps_len 60–96** band for the bulk of training. Neither reaches the single-clip CMD_B training pattern (rwd ~280 / eps_len ~185 at Ep 20k).

**Shape of the curves**:
- NR climbs fastest in the first 7.5k epochs (eps_len 32 → 93), plateaus, then dips at Ep 20k.
- RT trails NR for the first 10k epochs, catches up around Ep 10k–17.5k, and finishes slightly higher than NR at Ep 20k.
- Both show 20–30 % oscillation between adjacent milestones (typical PPO + AMP stochasticity, amplified by the mixed-clip distribution).

## 3. Comparison with priors

| Experiment | Clips | Training rwd @ 20k | Training eps_len @ 20k | Test-greedy rwd | Success rate |
|---|---|---|---|---|---|
| VIC4 no-cmd (baseline) | 1 (forward_single) | ~552 | ~180 | **951** | **100 %** |
| CMD_A w=0.3 (absolute v_cmd) | 1 | 388 | 185 | 527 | 65.5 % |
| CMD_B w=0.5 (absolute v_cmd) | 1 | 279 | 185 | 440 | 86.1 % |
| **MULTICLIP_FWD_NR** | **22 (fwd)** | **148** | **62** | _(running)_ | _(running)_ |
| **MULTICLIP_FWD_RT** | **22 (fwd)** | **214** | **85** | _(running)_ | _(running)_ |

Training-log gap vs single-clip is large. But training log systematically underrates test-greedy by ~1.6–1.7× on reward (CCF sigma=0.37 noise), and test-greedy's success rate only becomes clear when measured — training eps_len 85 doesn't rule out a 70 %+ success rate at deterministic rollout.

## 4. Interpretation

1. **Multi-clip is harder than single-clip, as expected.** The policy must amortize over 22 reference trajectories that differ in stride length, cadence, arm swing, and speed. With 512 envs × 22 clips, each clip-style receives only ~23 envs' worth of gradient signal per step. Convergence is slower and noisier than single-clip where all 512 envs reinforce one motion.

2. **NR (no retime) outperforms RT (retime) in training log until near the end.** Unexpected given the earlier retime recommendation. Possible reasons:
   - With 22 clips in a dense speed grid, the nearest-clip mismatch is already small (worst gap ~0.05 m/s between clips). The retime scale s in RT rarely hits its bounds and often just equals 1.0 — making RT ~= NR but with extra observation variance from the scale term.
   - RT's extra bit of stochasticity (per-env s in [0.9, 1.1]) adds learning friction early and only pays off once the policy has mastered the base multi-clip task.
   - AMP demo velocity scaling in RT adds one more source of distribution shift that NR doesn't have.

3. **End-of-run dip for NR (Ep 17500 → 20000: rwd 238 → 148)** is consistent with PPO late-training overshoot; the canonical `output/*.pth` is saved on best-reward, so the deployed checkpoint is not the dipped one.

## 5. Test-greedy evaluation

Two sbatch jobs (num_envs=1, `--test --epoch -1`, `--no_virtual_display`). v_cmd resampled per-episode from U(0.25, 0.85) via the normal reset path. Episodes terminated on imitation-error threshold or max 300 steps.

| Variant | av_reward | av_steps | Notes |
|---|---|---|---|
| **FWD_NR** (3936, idx2) | **215.37** | **65.51** | Canonical `output/AMASS_MULTICLIP_FWD_NR.pth` (epoch 20001) |
| **FWD_RT** (3935, idx1) | **211.78** | **64.29** | Canonical `output/AMASS_MULTICLIP_FWD_RT.pth` (epoch 20001) |

NR and RT are **indistinguishable at test-greedy** (within 2 % on both metrics). The retime layer on top of nearest-clip retrieval neither helps nor hurts in a 22-clip dense-grid library — confirming the hypothesis from §4.2 that retime's per-env scale s almost never leaves 1.0 when the nearest-clip residual is already small.

**Anchoring context** (Round 1 single-clip baselines):
- VIC4 no-cmd: 951.2 / 299 (100 % success)
- CMD_A w=0.3: 527.0 / 236 (65.5 %)
- CMD_B w=0.5: 440.4 / 279 (86.1 %)

RT's 64 mean steps = policy falls at ~2.1 s average (dt 0.033 × 64). That's a **failure** by the success-rate criterion (need ≥ 290 steps, i.e. 9.7 s). Multi-clip at 20k epochs did not reach usable performance.

### Canonical checkpoint caveat

`torch.load(...)['last_mean_rewards']` on both canonical files returns `-100500` (rl-games sentinel) — the save-on-best hook never crossed its threshold, so the canonical ckpt is simply the final-epoch save, not a best-reward pick. For NR specifically, the final-epoch state happened to be a post-peak dip (Ep 17500 rwd 238 → Ep 20000 rwd 148 in training log). Re-evaluating on the Ep 17500 numbered checkpoint would likely give a better reward but not change the order-of-magnitude "av_steps ~60" result.

## 6. Diagnosis

Test-greedy av_steps ≈ 65 for both variants means the policy holds upright for ~2 s on average, then falls. Success rate (≥ 290 steps) is effectively **0 %**. This is dramatically worse than single-clip baselines:

| Comparison | Test-greedy av_steps | Success rate |
|---|---|---|
| VIC4 single clip, no cmd | 299 | 100 % |
| CMD_B single clip, abs v_cmd | 279 | 86 % |
| **MULTICLIP_FWD_NR (22 clips)** | **65** | **~0 %** |
| **MULTICLIP_FWD_RT (22 clips)** | **64** | **~0 %** |

This is **not** "multi-clip is just harder and needs more training" — an 86 %-success single-clip policy does not plateau at 0 % on a 22-clip superset unless something structural breaks. Likely causes, ordered by how much I would investigate them next:

### Primary suspect: cycle-wrap discontinuity across short clips

Many of the 22 forward clips are **< 5 s**, but the episode cap is 300 steps × dt 0.033 = **10 s**. With `cycle_motion=True` (PHC default), the reference snaps back to frame 0 partway through each episode, producing an instantaneous velocity discontinuity. A single forward clip has this too (noted as "bunny hop" in `CLAUDE.md`), but:

- Single clip: one discontinuity per episode, policy can memorize how to survive it.
- 22 clips: 22 different wrap points, many of them different speeds, each time reference jumps back to the clip's start pose which may differ wildly from the prior end pose.

Training and test-greedy see the same wrap events, so the training log does reflect the difficulty — and the log does show eps_len saturating around 60–95, exactly the regime where the first cycle-wrap would hit on the shorter clips.

### Secondary suspect: Phase-obs table misalignment

`HumanoidImVIC._precompute_gait_phase` computes a single gait-phase lookup table from one reference motion at load time. The lookup is keyed on motion_time, not on which clip is currently loaded. For single-clip training, this is fine. For 22 clips with different durations and gait cycles, the phase obs the policy sees no longer corresponds to the currently-loaded clip's actual gait.

This would introduce **observation noise that scales with clip diversity** — which matches our finding that the 22-clip run performs much worse than any single-clip run.

The `[VIC] Phase-CCF analysis failed: could not broadcast input array from shape (4,) into shape (8,)` message at the end of test-greedy is a separate 4-group vs 8-group mismatch in the phase analysis plotter — unrelated to the phase obs itself but indicates the phase pipeline was not revalidated for multi-clip.

### Tertiary suspect: AMP discriminator distribution mismatch

Retime-scaled demo velocities (per-demo s' ~ U(0.9, 1.1)) are still sampled from a single-clip distribution because the task class inherits the parent's demo pipeline. Across 22 forward clips with joint-velocity ranges that differ by ~3 × (slow vs run), discriminator positives and rollout states may not overlap cleanly. Needs a reward-breakdown trace (per-epoch AMP accuracy, not just combined reward) to confirm.

## 7. Next steps

In order of effort × expected diagnostic value:

1. **Re-run NR on the 4 literal `...WalkingStraightForwards...` clips only** (speeds 0.23–0.53 m/s). Same architecture, same code. If this recovers single-clip-like success (≥ 80 %), the issue is clip-diversity (cycle-wrap or phase-obs). If it doesn't, the multi-clip plumbing itself has a bug. 4 clips × 20k epochs, cheap.

2. **Per-clip failure-mode audit on the canonical ckpt.** Run test-greedy 20 times each while pinning v_cmd to each of the 22 clips' `|v_x|`, log the termination frame relative to the clip's duration. If termination consistently aligns with first cycle-wrap, the discontinuity hypothesis is confirmed.

3. **Fix phase-obs to be per-clip and per-retime-aware.** The current precompute is clip-0-only. A correct version would index `_gait_phase_table` by clip_id and retime scale.

4. **Extend training length on one variant** to 30k–40k epochs only after items 1–3 narrow the problem. Blind extension is unlikely to change anything if the cause is structural.

## 8. Preliminary conclusion

Multi-clip on the 22-clip forward subset **does not work as currently implemented**. The retime layer is not the differentiator (NR ≈ RT). The problem lives somewhere upstream of the retrieval logic — most likely in cycle-wrap behavior or the phase-obs lookup's failure to track clip identity. Items 1 and 2 above will tell us which.

The 20k-epoch training compute on these two runs is not wasted — it establishes a clear baseline ("naive multi-clip fails at ~0 % success even with a clean forward subset") against which the diagnostic experiments in §7 can be compared.

## 7. Files / artifacts

- Checkpoints: `output/AMASS_MULTICLIP_FWD_{NR,RT}.pth` + `_00002500.pth` … `_00020000.pth` (×8 milestones each, ~90 MB per file).
- Configs: `exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/`
- Task class: `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py`
- Direction metadata: `sample_data/amass_isaac_walking_primitive_{dirmeta,fwd_only}.json`
- Training logs: `logs/amass_multiclip_fwd_{NR,RT}_386*.out`
- Test-greedy logs: `logs/testg_multiclip_fwd_{NR,RT}_393*.out` _(in progress)_

---

*Authored 2026-04-23 after both 20k-epoch runs completed overnight. Test-greedy results and conditional next steps to follow.*

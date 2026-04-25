# KIT_425 Continuous Walking — 3-Slot Training Design

**Date**: 2026-04-26
**Author**: Youn Jimin (with Claude)
**Predecessor**: `260424_KIT425_PERSONAL_VCMD/` (3-way ablation A/B/C, ep_len 44-59% of ceiling)
**Status**: Approved (pending user review of this spec)

---

## 1. Motivation

The previous KIT_425 PERSONAL ablation found:

| Result | All 4 ckpts |
|---|---|
| Mean episode length | 60-81 / 137 frames (44-59% of ceiling) |
| Termination cause | **98% drift > 0.25 m from reference**, **0% fall** |
| Curriculum switch (Ep 10 000) | NOT the dominant cause — same plateau pre/post switch |

The bottleneck is **terminationDistance=0.25 m**, not policy quality. The humanoid stays upright but accumulates positional drift against the retimed reference until the tight 25 cm threshold trips. Loosening or disabling this termination should unlock substantially longer episodes.

The user's end goal is an **interactive Isaac Gym demo** where keyboard input drives a velocity command (v_cmd) and the humanoid walks indefinitely at that speed. To support that:
1. Continuous walking (no termination at clip end → `cycle_motion=True`)
2. Robustness to v_cmd changes during a rollout (cycle-aligned re-sampling)

This design tests both directions in one 3-slot batch.

---

## 2. Slot Plan

| Property | **Slot 1: TERM_OFF** | **Slot 2: CYCLE_BASE** | **Slot 3: CYCLE_VCMD_HOP** |
|---|---|---|---|
| **Hypothesis** | terminationDistance alone is the bottleneck; current policy class can reach ceiling once termination is loosened | cycle_motion=True enables indefinite walking; tests if v_cmd-conditioned policy stays stable across cycles | mid-rollout v_cmd changes don't break learning, demonstrating policy ready for interactive demo |
| `terminationDistance` | **disabled** (set to `100.0`) | **disabled** | **disabled** |
| `enableEarlyTermination` | True (height 0.15 only) | True (height only) | True (height only) |
| `cycle_motion` | False (matches B canonical) | **True** | **True** |
| `fallInitProb` | **0.0** (was 0.3) | **0.0** | **0.0** |
| `hybridInitProb` | **0.0** (was 0.5) | **0.0** | **0.0** |
| v_cmd range | continuous [0.40, 0.82] m/s | [0.40, 0.82] | [0.40, 0.82] |
| v_cmd resample | episode reset only | episode reset only | episode reset **+ every cycle boundary** |
| Retime | enabled, [0.7, 1.4] | enabled, [0.7, 1.4] | enabled, [0.7, 1.4] |
| `episode_length` | 300 (B canonical) | **3000** (long enough for many cycles) | **3000** |
| `max_epochs` | 20000 | 20000 | 20000 |
| Reward curriculum | switch @ Ep 10000 (B canonical) | switch @ Ep 10000 | switch @ Ep 10000 |
| All other env / learning yaml | identical to `env_im_walk_vic_kit425_personal_B.yaml` | identical, with overrides above | identical, with overrides above |

**Why these knobs and not others** (YAGNI applied):
- Reward weights, k_pos, AMP disc weights, VIC stage, CCF groups, MLP arch — kept identical to B canonical so any improvement is attributable to the changes above.
- Retime kept (not removed) — within [0.7, 1.4] it's a known-working component; removing would conflate effects.
- v_cmd range kept narrow [0.40, 0.82] — backward / stop / extended range is a separate future milestone.

**Episode length increase to 3000 for Slot 2/3**: with cycle_motion=True and termination disabled, episodes can run indefinitely. Cap at 3000 control steps (~100 sec, ~18 cycles of 5.4s motion clips) to avoid stuck/looping envs hogging GPU.

---

## 3. Code Changes

### 3.1 Config files (yaml only — no Python changes for Slot 1 or Slot 2)

Three new env yaml files under `exp_config/forward_walking/260426_KIT425_CONT/`:
- `env_im_walk_vic_kit425_cont_S1_termoff.yaml`
- `env_im_walk_vic_kit425_cont_S2_cyclebase.yaml`
- `env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml`

Each starts as a copy of `env_im_walk_vic_kit425_personal_B.yaml` with the overrides from the table above.

The learning yaml is also snapshotted into the new directory per the project convention of backing up all configs before training:
- `exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml` — copy of `im_walk_vic_kit425_personal.yaml`, no content changes (identical training hyperparameters across all 3 slots).

### 3.2 Cycle-boundary v_cmd resampling (Slot 3 only)

Modify `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` (the VIC subclass — does not touch base PHC files per project rule).

**Where**: extend `_compute_reset()` (or add a new hook called from there). Detect cycle boundary via `pass_time_motion_len` (already computed in parent `HumanoidImVIC._compute_reset`).

**What**: when an env crosses a cycle boundary in cycle-motion mode:
1. Re-sample v_cmd for that env from `multiclip_v_cmd_range`
2. Re-pick clip via the existing closest-clip logic
3. Recompute retime scale `s = clamp(v_cmd / v_nat, s_lo, s_hi)`
4. Update internal `_cmd_v` tensor

**Gating**: behavior gated by a new env yaml flag (default off):
```yaml
multiclip_resample_on_cycle: True
```
Slot 1, 2 leave this False (or absent). Slot 3 sets True. Keeps Slot 1/2 yaml-only.

**Code change scope**: ~30-50 lines added to `humanoid_im_vic_cmd_multiclip.py`. New private method `_resample_vcmd_on_cycle(env_ids)` that mirrors the existing reset-time sampling logic, called from a hook in `_compute_reset` after the parent runs.

### 3.3 No changes to base PHC code
- `phc/env/tasks/humanoid_im.py`: untouched
- `phc/env/tasks/humanoid_im_vic.py`: untouched
- `phc/learning/*.py`: untouched
- `phc/run.py`: untouched

---

## 4. Server Execution Plan

### 4.1 Resources
- 3 jobs, 1 GPU each (idx0, idx1, idx2 of the 4×A5000 server)
- Memory: 15G/job; CPUs: 8/job; time limit: 36h/job (extended from 24h to allow margin)
- Total: 3 GPUs out of 4 used in parallel

### 4.2 sbatch templates

Three scripts under `exp_config/forward_walking/260426_KIT425_CONT/`:
- `train_S1_termoff_gpu0.sh`
- `train_S2_cyclebase_gpu1.sh`
- `train_S3_vcmdhop_gpu2.sh`

Each follows the proven template from `train_personal_B_gpu1.sh`:
```bash
#SBATCH -J kit425_cont_<slot>
#SBATCH -p idx<N>
#SBATCH --gres=gpu:idx<N>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd ~/PHC
python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_<slot>.yaml \
  --cfg_train exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_CONT_<slot>
```

### 4.3 Pre-flight checklist (run on server before sbatch)
1. `git pull origin Jimin` — fetch new env yamls and code change
2. `conda activate phc` — verify env active
3. `python -c "import smpl_sim, isaacgym; print('OK')"` — verify deps importable
4. Smoke test Slot 3 code change locally first (num_envs=2, max_epochs=10) to catch import / API errors before submitting 36h job
5. `sbatch train_S1_termoff_gpu0.sh && sbatch train_S2_cyclebase_gpu1.sh && sbatch train_S3_vcmdhop_gpu2.sh`
6. `squeue -u $USER` — confirm 3 jobs queued
7. Tail logs: `tail -F logs/kit425_cont_*.out`

### 4.4 Failure modes and mitigations
- **Slot 3 import error / NameError in new code**: smoke test in step 4 above catches this before sbatch submission.
- **OOM**: A5000 has 24GB. B canonical used ~12GB at num_envs=512. With cycle_motion=True trajectories may be longer in memory but no architectural change → expect ≤16GB. If OOM, drop num_envs to 256.
- **Job pending due to GPU contention**: use generic `--gres=gpu:1` instead of `--gres=gpu:idxN:1` if a specific GPU is held.
- **Diverging policy / exploding loss**: same risk as B canonical run; if any slot diverges in the first 1000 epochs, kill that job and review the new code path.

---

## 5. Evaluation Plan

After all 3 jobs finish (~24-36h):

### 5.1 Quantitative
- Run `scripts/test_termination_stats.py --slot S1/S2/S3 --epoch -1 --num_envs 32 --max_resets 1500` for each slot
- Also re-run baseline `scripts/test_termination_stats.py --slot B` for reference
- Report 4-way table: % fall, % drift (if applicable), % clip_end, % max_episode, mean ep_len, max ep_len
- Plot training curves with the existing `scripts/plot_kit425_personal_training_v2.py` adapted to the new log filenames

### 5.2 Success criteria
- **Slot 1 success**: mean ep_len ≥ 130 (≥95% of 137 ceiling) → confirms terminationDistance was the bottleneck
- **Slot 2 success**: mean ep_len > 200 (multiple cycles completed) AND fall rate < 5%
- **Slot 3 success**: same as Slot 2 AND v_cmd transition does not collapse the policy (post-transition ep_len within 20% of pre-transition)

### 5.3 v_cmd tracking quantification (if Slot 2/3 succeed)
- Greedy eval at fixed v_cmd ∈ {0.40, 0.55, 0.70, 0.82}; measure actual humanoid forward speed; report mean tracking error.
- This is a sanity check, not a primary success criterion.

### 5.4 Documentation
- Write result analysis to `02_research_dev/260427_kit425_continuous_walking_results.md` (or appropriate date) following the existing PHC analysis-doc format.

---

## 6. Out of Scope (deferred to next milestone)

- Interactive keyboard demo (`scripts/keyboard_demo.py` — explicitly deferred per user request: "let's first make it learn first")
- v_cmd range extension (stop, backward, > 0.82 m/s)
- Random-timing v_cmd transitions (option U from brainstorm) — deferred until cycle-aligned (W) is validated
- Recovery skill from real falls — sacrificed by disabling fallInit; revisit if real-world demo needs it

---

## 7. Memory / Decision Log

Project memory `project_continuous_walking_design.md` records the following decisions, all approved by user during brainstorm:
- Option **B** for slot allocation (1 termination + 2 continuous variants)
- v_cmd: continuous [0.40, 0.82], NOT discrete
- fallInit / hybridInit: disabled (recovery sacrificed deliberately, since terminationDistance also off makes recovery grace redundant)
- mid-episode transition: cycle-aligned (W), not random-timing (U) — prioritizing success rate over deployment-distribution match for first iteration

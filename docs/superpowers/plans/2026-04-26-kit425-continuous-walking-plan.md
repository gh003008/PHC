# KIT_425 Continuous Walking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-26-kit425-continuous-walking-design.md` (commit `f778b3f`)

**Goal:** Produce 3 server-runnable training jobs (yaml + sbatch + Slot 3 Python change) that test loosened terminationDistance + cycle_motion + cycle-aligned v_cmd resampling on KIT_425 PERSONAL.

**Architecture:** Three env yamls (S1/S2/S3) deriving from `env_im_walk_vic_kit425_personal_B.yaml` with targeted overrides. Slot 3 needs a ~30-line addition to `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` that adds an `_compute_reset` override + `_resample_vcmd_on_cycle` method, gated by a new yaml flag `multiclip_resample_on_cycle: True`. No base PHC files modified. Three sbatch scripts target `idx0/1/2` GPUs on `server1-jiminyoun`.

**Tech Stack:** PHC + IsaacGym + rl_games + AMP + VIC. Slurm scheduler on server. Python 3.8, torch 2.4.1+cu118, conda env `phc`.

---

## File Structure

**Created (yaml/scripts):**
- `exp_config/forward_walking/260426_KIT425_CONT/README.md` — describes the 3-slot batch
- `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml`
- `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml`
- `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml`
- `exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml` — snapshot of `im_walk_vic_kit425_personal.yaml`, no content edits
- `exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh`
- `exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh`
- `exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh`
- `scripts/smoketest_cont_s3.py` — local smoke test for Slot 3 code change

**Modified (Python):**
- `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` — adds `_resample_on_cycle` flag + `_compute_reset` override + `_resample_vcmd_on_cycle` method (~35 lines)

**Untouched:**
- All base PHC files (`humanoid_im.py`, `humanoid_im_vic.py`, `humanoid_im_vic_cmd.py`, `humanoid_im_vic_cmd_retime.py`, `phc/learning/*.py`, `phc/run.py`)

---

## Task 1: Create directory + README

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/README.md`

- [ ] **Step 1: Create directory and README**

```bash
mkdir -p /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT
```

Then create `exp_config/forward_walking/260426_KIT425_CONT/README.md` with this content:

```markdown
# 260426 KIT_425 Continuous Walking — 3-Slot Batch

Spec: `docs/superpowers/specs/2026-04-26-kit425-continuous-walking-design.md`

| Slot | Env yaml | Sbatch | Hypothesis |
|---|---|---|---|
| S1 TERM_OFF | `env_im_walk_vic_kit425_cont_S1_termoff.yaml` | `train_S1_termoff_gpu0.sh` | terminationDistance disabling alone unlocks ceiling |
| S2 CYCLE_BASE | `env_im_walk_vic_kit425_cont_S2_cyclebase.yaml` | `train_S2_cyclebase_gpu1.sh` | cycle_motion=True enables continuous walking |
| S3 CYCLE_VCMD_HOP | `env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml` | `train_S3_vcmdhop_gpu2.sh` | cycle-boundary v_cmd resampling robust |

All three: terminationDistance disabled (100.0), fallInit/hybridInit OFF, v_cmd ∈ [0.40, 0.82], retime [0.7, 1.4], 20k epochs.

Predecessor: `260424_KIT425_PERSONAL_VCMD/` (3-way A/B/C ablation, all 44-59% of ceiling).
```

- [ ] **Step 2: No commit yet — combined commit at end of Phase A (Task 6)**

---

## Task 2: Slot 1 (TERM_OFF) env yaml

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml`

- [ ] **Step 1: Copy B canonical env yaml as base**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_B.yaml \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml
```

- [ ] **Step 2: Apply Slot 1 overrides via Edit tool**

In `env_im_walk_vic_kit425_cont_S1_termoff.yaml`:

Replace the `project_name` and `notes` lines:
```yaml
project_name: "PHC_Walk_KIT425_CONT"
notes: "Slot S1 TERM_OFF: terminationDistance disabled (100.0), fallInit/hybrid OFF. cycle_motion=False (B canonical clone with termination loosened)."
```

Replace `terminationDistance: 0.25` with:
```yaml
  terminationDistance: 100.0
```

Replace `fallInitProb: 0.3` with:
```yaml
  fallInitProb: 0.0
```

Replace `hybridInitProb: 0.5` with:
```yaml
  hybridInitProb: 0.0
```

- [ ] **Step 3: Verify the diff**

```bash
diff /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_B.yaml \
     /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml
```

Expected diff: only `project_name`, `notes`, `terminationDistance`, `fallInitProb`, `hybridInitProb` lines change. Everything else identical.

- [ ] **Step 4: No commit yet**

---

## Task 3: Slot 2 (CYCLE_BASE) env yaml

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml`

- [ ] **Step 1: Copy Slot 1 yaml as base**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml
```

- [ ] **Step 2: Apply Slot 2 overrides**

Replace `notes:` line with:
```yaml
notes: "Slot S2 CYCLE_BASE: S1 + cycle_motion=True + episode_length 3000. v_cmd at reset only."
```

Replace `cycle_motion: False` with:
```yaml
  cycle_motion: True
```

Replace `episode_length: 300` with:
```yaml
  episode_length: 3000
```

Replace `episodeLength: 300` with:
```yaml
  episodeLength: 3000
```

- [ ] **Step 3: Verify diff**

```bash
diff /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml \
     /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml
```

Expected: only `notes`, `cycle_motion`, `episode_length`, `episodeLength` changed.

- [ ] **Step 4: No commit yet**

---

## Task 4: Slot 3 (CYCLE_VCMD_HOP) env yaml

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml`

- [ ] **Step 1: Copy Slot 2 yaml as base**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml
```

- [ ] **Step 2: Apply Slot 3 overrides**

Replace `notes:` line with:
```yaml
notes: "Slot S3 CYCLE_VCMD_HOP: S2 + multiclip_resample_on_cycle=True. v_cmd resampled at every cycle boundary."
```

Add a new line in the env block (right after `multiclip_v_nat_path: ...`):
```yaml
  multiclip_resample_on_cycle: True
```

- [ ] **Step 3: Verify diff**

```bash
diff /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml \
     /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml
```

Expected: `notes` changed, one new line `multiclip_resample_on_cycle: True` added.

- [ ] **Step 4: No commit yet**

---

## Task 5: Snapshot learning yaml

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml`

- [ ] **Step 1: Copy without changes**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml
```

- [ ] **Step 2: Verify identical content**

```bash
diff /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
     /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml
```

Expected: no output (files identical).

- [ ] **Step 3: No commit yet**

---

## Task 6: Commit Phase A (configs)

- [ ] **Step 1: Stage and commit yamls + README**

```bash
cd /home/exolab/Documents/GitHub/PHC
git add exp_config/forward_walking/260426_KIT425_CONT/README.md \
        exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml \
        exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml \
        exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml \
        exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml
git commit -m "$(cat <<'EOF'
exp(kit425_cont): 3-slot env configs for continuous walking

S1 TERM_OFF disables terminationDistance + fallInit. S2 adds
cycle_motion=True. S3 adds multiclip_resample_on_cycle (Python
support added in next commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds; `git log -1 --stat` shows 5 new files.

---

## Task 7: Smoke test for Slot 3 cycle resample

**Files:**
- Create: `scripts/smoketest_cont_s3.py`

This is a fail-first test that exercises the new code path: it sets up an env with Slot 3 config + `multiclip_resample_on_cycle: True`, advances 200 steps with a fixed action policy (zeros), and asserts that `_current_cmd[0, 0]` (env 0's effective v_cmd) changes value at least once during the run (cycle boundary should trigger resample).

- [ ] **Step 1: Create the smoke test script**

Create `scripts/smoketest_cont_s3.py` with this content:

```python
"""Smoke test for Slot 3 cycle-boundary v_cmd resampling.

Verifies that with multiclip_resample_on_cycle=True, env._current_cmd[0, 0]
changes value at least once during a 200-step rollout (covering >= one
clip cycle of ~5.4s = 162 control steps).
"""
from __future__ import annotations
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for p in (_PHC_PKG, _PHC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
import torch


def main():
    cfg_env = 'exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml'
    cfg_train = 'exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml'

    sys.argv = [
        'run.py',
        '--task', 'HumanoidImVICCmdMultiClip',
        '--cfg_env', cfg_env,
        '--cfg_train', cfg_train,
        '--num_envs', '4',
        '--test', '--epoch', '-1',
        '--no_virtual_display', '--headless',
        '--experiment', 'KIT425_PERSONAL_B',  # reuse existing ckpt for the smoke test
    ]

    import argparse
    from phc import run as phc_run
    from phc.utils import config as phc_config

    # We piggyback on phc.run.main() to set up env+policy, but intercept
    # before the rl_games player loop. Patch BasePlayer.run() to step a
    # few times and return.

    from rl_games.common import player as rl_player

    captured = {}

    orig_run = rl_player.BasePlayer.run
    def shortened_run(self, *args, **kwargs):
        env = self.env
        captured['env'] = env
        # Step ~200 times with zero actions
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs.get('obs', obs)
        action_dim = env.action_space.shape[0]
        zero_action = torch.zeros(env.num_envs, action_dim, device=self.device)
        v_cmd_history = [env._current_cmd[0, 0].item()]
        for step in range(200):
            obs, rew, done, info = env.step(zero_action)
            v_cmd_history.append(env._current_cmd[0, 0].item())
        captured['v_cmd_history'] = v_cmd_history
        # Don't actually run the player loop — break out.
        return

    rl_player.BasePlayer.run = shortened_run

    # Run with patched BasePlayer.
    phc_run.main()

    history = captured.get('v_cmd_history', [])
    if not history:
        print('[SMOKE TEST] FAIL: no v_cmd history captured')
        sys.exit(1)
    initial = history[0]
    changed = any(abs(v - initial) > 1e-6 for v in history[1:])
    unique_vals = sorted(set(round(v, 4) for v in history))
    print(f'[SMOKE TEST] v_cmd history: {len(history)} steps')
    print(f'[SMOKE TEST] initial v_cmd: {initial:.4f}')
    print(f'[SMOKE TEST] unique values: {unique_vals[:10]}{"..." if len(unique_vals) > 10 else ""}')
    print(f'[SMOKE TEST] cycle resample triggered: {changed}')
    if not changed:
        print('[SMOKE TEST] FAIL: v_cmd did not change during 200 steps')
        sys.exit(1)
    print('[SMOKE TEST] PASS')
    sys.exit(0)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the test BEFORE the code change to confirm it fails**

```bash
source /home/exolab/miniconda3/etc/profile.d/conda.sh && conda activate phc && export PYTHONUNBUFFERED=1
cd /home/exolab/Documents/GitHub/PHC
timeout 300 python scripts/smoketest_cont_s3.py 2>&1 | tee /tmp/smoke_before.log
```

Expected: FAIL with `v_cmd did not change during 200 steps` (since cycle resample logic isn't implemented yet) — OR fail at config-load with "unrecognized key `multiclip_resample_on_cycle`" (the parent yaml-loader may or may not warn on unknown keys; either way we proceed to add the code that USES the key).

- [ ] **Step 3: No commit yet — wait for Task 8 code change**

---

## Task 8: Add cycle-resample code to multiclip task

**Files:**
- Modify: `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py`

- [ ] **Step 1: Read current state**

Read the full file. Note the `__init__` ends at line 104 (the print statement) and class methods continue from line 112 (`_resample_motion_scale`). The new flag goes at the top of `__init__` near `self._multiclip_retime_enabled`.

- [ ] **Step 2: Add `_resample_on_cycle` flag in __init__**

Find this block (around line 31-44):

```python
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._multiclip_retime_enabled = env_cfg.get("multiclip_retime_enabled", True)
        self._multiclip_v_cmd_range = env_cfg.get("multiclip_v_cmd_range", [0.25, 0.85])
```

Add a new line right after `self._multiclip_v_cmd_range`:

```python
        self._resample_on_cycle = env_cfg.get("multiclip_resample_on_cycle", False)
```

- [ ] **Step 3: Add `_compute_reset` override and `_resample_vcmd_on_cycle` method**

At the END of the `HumanoidImVICCmdMultiClip` class (after `_sample_ref_state` ends), add:

```python

    # ------------------------------------------------------------------
    # Slot 3: cycle-boundary v_cmd resampling
    # ------------------------------------------------------------------

    def _compute_reset(self):
        """Override: detect cycle boundary BEFORE parent runs cycle logic, so
        when we change motion_id here, parent's _sample_time / _global_offset
        update uses the NEW clip.
        """
        if self._resample_on_cycle and getattr(self, 'cycle_motion', False):
            time_now = (self.progress_buf * self.dt
                        + self._motion_start_times
                        + self._motion_start_times_offset)
            cycle_mask = time_now >= self._motion_lib._motion_lengths
            if cycle_mask.any():
                cycle_env_ids = torch.where(cycle_mask)[0]
                self._resample_vcmd_on_cycle(cycle_env_ids)
        super()._compute_reset()

    def _resample_vcmd_on_cycle(self, env_ids):
        """Sample new v_cmd, re-pick closest clip, recompute retime scale + obs cmd.

        Called at cycle boundary (motion_lib reaches end). Does NOT reset motion
        start times — that's handled by parent's cycle_motion logic which runs
        AFTER this. Updates self._sampled_motion_ids so parent picks the new
        clip's start time and global offset.
        """
        n = env_ids.shape[0]
        if n == 0:
            return
        v_lo, v_hi = self._multiclip_v_cmd_range
        new_v_cmd = torch.rand(n, device=self.device) * (v_hi - v_lo) + v_lo  # [n]

        # Re-pick closest eligible clip per env.
        v_diff = (self._clip_v_nat.unsqueeze(0) - new_v_cmd.unsqueeze(1)).abs()  # [n, num_clips]
        v_diff = v_diff.masked_fill(~self._clip_eligible.unsqueeze(0), float("inf"))
        chosen_ids = v_diff.argmin(dim=1)  # [n]
        self._sampled_motion_ids[env_ids] = chosen_ids

        # Per-env retime scale.
        if self._multiclip_retime_enabled:
            s_lo, s_hi = self._retime_scale_range
            ideal_s = new_v_cmd / self._clip_v_nat[chosen_ids].clamp(min=1e-6)
            s = ideal_s.clamp(s_lo, s_hi)
        else:
            s = torch.ones(n, device=self.device)
        self._motion_scale[env_ids] = s

        # Update observation v_cmd (effective forward speed after retime).
        effective_v = s * self._clip_v_nat[chosen_ids]
        self._current_cmd[env_ids, 0] = effective_v
        self._current_cmd[env_ids, 1] = 0.0
        self._current_cmd[env_ids, 2] = 0.0
```

- [ ] **Step 4: Verify file is syntactically valid**

```bash
source /home/exolab/miniconda3/etc/profile.d/conda.sh && conda activate phc
python -c "import ast; ast.parse(open('/home/exolab/Documents/GitHub/PHC/phc/env/tasks/humanoid_im_vic_cmd_multiclip.py').read())"
```

Expected: no output (parse succeeds).

- [ ] **Step 5: No commit yet — wait for Task 9 (smoke test pass)**

---

## Task 9: Run smoke test, confirm pass

- [ ] **Step 1: Run smoke test**

```bash
source /home/exolab/miniconda3/etc/profile.d/conda.sh && conda activate phc && export PYTHONUNBUFFERED=1
cd /home/exolab/Documents/GitHub/PHC
timeout 300 python scripts/smoketest_cont_s3.py 2>&1 | tee /tmp/smoke_after.log
```

Expected: `[SMOKE TEST] PASS` printed at end. The history shows v_cmd changing at least once (cycle boundary fires around step ~162 with default reference clip lengths).

- [ ] **Step 2: If FAIL, debug**

Likely issues:
- Smoke test patches BasePlayer.run incorrectly — check `import` paths
- `cycle_motion` attribute name differs — search `humanoid_im_vic.py` for the actual attr (`self.cycle_motion`)
- `_motion_start_times_offset` not initialized — adjust the cycle_mask computation to defer until after super has been called once

If debugging needed, do NOT modify the implementation in `humanoid_im_vic_cmd_multiclip.py` to make the test pass artificially. Either fix the test or recognize a real issue with the implementation.

- [ ] **Step 3: Commit Phase B (code change + smoke test)**

```bash
cd /home/exolab/Documents/GitHub/PHC
git add phc/env/tasks/humanoid_im_vic_cmd_multiclip.py scripts/smoketest_cont_s3.py
git commit -m "$(cat <<'EOF'
feat(multiclip): cycle-boundary v_cmd resampling for Slot 3

Adds multiclip_resample_on_cycle flag (default False) that, when set,
resamples v_cmd + re-picks closest clip + recomputes retime scale at
each motion cycle boundary. Called from a _compute_reset override
that runs BEFORE parent cycle logic so the new motion_id is picked
up by parent's _sample_time. Smoke test verifies v_cmd changes within
200 steps of rollout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Slot 1 sbatch script

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh`

- [ ] **Step 1: Write the sbatch script**

Create `exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh`:

```bash
#!/bin/bash
#SBATCH -J kit425_cont_S1
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S1 TERM_OFF: terminationDistance disabled, fallInit/hybrid OFF.

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml \
  --cfg_train exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_CONT_S1

echo "=== Job finished: $(date) ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh
```

- [ ] **Step 3: No commit yet**

---

## Task 11: Slot 2 sbatch script

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh`

- [ ] **Step 1: Copy Slot 1 script and modify**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh
```

In the new file, change these 4 things via Edit:
- `#SBATCH -J kit425_cont_S1` → `#SBATCH -J kit425_cont_S2`
- `#SBATCH -p idx0` → `#SBATCH -p idx1`
- `#SBATCH --gres=gpu:idx0:1` → `#SBATCH --gres=gpu:idx1:1`
- `# Slot S1 TERM_OFF...` comment → `# Slot S2 CYCLE_BASE: S1 + cycle_motion=True + episode_length 3000.`
- `--cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S1_termoff.yaml` → `--cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml`
- `--experiment KIT425_CONT_S1` → `--experiment KIT425_CONT_S2`

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh
```

- [ ] **Step 3: No commit yet**

---

## Task 12: Slot 3 sbatch script

**Files:**
- Create: `exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh`

- [ ] **Step 1: Copy Slot 2 script and modify**

```bash
cp /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh \
   /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh
```

In the new file:
- `#SBATCH -J kit425_cont_S2` → `#SBATCH -J kit425_cont_S3`
- `#SBATCH -p idx1` → `#SBATCH -p idx2`
- `#SBATCH --gres=gpu:idx1:1` → `#SBATCH --gres=gpu:idx2:1`
- `# Slot S2 CYCLE_BASE...` comment → `# Slot S3 CYCLE_VCMD_HOP: S2 + multiclip_resample_on_cycle=True. v_cmd hops at every cycle boundary.`
- `env_im_walk_vic_kit425_cont_S2_cyclebase.yaml` → `env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml`
- `--experiment KIT425_CONT_S2` → `--experiment KIT425_CONT_S3`

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/exolab/Documents/GitHub/PHC/exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh
```

- [ ] **Step 3: Commit Phase C (sbatch scripts)**

```bash
cd /home/exolab/Documents/GitHub/PHC
git add exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh \
        exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh \
        exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh
git commit -m "$(cat <<'EOF'
exp(kit425_cont): sbatch scripts for 3-slot training

S1/S2/S3 target idx0/1/2 GPUs respectively, 36h time limit, 15G mem,
8 cpus, num_envs 512. Mirrors the proven 260424_KIT425_PERSONAL_VCMD
template.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Push everything to origin

- [ ] **Step 1: Push**

```bash
cd /home/exolab/Documents/GitHub/PHC
git push origin Jimin
```

Expected: 3 new commits pushed (configs, code change, sbatch).

- [ ] **Step 2: Verify**

```bash
git log --oneline origin/Jimin -5
```

Expected: top 3-4 commits should be the ones from Tasks 6, 9, 12.

---

## Task 14: Server-side execution

This task requires SSH to `server1-jiminyoun`. The user can do it themselves or I can do it via Bash if the SSH key is configured (it is — earlier `rsync` calls worked).

- [ ] **Step 1: Pull on server**

```bash
ssh server1-jiminyoun "cd ~/PHC && git pull origin Jimin"
```

Expected: 3 commits fetched, fast-forward.

- [ ] **Step 2: Verify env yamls + new code path on server**

```bash
ssh server1-jiminyoun "cd ~/PHC && ls exp_config/forward_walking/260426_KIT425_CONT/ && grep -n 'multiclip_resample_on_cycle\\|_resample_vcmd_on_cycle' phc/env/tasks/humanoid_im_vic_cmd_multiclip.py"
```

Expected: lists 8 files in the new dir, shows 3-4 hits in the multiclip task source.

- [ ] **Step 3: Optional — server-side smoke test (if any concern about server-only env differences)**

```bash
ssh server1-jiminyoun "cd ~/PHC && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate phc && timeout 300 python scripts/smoketest_cont_s3.py" 2>&1 | tail -10
```

Expected: `[SMOKE TEST] PASS`. If FAIL on server but PASS locally, halt and investigate before submitting jobs.

- [ ] **Step 4: Submit 3 sbatch jobs**

```bash
ssh server1-jiminyoun "cd ~/PHC && mkdir -p logs && \
  sbatch exp_config/forward_walking/260426_KIT425_CONT/train_S1_termoff_gpu0.sh && \
  sbatch exp_config/forward_walking/260426_KIT425_CONT/train_S2_cyclebase_gpu1.sh && \
  sbatch exp_config/forward_walking/260426_KIT425_CONT/train_S3_vcmdhop_gpu2.sh"
```

Expected: 3 lines like `Submitted batch job <ID>`.

- [ ] **Step 5: Confirm jobs queued/running**

```bash
ssh server1-jiminyoun "squeue -u \$USER"
```

Expected: 3 jobs listed, status `R` (running) on partitions `idx0/1/2`.

- [ ] **Step 6: Tail first 60 sec of logs to verify clean startup**

```bash
ssh server1-jiminyoun "tail -F ~/PHC/logs/kit425_cont_S1_*.out ~/PHC/logs/kit425_cont_S2_*.out ~/PHC/logs/kit425_cont_S3_*.out" &
TAIL_PID=$!
sleep 60
kill $TAIL_PID
```

Expected: each log shows the standard PHC startup banner ("Loading extension module gymtorch...", "Setting seed:", "Found checkpoint" or "Started training", first `Ep: 1` line). No tracebacks.

If any traceback appears in the first 60 sec → cancel that job (`scancel <ID>`), debug locally, re-push, retry.

- [ ] **Step 7: Document submission**

Append to `02_research_dev/260426_kit425_continuous_walking_runs.md` (create if not exists):

```markdown
# KIT_425 Continuous Walking — 3 Job Submission

Submitted: 2026-04-26
Spec: docs/superpowers/specs/2026-04-26-kit425-continuous-walking-design.md
Plan: docs/superpowers/plans/2026-04-26-kit425-continuous-walking-plan.md

| Slot | Job ID | GPU | Started | Expected finish |
|---|---|---|---|---|
| S1 TERM_OFF | <fill> | idx0 | <fill> | +24h |
| S2 CYCLE_BASE | <fill> | idx1 | <fill> | +24h |
| S3 CYCLE_VCMD_HOP | <fill> | idx2 | <fill> | +24h |
```

Fill in job IDs from Step 4 output, started time from `squeue`, expected finish = started + 24h.

- [ ] **Step 8: Commit + push run-log**

```bash
git add 02_research_dev/260426_kit425_continuous_walking_runs.md
git commit -m "exp(kit425_cont): record S1/S2/S3 job submissions"
git push origin Jimin
```

---

## Post-execution (out of plan scope, deferred to next session)

After all 3 jobs finish (~24-36h):
1. `rsync` checkpoints from server to local: `KIT425_CONT_{S1,S2,S3}.pth`
2. Run `scripts/test_termination_stats.py --slot S1/S2/S3 --num_envs 32 --max_resets 1500` for each
3. Compare against B canonical baseline (already in `logs/test_kit425_personal/B_termstats.log`)
4. Plot training curves (adapt `scripts/plot_kit425_personal_training_v2.py` to new log filenames)
5. Write `02_research_dev/260427_kit425_continuous_walking_results.md` analysis doc
6. If Slot 2/3 succeed → next milestone: keyboard demo wrapper (out of scope for this plan)

---

## Self-Review Notes (already applied during plan write)

- Spec coverage: every section in the spec is mapped to a task. Slot allocation table → Tasks 2-4. Code change → Tasks 7-9. Sbatch → Tasks 10-12. Server execution → Task 14. Eval plan and out-of-scope sections noted explicitly as deferred.
- Placeholder scan: no TBD/TODO. All commands have full content. Sbatch script content is shown in full.
- Type consistency: flag name `multiclip_resample_on_cycle` used consistently in yaml, Python __init__, and code body. Method names `_resample_vcmd_on_cycle` and `_compute_reset` used consistently.
- Identified one minor risk: smoke test patches `rl_games.common.player.BasePlayer.run` which may not match the actual class hierarchy used by `IMAMPPlayerContinuous`. If the patched run isn't called, the test errors out with no v_cmd history captured. Task 9 Step 2 covers this debugging path.

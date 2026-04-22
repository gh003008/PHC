# PHC-Pain-v0 Spec #3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close PHC-Pain-v0 by retuning pain parameters to non-destructive values, copying the pretrained checkpoint to an isolated exp directory, running a short PPO fine-tune with `pain.mode=guard_and_reward`, and verifying measurable post-training behavior change on V3/V4-style probes.

**Architecture:** No structural changes. Spec #2 already installed the end-to-end pain pipeline (guard at `_action_to_pd_targets`, reward penalty at `_compute_reward`). Spec #3 adds (a) one new env config with retuned pain constants, (b) ~20 lines of body-group / tail-quantile logging in the task class, and (c) a sequenced verification run (calibration → sanity → survival gate → training → post-training probes).

**Tech Stack:** PHC (ZhengyiLuo/PHC), Isaac Gym + Hydra, rl_games PPO, SMPL humanoid, conda env `phc` (Python 3.8, PyTorch CUDA 11.6). RTX 3070 Ti, 8 GB VRAM — calibration compensates.

**Spec:** `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec3-finetune-design.md`
**Impl log:** `docs/superpowers/specs/phc_v0_spec3_impl_log.md`
**Starting commit:** `8f41e65` (design + log skeleton on `jinsu-pain_baseline_phc_v0`).

---

## File Structure

New files:
- `phc/data/cfg/env/env_im_pain_finetune.yaml` — copy of `env_im_pain.yaml` with the `pain:` block replaced (six value changes: `mode`, `lambda_p`, `pain_threshold`, `guard_gain`, `max_guard`, and the `notes:` string).

Modified files:
- `phc/env/tasks/humanoid_im_pain.py` — two additions inside the existing class:
  - `__init__`: one new helper call `self._group_dof_idx = self._build_body_group_idx()` plus the helper method body (~20 lines).
  - `_compute_reward`: ~8 lines appended after the existing `self.extras["pain_*"]` block.
- `docs/superpowers/specs/phc_v0_spec3_impl_log.md` — filled in across tasks 3–10.

Not modified:
- `phc/env/util/pain_baseline.py` — frozen, pure-torch.
- `phc/utils/parse_task.py` — already has `HumanoidImPain` import from Spec #2.
- `phc/env/tasks/humanoid_im.py`, `phc/env/tasks/humanoid.py`, any learning config, any network builder.

Total diff budget: 1 new yaml (~80 lines), ≤ 30 lines inserted in one py file, log growth only.

---

## Phase A — Implementation (Commit 2)

### Task 1: Create `env_im_pain_finetune.yaml`

**Files:**
- Create: `phc/data/cfg/env/env_im_pain_finetune.yaml`
- Reference: `phc/data/cfg/env/env_im_pain.yaml` (Spec #2 config — copy as starting point)

- [ ] **Step 1: Copy Spec #2 config as starting point**

Run:
```bash
cp /home/jinsu/Documents/GitHub/PHC/phc/data/cfg/env/env_im_pain.yaml \
   /home/jinsu/Documents/GitHub/PHC/phc/data/cfg/env/env_im_pain_finetune.yaml
```

Verify the file exists:
```bash
ls -la /home/jinsu/Documents/GitHub/PHC/phc/data/cfg/env/env_im_pain_finetune.yaml
```
Expected: a file of identical size to `env_im_pain.yaml`.

- [ ] **Step 2: Edit the `notes:` field**

Change the `notes:` line (near the top of the file) to:
```yaml
notes: "pain v0 Spec #3 finetune (guard_and_reward)"
```

Keep all other top-level fields (`project_name`, `task: HumanoidImPain`, `num_prim: 4`, etc.) unchanged — the only purpose of this file diff is the `pain:` block.

- [ ] **Step 3: Replace the `pain:` block with retuned values**

Find the `pain:` block (it sits above `plane:` at the bottom of the file). Replace its content with exactly:

```yaml
pain:
  enabled: True
  mode: "guard_and_reward"       # Spec #3 is the first guard_and_reward run
  append_to_obs: False           # unchanged; v0 contract flag only
  use_external_contact: True

  lambda_p: 0.02                 # was 0.05 — softer reward penalty

  w_limit: 1.0                   # unchanged
  w_torque: 0.35                 # unchanged
  w_power: 0.15                  # unchanged
  w_contact: 0.50                # unchanged

  joint_limit_margin_ratio: 0.15 # unchanged
  torque_ref_scale: 0.50         # unchanged
  power_ref: 5.0                 # unchanged
  contact_force_ref: 150.0       # unchanged

  pain_threshold: 0.30           # was 0.10 — suppress most natural standing pain
  pain_cap: 3.0                  # unchanged

  rise_alpha: 0.25               # unchanged
  decay_alpha: 0.02              # unchanged

  guard_gain: 0.25               # was 0.60 — softer target pullback
  max_guard: 0.30                # was 0.75 — lower cap

  log_joint_pain: True           # unchanged; gated on flags.im_eval
  log_contact_pain: True         # unchanged; gated on flags.im_eval
```

Do NOT add, delete, or rename any other pain keys. The task class reads them by name.

- [ ] **Step 4: Verify the yaml parses**

Run (PHC's env uses PyYAML which ships with conda `phc`):
```bash
conda run -n phc python -c "
import yaml
with open('/home/jinsu/Documents/GitHub/PHC/phc/data/cfg/env/env_im_pain_finetune.yaml') as f:
    cfg = yaml.safe_load(f)
pain = cfg['pain']
assert pain['mode'] == 'guard_and_reward', pain['mode']
assert pain['pain_threshold'] == 0.30, pain['pain_threshold']
assert pain['guard_gain'] == 0.25, pain['guard_gain']
assert pain['max_guard'] == 0.30, pain['max_guard']
assert pain['lambda_p'] == 0.02, pain['lambda_p']
assert cfg['task'] == 'HumanoidImPain', cfg['task']
assert cfg['num_prim'] == 4, cfg['num_prim']
print('ok')
"
```
Expected: `ok` on stdout, exit 0. Any AssertionError means a value is wrong — re-check Step 3 against the block above.

- [ ] **Step 5: Verify `git diff` is exactly the six intended changes**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  diff -u phc/data/cfg/env/env_im_pain.yaml \
          phc/data/cfg/env/env_im_pain_finetune.yaml | head -60
```
Expected: exactly six differing lines (mode, lambda_p, pain_threshold, guard_gain, max_guard, notes). All other lines identical. If more lines diff, find and revert extra edits.

(This task commits as part of commit 2, alongside Task 2 — do NOT commit here yet.)

---

### Task 2: Add body-group + quantile logging to `HumanoidImPain`

**Files:**
- Modify: `phc/env/tasks/humanoid_im_pain.py` (add ~20 lines in `__init__` + ~8 lines in `_compute_reward`)

- [ ] **Step 1: Understand the baseline shape**

Read `phc/env/tasks/humanoid_im_pain.py` end-to-end (it is ~160 lines). Take special note of:
- `__init__` ends at the `pain_scalar` allocation (currently line ~78).
- `_compute_reward` ends after the `flags.im_eval` block (currently line ~137).
- The assertion `self.num_dof == 3 * (self.num_bodies - 1)` in `__init__` (SMPL layout).

You do NOT need to understand every line of the Spec #2 code — just verify the two insertion points exist where this plan expects.

- [ ] **Step 2: Add `_build_body_group_idx` helper method**

Append this method to the `HumanoidImPain` class (place it between `__init__` and `_update_pain_buffers`):

```python
    def _build_body_group_idx(self):
        """SMPL body-name groups -> DOF index tensors. Called once in __init__.

        Groups aggregate DOF-level pain_state into 5 interpretable buckets.
        Root body (`Pelvis`) is excluded — it has no DOFs under SMPL.
        """
        groups = {
            "legs":  ["L_Hip", "R_Hip", "L_Knee", "R_Knee",
                      "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"],
            "arms":  ["L_Thorax", "R_Thorax", "L_Shoulder", "R_Shoulder",
                      "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"],
            "torso": ["Torso", "Spine", "Chest"],
            "head":  ["Neck", "Head"],
            "hands": ["L_Hand", "R_Hand"],
        }

        # SMPL DOF layout: non-root body at index i -> DOFs [3*(i-1) : 3*i].
        name_to_body_idx = {n: i for i, n in enumerate(self._body_names)}
        out = {}
        for group_name, body_names in groups.items():
            dof_idx = []
            for bn in body_names:
                bi = name_to_body_idx.get(bn)
                if bi is None or bi == 0:
                    continue
                dof_idx.extend([3 * (bi - 1), 3 * (bi - 1) + 1, 3 * (bi - 1) + 2])
            out[group_name] = torch.tensor(
                dof_idx, device=self.device, dtype=torch.long
            )
        return out
```

Indent matches other methods (4 spaces). Do not add a blank line inside the method.

- [ ] **Step 3: Call the helper from `__init__`**

In `__init__`, insert ONE line right after the last `pain_scalar` buffer allocation (after `self.pain_scalar = torch.zeros(...)`):

```python
        self._group_dof_idx = self._build_body_group_idx()
```

Match existing indentation (8 spaces — two levels inside the method).

- [ ] **Step 4: Add the 8 new logging lines in `_compute_reward`**

Find the `flags.im_eval` block near the end of `_compute_reward`. Directly BEFORE that `if flags.im_eval:` line, insert:

```python
        for name, idx in self._group_dof_idx.items():
            self.extras[f"pain_{name}"] = float(self.pain_state[:, idx].mean().item())

        self.extras["pain_p90"] = float(torch.quantile(self.pain_scalar, 0.90).item())
        self.extras["pain_p99"] = float(torch.quantile(self.pain_scalar, 0.99).item())
        self.extras["pain_saturated_frac"] = float((self.pain_state >= 2.5).float().mean().item())
```

Indentation: 8 spaces (two levels inside `_compute_reward`). Insert a blank line before the `for` and after the `pain_saturated_frac` line so the block is visually separated.

- [ ] **Step 5: Sanity-check the diff shape**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff --stat phc/env/tasks/humanoid_im_pain.py
```
Expected: `1 file changed, <~30 insertions(+)`. 0 deletions. Any deletion here means existing code got clobbered — inspect and fix.

Also run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff phc/env/tasks/humanoid_im_pain.py | head -80
```
Sanity-check visually: additions are (a) `_group_dof_idx = ...` in `__init__`, (b) new `_build_body_group_idx` method, (c) the 5+3 logging lines before `flags.im_eval`.

- [ ] **Step 6: V1-style import smoke test (no Isaac Gym load)**

Still no Isaac Gym. Verify the file still parses:
```bash
conda run -n phc python -c "
import isaacgym
from phc.env.tasks.humanoid_im_pain import HumanoidImPain
print(HumanoidImPain.__name__)
print('has _build_body_group_idx:', hasattr(HumanoidImPain, '_build_body_group_idx'))
"
```
Expected:
```
HumanoidImPain
has _build_body_group_idx: True
```
Exit 0.

If ImportError fires, the insertion broke Python syntax — re-open the file and check indentation on the new block.

- [ ] **Step 7: Smoke test — Spec #2 V2 regression (baseline unchanged)**

This is the critical check: new logging code must not regress the `mode=off` / `enabled=False` path. The body-group helper builds a dict even when pain is disabled (init runs every time), but `_compute_reward`'s early `if not self.pain_enabled: return` still short-circuits the new logging block.

Run Spec #2 V2's command (verbatim):
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2 2>&1 | tail -30
```

Expected: final stdout includes `reward: 941.05078125 steps: 999.0` (or within 1%). Exit 0.

If reward diverges: the body-group helper or DOF indexing broke the init. Revert the `_build_body_group_idx` call from `__init__` temporarily and re-run to bisect.

- [ ] **Step 8: Commit 2 — yaml + task class changes together**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
git add phc/data/cfg/env/env_im_pain_finetune.yaml \
        phc/env/tasks/humanoid_im_pain.py && \
git status --short
```
Expected: two files staged (A and M), no other diff.

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
git commit -m "$(cat <<'EOF'
Spec #3: add retuned finetune config + body-group pain logging

env_im_pain_finetune.yaml copies env_im_pain.yaml with six value changes
in the pain: block — mode=guard_and_reward, pain_threshold=0.30,
guard_gain=0.25, max_guard=0.30, lambda_p=0.02 (Codex-validated Option C
from brainstorm), plus notes update.

HumanoidImPain gains a _build_body_group_idx helper that maps SMPL body
names to DOF index tensors for 5 groups (legs/arms/torso/head/hands),
called once in __init__. _compute_reward now also writes pain_{legs,arms,
torso,head,hands} means plus pain_p90/p99/saturated_frac scalars to
self.extras per step — wandb-cheap (scalar floats, no per-DOF arrays).

Smoke test: Spec #2 V2 command still produces reward 941.05 / steps 999
when pain.enabled=False. No obs dim / checkpoint / network changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify with `git log --oneline -3`. Expected: new commit on top referencing "retuned finetune config + body-group pain logging".

---

## Phase B — Pre-training verification (commits into log, then Commit 3)

### Task 3: V5.0 GPU calibration ladder

**Files:**
- Read/execute: shell
- Modify (append results): `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V5.0` section.

Calibration goal: find the largest `(num_envs, horizon_length)` rung where peak VRAM stays ≤ 85% of the 3070 Ti's 8 GB (~6.8 GB ≈ 6960 MB reported by `nvidia-smi` as `memory.used`).

- [ ] **Step 1: Pre-flight — verify conda env, GPU, pristine pretrained checkpoint**

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -la /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth
conda run -n phc python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: 3070 Ti with ~8192 MiB, `Humanoid.pth` exists (size ~hundreds of MB), torch reports CUDA available on the 3070 Ti.

If any of these fails, stop and fix environment before proceeding.

- [ ] **Step 2: Run Rung 1 probe — `num_envs=1024, horizon_length=32`**

Launch background VRAM monitor first:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
  > /tmp/phc_spec3_v5.0_mem_rung1.csv &
MONITOR_PID=$!
echo "monitor pid: $MONITOR_PID"
```

Then run the probe:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_spec3_calib_rung1 test=False no_log=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1024 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.horizon_length=32 \
  learning.params.config.max_epochs=10 2>&1 | tee /tmp/phc_spec3_v5.0_rung1.log | tail -20
```

Stop the monitor:
```bash
kill $MONITOR_PID
```

Compute peak VRAM:
```bash
awk -F, 'NR>0 {v=$1+0; if (v>max) max=v} END {print "peak_used_MB=" max}' \
  /tmp/phc_spec3_v5.0_mem_rung1.csv
```

Cleanup throwaway output:
```bash
rm -rf /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_spec3_calib_rung1
```

Three outcomes:
- **Peak VRAM ≤ 6960 MB** AND exit 0: Rung 1 accepted — it's already the largest ladder entry. Jump directly to Step 8 to record the result, then Task 4.
- **Peak VRAM > 6960 MB** but exit 0: rung 1 technically fit but too tight for a 30 min run; drop to Rung 2 (Step 3).
- **OOM or exit nonzero** with CUDA OOM traceback: drop to Rung 2 (Step 3).

- [ ] **Step 3: Run Rung 2 probe — `num_envs=512, horizon_length=64` (only if Rung 1 failed/too tight)**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
  > /tmp/phc_spec3_v5.0_mem_rung2.csv &
MONITOR_PID=$!
```

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_spec3_calib_rung2 test=False no_log=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=512 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.horizon_length=64 \
  learning.params.config.max_epochs=10 2>&1 | tee /tmp/phc_spec3_v5.0_rung2.log | tail -20
```

```bash
kill $MONITOR_PID
awk -F, 'NR>0 {v=$1+0; if (v>max) max=v} END {print "peak_used_MB=" max}' /tmp/phc_spec3_v5.0_mem_rung2.csv
rm -rf /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_spec3_calib_rung2
```

Accept iff exit 0 AND peak ≤ 6960 MB. If fit is comfortable (peak < 6500 MB), Rung 1 might have been *nearly* fine — but we defer to Rung 2 for safety margin on long runs. Go to Step 8 if accepted, Step 4 if failed.

- [ ] **Step 4: Run Rung 3 probe — `num_envs=256, horizon_length=128`**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
  > /tmp/phc_spec3_v5.0_mem_rung3.csv &
MONITOR_PID=$!

cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_spec3_calib_rung3 test=False no_log=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=256 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.horizon_length=128 \
  learning.params.config.max_epochs=10 2>&1 | tee /tmp/phc_spec3_v5.0_rung3.log | tail -20

kill $MONITOR_PID
awk -F, 'NR>0 {v=$1+0; if (v>max) max=v} END {print "peak_used_MB=" max}' /tmp/phc_spec3_v5.0_mem_rung3.csv
rm -rf /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_spec3_calib_rung3
```

Accept iff exit 0 AND peak ≤ 6960 MB. Go to Step 8 if accepted, Step 5 if not.

- [ ] **Step 5: Run Rung 4 probe — `num_envs=128, horizon_length=256`**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
  > /tmp/phc_spec3_v5.0_mem_rung4.csv &
MONITOR_PID=$!

cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_spec3_calib_rung4 test=False no_log=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.horizon_length=256 \
  learning.params.config.max_epochs=10 2>&1 | tee /tmp/phc_spec3_v5.0_rung4.log | tail -20

kill $MONITOR_PID
awk -F, 'NR>0 {v=$1+0; if (v>max) max=v} END {print "peak_used_MB=" max}' /tmp/phc_spec3_v5.0_mem_rung4.csv
rm -rf /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_spec3_calib_rung4
```

Accept iff exit 0 AND peak ≤ 6960 MB. If Rung 4 also fails (still OOM), escalate to Step 6; otherwise Step 8.

- [ ] **Step 6: Escalation — drop `minibatch_size` (only if Rung 4 OOMs)**

If Rung 4 still OOMs, retry Rung 4 with `learning.params.config.minibatch_size=8192` (default is 16384). Same command template; add the override. If that also OOMs, try `minibatch_size=4096`. Record every rung + knob combination's peak VRAM in the log.

If `minibatch_size=4096` at Rung 4 still OOMs: STOP this plan and flag to user — the GPU is genuinely under-sized for this spec. User decides whether to escalate to cloud GPU or descope.

- [ ] **Step 7: (Skip if any rung already accepted)**

Placeholder only — continue to Step 8.

- [ ] **Step 8: Record V5.0 result in impl log**

Open `docs/superpowers/specs/phc_v0_spec3_impl_log.md`. Under `## V5.0 — GPU calibration`, fill in the accepted rung with:
- Exact command run (copy-paste from the ladder step above, with `<rung>`/`<N>`/`<H>` substituted).
- `peak_used_MB` value from `awk` output.
- Exit code.
- One-line interpretation — e.g. "Rung 2 selected. Peak 6320 MB = 77% of 8192 MB. Rung 1 OOMed with traceback at 7890 MB. Rungs 3/4 skipped."
- A one-line recording of *every rung attempted*, even the failed ones, so the decision is auditable.

Also update the "Selected rung" line at the top of the V5.0 section to the chosen `num_envs` and `horizon_length`.

DO NOT commit yet — V5.1/V5.2 go in the same log commit.

- [ ] **Step 9: Export chosen values for Task 4+ use**

For convenience in subsequent tasks, set shell vars (or write a file) capturing:
```bash
SPEC3_NUM_ENVS=<chosen>
SPEC3_HORIZON=<chosen>
echo "SPEC3_NUM_ENVS=$SPEC3_NUM_ENVS"
echo "SPEC3_HORIZON=$SPEC3_HORIZON"
```

Write these to `/tmp/phc_spec3_calib.env` for re-source on resumption:
```bash
{
  echo "SPEC3_NUM_ENVS=$SPEC3_NUM_ENVS"
  echo "SPEC3_HORIZON=$SPEC3_HORIZON"
} > /tmp/phc_spec3_calib.env
cat /tmp/phc_spec3_calib.env
```

---

### Task 4: V5.1 — Copy checkpoint & sanity-reproduce baseline

**Files:**
- Create (copy): `output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth`
- Append: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V5.1`.

- [ ] **Step 1: Create new exp dir and copy checkpoint**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  mkdir -p output/HumanoidIm/phc_shape_pnn_iccv_pain && \
  cp output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth \
     output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth && \
  ls -la output/HumanoidIm/phc_shape_pnn_iccv_pain/
```
Expected: `Humanoid.pth` listed with the same size as in the pristine dir.

Verify byte-identical copy:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  md5sum output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth \
         output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth
```
Expected: two identical MD5 hashes.

- [ ] **Step 2: Run the sanity probe**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2 2>&1 | tee /tmp/phc_spec3_v5.1.log | tail -15
```
Expected tail: `reward: 941.05078125 steps: 999.0` twice, `av reward: 941.05...` at end. Exit 0.

If reward is not within 1% of 941.05: `rm -rf output/HumanoidIm/phc_shape_pnn_iccv_pain` and re-copy. If still wrong, the pristine pretrained might itself be missing — stop and escalate.

- [ ] **Step 3: Record in impl log**

Under `## V5.1` in `docs/superpowers/specs/phc_v0_spec3_impl_log.md`, append:
- The `mkdir` / `cp` / `md5sum` commands and their output.
- The probe command (copy-paste verbatim).
- Exit code.
- Last ~15 lines of stdout.
- One-line interpretation: "Reward 941.05 matches Spec #2 V2 to 0.00%. Copy verified lossless."

DO NOT commit yet.

---

### Task 5: V5.2 — Retuned guard_only survival gate

**Files:**
- Execute: probe against `env_im_pain_finetune.yaml` with `pain.mode=guard_only`.
- Append: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V5.2`.

- [ ] **Step 1: Run the retuned guard_only probe**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=guard_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2 2>&1 | tee /tmp/phc_spec3_v5.2.log | tail -20
```

Expected tail: `reward: <value>` lines where `<value>` ≥ ~470, `steps: <value>` where value ≥ 300.

- [ ] **Step 2: Interpret outcome**

Three cases:

**Case A — PASS (steps ≥ 300 AND reward ≥ 470):**
Proceed to Task 6. Training has a viable starting point.

**Case B — FAIL (steps < 300 OR reward < 470):**
Retune is still too aggressive. DO NOT proceed to training. Instead:
1. Edit `phc/data/cfg/env/env_im_pain_finetune.yaml`:
   - Raise `pain_threshold` by 0.05 (e.g., 0.30 → 0.35).
   - OR lower `guard_gain` by 0.05 (e.g., 0.25 → 0.20).
   - OR lower `max_guard` by 0.05 (e.g., 0.30 → 0.25).
2. Re-run Step 1. Record both the failed original and the retry in the log.
3. Up to 3 retries. If 3 fail, escalate to user — retune strategy needs rethinking.

**Case C — passes but borderline (steps in [300, 400] or reward in [470, 600]):**
Accept for now. Note in the log that survival margin is thin; monitor V5.3 reward carefully for early collapse.

- [ ] **Step 3: Record in impl log**

Under `## V5.2` in the impl log, append:
- Command(s) run (including any retries).
- Exit codes.
- Last ~15 lines of stdout per run.
- One-line interpretation per run (pass/fail + the numbers).
- Final decision: "PASS with <X> steps / <Y> reward" OR "PASS after retune X (threshold 0.35)".

DO NOT commit yet — V5.0 + V5.1 + V5.2 will commit together in Task 6.

---

### Task 6: Commit 3 — populated log V5.0–V5.2

**Files:**
- Commit: `docs/superpowers/specs/phc_v0_spec3_impl_log.md`

- [ ] **Step 1: Review the log diff**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff docs/superpowers/specs/phc_v0_spec3_impl_log.md | head -100
```

Verify:
- V5.0 has selected rung + peak VRAM numbers recorded.
- V5.1 has reward 941.05 reproduction confirmed.
- V5.2 has pass status and concrete numbers.
- No stray edits elsewhere in the file.

- [ ] **Step 2: Stage and commit**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
git add docs/superpowers/specs/phc_v0_spec3_impl_log.md && \
git commit -m "$(cat <<'EOF'
Spec #3: populate log with V5.0-V5.2 results

V5.0 GPU calibration selected num_envs=<N>, horizon_length=<H> on
3070 Ti (peak <M> MB). V5.1 checkpoint copy reproduces reward 941.05.
V5.2 retuned guard_only survival gate passes with <STEPS> steps / <R>
reward — training is cleared to start.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<N>`, `<H>`, `<M>`, `<STEPS>`, `<R>` with the actual numbers from the log.

Verify with `git log --oneline -4`.

---

## Phase C — Fine-tune and post-training verification

### Task 7: V5.3 — Short PPO fine-tune (~30 min)

**Files:**
- Execute: long-running training command, monitored.
- Append: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V5.3`.
- Writes: `output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth` (overwritten with fine-tuned weights).

- [ ] **Step 1: Pre-flight — re-load calibrated values, sanity-check wandb**

```bash
source /tmp/phc_spec3_calib.env
echo "num_envs=$SPEC3_NUM_ENVS horizon=$SPEC3_HORIZON"

# Check wandb is logged in (for online mode). If not, run `wandb login` and paste API key.
conda run -n phc python -c "import wandb; print('wandb version:', wandb.__version__); print('logged-in user:', wandb.api.viewer().get('username', '<none>'))" 2>&1 | head -5
```

If wandb not logged in AND user prefers online mode, stop and ask user to `wandb login`. If offline is acceptable, export `WANDB_MODE=offline` before the next step.

- [ ] **Step 2: Launch V5.3 training as a Monitor background job**

Use the Monitor tool (see writing-plans skill's tool inventory — not a shell command). Construct the training invocation:

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=False \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=$SPEC3_NUM_ENVS headless=True \
  env.pain.enabled=True env.pain.mode=guard_and_reward \
  env.notes="pain v0 Spec #3 short fine-tune (guard_and_reward)" \
  learning.params.config.horizon_length=$SPEC3_HORIZON \
  learning.params.config.max_epochs=2000 2>&1 | \
  tee /tmp/phc_spec3_v5.3.log
```

Dispatch via Monitor with:
- `description`: `"V5.3 Spec #3 fine-tune training"`
- `timeout_ms`: `3600000` (1 hour, 2× the 30 min target)
- Filter: `grep -E --line-buffered "epoch=|reward:|Traceback|CUDA|OOM|assert|Killed|a_loss|v_loss|Error"`

Filter rationale: progress events (`epoch=`, `reward:`), loss spike/NaN (`a_loss`, `v_loss`), and all failure signatures. Do not narrow to success-only — silence would hide a crashloop.

- [ ] **Step 3: Monitor while training runs**

Events stream in. Watch for:
- Steady increase in epoch number.
- `reward:` line values staying > ~200 (i.e., not collapsing to 0). In early fine-tune, reward may dip briefly — OK as long as recovery is visible.
- Absence of `Traceback` / `OOM`.

After training ends (either 2000 epochs reached, or manually stopped at ~30 min wall-clock):
- Final `reward:` line should be ≥ 200 at minimum.
- `output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth` should have a recent mtime (not the copy's mtime).

Verify the checkpoint was overwritten:
```bash
ls -la /home/jinsu/Documents/GitHub/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth
```
Expected: modification time after the training started.

- [ ] **Step 4: Snapshot wandb run URL + training tail**

Grab last 50 lines of the training log for the impl log:
```bash
tail -50 /tmp/phc_spec3_v5.3.log
```

If wandb was online, grab the run URL from the training log — rl_games prints it near the start (`wandb: View run ... at: https://wandb.ai/...`). Search:
```bash
grep "wandb: View run" /tmp/phc_spec3_v5.3.log | head -1
```

- [ ] **Step 5: Record V5.3 in impl log**

Under `## V5.3`, append:
- Calibrated values (`num_envs`, `horizon_length`).
- Full training command (with actual values substituted).
- Exit outcome (completed 2000 epochs / stopped at wall-clock / crashed).
- Last ~50 lines of stdout.
- wandb run URL (if online mode).
- Final reward at last epoch.
- One-line interpretation: "Training completed <N> epochs in <T> min. Final reward <R>. Checkpoint saved."

DO NOT commit yet — V6 and V7 append before commit 4.

---

### Task 8: V6 — Post-training log_only probe

**Files:**
- Execute: probe with fine-tuned checkpoint + `pain.mode=log_only`.
- Temporary: add and later remove a `print(self.pain_scalar.max().item())` line in `_compute_reward` to see per-step pain on stdout.
- Append: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V6`.

- [ ] **Step 1: Add temporary print inside `_compute_reward`**

Same approach Spec #2 V3 used (recorded in Spec #2 impl log). In `phc/env/tasks/humanoid_im_pain.py` `_compute_reward`, after the `self.pain_scalar[:] = ...` line, add (temporarily):

```python
        if self.pain_enabled:
            print(f"[pain-debug] max={self.pain_scalar.max().item():.3f} mean={self.pain_scalar.mean().item():.3f}", flush=True)
```

Indentation 8 spaces. This MUST be reverted before final commit.

- [ ] **Step 2: Run the probe**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=log_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2 2>&1 | tee /tmp/phc_spec3_v6.log
```

Expected: `[pain-debug]` lines across ~999 steps × 2 episodes. Final steady-state `max=<X>` where `X ≤ 0.35` (acceptance). Final `reward: <Y>` near baseline ~900 (log_only mode doesn't apply penalty to the printed reward, but the policy has been fine-tuned so it may track slightly differently — don't require exact 941).

Extract the steady-state max:
```bash
grep '\[pain-debug\]' /tmp/phc_spec3_v6.log | tail -20
grep 'reward:' /tmp/phc_spec3_v6.log | tail -4
```

- [ ] **Step 3: Revert the temporary print**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff phc/env/tasks/humanoid_im_pain.py
```
Expected: exactly the 2-line addition from Step 1.

Remove the print:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git checkout -- phc/env/tasks/humanoid_im_pain.py
```

Verify clean:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff phc/env/tasks/humanoid_im_pain.py
```
Expected: no output (file restored to commit 2 state).

- [ ] **Step 4: Interpret outcome**

- **PASS**: steady-state `pain_max ≤ 0.35`. Proceed to Task 9.
- **FAIL**: `pain_max > 0.35`. Two sub-cases:
  - Still close (e.g., 0.36–0.45): record as "near-miss"; can either accept with note or re-train with higher `lambda_p`. User decides.
  - Much higher (> 0.50): fine-tune did not meaningfully change pain-avoidance behavior. Likely insufficient training budget. Record as FAIL, surface to user before continuing.

- [ ] **Step 5: Record V6 in impl log**

Under `## V6`, append:
- Probe command (copy-paste).
- Exit code.
- Last ~20 `[pain-debug]` lines.
- Final reward lines.
- Decision (PASS/near-miss/FAIL) with comparison to Spec #2 V3's 0.522.

DO NOT commit yet — V7 appends before commit 4.

---

### Task 9: V7 — Post-training guard_only probe

**Files:**
- Execute: probe with fine-tuned checkpoint + `pain.mode=guard_only`.
- Append: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` under `## V7`.

- [ ] **Step 1: Run the probe (no temporary print — the stdout already shows reward and steps)**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=guard_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2 2>&1 | tee /tmp/phc_spec3_v7.log | tail -20
```

Expected tail: two `reward: <R> steps: <S>` lines. Acceptance: `S ≥ 500` AND `R ≥ 470`.

- [ ] **Step 2: Interpret outcome**

- **PASS**: `S ≥ 500 AND R ≥ 470`. v0 acceptance closed.
- **FAIL (steps improved but reward low)**: policy survives but imitation tracking degraded. Flag in log; user decides whether to raise `lambda_p` and retrain, or accept v0 with caveat.
- **FAIL (no survival improvement)**: training did not help guard-tolerance. Likely insufficient budget or wrong hyper. Surface to user.

- [ ] **Step 3: Record V7 in impl log**

Under `## V7`, append:
- Probe command.
- Exit code.
- Last ~20 lines of stdout (both `reward: ...` and any `av reward: ...`).
- Decision with comparison to Spec #2 V4's 38 steps / 27.15 reward.

---

### Task 10: Commit 4 — final log + exit-gate signoff

**Files:**
- Modify: `docs/superpowers/specs/phc_v0_spec3_impl_log.md` — tick exit-gate checkboxes.
- Commit: that file.

- [ ] **Step 1: Fill in exit-gate checkboxes**

Open `docs/superpowers/specs/phc_v0_spec3_impl_log.md`. Under `## Exit gate`, change every `- [ ]` to `- [x]` for all criteria that passed. Any FAIL/near-miss items stay `- [ ]` with a parenthetical note linking to the section that records the failure.

Verify `git diff` of the impl log is clean (no stray edits):
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff docs/superpowers/specs/phc_v0_spec3_impl_log.md | head -100
```

- [ ] **Step 2: Run the structural constraint check from spec §14**

Verify diff since Spec #2's last impl commit:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  git diff --stat a72f5b0..HEAD -- phc/
```
Expected output (approximately):
```
 phc/data/cfg/env/env_im_pain_finetune.yaml      | 80+ insertions
 phc/env/tasks/humanoid_im_pain.py               | <30 insertions, 0 deletions
```
Exactly 2 files under `phc/`. Any other files → a stray edit leaked in; investigate and remove.

- [ ] **Step 3: Commit**

```bash
cd /home/jinsu/Documents/GitHub/PHC && \
git add docs/superpowers/specs/phc_v0_spec3_impl_log.md && \
git commit -m "$(cat <<'EOF'
Spec #3: populate log with V5.3/V6/V7 results + exit-gate signoff

V5.3 fine-tune completed <N> epochs in <T> min. V6 post-training
pain_max <X> (was 0.522 in Spec #2 V3). V7 post-training <S> steps /
<R> reward (was 38 / 27.15 in Spec #2 V4). All acceptance conditions
from spec §1 recorded. PHC-Pain-v0 complete — any further work on a
new pain_baseline_phc_v1_* branch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace the `<N>`, `<T>`, `<X>`, `<S>`, `<R>` placeholders with actual numbers before running.

Verify with `git log --oneline -6`. Expected: 4 commits for Spec #3 on top of the Spec #2 commits.

- [ ] **Step 4: Final hand-off**

Tell the user:
- v0 is complete on branch `jinsu-pain_baseline_phc_v0`.
- 4 Spec #3 commits landed on top of Spec #2's 4.
- Impl log has full results; wandb run linked (if online mode was used).
- Next work (obs-appended pain, MPJPE, multi-motion) is v1 scope and goes on a new branch.

---

## Post-completion review hooks

After Task 10 commit, offer two optional follow-ups to the user (do NOT run without explicit ask):

1. `superpowers:requesting-code-review` on the branch's Spec #3 delta.
2. `ultrareview` if the user wants a multi-agent cloud review of the whole v0 branch.

---

## Summary table of commits produced by this plan

| # | Title | Files |
|---|---|---|
| (already landed) | Spec #3: add finetune design + impl log skeleton | design + log skeleton |
| Commit 2 (Task 2) | Spec #3: add retuned finetune config + body-group pain logging | yaml + py |
| Commit 3 (Task 6) | Spec #3: populate log with V5.0-V5.2 results | log |
| Commit 4 (Task 10) | Spec #3: populate log with V5.3/V6/V7 results + exit-gate signoff | log |

Total branch delta since Spec #2: 2 code files + 1 new spec doc + 1 impl log.

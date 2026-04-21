# PHC-Pain-v0 — Spec #1 Implementation Plan (Environment Bring-up + Sanity)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up the PHC repo on this machine (`phc` conda env, isaacgym preview4 linked, SMPL assets placed, checkpoints downloaded) and verify by running the official `phc_shape_pnn_iccv` single-primitive checkpoint in a viewer.

**Architecture:** Environment setup only — zero code changes under `phc/`. The plan has three phases: (1) pre-flight discovery (find existing isaacgym path, find existing SMPL pkls), (2) env + data setup, (3) three sanity checks (`vis_motion_mj.py`, `joint_monkey_smpl.py`, `run_hydra.py test=True`). All outputs recorded in a setup log.

**Tech Stack:** conda (python 3.8), PyTorch + CUDA 11.6, NVIDIA isaacgym preview4, SMPL/SMPL-X pickles, Hydra configs (upstream PHC).

**Spec reference:** `docs/superpowers/specs/2026-04-22-phc-pain-baseline-v0-design.md`
**Branch:** `jinsu-pain_baseline_phc_v0` (already created)
**Repo root:** `/home/jinsu/Documents/GitHub/PHC`
**Human-in-the-loop:** Several steps require the user to (a) provide info (SMPL pkl locations), (b) visually confirm viewer windows opened, (c) accept SMPL license re-downloads if needed. These are flagged `[HITL]`.

---

## File Structure

This plan creates or touches only these files (zero code):

| Path | Role |
|---|---|
| `docs/superpowers/specs/phc_v0_setup_log.md` | **Create** — setup log skeleton, then populated |
| `data/smpl/` | Populated with 6 pkl files |
| `sample_data/` | Populated by `download_data.sh` |
| `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth` | Populated by `download_data.sh` |

`.gitignore` handling: `data/`, `sample_data/`, `output/` should already be
ignored (repo-default). Verify in Task 1.

---

## Task 1: Pre-flight — verify repo state

**Files:**
- Read only (no writes)

- [ ] **Step 1.1: Confirm working directory and branch**

Run:
```bash
pwd && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD
```

Expected output (all three lines):
```
/home/jinsu/Documents/GitHub/PHC
jinsu-pain_baseline_phc_v0
846988d433ce1f341e85ac6fbd2cd51911bb3341
```

If any mismatch, stop and fix before proceeding.

- [ ] **Step 1.2: Confirm `data`, `sample_data`, `output` are gitignored**

Run:
```bash
git check-ignore -v data sample_data output 2>&1 || echo "NOT IGNORED"
```

Expected: each dir matches a `.gitignore` rule. If any dir prints `NOT IGNORED`, add to `.gitignore` before any download happens. (Prevents accidental commit of multi-GB binaries.)

- [ ] **Step 1.3: Probe display environment**

Run:
```bash
echo "DISPLAY=$DISPLAY"
xdpyinfo >/dev/null 2>&1 && echo "X OK" || echo "NO X"
```

Expected: `X OK`. User confirmed a display is available; this makes it explicit.

- [ ] **Step 1.4: Confirm no conda env `phc` exists yet** (collision check)

Run:
```bash
conda env list | grep -E "^phc\b" && echo "COLLISION" || echo "CLEAR"
```

Expected: `CLEAR`. If `COLLISION`, ask user whether to reuse or rename (do not destroy an existing env silently).

---

## Task 2: Pre-flight — locate isaacgym preview4

**Files:** Read only

- [ ] **Step 2.1: Extract isaacgym path via existing `rl_mpc` env**

Run:
```bash
conda run -n rl_mpc python -c "import isaacgym, os; print(os.path.dirname(os.path.dirname(isaacgym.__file__)))" 2>&1
```

Expected: a single line, an absolute path ending in `.../isaacgym` or `.../isaacgym/python`. The printed path is the **isaacgym python-package root**. Its parent is the **source tree** (contains `setup.py`, `assets/`, `python/`).

- [ ] **Step 2.2: Verify the source tree has `python/setup.py`**

Let `IG_PY_ROOT` = output of Step 2.1. Run:
```bash
IG_SOURCE="$(dirname "$IG_PY_ROOT")"
ls "$IG_SOURCE/setup.py" 2>&1
```

Expected: a valid file path, not "No such file". Record `IG_SOURCE` for use in Task 6.

- [ ] **Step 2.3 [HITL]: Report the discovered path to user**

Print:
```
Found isaacgym source at: <IG_SOURCE>
Will use this in Task 6 (pip install -e .)
Proceed? [y/n]
```

If user says no or path looks wrong, fall back to manual re-download from https://developer.nvidia.com/isaac-gym (requires NVIDIA dev account login). Do not proceed without confirmation.

---

## Task 3: Pre-flight — locate existing SMPL pickles

**Files:** Read only

- [ ] **Step 3.1: Probe home for SMPL-family pickles**

Run:
```bash
find "$HOME" -maxdepth 6 \
  \( -iname 'SMPL_NEUTRAL.pkl' -o -iname 'SMPL_MALE.pkl' -o -iname 'SMPL_FEMALE.pkl' \
     -o -iname 'SMPLX_NEUTRAL.pkl' -o -iname 'SMPLX_MALE.pkl' -o -iname 'SMPLX_FEMALE.pkl' \
     -o -iname 'basicmodel_neutral_lbs_10_207_0_v1.1.0*' \
     -o -iname 'basicmodel_m_lbs_10_207_0_v1.1.0*' \
     -o -iname 'basicmodel_f_lbs_10_207_0_v1.1.0*' \) \
  2>/dev/null
```

Expected: zero or more paths. Record them.

- [ ] **Step 3.2 [HITL]: Classify findings and decide**

Based on Step 3.1 output:

| Case | Action |
|---|---|
| All 6 final-named (`SMPL_NEUTRAL.pkl` etc.) found | Use them in Task 7, skip re-download |
| Original-named basicmodel_*.pkl found | Copy + rename in Task 7 (see spec §3.1 table) |
| SMPLX files found but not SMPL (or vice versa) | Re-download the missing set |
| None found | Full re-download of both sets |

Print the decision to the user. If re-download needed, ask user to confirm they will log into smpl.is.tue.mpg.de (SMPL v1.1.0) and smpl-x.is.tue.mpg.de (SMPL-X v1.1) and download. **Block here until user provides the download paths.**

---

## Task 4: Create `phc` conda env + install PyTorch

**Files:** Read only (env state)

- [ ] **Step 4.1: Create env**

Run:
```bash
conda create -n phc python=3.8 -y
```

Expected: env creates without error, last line is "done".

- [ ] **Step 4.2: Install PyTorch with CUDA 11.6**

Run:
```bash
conda install -n phc pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
```

Expected: installs without conflict resolution spiral. If conda's solver stalls over 5 minutes, Ctrl-C and try `conda install -n phc pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y` (pinned version known compatible with isaacgym preview4).

- [ ] **Step 4.3: Verify torch + CUDA**

Run:
```bash
conda run -n phc python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

Expected: `cuda True`, `device_count` >= 1. If False, check `nvidia-smi` and driver compat before continuing.

---

## Task 5: Install PHC python requirements

**Files:** Read only (env state)

- [ ] **Step 5.1: Install `requirement.txt`**

Run:
```bash
conda run -n phc pip install -r /home/jinsu/Documents/GitHub/PHC/requirement.txt 2>&1 | tail -30
```

Expected: exits 0. If torch version gets downgraded by a pinned line, ignore and reconfirm with Step 4.3. If a line is unreachable (e.g. a git URL 404s), log the exact line and ask the user.

- [ ] **Step 5.2: Install SMPLSim (README line 49)**

Run:
```bash
conda run -n phc pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master 2>&1 | tail -10
```

Expected: `Successfully installed smpl-sim-...`.

- [ ] **Step 5.3: Verify SMPLSim import**

Run:
```bash
conda run -n phc python -c "import smpl_sim; print(smpl_sim.__file__)"
```

Expected: a path under site-packages. If `ModuleNotFoundError`, the package name may differ — run `conda run -n phc pip show smpl-sim` and import the true module name.

---

## Task 6: Link isaacgym preview4 into `phc` env

**Files:** Read only (env state)

- [ ] **Step 6.1: Install isaacgym in editable mode**

Using `IG_SOURCE` from Task 2:
```bash
conda run -n phc pip install -e "$IG_SOURCE/python" 2>&1 | tail -10
```

Expected: `Successfully installed isaacgym-...` or a no-op saying it's already linked.

- [ ] **Step 6.2: Verify isaacgym import from `phc` env**

Run:
```bash
conda run -n phc python -c "import isaacgym; print(isaacgym.__file__)"
```

Expected: a path identical to (or one `python/` subdir deep under) `IG_SOURCE`.

- [ ] **Step 6.3: Verify isaacgym can load a gym handle (no sim spawn)**

Run:
```bash
conda run -n phc python -c "from isaacgym import gymapi; g = gymapi.acquire_gym(); print('gym', g is not None)"
```

Expected: `gym True`. If this fails, it is usually a CUDA/driver or libpython mismatch; debug here before moving on.

---

## Task 7: Place SMPL / SMPL-X pickles

**Files:**
- Create: `data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` (6 files total incl. SMPL-X)

- [ ] **Step 7.1: Ensure `data/smpl/` exists**

Run:
```bash
mkdir -p /home/jinsu/Documents/GitHub/PHC/data/smpl && ls -la /home/jinsu/Documents/GitHub/PHC/data/smpl
```

Expected: empty dir listing (just `.` and `..`).

- [ ] **Step 7.2: Copy + rename according to Task 3.2 decision**

If found existing final-named:
```bash
cp /path/to/SMPL_NEUTRAL.pkl /home/jinsu/Documents/GitHub/PHC/data/smpl/SMPL_NEUTRAL.pkl
# ... repeat for all 6
```

If found original basicmodel names (SMPL v1.1.0 zip contents):
```bash
cp /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl /home/jinsu/Documents/GitHub/PHC/data/smpl/SMPL_NEUTRAL.pkl
cp /path/to/basicmodel_m_lbs_10_207_0_v1.1.0.pkl       /home/jinsu/Documents/GitHub/PHC/data/smpl/SMPL_MALE.pkl
cp /path/to/basicmodel_f_lbs_10_207_0_v1.1.0.pkl       /home/jinsu/Documents/GitHub/PHC/data/smpl/SMPL_FEMALE.pkl
# SMPL-X v1.1 files are already correctly named:
cp /path/to/SMPLX_NEUTRAL.pkl /home/jinsu/Documents/GitHub/PHC/data/smpl/
cp /path/to/SMPLX_MALE.pkl    /home/jinsu/Documents/GitHub/PHC/data/smpl/
cp /path/to/SMPLX_FEMALE.pkl  /home/jinsu/Documents/GitHub/PHC/data/smpl/
```

Note: The v1.1.0 zip may ship files *without* the `.pkl` extension. If so, add `.pkl` during the copy: `cp SOURCE_WITHOUT_EXT data/smpl/SMPL_NEUTRAL.pkl`.

- [ ] **Step 7.3: Verify all 6 files are pickles, not text/html**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && ls -la data/smpl/ && file data/smpl/*.pkl
```

Expected: 6 files listed, each `file` line contains "data" or "Python pickle data" (the `file` command may classify them as generic data). Each file should be ≥ 5 MB (SMPL pkls are typically 10-40 MB). A 1 KB file is almost certainly an HTML download-page saved by accident.

---

## Task 8: Run `download_data.sh`

**Files:**
- Create: `sample_data/*`, `output/HumanoidIm/**/Humanoid.pth`

- [ ] **Step 8.1: Verify `gdown` is installed in `phc` env**

Run:
```bash
conda run -n phc gdown --version 2>&1 | head -1
```

Expected: a version string. If missing, `conda run -n phc pip install gdown`.

- [ ] **Step 8.2: Execute the downloader**

Run from repo root, **inside phc env**:
```bash
cd /home/jinsu/Documents/GitHub/PHC && conda run -n phc bash download_data.sh 2>&1 | tee /tmp/phc_download.log
```

Expected: ~12 `gdown` calls, each completing without 403. Total download ~5-10 GB.

If any single link 403s (Google Drive quota), the log will show "Permission denied" or "Cannot retrieve the public link". Mitigation: open that specific gdown URL in a browser, solve any captcha, download manually, place at the expected path. Then continue.

- [ ] **Step 8.3: Verify the two critical files for Task 11**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && \
  ls -la sample_data/amass_isaac_standing_upright_slim.pkl && \
  ls -la output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth
```

Expected: both files exist, `Humanoid.pth` is on the order of tens of MB, not < 1 MB.

---

## Task 9: Sanity check — `vis_motion_mj.py`

**Files:** Read only

- [ ] **Step 9.1 [HITL]: Run visualization script**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && conda run -n phc python scripts/vis/vis_motion_mj.py 2>&1 | tee /tmp/phc_vis_motion.log
```

Expected: a MuJoCo viewer window opens showing an SMPL humanoid. Script exits cleanly when user closes the window.

Common failure: `ModuleNotFoundError: No module named 'mujoco'` — install with `conda run -n phc pip install mujoco`. (README line 67 mentions mujoco is installed via pip.)

- [ ] **Step 9.2 [HITL]: Confirm success**

Ask user: "Did the MuJoCo viewer open and show the humanoid mesh? [y/n]". Block until answer.

---

## Task 10: Sanity check — `joint_monkey_smpl.py`

**Files:** Read only

- [ ] **Step 10.1 [HITL]: Run joint monkey**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && conda run -n phc python scripts/joint_monkey_smpl.py 2>&1 | tee /tmp/phc_joint_monkey.log
```

Expected: an isaacgym viewer with an SMPL humanoid animating through DOFs. Close the window to exit.

Common failure: no viewer (isaacgym GL/Vulkan issue) — confirm `echo $DISPLAY` still set, and `nvidia-smi` lists the GPU. If stuck, try `conda run -n phc python scripts/joint_monkey_smpl.py --sim_device cuda:0 --graphics_device_id 0` or similar.

- [ ] **Step 10.2 [HITL]: Confirm success**

Ask user: "Did the isaacgym viewer open with the humanoid cycling through joint poses? [y/n]". Block until answer.

---

## Task 11: Sanity check — pretrained PHC checkpoint in viewer

**Files:** Read only

- [ ] **Step 11.1 [HITL]: Run the full pretrained eval**

Run from repo root:
```bash
cd /home/jinsu/Documents/GitHub/PHC && conda run -n phc python phc/run_hydra.py \
  learning=im_pnn \
  exp_name=phc_shape_pnn_iccv \
  epoch=-1 \
  test=True \
  env=env_im_pnn \
  robot=smpl_humanoid_shape \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 \
  headless=False \
  2>&1 | tee /tmp/phc_pretrained_sanity.log
```

Expected (gate — all must hold, upstream guide §5.2):
1. Humanoid loads in viewer.
2. Simulation loop runs (step counter or stdout progresses).
3. No "size mismatch" / "state_dict" errors on checkpoint load.
4. No SMPL asset errors ("file not found", "cannot open pkl").
5. No Hydra config errors ("Could not override", "Missing mandatory value").

Close viewer to exit cleanly.

- [ ] **Step 11.2 [HITL]: Confirm success**

Ask user: "Viewer showed the humanoid standing/balancing from the pretrained checkpoint, no errors in the log? [y/n]". Block until answer. If no, paste the relevant log excerpt and debug per spec §7.

---

## Task 12: Write setup log

**Files:**
- Create: `docs/superpowers/specs/phc_v0_setup_log.md`

- [ ] **Step 12.1: Collect evidence into the log**

Write to `/home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/phc_v0_setup_log.md`:

```markdown
# PHC-Pain-v0 Spec #1 — Setup Log

**Date:** 2026-04-22
**Branch:** `jinsu-pain_baseline_phc_v0`
**Commit at setup start:** 846988d
**Spec:** `docs/superpowers/specs/2026-04-22-phc-pain-baseline-v0-design.md`

## Env facts

- `which python` (phc env): <paste Step 4.3 output>
- `python --version` (phc env): 3.8.x
- `torch.__version__`: <paste>
- `torch.cuda.is_available()`: True
- `isaacgym` path (phc env): <paste Step 6.2 output>
- `isaacgym` source root: <paste IG_SOURCE>
- SMPL pkl source decision: <in-place / copied-renamed / re-downloaded>

## Directory listings

### `data/smpl/`
<paste `ls -la data/smpl/`>

### `sample_data/`
<paste `ls sample_data/ | head -20`>

### `output/HumanoidIm/phc_shape_pnn_iccv/`
<paste `ls -la output/HumanoidIm/phc_shape_pnn_iccv/`>

## Sanity command outputs (last ~50 lines each)

### `vis_motion_mj.py`
<paste `tail -50 /tmp/phc_vis_motion.log`>

User confirmation: viewer opened successfully → yes/no

### `joint_monkey_smpl.py`
<paste `tail -50 /tmp/phc_joint_monkey.log`>

User confirmation: viewer opened successfully → yes/no

### `run_hydra.py` pretrained sanity
<paste `tail -50 /tmp/phc_pretrained_sanity.log`>

User confirmation: humanoid loaded, no errors → yes/no

## Exit gate

- [ ] All 7 acceptance criteria from spec §1 hold
- [ ] `git diff phc/` is empty (no code changes under phc/)
- [ ] Setup log committed

## Handoff

Spec #1 complete. Entering Spec #2 (PHC-Pain-v0 implementation, upstream
guide §§6–18).
```

- [ ] **Step 12.2: Verify phc/ untouched**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && git diff --stat phc/ && git status phc/
```

Expected: empty diff. If any file under `phc/` has been modified, revert it — Spec #1 is bring-up only.

---

## Task 13: Commit setup log

- [ ] **Step 13.1: Stage and commit**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && git add docs/superpowers/specs/phc_v0_setup_log.md && \
  git commit -m "$(cat <<'EOF'
Add Spec #1 setup log: PHC env bring-up complete

Records env facts, SMPL/checkpoint file listings, and last ~50 lines of
stdout for vis_motion_mj.py, joint_monkey_smpl.py, and the pretrained
phc_shape_pnn_iccv sanity command. All 7 acceptance criteria met. Ready
to enter Spec #2 (PHC-Pain-v0 implementation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit created, working tree clean.

- [ ] **Step 13.2: Verify branch state**

Run:
```bash
cd /home/jinsu/Documents/GitHub/PHC && git log --oneline -5 && git status
```

Expected: last commit is the setup log commit; one commit back is the spec commit (`ea9855c`); working tree clean (except possibly untracked `reference/`).

---

## Task 14: Exit-gate checklist

- [ ] **Step 14.1: Verify each acceptance criterion from spec §1**

Walk through the spec's 7 acceptance criteria one by one and confirm each in the setup log. If any is missing or ambiguous, go back to the relevant task and re-collect evidence.

1. Humanoid loads in viewer. → Task 11
2. Simulation loop runs. → Task 11
3. No config errors. → Task 11
4. No checkpoint loading errors. → Task 11
5. No SMPL asset errors. → Tasks 9, 10, 11
6. `vis_motion_mj.py` runs without error. → Task 9
7. `joint_monkey_smpl.py` runs without error. → Task 10

- [ ] **Step 14.2 [HITL]: Final user sign-off**

Report to the user:
```
Spec #1 complete. All acceptance gates passed. Branch: jinsu-pain_baseline_phc_v0.
Commits: <spec commit sha>, <setup log commit sha>.
Ready to start Spec #2 (PHC-Pain-v0 implementation)?
```

Block until user gives go-signal.

---

## Out of scope for this plan (forbidden here)

Do **not** during execution of this plan:
- Create any file under `phc/env/` or `phc/data/cfg/env/`
- Edit `phc/utils/parse_task.py`
- Run `scripts/eval_in_isaaclab.py`
- Touch the IsaacLab path at all
- Run any MCP / composer checkpoint command
- Start any training (all commands here use `test=True`)

If any of these comes up during execution, stop and escalate to the user.

---

## Notes for the executor

1. Many steps are `[HITL]` — they require the user to visually confirm viewer windows or to provide file paths. Do not fabricate "y" answers; block and wait.
2. `conda run -n phc <cmd>` is preferred over `conda activate phc` across steps because it is stateless per-shell and reliable inside an agentic runner. Use `conda activate` only if you know the shell persists.
3. Tee every command's output to `/tmp/phc_*.log` so Task 12 can paste from them.
4. Do not `git add -A` anywhere; stage specific files only, to avoid catching multi-GB asset downloads.

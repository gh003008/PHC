# PHC-Pain-v0 — Spec #1: Environment bring-up + Pretrained sanity

**Date:** 2026-04-22
**Author:** jinsu (via Claude Code brainstorming)
**Upstream guide:** `docs/phc_latest_commit_pain_baseline_quickstart_for_claude_code.md`
**Repo:** `ZhengyiLuo/PHC`
**Pinned commit:** `846988d433ce1f341e85ac6fbd2cd51911bb3341` (already the current HEAD)
**Target branch:** `jinsu-pain_baseline_phc_v0`

---

## 0. Series overview (why this is only Spec #1)

The upstream guide covers four phases: environment setup (A), sanity check (B),
PHC-Pain-v0 implementation (C), short fine-tuning (D). The user is brand-new to
this repo, so we separate the work into three sequential specs:

- **Spec #1 (this doc)** — A + B. Exit condition: the official single-primitive
  pretrained checkpoint runs in a viewer on this machine, zero code changes.
- **Spec #2 (future)** — C. PHC-Pain-v0 implementation. Covers guide §§6–18.
- **Spec #3 (future)** — D. `guard_and_reward` fine-tuning. Covers guide §15.

Specs #2 and #3 are **not** in scope here. Any attempt to add pain code or
observation changes in this spec is a scope violation.

---

## 1. Goal & acceptance

### Goal
Bring up a reproducible environment on this machine and confirm the upstream
PHC single-primitive checkpoint runs end-to-end in a viewer. This is the sole
entry gate to the pain implementation work.

### Acceptance (upstream guide §5.2, unchanged)
All of the following must hold on this machine:
1. Humanoid loads in the viewer.
2. Simulation loop runs (non-zero steps, no crash).
3. No config errors.
4. No checkpoint loading errors.
5. No SMPL asset errors.
6. `scripts/vis/vis_motion_mj.py` runs without error.
7. `scripts/joint_monkey_smpl.py` runs without error.

All seven recorded in `docs/superpowers/specs/phc_v0_setup_log.md` as stdout
excerpts (last ~50 lines per command).

---

## 2. Branch & commit strategy

- Create new branch: `git switch -c jinsu-pain_baseline_phc_v0` from current
  `jinsu` branch (HEAD is already `846988d`, the pinned commit — no `git
  checkout <sha>` needed).
- **This spec makes zero code changes.** Changes are limited to:
  - The spec file itself (this document).
  - The setup log file (`docs/superpowers/specs/phc_v0_setup_log.md`).
  - Local untracked data directories (`data/smpl/`, `sample_data/`, `output/`)
    which must NOT be committed (should be in `.gitignore` already; verify).
- Commit boundary for Spec #1:
  1. Commit this design doc + empty setup log skeleton.
  2. Commit populated setup log after all 9 steps pass.

---

## 3. Nine-step execution order (fixed)

Each step has a verify command. If the verify fails, stop and debug before
moving on. Do not skip ahead.

| # | Action | Verify |
|---|---|---|
| 1 | `conda create -n phc python=3.8 -y && conda activate phc` | `conda env list` shows `phc` as active env |
| 2 | `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y` | `python -c "import torch; print(torch.cuda.is_available())"` → `True` |
| 3 | `pip install -r requirement.txt` | exit 0, no dependency-resolver errors |
| 4 | Locate + link isaacgym preview4 source. See §3.0 below. | `python -c "import isaacgym; print(isaacgym.__file__)"` (inside `phc` env) prints a path |
| 5 | Copy + rename the 6 SMPL / SMPL-X pickles into `data/smpl/` (see §3.1 below) | `ls data/smpl/` shows all 6 files |
| 6 | `bash download_data.sh` | `sample_data/amass_isaac_standing_upright_slim.pkl` **and** `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth` exist |
| 7 | `python scripts/vis/vis_motion_mj.py` | viewer opens, clean exit on close |
| 8 | `python scripts/joint_monkey_smpl.py` | viewer opens, clean exit on close |
| 9 | Upstream guide §5.2 command (full line in §3.2 below) | viewer shows humanoid, sim loop runs, no errors |

### 3.0 Locating isaacgym preview4 (step 4)

The user does not remember the install path but uses isaacgym through the
existing `rl_mpc` conda env. Discover the path once, reuse it:

```bash
conda activate rl_mpc
python -c "import isaacgym, os; print(os.path.dirname(os.path.dirname(isaacgym.__file__)))"
conda deactivate
```

That prints the isaacgym preview4 **python package root**. The source tree's
`setup.py` lives one level up (the dir that contains `python/`, `assets/`,
etc.). Navigate there, then inside the `phc` env:

```bash
conda activate phc
cd <isaacgym_source_root>/python
pip install -e .
```

If the discovery fails (e.g. `rl_mpc` import broken), fall back to manual
re-download from https://developer.nvidia.com/isaac-gym. Note that the
preview-4 download requires NVIDIA developer account login, so the re-download
path is slow.

### 3.1 SMPL file placement (step 5)

User already has SMPL licenses but is not certain which files are on disk
and in what form. Execution plan should first `find ~ -iname 'SMPL_*.pkl'
-o -iname 'basicmodel_*.pkl' 2>/dev/null` to probe for existing copies before
deciding to re-download from smpl.is.tue.mpg.de / smpl-x.is.tue.mpg.de.

Required files in `data/smpl/`:

| Final filename | Source | Original filename |
|---|---|---|
| `SMPL_NEUTRAL.pkl` | smpl.is.tue.mpg.de, v1.1.0 | `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` |
| `SMPL_MALE.pkl` | smpl.is.tue.mpg.de, v1.1.0 | `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` |
| `SMPL_FEMALE.pkl` | smpl.is.tue.mpg.de, v1.1.0 | `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` |
| `SMPLX_NEUTRAL.pkl` | smpl-x.is.tue.mpg.de, v1.1 | `SMPLX_NEUTRAL.pkl` (no rename) |
| `SMPLX_MALE.pkl` | smpl-x.is.tue.mpg.de, v1.1 | `SMPLX_MALE.pkl` (no rename) |
| `SMPLX_FEMALE.pkl` | smpl-x.is.tue.mpg.de, v1.1 | `SMPLX_FEMALE.pkl` (no rename) |

### 3.2 Step 9 — sanity command (exact)

```bash
python phc/run_hydra.py \
  learning=im_pnn \
  exp_name=phc_shape_pnn_iccv \
  epoch=-1 \
  test=True \
  env=env_im_pnn \
  env.num_prim=4 \
  robot=smpl_humanoid_shape \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 \
  headless=False
```

Same as upstream guide §5.2 and README lines 250–251, **plus `env.num_prim=4`
added after execution discovered the shape mismatch** — see Task 11 note in
`docs/superpowers/specs/phc_v0_setup_log.md`. The downloaded
`phc_shape_pnn_iccv/Humanoid.pth` checkpoint has 4 PNN primitives
(`pnn.actors.0..3`) but `env_im_pnn.yaml` defaults to `num_prim: 3`; without
the override, `load_state_dict` throws `Unexpected key(s): ...actors.3...`.
Both upstream guide §5.2 and README line 250 omit this override — treat as
upstream documentation oversight.

Single-primitive-mode checkpoint (`phc_shape_pnn_iccv`), not MCP/composer.
env `env_im_pnn`, not `env_im_getup_mcp`. ("Single primitive" here means
"without MCP composer", not "1 PNN primitive" — PNN count is 4.)

---

## 4. Out of scope (Spec #1)

Do not do any of these here; they belong to Spec #2 or Spec #3:

- Creating `phc/env/tasks/humanoid_im_pain.py`
- Creating `phc/env/util/pain_baseline.py`
- Creating `phc/data/cfg/env/env_im_pain.yaml`
- Editing `phc/utils/parse_task.py`
- Running `scripts/eval_in_isaaclab.py` (upstream guide §1.1 forbids using the
  IsaacLab eval path for the pain baseline)
- Any MCP / getup composer commands (`env.models=[...]`)
- AMASS full-dataset processing or training
- Any change to observation size, action space, or network builder

If during Spec #1 execution you find yourself editing any file under `phc/`,
stop and re-read this section.

---

## 5. Risks and mitigations

1. **Google Drive `gdown` quota (step 6).** `gdown` can 403 when a file hits
   its 24h download limit. Mitigation: open the Google Drive URL in a browser,
   confirm manually, save the file to the expected path. Re-run
   `download_data.sh` to fill the rest.

2. **isaacgym preview4 Python compatibility.** preview4 officially supports
   Python 3.6–3.8. We use 3.8, so this is safe, but the `pip install -e .`
   step must run inside the activated `phc` env.

3. **SMPL pickle filename drift.** The SMPLSim code loads by exact filename.
   If a file is named `SMPL_Neutral.pkl` (wrong case) or the base model pickle
   lacks the `.pkl` extension (SMPL v1.1.0 zip ships them without an
   extension), loading will silently fail or raise `FileNotFoundError`.
   Mitigation: `file data/smpl/*.pkl` should report "Python pickle data" for
   all six; if any line lacks the extension, rename.

4. **Display environment.** User confirmed display is available, so
   `headless=False` should work. If running over SSH later, remember
   `pyvirtualdisplay` / Xvfb fallback exists (README line 189) but we aren't
   using it in Spec #1.

5. **`requirement.txt` vs PyTorch CUDA wheel mismatch.** `requirement.txt`
   may pin versions that conflict with the separately-installed torch. If pip
   complains about torch version, prefer the torch we installed in step 2 and
   use `pip install --no-deps` for offending lines, or simply ignore if torch
   stays importable and CUDA is available.

6. **SMPLSim install.** README line 49 mentions an explicit SMPLSim install
   (`pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master`). If
   `requirement.txt` does not already include it, run that manually. Verify
   with `python -c "import smpl_sim"` (or the correct module name; check the
   SMPLSim README).

---

## 6. Deliverables

1. **This design doc** at
   `docs/superpowers/specs/2026-04-22-phc-pain-baseline-v0-design.md`.
2. **New branch** `jinsu-pain_baseline_phc_v0` created and checked out.
3. **Setup log** at `docs/superpowers/specs/phc_v0_setup_log.md` containing:
   - Output of `which python` and `python --version` inside the `phc` env.
   - Output of `ls data/smpl/` (expect 6 files).
   - Output of `ls sample_data/` (expect the standing pkl and friends).
   - Output of `ls output/HumanoidIm/phc_shape_pnn_iccv/` (expect
     `Humanoid.pth`).
   - Last ~50 lines of stdout for each of: `vis_motion_mj.py`,
     `joint_monkey_smpl.py`, the §3.2 sanity command.
4. **Two commits** on branch `jinsu-pain_baseline_phc_v0`:
   - Commit 1: add this design doc + empty log skeleton.
   - Commit 2: populated setup log after step 9 passes.
5. **Handoff note** at the bottom of the setup log: "Spec #1 complete,
   entering Spec #2 (PHC-Pain-v0 implementation)."

---

## 7. Debug hints (for the executing session)

Mapped from upstream guide §19, expanded:

- **Checkpoint load error "size mismatch"**
  → wrong learning config. Re-check that step 9 uses `learning=im_pnn` and
  `env=env_im_pnn`, not `im_mcp` / `env_im_getup_mcp`.

- **`RuntimeError: CUDA error` on isaacgym import**
  → CUDA/driver mismatch. Run `nvidia-smi`, confirm driver ≥ required by
  cuda 11.6.

- **`FileNotFoundError` on `sample_data/amass_...pkl`**
  → step 6 didn't finish. Re-run or download the one file manually.

- **Viewer black / frozen**
  → pyvirtualdisplay kicked in inadvertently. Add `no_virtual_display=True`
  to the Hydra command.

- **`ImportError: No module named 'isaacgym'`**
  → step 4 was done in wrong conda env. `conda activate phc` first, then
  re-run `pip install -e .` in the isaacgym preview4 source dir.

- **`ModuleNotFoundError: No module named 'smpl_sim'` (or similar)**
  → SMPLSim not installed. See risk §5.6.

- **Hydra error "cannot find config"**
  → you are not running from the repo root. `cd` to the PHC repo root
  before running.

---

## 8. Exit gate to Spec #2

Spec #1 is DONE when:

- All 9 steps in §3 have verify-pass outputs in the setup log.
- No files under `phc/` have been modified (`git diff phc/` is empty).
- Commit 2 (populated log) has been made on `jinsu-pain_baseline_phc_v0`.
- User has read the setup log and given a go-signal to proceed to Spec #2.

Only then do we begin Spec #2 (PHC-Pain-v0 implementation).

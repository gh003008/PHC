# PHC-Pain-v0 Spec #1 — Setup Log

**Date:** 2026-04-22
**Branch:** `jinsu-pain_baseline_phc_v0`
**Commit at setup start:** `846988d` (HEAD of `jinsu` branch before we began)
**Spec:** `docs/superpowers/specs/2026-04-22-phc-pain-baseline-v0-design.md`
**Plan:** `docs/superpowers/plans/2026-04-22-phc-pain-v0-spec1-env-bringup.md`

---

## Env facts

- `which python` (phc env): `/home/jinsu/miniconda3/envs/phc/bin/python`
- `python --version`: `Python 3.8.20`
- `torch.__version__`: `1.13.1`
- `torch.cuda.is_available()`: `True`
- `numpy.__version__`: `1.23.5` (downgraded from 1.24.3 — see Task 5 note)
- `isaacgym.__file__`: `/home/jinsu/isaacgym/python/isaacgym/__init__.py`
- `isaacgym` source root: `/home/jinsu/isaacgym/`
- `conda env config vars list -n phc`: `LD_LIBRARY_PATH = /home/jinsu/miniconda3/envs/phc/lib`
- SMPL pkl source: `SMPL_python_v.1.1.0.zip` + `models_smplx_v1_1.zip` unzipped to `/tmp/phc_smpl_extract/`, then copy+renamed into `data/smpl/`.

## Directory listings

### `data/smpl/` (2.3G)
```
-rw-r--r-- 247530000 SMPL_FEMALE.pkl
-rw-r--r-- 247101031 SMPL_MALE.pkl
-rwxr-xr-- 247186228 SMPL_NEUTRAL.pkl
-rwxr-x--- 544434140 SMPLX_FEMALE.pkl
-rwxr-x--- 544477159 SMPLX_MALE.pkl
-rwxr-x--- 544173380 SMPLX_NEUTRAL.pkl
```
All 6 pkls load cleanly with `pickle.load(..., encoding='latin1')`, top-level keys include
`J_regressor`, `kintree_table`, `v_template`, `weights`, `shapedirs`, `posedirs`, `f`.

### `sample_data/` (8.8M)
```
amass_copycat_occlusion_v3.pkl
amass_isaac_gender_betas.pkl
amass_isaac_gender_betas_unique.pkl
amass_isaac_standing_upright_slim.pkl    ← used by sanity cmd
dance_sample_g1.pkl
dance_sample_h1.pkl
standing_x.pkl
```

### `output/HumanoidIm/phc_shape_pnn_iccv/` (94 MB pth)
```
-rw-rw-r-- 98832209 Humanoid.pth
drwxrwxr-x        0 .hydra/
-rw-rw-r--        0 phc_shape_pnn_iccv.log
```
Plus 9 other checkpoint dirs under `output/HumanoidIm/` (phc_3, phc_kp_2, phc_x_pnn, etc.).
Total `output/` = 2.3 GB.

## Sanity command outputs (last ~50 lines each)

### Task 9 — `vis_motion_mj.py` (MuJoCo viewer)

```
logger initialized
(progress bars 1/1)
Xlib:  extension "NV-GLX" missing on display ":1".    <-- benign: mesa/nvidia mismatch warning
MOVING MOTION DATA TO GPU, USING CACHE: False
!!!! Using modified SMPL starting pose !!!!
SIM FPS: 30.0
Loaded 1 motions with a total length of 33.300s and 1000 frames.
```
Exit code: 0. User visually confirmed MuJoCo viewer opened with SMPL humanoid.

### Task 10 — `joint_monkey_smpl.py` (isaacgym viewer)

```
Animating DOF 0 ('L_Hip_x') ... Animating DOF 68 ('R_Hand_z')
Animating DOF 0 ('L_Hip_x') ... Animating DOF 12 ('R_Hip_x')
Done
WARNING: Forcing CPU pipeline.   <-- benign: isaacgym default for vis scripts
```
Exit code: 0. User visually confirmed isaacgym viewer cycled through all 69 DOFs.

### Task 11 — pretrained `phc_shape_pnn_iccv` eval

**Command (as actually run — note `env.num_prim=4` override added to the spec's §3.2 command):**
```bash
conda run -n phc python phc/run_hydra.py \
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

Last ~30 lines of `/tmp/phc_pretrained_sanity.log`:
```
(all 4 primitives' actor weights listed: pnn.actors.0..3)
RunningMeanStd:  (2070,)
=> loading checkpoint 'output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'
=> loading checkpoint 'output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'
reward: 941.05078125 steps: 999.0
```

Exit code: 0. No Traceback, no state_dict mismatch, no SMPL asset errors.
User visually confirmed humanoid balanced in the isaacgym viewer.

**Observation_space:** `Box(-inf, inf, (945,), float32)` — important: this is the obs dim
that Spec #2 must not change (pain must NOT be appended to obs in v0).

## Deviations from the original plan

These are worth recording for Spec #2 and for future bring-ups on this machine:

1. **conda solver picks CPU pytorch by default.** Plan's `conda install pytorch torchvision
   torchaudio pytorch-cuda=11.6` resolved to `pytorch-mutex=cpu` → `torch.cuda.is_available()
   == False`. Fix: explicitly pin `'pytorch=1.13.1=py3.8_cuda11.6*'` + `'pytorch-mutex=1.0=cuda'`.
   Saved to memory `feedback_conda_pytorch_pin.md`.

2. **isaacgym needs `LD_LIBRARY_PATH` on the env.** `conda run` and `conda activate` do NOT
   propagate `<env>/lib` onto `LD_LIBRARY_PATH`. Symptom: `ImportError: libpython3.8.so.1.0`.
   Fix: `conda env config vars set LD_LIBRARY_PATH=/home/jinsu/miniconda3/envs/phc/lib -n phc`.

3. **`requirement.txt` pulls numpy 1.24.3** which breaks isaacgym preview4 (`np.float` removed).
   Fix: `pip install 'numpy<1.24'` → numpy 1.23.5. Same fix handles chumpy's `np.bool`/`np.int`
   deprecations.

4. **Spec's §3.2 sanity command needs `env.num_prim=4`.** The downloaded
   `phc_shape_pnn_iccv/Humanoid.pth` checkpoint has 4 PNN primitives (`actors.0..3`) but
   `env_im_pnn.yaml` defaults `num_prim: 3`. Without override: `RuntimeError: Error(s) in
   loading state_dict for Network: Unexpected key(s) ...pnn.actors.3...`. The README's
   exact command (line 250) also omits `env.num_prim=4` — treat as upstream doc oversight.
   **Spec #2 and any future pretrained eval must include `env.num_prim=4`.**

5. **`.gitignore` gaps.** Plan Task 1.2 flagged `sample_data/` not explicitly gitignored and
   `*.zip` missing. Both added in commits `6abe5c7` and `529bd77`.

6. **Warning: `pytorch_lightning==2.4.0`** pulled by `requirement.txt` requires torch >= 2.1,
   but we force torch 1.13.1. No PHC code actually imports `pytorch_lightning` (grep confirmed,
   2026-04-22), so this is inert. If Spec #2 code ever imports PL, downgrade with
   `pip install 'pytorch_lightning<2.0' --no-deps`.

## Exit gate (spec §1 acceptance criteria)

- [x] 1. Humanoid loads in viewer (Task 11, user confirmed)
- [x] 2. Simulation loop runs — episode ran 999 steps with reward 941.05 (Task 11)
- [x] 3. No config errors (Task 11)
- [x] 4. No checkpoint loading errors (Task 11, after `env.num_prim=4` fix)
- [x] 5. No SMPL asset errors (Tasks 9, 10, 11 all clean)
- [x] 6. `vis_motion_mj.py` runs without error (Task 9)
- [x] 7. `joint_monkey_smpl.py` runs without error (Task 10)

- [x] `git diff phc/` is empty (no code changes under `phc/` — verified: "작업 폴더 깨끗함")
- [x] Setup log committed (this file, in a follow-up commit)

## Handoff

Spec #1 complete. Entering Spec #2 (PHC-Pain-v0 implementation, upstream guide §§6–18).

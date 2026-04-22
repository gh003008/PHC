# PHC-Pain-v0 Spec #3 — Implementation Log

**Date started:** 2026-04-22
**Branch:** `jinsu-pain_baseline_phc_v0`
**Spec:** `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec3-finetune-design.md`
**Plan:** `docs/superpowers/plans/2026-04-22-phc-pain-v0-spec3-impl.md` (written after spec review).

Format mirrors `phc_v0_spec2_impl_log.md`.

---

## V5.0 — GPU calibration (RTX 3070 Ti, 8 GB)

_Rungs run in order; accept largest rung with peak VRAM ≤ 85%._

### Rung 1 — num_envs=1024, horizon_length=32

_Pending._

### Rung 2 — num_envs=512, horizon_length=64

_Pending._

### Rung 3 — num_envs=256, horizon_length=128

_Pending._

### Rung 4 — num_envs=128, horizon_length=256

_Pending._

**Selected rung:** _TBD after calibration._

---

## V5.1 — Checkpoint copy + sanity regression

_Verify `output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth` reproduces
Spec #1 baseline (reward within ~1% of 941.05)._

_Pending._

---

## V5.2 — Retuned guard_only survival gate

_Accept iff `steps ≥ 300` AND `reward ≥ 470` under `env=env_im_pain_finetune`,
`pain.mode=guard_only`._

_Pending._

---

## V5.3 — Fine-tuning run

_2000 epochs OR 30 min wall-clock under `pain.mode=guard_and_reward`. Accept
iff training completes the budget without reward collapse and a new
`Humanoid.pth` is saved._

_Pending._

---

## V6 — Post-training log_only probe

_Accept iff steady-state `pain_max ≤ 0.35` (Spec #2 V3 was 0.522)._

_Pending._

---

## V7 — Post-training guard_only probe

_Accept iff `steps ≥ 500` AND `reward ≥ 470` (Spec #2 V4 was 38 / 27.15)._

_Pending._

---

## Deviations from the spec

_None yet — filled in as implementation progresses._

---

## Exit gate

- [ ] V5.0 records calibrated `num_envs` / `horizon_length`.
- [ ] V5.1 checkpoint-copy reward within 1% of 941.05.
- [ ] V5.2 survival gate passes (≥300 steps, ≥470 reward).
- [ ] V5.3 training completes 2000 epochs OR 30 min without reward collapse;
      new `Humanoid.pth` present under
      `output/HumanoidIm/phc_shape_pnn_iccv_pain/`.
- [ ] wandb run exists and contains pain_* scalars (mean/max/p90/p99/sat/5
      body-group means).
- [ ] V6 `pain_max ≤ 0.35`.
- [ ] V7 `steps ≥ 500` AND `reward ≥ 470`.
- [ ] `git diff phc/` since Spec #2's last impl commit touches only
      1 new yaml + ≤30 lines edit to `humanoid_im_pain.py`.
- [ ] User has reviewed this log and approved v0 as shipped.

---

## Handoff

Spec #3 closes PHC-Pain-v0. Next work (if pursued) moves to a new branch
`pain_baseline_phc_v1_*` — obs-appended pain, MPJPE gating, or multi-motion
fine-tune. All three are out of scope for v0.

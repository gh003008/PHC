# Phase 4 Summary: Training Conditions And Ablations

**Status:** Complete
**Completed:** 2026-04-27

## What Changed

- Added `docs/superpowers/phc_pain_v1_ablation_commands.md` with bounded Hydra
  commands for the main condition, no-obs ablation, no-reward ablation, and
  left-knee side-specific run.
- Adjusted `HumanoidImPainV1` so reward computation remains active when pain
  observation is disabled.
- Added extras flags showing whether v1 pain observation and reward cost are
  enabled.

## Verification

- `python3 -m py_compile phc/env/tasks/humanoid_im_pain.py`
- Command matrix grep for required conditions.
- `git diff --check`

## Deferred

- Phase 5 owns the evaluation evidence package.

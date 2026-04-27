# Phase 3 Summary: Expanded-Observation Checkpoint Adaptation

**Status:** Complete
**Completed:** 2026-04-27

## What Changed

- Added conditional PHC-Pain-v1 checkpoint adaptation in `AMPAgent`.
- Expanded model input tensors by copying pretrained leading observation columns
  and zero-initializing new pain-observation columns.
- Expanded input running mean/std with zero means and unit variances for new
  pain observation dimensions.
- Reset optimizer state when an expanded-observation checkpoint adaptation occurs.
- Added checkpoint adaptation config documentation to `env_im_pain_v1.yaml`.

## Verification

- `python3 -m py_compile phc/learning/amp_agent.py phc/env/tasks/humanoid_im_pain.py`
- Phase 3 config sanity check.
- `git diff --check`

## Deferred

- Phase 4 owns training condition commands and ablation matrix.

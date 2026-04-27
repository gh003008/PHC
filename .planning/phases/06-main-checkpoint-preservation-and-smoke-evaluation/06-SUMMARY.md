# Phase 6 Summary: Main Checkpoint Preservation And Smoke Evaluation

**Status:** Complete
**Completed:** 2026-04-27
**Verdict:** MAIN_CHECKPOINT_USABLE

## What Ran

Preserved the current main-condition checkpoint on `server1-jinsu` and ran a
no-training `test=True` evaluation smoke on Slurm `idx1`.

## Server Artifacts

- Checkpoint backup:
  `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/evidence/Humanoid_short50_20260427_2314.pth`
- Smoke log backup:
  `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/evidence/phc_pain_v1_smoke_3967.out`
- Short50 log backup:
  `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/evidence/phc_pain_v1_short50_3968.out`
- Eval smoke logs:
  `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/evidence/phc_pain_v1_eval_3969.out`
  `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/evidence/phc_pain_v1_eval_3969.err`

## Eval Smoke

- Slurm job: `3969`
- Partition/GRES: `idx1`, `gres:gpu:idx1:1`
- Mode: `test=True`, `env.num_envs=1`, `games_num=1`
- Observation shape: `963`
- Network build shape: `build mlp: 963`
- Checkpoint restore: `output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/Humanoid.pth`
- Result: `reward: 936.7422485351562`, `steps: 999.0`

## Notes

The eval smoke confirms that the expanded pain-observation checkpoint can be
restored by the player path and complete a no-training rollout. It does not by
itself prove the pain mechanism.

# VIC4 + v_cmd — 3-Slot Training Snapshot

**Date**: 2026-04-27
**Spec**: `docs/superpowers/specs/2026-04-27-vic4-vcmd-3slot-design.md`
**Plan**: `docs/superpowers/plans/2026-04-27-vic4-vcmd-3slot-plan.md`

## Slots

| Slot | Task | Motion | termDist |
|---|---|---|---|
| S4 | HumanoidImVICCmdRetime | single (KIT_11) | 0.4 |
| S5 | HumanoidImVICCmdMultiClip | 3 clips spaced | 0.4 |
| S6 | HumanoidImVICCmdMultiClip | 3 clips spaced | 0.6 |

All slots: stateInit=Start (t₀=0), retime [0.85, 1.15], cmd_tracking_w=0.0,
vic_ccf_num_groups=4, cycle_motion=True, no action smoothness penalty.

## Submission

```bash
sbatch exp_config/forward_walking/260427_VIC4_VCMD/train_S4_gpu0.sh
sbatch exp_config/forward_walking/260427_VIC4_VCMD/train_S5_gpu1.sh
sbatch exp_config/forward_walking/260427_VIC4_VCMD/train_S6_gpu2.sh
```

## Post-training analysis

See `02_research_dev/260428_vic4_vcmd_results.md` (to be written).

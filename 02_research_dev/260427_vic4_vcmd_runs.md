# VIC4+v_cmd 3-Slot Job Submission

Submitted: 2026-04-27 (KST)
Spec: `docs/superpowers/specs/2026-04-27-vic4-vcmd-3slot-design.md`
Plan: `docs/superpowers/plans/2026-04-27-vic4-vcmd-3slot-plan.md`

## Submitted jobs

| Slot | Job ID | GPU | Started | Sbatch |
|---|---|---|---|---|
| S4 INSURANCE | 3957 | idx0 | 2026-04-27 (server1) | train_S4_gpu0.sh |
| S5 PRIMARY | 3958 | idx1 | 2026-04-27 (server1) | train_S5_gpu1.sh |
| S6 MARGIN | 3959 | idx2 | 2026-04-27 (server1) | train_S6_gpu2.sh |

## Setup

All slots: 20000 epochs, num_envs=512, stateInit=Start, retime [0.85, 1.15],
cmd_tracking_w=0.0, vic_ccf_num_groups=4, fallInit/hybridInit unused
(HumanoidImVIC has no getup logic).

## Slot differences

| Item | S4 | S5 | S6 |
|---|---|---|---|
| Task | Retime | MultiClip | MultiClip |
| Motion clips | 1 (KIT_11) | 3 spaced (walking_03/medium08/medium05) | 3 spaced (same as S5) |
| terminationDistance | 0.4 | 0.4 | 0.6 |
| v_cmd range | [0.4675, 0.6325] auto | [0.32, 0.75] | [0.32, 0.75] |

## Motion data (S5/S6)

User-selected 3 KIT_425 clips per option B-1+c (signed v_x < 0 = forward in dataset frame):

| Role | Clip | Frames | end-to-end \|v_x\| | mid-clip \|v_x_mean_mid\| |
|---|---|---|---|---|
| Slow | 0-KIT_425_walking_03_poses | 195 | 0.312 | 0.371 |
| Medium | 0-KIT_425_walking_medium08_poses | 162 | 0.528 | 0.652 |
| Fast | 0-KIT_425_walking_medium05_poses | 127 | 0.683 | 0.845 |

Coverage gap [0.427, 0.554] (0.127 m/s) accepted to keep retime narrow ±15%.

## Initial verification (Ep ~10)

| Slot | rwd | eps_len | per-frame rwd |
|---|---|---|---|
| S4 | 6.2 | 2.0 | 3.1 |
| S5 | 14.1 | 4.4 | 3.2 |
| S6 | 75.1 | 23.9 | 3.1 |

All 3 starting per-frame rewards match VIC4 baseline ~3.19.
S6 has longest eps_len due to looser termDist=0.6.

## Post-training plan

1. rsync canonical + Ep10000 milestones from server
2. `python scripts/test_termination_stats.py --slot S{4,5,6} --num_envs 32 --max_resets 1500`
3. `python scripts/vis_vic4_vcmd.py --slot S{4,5,6} --num_envs 8`
4. Write `02_research_dev/260428_vic4_vcmd_results.md` per spec §10

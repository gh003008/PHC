# 260426 KIT_425 Continuous Walking — 3-Slot Batch

Spec: `docs/superpowers/specs/2026-04-26-kit425-continuous-walking-design.md`

| Slot | Env yaml | Sbatch | Hypothesis |
|---|---|---|---|
| S1 TERM_OFF | `env_im_walk_vic_kit425_cont_S1_termoff.yaml` | `train_S1_termoff_gpu0.sh` | terminationDistance disabling alone unlocks ceiling |
| S2 CYCLE_BASE | `env_im_walk_vic_kit425_cont_S2_cyclebase.yaml` | `train_S2_cyclebase_gpu1.sh` | cycle_motion=True enables continuous walking |
| S3 CYCLE_VCMD_HOP | `env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml` | `train_S3_vcmdhop_gpu2.sh` | cycle-boundary v_cmd resampling robust |

All three: terminationDistance disabled (100.0), fallInit/hybridInit OFF, v_cmd ∈ [0.40, 0.82], retime [0.7, 1.4], 20k epochs.

Predecessor: `260424_KIT425_PERSONAL_VCMD/` (3-way A/B/C ablation, all 44-59% of ceiling).

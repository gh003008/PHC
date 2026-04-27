# Phase 7 Summary: Main Run Evidence Extraction

**Status:** Complete
**Completed:** 2026-04-27
**Verdict:** METRICS_BLOCKED

## What Was Extracted

The main run evidence was extracted from server stdout logs and TensorBoard
events for `phc_shape_pnn_iccv_pain_v1`.

## Main Run Competence Signals

From TensorBoard event
`~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/Humanoid_27-23-11-45/summaries/events.out.tfevents.1777299180.server1`:

| Metric | First | Last | Last 10 Avg |
|---|---:|---:|---:|
| reward | 15.381918907165527 | 350.2726135253906 | 335.4804260253906 |
| episode length | 19.0 | 461.21697998046875 | 441.7922332763672 |
| body position reward | 0.9596408605575562 | 0.9402341842651367 | 0.9408837258815765 |
| body rotation reward | 0.7583709359169006 | 0.7526975274085999 | 0.7542606890201569 |
| power reward | -0.07893506437540054 | -0.08625558018684387 | -0.08669018968939782 |

## Pain Metric Availability

Required pain keys were not present in stdout, stderr, or TensorBoard scalar
tags:

- `pain_v1_right_knee_load`
- `pain_v1_right_knee_drive`
- `pain_v1_right_knee_state`
- `pain_v1_reward_cost_mean`

TensorBoard scalar tags contained reward, episode length, discriminator, and
imitation reward components, but no tag containing `pain`.

## Evidence Artifact

- Local summary JSON:
  `.planning/phases/07-main-run-evidence-extraction/main_run_summary.json`

## Interpretation

The main run is usable as a restore/training-stability signal. It is not yet
usable for the PHC-Pain-v1 mechanism claim because the pain/load metrics needed
for Gate 1 and Gate 3 were not logged.

Next step before ablations: add or enable training scalar logging for the
`pain_v1_*` extras, then rerun a short instrumentation probe before spending GPU
time on the full ablation matrix.

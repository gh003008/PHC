# Phase 8 Summary: Minimal Ablation Runs

**Status:** Blocked
**Completed:** 2026-04-27
**Verdict:** BLOCKED_BY_MISSING_PAIN_METRICS

## What Happened

Phase 8 did not launch ablation jobs. This is intentional: Phase 7 found that
the current training outputs do not contain the required `pain_v1_*` metrics.
Running `main/no_obs/no_reward` ablations now would consume GPU time without
producing the pain/load evidence required by the evaluation protocol.

## Blocking Evidence

Missing required scalar metrics:

- `pain_v1_right_knee_load`
- `pain_v1_right_knee_drive`
- `pain_v1_right_knee_state`
- `pain_v1_reward_cost_mean`

Available evidence is limited to reward, episode length, imitation reward
components, and eval-smoke restore success.

## Required Before Retry

Add or enable scalar logging for the `pain_v1_*` extras during training, then
run a small instrumentation probe. Only after pain metrics appear in logs or
event files should the minimal ablation matrix be launched.

## Deferred Ablations

- main: obs on, reward on
- no_obs: obs off, reward on
- no_reward: obs on, reward off

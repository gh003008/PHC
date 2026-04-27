# Phase 8 Plan Verification

**Status:** PASS

The plan is executable because it starts from the documented main run, requires
Phase 7 metric readiness before launching more jobs, and isolates each ablation
into its own checkpoint/log namespace.

## Checks

- It covers the minimum causal matrix: main, no_obs, no_reward.
- It preserves Slurm GPU isolation.
- It includes the known command fixes from the smoke run.
- It blocks ablations if pain metrics are not extractable.

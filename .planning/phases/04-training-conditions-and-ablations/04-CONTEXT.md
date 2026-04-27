# Phase 4: Training Conditions And Ablations - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Autonomous auto-discuss

<domain>
## Phase Boundary

Define the v1.0 condition matrix so the mechanism proof can distinguish pain
observability, reward pressure, and side-specific impairment effects.
</domain>

<decisions>
## Implementation Decisions

### Use Hydra overrides for ablations

Keep one canonical `env_im_pain_v1.yaml` and express main/ablation differences
with Hydra overrides. This avoids duplicating large env yaml files.

### Do not run long training locally

Phase 4 creates reproducible bounded commands. Full PPO runs are expensive and
belong to the operator/GPU execution step after condition wiring is reviewed.
</decisions>

<code_context>
## Existing Code Insights

- `pain_obs_off + pain_reward_on` requires reward computation even when
  observation append is disabled.
- `env.pain.mode=log_only` is the clean reward-off ablation because pain
  computation/logging still runs while reward cost is disabled.
</code_context>

<specifics>
## Specific Ideas

- Add a command matrix under `docs/superpowers/`.
- Include main, no-obs, no-reward, and left-side runs.
- Mark guard-enabled runs as debug-only.
</specifics>

<deferred>
## Deferred Ideas

- Actual result interpretation belongs to Phase 5.
</deferred>

# Phase 2: Synthetic Pain And Reward Mechanism - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Autonomous auto-discuss

<domain>
## Phase Boundary

Implement the v1.0 synthetic unilateral medial tibiofemoral knee pain drive and
wire it as reward-only training pressure.

This phase must keep the pain model wording narrow: it is a thresholded,
sensitivity-weighted mechanical-provocation proxy, not perceived pain.
</domain>

<decisions>
## Implementation Decisions

### Use available PHC tensors only

Compute the first v1.0 knee load proxy from reliable existing tensors:

- knee flexion-axis torque;
- positive flexion torque proxy;
- near-ROM-limit proxy;
- positive knee work proxy.

Medial KAM/contact force is not directly available in PHC tensors, so it is
logged as unavailable rather than invented.

### Main reward path excludes action guard

`HumanoidImPainV1._compute_reward()` should compute the base PHC imitation reward,
update v1 pain buffers, and subtract `lambda_p * affected_knee_pain_state`.
The inherited action guard remains available only when a guard mode is selected,
but v1 config uses `reward_only`.
</decisions>

<code_context>
## Existing Code Insights

- v0 `HumanoidImPain._compute_reward()` subtracts mean DOF pain. v1 needs a
  separate reward path so the main cost is affected body-part knee pain.
- `Humanoid` already exposes knee flexion DOF conventions through
  `L_Knee/R_Knee` body-name indices and flexion axis `* 3 + 1`.
- Existing pure-torch helpers can be reused for ROM near-limit pain.
</code_context>

<specifics>
## Specific Ideas

- Add `pain_body_drive`, side-specific `knee_load_proxy`, and `knee_drive`
  buffers.
- Add `pain.knee_mechanism` config for sensitivity, threshold, component
  weights, and memory smoothing.
- Expose affected and unaffected side metrics through `extras`.
</specifics>

<deferred>
## Deferred Ideas

- Checkpoint adaptation belongs to Phase 3.
- Training matrix belongs to Phase 4.
- Evidence package belongs to Phase 5.
</deferred>

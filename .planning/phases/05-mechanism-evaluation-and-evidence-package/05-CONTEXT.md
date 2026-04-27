# Phase 5: Mechanism Evaluation And Evidence Package - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Autonomous auto-discuss

<domain>
## Phase Boundary

Create the evaluation protocol and evidence package structure that lets v1.0
results be judged as mechanism proof, failure, or inconclusive without drifting
into patient-specific claims.
</domain>

<decisions>
## Implementation Decisions

### Gate aggregate metrics, not raw logs

Use a small evaluator over extracted aggregate metrics. Raw IsaacGym logs vary
by run setup; the final v1.0 claim should be based on explicit condition-level
metrics.

### Separate success gates from diagnostics

Pain/load reduction, locomotion competence, and side-specificity are gates.
Gait pattern features are diagnostics until patient data exists.
</decisions>

<code_context>
## Existing Code Insights

- Phase 4 documents the required condition matrix.
- v1.0 does not yet include actual PPO outputs, so Phase 5 should provide the
  protocol and template rather than fabricating evidence.
</code_context>

<specifics>
## Specific Ideas

- Add `scripts/phc_pain_v1_evaluate.py`.
- Add evaluation protocol documentation.
- Add evidence template documentation.
</specifics>

<deferred>
## Deferred Ideas

- Populate the evidence template after real training/ablation runs complete.
</deferred>

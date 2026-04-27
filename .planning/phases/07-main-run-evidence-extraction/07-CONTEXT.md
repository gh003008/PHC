# Phase 7: Main Run Evidence Extraction - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Evidence extraction planning

<domain>
## Phase Boundary

Extract the minimum defensible evidence from the main condition before any
ablation comparison. This phase determines what the existing main run can and
cannot support.
</domain>

<decisions>
## Implementation Decisions

### Separate available evidence from missing instrumentation

The current stdout log clearly provides reward, FPS, frame count, and episode
length. Pain extras may not be printed by the current rl_games stats path, so
the phase must explicitly check whether `pain_v1_*` metrics exist before using
them.

### Do not over-interpret standing-only runs

The current motion file is `sample_data/amass_isaac_standing_upright_slim.pkl`.
Any evidence from it is a restore/training-stability and synthetic pain signal
check, not a gait-pattern claim.

### Evidence package before claims

Populate a machine-readable summary and the existing evidence template before
running `scripts/phc_pain_v1_evaluate.py`.
</decisions>

<code_context>
## Existing Code Insights

- `docs/superpowers/phc_pain_v1_evaluation_protocol.md` defines required
  aggregate metrics.
- `docs/superpowers/phc_pain_v1_evidence_template.md` defines the evidence
  package format.
- `scripts/phc_pain_v1_evaluate.py` consumes aggregate JSON, not raw logs.
</code_context>

<specifics>
## Specific Ideas

- Parse reward and episode length from `phc_pain_v1_short50_idx1_3968.out`.
- Search stdout/stderr and any tensorboard/event outputs for `pain_v1_` keys.
- If pain metrics are absent, mark pain/load evidence `BLOCKED` and add an
  instrumentation task before ablations.
</specifics>

<deferred>
## Deferred Ideas

Condition comparisons belong to Phase 8. This phase handles only the main run.
</deferred>

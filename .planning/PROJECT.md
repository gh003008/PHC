# PHC-Pain-v1 Mechanism Proof

## What This Is

PHC-Pain-v1.0 is a synthetic mechanism proof inside PHC / IsaacGym / rl_games PPO on branch `jinsu-pain_baseline_phc_v0`. It tests whether a PHC-based controller with body-part pain observation and reward-only unilateral medial tibiofemoral knee pain cost can reduce mechanically provocative affected-knee pain/load while preserving locomotion.

This milestone is developer-facing research infrastructure and evaluation, not a patient-specific gait reproduction system.

## Core Value

Show that a pain-conditioned PHC controller can reduce synthetic unilateral knee pain/load without locomotion collapse, and that the effect is side-specific.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] Add a v1 task boundary that preserves the v0 task while making the observation-dimension change explicit.
- [ ] Append a general body-part pain map with knee-only synthetic activation for v1.0.
- [ ] Implement a defensible synthetic medial tibiofemoral knee pain drive/state based on available mechanical proxies.
- [ ] Use reward-only pain cost for the main mechanism claim, with action guard excluded from main evidence.
- [ ] Adapt pretrained PHC checkpoints to the expanded observation space while preserving initial motor-prior behavior.
- [ ] Run required pain observation/reward and left/right knee ablations.
- [ ] Evaluate pain/load reduction, locomotion competence, and side specificity with diagnostic logs.

### Out of Scope

- Patient-specific gait imitation - v1.0 has no patient dataset and must not claim clinical gait reproduction.
- Validated pain perception model - v1.0 uses a synthetic mechanical-provocation proxy only.
- Patient digital-twin validity - deferred until patient data and parameter identification exist.
- Full mechanistic decomposition of impairment physiology - too large for the v1.0 mechanism proof.
- Main-claim action clipping/governor - may exist only as fallback/debug ablation because it obscures learned compensation.
- v1.1 one-patient parameter ID, v1.2 held-out validation, and v1.3 counterfactual intervention - preserved as future scope.

## Context

- Source spec: `docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md`.
- Intel entry point: `.planning/intel/SYNTHESIS.md`.
- The ingest synthesizer extracted constraints but no requirements, so the source spec is authoritative for v1.0 requirement derivation.
- v0 partially succeeded at guard tolerance after retune/fine-tune, but `log_only` pain did not decrease. v1.0 changes the question from external guard tolerance to learned pain-conditioned compensation.
- The active milestone is v1.0 only: synthetic unilateral medial tibiofemoral knee pain mechanism proof.

## Constraints

- **Runtime**: PHC / IsaacGym / rl_games PPO on branch `jinsu-pain_baseline_phc_v0` - implementation must fit the existing training stack.
- **Task boundary**: Add an explicit v1 task such as `HumanoidImPainV1` - v0 must remain reproducible with its original observation contract.
- **Observation schema**: Use body-part pain channels for left/right hip, knee, ankle, foot, and back - v1.0 activates only one knee channel.
- **Pain model wording**: Treat pain as a synthetic thresholded, sensitivity-weighted mechanical-provocation proxy - not as perceived pain.
- **Checkpoint compatibility**: Copy original observation weights and zero-initialize new pain-observation input columns - initial behavior should match the pretrained motor prior.
- **Objective**: Main experiment is reward-only pain cost; action guard is off for main claims.
- **Evaluation**: Claims require pain/load reduction, locomotion competence, and side-specificity gates.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Scope v1.0 to synthetic mechanism proof | No patient dataset is available, and the mechanism must be shown before patient claims | - Pending |
| Use body-map architecture with knee-only activation | Avoids another observation schema change in v1.1 while keeping v1.0 controlled | - Pending |
| Preserve v0 and add explicit v1 task | Keeps checkpoint-compatible v0 reproducible and isolates v1 observation/checkpoint/evaluation changes | - Pending |
| Use reward-only pain cost for main claim | Tests learned compensation rather than external action clipping | - Pending |
| Preserve v1.1-v1.3 as future scope | Current milestone should not absorb patient ID, held-out validation, or counterfactual intervention work | - Pending |

---
*Last updated: 2026-04-27 after new-project-from-ingest initialization*

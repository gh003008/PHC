# Roadmap: PHC-Pain-v1 Mechanism Proof

## Overview

This v1.0 roadmap proves the synthetic unilateral medial tibiofemoral knee pain mechanism in the smallest defensible path: isolate a v1 task and body-map observation contract, make the synthetic pain/reward signal measurable, preserve pretrained PHC motor behavior through partial checkpoint loading, run the required reward/observation/side ablations, then evaluate mechanism, competence, and side-specificity without patient-specific claims.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: V1 Task And Pain Observation Contract** - Create the explicit v1 task boundary and body-part pain observation schema.
- [x] **Phase 2: Synthetic Pain And Reward Mechanism** - Compute unilateral knee pain drive/state and wire reward-only pain cost for the main experiment.
- [x] **Phase 3: Expanded-Observation Checkpoint Adaptation** - Load pretrained PHC checkpoints into the expanded observation policy while preserving initial motor-prior behavior.
- [x] **Phase 4: Training Conditions And Ablations** - Run the main condition plus required pain-observation, pain-reward, and left/right impairment ablations.
- [x] **Phase 5: Mechanism Evaluation And Evidence Package** - Report pain/load reduction, locomotion competence, side-specificity, and diagnostics with v1.0 scope boundaries.
- [x] **Phase 6: Main Checkpoint Preservation And Smoke Evaluation** - Preserve the current main checkpoint and verify the trained policy can run in evaluation mode.
- [x] **Phase 7: Main Run Evidence Extraction** - Extract and package evidence from the main condition run before interpreting mechanism quality.
- [x] **Phase 07.1: Pain Metric Scalar Logging Instrumentation (INSERTED)** - Promote PHC-Pain-v1 env extras into training scalar logs before ablations.
- [ ] **Phase 8: Minimal Ablation Runs** - Run the smallest condition matrix needed to separate pain observability and reward pressure.

## Phase Details

### Phase 1: V1 Task And Pain Observation Contract
**Goal**: Developer can run a distinct v1 PHC pain task that exposes a body-part pain map while preserving v0 reproducibility.
**Depends on**: Nothing (first phase)
**Requirements**: TASK-01, TASK-02, OBS-01, OBS-02, OBS-03, OBS-04
**Success Criteria** (what must be TRUE):
  1. Developer can select the v1 task separately from the v0 task without changing the v0 observation contract.
  2. The v1 task observation includes the original PHC observation plus the full body-part pain map.
  3. Left-knee and right-knee synthetic impairment can each be activated while all non-target channels remain present and inactive.
  4. A human inspecting logged observation metadata can verify current pain state and short temporal memory per channel.
**Plans**: 01-01-PLAN.md

### Phase 2: Synthetic Pain And Reward Mechanism
**Goal**: The v1 task produces a defensible unilateral knee pain signal and uses it as reward-only training pressure.
**Depends on**: Phase 1
**Requirements**: PAIN-01, PAIN-02, PAIN-03, PAIN-04, REW-01, REW-02, REW-03
**Success Criteria** (what must be TRUE):
  1. Developer can run the v1 task and observe affected/unaffected knee `pain_drive`, `pain_state`, and available load proxy components.
  2. Pain state rises and decays from the thresholded sensitivity-weighted pain drive rather than from an action guard.
  3. Unavailable mechanical proxy terms are explicitly logged as unavailable.
  4. The main training reward includes affected-knee pain cost with action guard off.
  5. Any guard-enabled run is visibly separated as fallback/debug and cannot be confused with main evidence.
**Plans**: 02-01-PLAN.md

### Phase 3: Expanded-Observation Checkpoint Adaptation
**Goal**: Pretrained PHC policies initialize into the expanded pain-observation policy without destroying the motor prior.
**Depends on**: Phase 2
**Requirements**: CKPT-01, CKPT-02, CKPT-03
**Success Criteria** (what must be TRUE):
  1. Developer can load a pretrained PHC checkpoint into the v1 policy despite the expanded observation dimension.
  2. Original observation input weights match the pretrained checkpoint after adaptation.
  3. New pain-observation input weights are zero-initialized at load time.
  4. Before pain fine-tuning, the adapted policy matches the original PHC motor-prior behavior within an explicit equivalence check.
**Plans**: 03-01-PLAN.md

### Phase 4: Training Conditions And Ablations
**Goal**: The mechanism proof has the required condition matrix to distinguish observability, reward pressure, and impairment side effects.
**Depends on**: Phase 3
**Requirements**: ABL-01, ABL-02, ABL-03, ABL-04
**Success Criteria** (what must be TRUE):
  1. Developer can run the main `pain_obs_on + pain_reward_on` condition.
  2. Developer can run `pain_obs_off + pain_reward_on` and compare it against the main condition.
  3. Developer can run `pain_obs_on + pain_reward_off` and compare it against the main condition.
  4. Left-knee and right-knee impairment runs both produce comparable logs for evaluation.
**Plans**: 04-01-PLAN.md

### Phase 5: Mechanism Evaluation And Evidence Package
**Goal**: The v1.0 result can be judged as mechanism proof, failure, or inconclusive without drifting into patient-specific claims.
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. Evaluator can see whether affected-knee pain/load drops versus baseline or ablations using episode average and stance-phase peak or p95 where available.
  2. Evaluator can see whether locomotion competence remains intact through no-fall rate, episode length, forward progress or speed, and imitation/style reward.
  3. Runs that reduce pain by stopping, freezing, collapse, or non-locomotor posture are labeled as failures.
  4. Evaluator can tell whether effects are side-specific rather than global freezing or bilateral shutdown.
  5. Diagnostic gait evidence is available for interpretation but not promoted to clinical-pattern success claims.
**Plans**: 05-01-PLAN.md

## Future Scope

Preserved from the source design but not part of the active v1.0 milestone:

- **v1.1: One-Patient Parameter ID** - introduce one real patient dataset and identify physical, pain, and control latent parameters.
- **v1.2: Held-Out Validation** - test identified parameters on held-out speeds, turns, or gait segments.
- **v1.3: Counterfactual Intervention** - test directional causal predictions under simulated interventions.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. V1 Task And Pain Observation Contract | 1/1 | Complete | 2026-04-27 |
| 2. Synthetic Pain And Reward Mechanism | 1/1 | Complete | 2026-04-27 |
| 3. Expanded-Observation Checkpoint Adaptation | 1/1 | Complete | 2026-04-27 |
| 4. Training Conditions And Ablations | 1/1 | Complete | 2026-04-27 |
| 5. Mechanism Evaluation And Evidence Package | 1/1 | Complete | 2026-04-27 |
| 6. Main Checkpoint Preservation And Smoke Evaluation | 1/1 | Complete | 2026-04-27 |
| 7. Main Run Evidence Extraction | 1/1 | Complete | 2026-04-27 |
| 07.1. Pain Metric Scalar Logging Instrumentation | 1/1 | Complete | 2026-04-28 |
| 8. Minimal Ablation Runs | 0/1 | Ready | - |

### Phase 6: Main Checkpoint Preservation And Smoke Evaluation

**Goal:** Preserve the current main checkpoint and verify the trained policy can run in evaluation mode.
**Requirements**: CKPT-01, CKPT-02, CKPT-03, EVAL-02
**Depends on:** Phase 5
**Plans:** 1 plan

Plans:
- [x] 06-01-PLAN.md

### Phase 7: Main Run Evidence Extraction

**Goal:** Extract and package evidence from the main condition run before interpreting mechanism quality.
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-05
**Depends on:** Phase 6
**Plans:** 1 plan

Plans:
- [x] 07-01-PLAN.md

### Phase 07.1: Pain Metric Scalar Logging Instrumentation (INSERTED)

**Goal:** Promote PHC-Pain-v1 env extras into training scalar logs before ablations.
**Requirements**: PAIN-04, EVAL-01, EVAL-03
**Depends on:** Phase 7
**Plans:** 1 plan

Plans:
- [x] 07.1-01-PLAN.md

### Phase 8: Minimal Ablation Runs

**Goal:** Run the smallest condition matrix needed to separate pain observability and reward pressure.
**Requirements**: ABL-01, ABL-02, ABL-03, EVAL-01, EVAL-02, EVAL-03
**Depends on:** Phase 07.1
**Plans:** 1 plan

Plans:
- [ ] 08-01-PLAN.md

# Requirements: PHC-Pain-v1 Mechanism Proof

**Defined:** 2026-04-27
**Core Value:** Show that a pain-conditioned PHC controller can reduce synthetic unilateral knee pain/load without locomotion collapse, and that the effect is side-specific.

## v1 Requirements

Requirements for the v1.0 synthetic mechanism proof. Each maps to exactly one roadmap phase.

### Task Boundary

- [ ] **TASK-01**: Developer can select a new explicit v1 pain task, tentatively `HumanoidImPainV1`, without changing the v0 task contract.
- [ ] **TASK-02**: The v1 task owns the observation-dimension change, v1 checkpoint adaptation path, and v1 evaluation gates.

### Pain Observation

- [ ] **OBS-01**: The v1 policy observation appends a body-part pain map to the existing PHC observation.
- [ ] **OBS-02**: The pain map includes left/right hip, left/right knee, left/right ankle, back, left/right foot channels.
- [ ] **OBS-03**: Each pain channel supports current pain state and short temporal memory.
- [ ] **OBS-04**: v1.0 activates only one synthetic left-knee or right-knee impairment channel while inactive channels remain structurally present and zero or inactive.

### Pain Mechanism

- [ ] **PAIN-01**: A unilateral medial tibiofemoral knee pain drive is computed as a thresholded, sensitivity-weighted proxy for mechanically provocative knee loading.
- [ ] **PAIN-02**: Pain state accumulates from pain drive with rise and decay dynamics.
- [ ] **PAIN-03**: The mechanical load proxy uses only reliably available PHC/IsaacGym tensors and logs unavailable candidate terms instead of inventing weak replacements.
- [ ] **PAIN-04**: Pain metrics expose affected-side and unaffected-side `pain_drive`, `pain_state`, and available load proxy components for evaluation.

### Reward Mechanism

- [ ] **REW-01**: The main v1.0 training objective adds reward-only affected-knee pain cost to the existing PHC task and style terms.
- [ ] **REW-02**: Action guard is off for the main v1.0 experiment and cannot support the main mechanism claim.
- [ ] **REW-03**: Any action-guard condition is marked as optional fallback/debug ablation and separated from main results.

### Checkpoint Adaptation

- [ ] **CKPT-01**: Expanded-observation policy initialization copies pretrained weights for original PHC observation columns.
- [ ] **CKPT-02**: New pain-observation input columns are initialized to zero.
- [ ] **CKPT-03**: Initialization equivalence is checked so the adapted policy behaves like the original PHC motor prior before pain fine-tuning.

### Training And Ablations

- [ ] **ABL-01**: Main condition runs with `pain_obs_on + pain_reward_on`.
- [ ] **ABL-02**: Ablation runs with `pain_obs_off + pain_reward_on`.
- [ ] **ABL-03**: Ablation runs with `pain_obs_on + pain_reward_off`.
- [ ] **ABL-04**: Left-knee and right-knee impairment conditions are both supported and compared.

### Evaluation

- [ ] **EVAL-01**: Evaluation reports affected-knee pain/load reduction versus baseline or ablations using episode average and stance-phase peak or p95 where available.
- [ ] **EVAL-02**: Evaluation reports locomotion competence using no-fall rate, episode length, forward progress or speed, and imitation/style reward.
- [ ] **EVAL-03**: Evaluation rejects pain reduction achieved by stopping, freezing, collapse, or non-locomotor posture.
- [ ] **EVAL-04**: Evaluation reports side-specificity and rejects global freezing or bilateral shutdown as mechanism proof.
- [ ] **EVAL-05**: Diagnostics log gait evidence such as stance asymmetry, knee ROM, step width, pelvis/trunk compensation, knee torque/work, foot progression angle, velocity, and cadence when available.

## v2 Requirements

Deferred beyond the current v1.0 roadmap. These are preserved from the source spec as future scope, not active milestone work.

### v1.1 One-Patient Parameter ID

- **FUT-01**: Introduce one real patient dataset when available.
- **FUT-02**: Use patient gait mainly as inverse-problem observation and validation signal, not a strong imitation target.
- **FUT-03**: Identify candidate physical, pain, and control latent parameters.

### v1.2 Held-Out Validation

- **FUT-04**: Test identified parameters on held-out speeds, turns, or gait segments.
- **FUT-05**: Compare speed, cadence, stride length, step width, stance asymmetry, pelvis/trunk angles, knee/ankle ROM, force/pressure metrics, EMG timing, predicted pain trajectory, and MPJPE where available.

### v1.3 Counterfactual Intervention

- **FUT-06**: Simulate intervention candidates such as pain sensitivity/threshold changes, brace or exoskeleton assistance, stiffness/alignment changes, or gait-retraining changes.
- **FUT-07**: Evaluate directional agreement between predicted and real or later experimental response.

## Out of Scope

Explicitly excluded from v1.0 to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Patient-specific digital twin | Requires patient data and parameter identification; deferred to v1.1+ |
| Clinical gait reproduction claim | v1.0 is a synthetic mechanism proof only |
| Validated perceived-pain model | v1.0 uses a mechanical-provocation proxy, not pain perception |
| Full mechanistic decomposition | Too broad for the first v1 mechanism proof |
| Main-claim action guard | Would blur learned compensation with external action clipping |
| v1.1-v1.3 implementation | Preserved as future roadmap/backlog, not current milestone scope |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TASK-01 | Phase 1 | Pending |
| TASK-02 | Phase 1 | Pending |
| OBS-01 | Phase 1 | Pending |
| OBS-02 | Phase 1 | Pending |
| OBS-03 | Phase 1 | Pending |
| OBS-04 | Phase 1 | Pending |
| PAIN-01 | Phase 2 | Pending |
| PAIN-02 | Phase 2 | Pending |
| PAIN-03 | Phase 2 | Pending |
| PAIN-04 | Phase 2 | Pending |
| REW-01 | Phase 2 | Pending |
| REW-02 | Phase 2 | Pending |
| REW-03 | Phase 2 | Pending |
| CKPT-01 | Phase 3 | Pending |
| CKPT-02 | Phase 3 | Pending |
| CKPT-03 | Phase 3 | Pending |
| ABL-01 | Phase 4 | Pending |
| ABL-02 | Phase 4 | Pending |
| ABL-03 | Phase 4 | Pending |
| ABL-04 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| EVAL-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0

---
*Requirements defined: 2026-04-27*
*Last updated: 2026-04-27 after roadmap creation*

# PHC-Pain-v1 Mechanism Design

**Date:** 2026-04-27
**Branch:** `jinsu-pain_baseline_phc_v0`
**Status:** Design spec
**Builds on:** `PHC-Pain-v0` partial success

---

## 1. Thesis

PHC-Pain-v1 should not try to copy one patient's limp through strong imitation.
The research goal is to test whether a pain-conditioned controller can produce
compensatory gait as an emergent consequence of patient-like constraints.

For v1.0, no patient dataset is available. The first deliverable is therefore a
synthetic mechanism proof:

> Given a synthetic unilateral medial tibiofemoral knee pain impairment, can a
> PHC-based controller reduce mechanically provocative knee pain/load while
> preserving locomotion competence?

This keeps the claim narrow. v1.0 does not claim clinical gait reproduction,
patient-specific digital-twin validity, or a validated pain perception model. It
tests whether the mechanism can steer behavior.

---

## 2. Current Baseline

PHC-Pain-v0 implemented the pain estimator, action guard, reward penalty, and
fine-tune path while preserving the original PHC observation dimension.

The final v0 disposition was partial success:

- `guard_only` survival was achieved after retune/fine-tune.
- `log_only` pain did not decrease: steady-state pain remained at `0.522`.
- The controller learned or retained guard tolerance, but did not demonstrate
  internalized pain avoidance.

v1 changes the question. Instead of using pain only as an external action
governor, v1 makes pain observable and uses reward-only pain cost as the primary
learning signal.

---

## 3. Scope Decision

PHC-Pain-v1.0 is **Approach 2: Body-map architecture, knee-only experiment**.

The observation schema is designed as a general body-part pain map so later work
can support ankle weakness, hip/back pain, exoskeleton pressure, and other
impairments. The v1.0 experiment activates only one synthetic unilateral knee
pain channel to keep the mechanism proof controlled.

Rejected alternatives:

- **Minimal knee-only obs:** faster, but likely forces another obs/schema change
  in v1.1.
- **Full mechanistic decomposition:** scientifically richer, but too large for
  the first v1 mechanism proof.

---

## 4. v1.0 Goal

PHC-Pain-v1.0 uses PHC as a motor competence prior. It adds body-part pain
observation and a reward-only unilateral knee pain cost. A one-sided synthetic
knee impairment should cause the fine-tuned policy to reduce affected-knee pain
or load while still walking.

Success is not "looks like a patient." Success is:

1. affected-knee pain/load decreases versus baseline or ablations;
2. locomotion competence does not collapse;
3. the effect is side-specific.

Patient gait imitation is not a v1.0 training target.

---

## 5. Pain Model

Use the following wording throughout v1.0:

> Synthetic unilateral medial tibiofemoral knee pain drive is a thresholded,
> sensitivity-weighted proxy for mechanically provocative knee loading, not a
> validated direct model of perceived pain.

The model should avoid the stronger and less defensible statement "higher knee
load means higher pain." Real knee pain involves load, tissue state,
inflammation, sensitization, fear/avoidance, and history. v1.0 only needs a
controlled mechanical-provocation cost that a policy can learn to avoid.

Initial definition:

```text
pain_drive =
  sensitivity_side *
  ReLU(weighted_mechanical_load_proxy - threshold_side)

pain_state_t =
  decay * pain_state_{t-1}
  + rise * pain_drive_t
```

Initial mechanical load proxy:

```text
weighted_mechanical_load_proxy =
  w_kam  * medial_load_or_KAM_proxy
+ w_flex * knee_flexion_torque_proxy
+ w_rom  * near_ROM_limit_or_passive_torque
+ w_work * positive_knee_work
```

Implementation may start with the terms easiest to compute reliably from the
existing PHC/IsaacGym tensors, then log missing terms as unavailable rather than
inventing weak proxies.

---

## 6. Architecture

### 6.1 Task Boundary

Preserve v0's checkpoint-compatible task. v1.0 should use a new explicit task
subclass, tentatively:

```text
HumanoidImPainV1
```

Rationale:

- v0 keeps its original observation contract and remains reproducible.
- v1's obs-dimension change is obvious in task selection.
- v1-specific checkpoint adaptation and evaluation gates stay isolated.

`parse_task.py` should register the new task name when implementation begins.

### 6.2 Observation

Append body-part pain observation to the existing PHC observation.

Initial body-part channels:

```text
left_hip, right_hip,
left_knee, right_knee,
left_ankle, right_ankle,
back,
left_foot, right_foot
```

Each channel should support at least current pain state and short temporal
memory. v1.0 activates only `left_knee` or `right_knee` synthetic impairment.
The other channels remain structurally present but zero or inactive.

### 6.3 Checkpoint Adaptation

Use partial weight loading:

- copy pretrained weights for the original PHC observation columns;
- initialize new pain-observation input columns to zero;
- keep the pretrained motor prior behavior intact at initialization.

This makes the initial v1 policy behave like the original PHC policy before
fine-tuning, then allows training to open pain-conditioned behavior through the
new input weights.

### 6.4 Objective

The main v1.0 experiment uses reward-only pain cost:

```text
reward = phc_task_reward
       + style_terms
       - lambda_p * affected_knee_pain_state_or_drive
```

Action guard is off in the main experiment. It may exist only as a fallback or
debug ablation. Main claims must not rely on action clipping because that would
blur whether compensation emerged from policy learning or from an external
governor.

### 6.5 Data Flow

```text
PHC obs + pain_map/memory
  -> policy
  -> action
  -> sim step
  -> internal knee load proxy
  -> pain_drive/pain_state update
  -> reward-only pain cost
  -> PPO update
```

---

## 7. Evaluation

v1.0 evaluates mechanism and competence, not patient matching.

### 7.1 Primary Gates

1. **Pain/load reduction**
   The pain-conditioned policy lowers affected-knee `pain_state`, `pain_drive`,
   and available load proxy metrics versus baseline or ablation. Report episode
   average plus stance-phase peak or p95.

2. **Locomotion competence**
   The policy must avoid collapse. Track no-fall rate, episode length, forward
   progress or speed, and imitation/style reward. Reducing pain by stopping,
   freezing, or adopting a non-locomotor posture is failure.

3. **Side-specificity**
   A right-knee impairment should primarily alter/reduce right-knee pain/load.
   A left-knee impairment should primarily alter/reduce left-knee pain/load.
   Global freezing or bilateral shutdown is failure.

### 7.2 Required Ablations

- `pain_obs_on + pain_reward_on`: main condition.
- `pain_obs_off + pain_reward_on`: tests whether observability matters.
- `pain_obs_on + pain_reward_off`: verifies obs expansion alone is not causing
  behavior change.
- left-knee impairment vs right-knee impairment.

Optional:

- weak action guard debug condition, excluded from the main mechanism claim.

### 7.3 Diagnostics, Not Success Claims

Log these as diagnostic evidence:

- stance time asymmetry;
- knee ROM;
- step width;
- pelvis/trunk compensation;
- knee torque/work;
- foot progression angle;
- forward velocity and cadence if available.

Without patient data, these are not clinical-pattern success gates. They become
evidence maps for v1.1 patient-specific work.

---

## 8. v1 Roadmap

### v1.0: Synthetic Mechanism Proof

Goal: show that a unilateral medial knee pain cost can change learned behavior
while preserving locomotion.

Deliverables:

- `HumanoidImPainV1` task with pain-map observation;
- partial checkpoint loading for expanded obs;
- reward-only unilateral knee pain training;
- left/right knee impairment ablations;
- mechanism and competence evaluation logs.

### v1.1: One-Patient Parameter ID

Goal: introduce one real patient once data exists.

Patient gait should be used mainly as an inverse-problem observation and
validation signal, not as a strong imitation target. Candidate latent parameters:

- `theta_phys`: weakness, ROM limit, passive stiffness, passive torque, reflex
  gain, delay;
- `theta_pain`: side/body sensitivity, threshold, saturation, recovery time
  constant;
- `theta_control`: pain weight, energy weight, symmetry preference, speed
  preference, avoidance gain.

Candidate data:

- self-selected, slow, fast walking;
- turning;
- sit-to-stand;
- mocap or OpenCap/IMU;
- pressure/force data if available;
- EMG if available;
- task-specific pain score and pain-location map.

### v1.2: Held-Out Validation

Goal: test whether identified parameters explain unseen behavior.

Use held-out speeds, turns, or gait segments. Compare:

- speed, cadence, stride length, step width;
- stance asymmetry;
- pelvis/trunk angles;
- knee/ankle ROM;
- GRF or pressure metrics if available;
- EMG timing if available;
- predicted pain trajectory;
- MPJPE as a tracking diagnostic, not the only metric.

### v1.3: Counterfactual Intervention

Goal: move beyond imitation by testing causal predictions.

Candidate interventions:

- lower synthetic pain sensitivity or raise pain threshold;
- alter brace/exoskeleton assistance;
- alter stiffness/alignment assumptions;
- simulate gait-retraining changes such as foot progression angle.

Success is directional agreement: the model predicts how pain/load and gait
change under intervention, and the real or later experimental response moves in
the same direction.

---

## 9. In Scope for v1.0

- New v1 task subclass.
- Body-part pain map and memory observation.
- Synthetic unilateral medial tibiofemoral knee pain drive.
- Partial checkpoint loading with zero-initialized pain-input weights.
- Reward-only pain cost training.
- Walking or locomotion distribution fine-tune.
- Pain/load reduction plus locomotion competence evaluation.
- Left/right side-specific ablations.
- Diagnostic gait metric logging.

---

## 10. Out of Scope for v1.0

- Real patient parameter ID.
- Patient gait imitation as a target.
- Clinical-pattern success claims.
- Validated physiological pain perception model.
- Full musculoskeletal knee contact force model.
- EMG/GRF matching as a required gate.
- Exoskeleton external pressure/shear pain.
- Reflex, fear, or anticipation state.
- Action guard as the main mechanism.

---

## 11. Open Risks

1. **Pain reduction by freezing**
   The policy may reduce knee pain by stopping or collapsing into low-load
   behavior. Locomotion competence gates must catch this.

2. **Pain obs ignored**
   Zero-initialized pain columns may remain unused if `lambda_p` is too low or
   training is too short. The `pain_obs_off + pain_reward_on` ablation helps
   distinguish this.

3. **Weak load proxy**
   PHC may not expose a clean medial tibiofemoral load estimate. Start with
   reliable torque/ROM/work proxies and clearly label them as synthetic.

4. **Overclaiming clinical validity**
   v1.0 must not claim patient-specific gait reproduction. That belongs to v1.1+
   after data and parameter identification exist.

5. **Checkpoint adaptation complexity**
   Expanded obs may require careful network weight surgery. The spec requires
   initialization equivalence checks before training claims.

---

## 12. Approval Gate

This design is ready for implementation planning when the user accepts:

- v1.0 as synthetic mechanism proof;
- unilateral medial knee pain as the first impairment;
- body-part pain map observation;
- partial checkpoint loading;
- reward-only pain cost as the main mechanism;
- v1.1-v1.3 as future roadmap rather than v1.0 scope.

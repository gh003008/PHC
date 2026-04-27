# Constraints Intel

## PHC-Pain-v1 Mechanism Design

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: nfr
- title: Narrow v1.0 Research Claim

Content:
PHC-Pain-v1.0 is a synthetic mechanism proof, not a clinical gait reproduction,
patient-specific digital-twin, or validated pain perception claim. Success is
limited to showing that a pain-conditioned PHC controller can reduce synthetic
unilateral medial tibiofemoral knee pain/load while preserving locomotion
competence and side specificity.

## Scope: Body-Map Architecture, Knee-Only Experiment

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: schema
- title: Body-Part Pain Map with Knee-Only Activation

Content:
The v1.0 task uses a general body-part pain observation schema while activating
only one synthetic unilateral knee pain channel. Initial channels are left_hip,
right_hip, left_knee, right_knee, left_ankle, right_ankle, back, left_foot, and
right_foot. Each channel must support at least current pain state and short
temporal memory. Non-target channels remain structurally present but zero or
inactive in v1.0.

## Task Boundary

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: api-contract
- title: New HumanoidImPainV1 Task

Content:
Implementation planning should preserve the v0 checkpoint-compatible task and
add a new explicit task subclass, tentatively HumanoidImPainV1. The new task
owns the observation-dimension change, v1 checkpoint adaptation, and v1
evaluation gates. parse_task.py should register the new task name when
implementation begins.

## Checkpoint Adaptation

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: protocol
- title: Partial Weight Loading for Expanded Observation

Content:
The v1 policy should copy pretrained weights for original PHC observation
columns, initialize new pain-observation input columns to zero, and preserve the
pretrained motor-prior behavior at initialization. Initialization equivalence
must be checked before training claims.

## Pain Model

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: nfr
- title: Synthetic Mechanical Provocation Proxy

Content:
The pain drive is a thresholded, sensitivity-weighted proxy for mechanically
provocative knee loading, not a validated direct model of perceived pain. The
initial model uses pain_drive = sensitivity_side * ReLU(weighted_load_proxy -
threshold_side), with decayed pain_state accumulation. Candidate load proxy
terms include medial load or KAM proxy, knee flexion torque proxy, near-ROM
limit or passive torque, and positive knee work. Unavailable terms should be
logged as unavailable rather than replaced with weak invented proxies.

## Objective

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: protocol
- title: Reward-Only Pain Cost Main Mechanism

Content:
The main v1.0 experiment uses PHC task reward plus style terms minus lambda_p
times affected-knee pain state or drive. Action guard is off in the main
experiment and may exist only as fallback or debug ablation. Main claims must
not rely on action clipping.

## Evaluation Gates

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: protocol
- title: Mechanism, Competence, and Side-Specificity Gates

Content:
Primary gates are pain/load reduction, locomotion competence, and side
specificity. Pain/load reduction must compare affected-knee pain_state,
pain_drive, and available load proxy metrics against baseline or ablation,
including episode average and stance-phase peak or p95 where available.
Locomotion competence must catch pain reduction by stopping, freezing, collapse,
or non-locomotor posture. Side-specificity must reject global freezing or
bilateral shutdown.

## Required Ablations

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/specs/2026-04-27-phc-pain-v1-mechanism-design.md
- type: protocol
- title: Pain Observation and Reward Ablations

Content:
Required ablations are pain_obs_on + pain_reward_on, pain_obs_off +
pain_reward_on, pain_obs_on + pain_reward_off, and left-knee versus right-knee
impairment. A weak action guard debug condition is optional and excluded from
the main mechanism claim.


# PHC-Pain-v1.0 Evaluation Protocol

This protocol judges the v1.0 synthetic mechanism proof after the Phase 4
condition matrix has produced aggregate metrics. It must not be used to claim
patient-specific or clinical validation.

## Required Conditions

- `main`: `pain_obs_on + pain_reward_on`
- `no_obs`: `pain_obs_off + pain_reward_on`
- `no_reward`: `pain_obs_on + pain_reward_off`
- left/right side-specific runs for interpretation

## Required Aggregate Metrics

Each condition summary should provide:

- `affected_pain_avg`
- `affected_pain_p95`
- `unaffected_pain_avg`
- `episode_length_avg`
- `no_fall_rate`
- `forward_progress_avg`

Recommended additional diagnostics:

- stance time asymmetry
- knee ROM
- step width
- pelvis/trunk compensation
- knee torque/work
- foot progression angle
- cadence or velocity
- imitation/style reward

## v1.5 OA Knee Load Proxy Metrics

Primary success metrics:

- `pain_v1_right_knee_load`: final reward-facing OA load proxy.
- `pain_v1_right_knee_state`: leaky pain state driven by OA load.
- `pain_v1_right_knee_contact_load`: compression and loaded-flexion contact component.
- `pain_v1_right_knee_moment_load`: KAM/KFM moment-arm component.
- `pain_v1_right_knee_kam`: medial-compartment loading surrogate.

Secondary diagnostics:

- `pain_v1_right_knee_kfm`: sagittal flexion loading surrogate.
- `pain_v1_right_knee_compression`: stance foot load normalized by body-weight reference.
- `pain_v1_right_knee_loaded_flex`: knee flexion under compressive load.
- `pain_v1_right_knee_tau_abs`, `pain_v1_right_knee_tau_rms`, `pain_v1_right_knee_tau_peak`: legacy actuator-torque diagnostics only.

Interpretation rule:
Do not claim OA pain reduction from reduced actuator torque alone. v1.5 claims
must be based on reduced OA load proxy, especially contact load, KAM, and pain
state. KAM/KFM are estimated PHC load proxies from GRF line-of-action geometry,
not true medial contact force or full inverse-dynamics joint contact force.

## Gate 1: Pain/Load Reduction

The main condition must reduce affected-knee pain/load relative to both:

- `no_reward`, showing reward pressure matters;
- `no_obs`, showing pain observability matters.

Default script thresholds:

- `main.affected_pain_avg` improves by at least `0.05` versus `no_reward`;
- `main.affected_pain_avg` improves by at least `0.01` versus `no_obs`.

## Gate 2: Locomotion Competence

Pain reduction by stopping, freezing, falling, or holding a non-locomotor pose is
failure. The default script thresholds are:

- `no_fall_rate >= 0.95`
- `episode_length_avg >= 250`
- `forward_progress_avg >= 0.0`

Projects should tighten these thresholds once a stable walking distribution is
available.

## Gate 3: Side-Specificity

The affected knee should show a larger pain/load reduction than the unaffected
knee. Global bilateral shutdown is not mechanism proof.

## Script

Use:

```bash
python3 scripts/phc_pain_v1_evaluate.py path/to/summary.json
```

The script prints `PHC_PAIN_V1_VERDICT=PASS|FAIL|BLOCKED` and one line per gate.

## Example Summary JSON

```json
{
  "main": {
    "affected_pain_avg": 0.20,
    "affected_pain_p95": 0.40,
    "unaffected_pain_avg": 0.10,
    "episode_length_avg": 299,
    "no_fall_rate": 1.0,
    "forward_progress_avg": 1.2
  },
  "no_obs": {
    "affected_pain_avg": 0.24,
    "affected_pain_p95": 0.45,
    "unaffected_pain_avg": 0.10,
    "episode_length_avg": 299,
    "no_fall_rate": 1.0,
    "forward_progress_avg": 1.2
  },
  "no_reward": {
    "affected_pain_avg": 0.30,
    "affected_pain_p95": 0.55,
    "unaffected_pain_avg": 0.11,
    "episode_length_avg": 299,
    "no_fall_rate": 1.0,
    "forward_progress_avg": 1.2
  }
}
```

## Claim Boundary

Passing this protocol supports only:

> A synthetic unilateral knee pain/load cost can steer a PHC-based controller
> while preserving locomotion competence.

It does not support:

- patient-specific gait reproduction;
- validated perceived-pain modeling;
- clinical treatment prediction;
- exoskeleton intervention claims.

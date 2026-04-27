# PHC-Pain-v1.0 Evidence Template

Use this file as the structure for the final evidence package after running the
Phase 4 condition matrix.

## Run Metadata

- Commit:
- Date:
- GPU:
- Motion set:
- Training budget:
- Checkpoint source:

## Conditions

| Condition | Exp Name | Active Side | Pain Obs | Pain Reward | Guard | Status |
|---|---|---|---|---|---|---|
| main |  | right/left | on | on | off |  |
| no_obs |  | right/left | off | on | off |  |
| no_reward |  | right/left | on | off | off |  |
| left_side |  | left | on | on | off |  |

## Gate Results

| Gate | Verdict | Evidence |
|---|---|---|
| Pain/load reduction | PASS/FAIL/BLOCKED |  |
| Locomotion competence | PASS/FAIL/BLOCKED |  |
| Side-specificity | PASS/FAIL/BLOCKED |  |

## Metrics

| Condition | affected pain avg | affected pain p95 | unaffected pain avg | episode length avg | no-fall rate | forward progress avg |
|---|---:|---:|---:|---:|---:|---:|
| main |  |  |  |  |  |  |
| no_obs |  |  |  |  |  |  |
| no_reward |  |  |  |  |  |  |

## Diagnostic Gait Evidence

- Stance time asymmetry:
- Knee ROM:
- Step width:
- Pelvis/trunk compensation:
- Knee torque/work:
- Foot progression angle:
- Cadence/velocity:
- Imitation/style reward:

## Interpretation

State one of:

- `PASS`: synthetic mechanism proof supported.
- `FAIL`: mechanism proof not supported.
- `INCONCLUSIVE`: rerun or additional instrumentation required.

Do not claim patient-specific or clinical validation from v1.0 evidence.

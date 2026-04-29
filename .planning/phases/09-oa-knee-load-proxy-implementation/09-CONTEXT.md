# Phase 9 Context: OA Knee Load Proxy Implementation

## Source

Ingested from `docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md` on 2026-04-29.

## Goal

Replace the torque-centric knee pain drive with an OA-style knee joint loading proxy while preserving `pain_body_state` as the reward-facing variable.

## Decisions

- v1.5 should not claim true medial contact force or full inverse-dynamics contact force.
- v1.5 should describe KAM/KFM as PHC-estimated load proxies from GRF line-of-action geometry.
- The legacy torque proxy remains available for ablation and diagnostics but is disabled in the default v1.5 OA load mixture.
- No server training should be launched as part of implementation; first validation is a local/headless play probe confirming finite `pain_v1_*` metrics.

## Requirements

- OA-01: `pain_body_state` remains reward-facing while the knee load input becomes OA-style joint loading.
- OA-02: Add pure-torch contact, moment, and OA load combiner helpers.
- OA-03: Preserve legacy torque proxy for backwards compatibility and ablation.
- OA-04: Add `knee_mechanism.proxy_mode` values `torque_v13`, `oa_contact_v14`, and `oa_contact_v15`.
- OA-05: Probe finite compression, loaded-flexion, KAM, KFM, contact-load, moment-load, torque-load, and final load metrics before training.
- OA-06: Default `env_im_pain_v1.yaml` to `oa_contact_v15`.
- OA-07: Treat OA load, pain state, contact load, moment load, and KAM as primary v1.5 metrics.
- OA-08: Keep proxy-validity boundaries explicit in reporting.

## Implementation Entry Point

Use the existing Superpowers plan as the detailed implementation scaffold:

`docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md`

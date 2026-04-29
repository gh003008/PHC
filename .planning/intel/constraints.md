# Constraints Intel

## PHC-Pain v1.5 Knee Load Proxy Implementation Plan

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: nfr
- title: Future v1.5 OA Knee Load Proxy Scope

Content:
PHC-Pain v1.5 replaces the v1 torque-centric knee pain mechanism with an
OA-style knee joint loading proxy without launching training. This is scoped as
future implementation work and must not rewrite completed v1.0 phases or
promote v1.0 evidence into OA-specific claims.

## Reward-Facing Pain State Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: api-contract
- title: Preserve pain_body_state as Reward-Facing Variable

Content:
The reward-facing variable remains `pain_body_state`, but the knee load input
driving it changes. v1.4 adds a contact/compression proxy from foot GRF and knee
geometry; v1.5 adds KAM/KFM-style moment-arm proxies so medial knee OA pain is
driven by estimated tibiofemoral compartment loading rather than actuator
torque.

## Pure-Torch Load Helper Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: api-contract
- title: OA Knee Load Proxy Helper Functions

Content:
`phc/env/util/pain_baseline.py` should add pure-torch helpers
`compute_knee_contact_load_proxy`, `compute_knee_moment_load_proxy`, and
`combine_knee_oa_load_proxy`. `compute_knee_torque_load_proxy` remains present
for backwards compatibility and ablation visibility.

## HumanoidImPainV1 Mode Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: protocol
- title: Mode-Aware Knee Proxy Selection

Content:
`HumanoidImPainV1` should support `knee_mechanism.proxy_mode` values
`torque_v13`, `oa_contact_v14`, and `oa_contact_v15`. The v1.5 mode computes
legacy torque diagnostics, contact/compression load, KAM/KFM moment load, and a
combined OA load while keeping component metrics visible.

## Environment Configuration Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: schema
- title: env_im_pain_v1 OA Load Configuration

Content:
`phc/data/cfg/env/env_im_pain_v1.yaml` should set
`knee_mechanism.proxy_mode: "oa_contact_v15"`, include reference and weight
values for compression, loaded flexion, loading rate, KAM, KFM, contact load,
moment load, and legacy torque load, and keep legacy torque weights present but
disabled by default in the final v1.5 OA load mixture.

## Probe Metric Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: protocol
- title: OA Load Probe Metrics Before Training

Content:
Before any future training job, a short headless play probe should confirm that
the JSON probe path captures finite `pain_v1_*` metrics including right-knee
compression, loaded flexion, KAM, KFM, contact load, moment load, torque load,
and final reward-facing knee load.

## Evaluation Interpretation Contract

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: protocol
- title: v1.5 OA Metric Hierarchy

Content:
The evaluation protocol should treat `pain_v1_right_knee_load`,
`pain_v1_right_knee_state`, `pain_v1_right_knee_contact_load`,
`pain_v1_right_knee_moment_load`, and `pain_v1_right_knee_kam` as primary v1.5
metrics. Reduced actuator torque alone must not be used to claim OA pain
reduction.

## Proxy Validity Boundary

- source: /home/jinsu/Documents/GitHub/PHC/docs/superpowers/plans/2026-04-29-phc-pain-v15-knee-load-proxy.md
- type: nfr
- title: Estimated Load Proxy, Not True Contact Force

Content:
The KAM/KFM proxy uses PHC world axes and simple GRF moment arms rather than
full inverse dynamics. v1.5 reporting should describe it as an estimated load
proxy, not true medial contact force.


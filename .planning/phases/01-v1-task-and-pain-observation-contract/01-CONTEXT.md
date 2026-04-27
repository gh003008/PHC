# Phase 1: V1 Task And Pain Observation Contract - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Autonomous auto-discuss

<domain>
## Phase Boundary

Create a distinct v1 PHC pain task that exposes a body-part pain map while
preserving v0 reproducibility.

This phase owns only the task/observation contract:

- v0 `HumanoidImPain` must remain checkpoint-compatible and unchanged in its
  observation contract.
- v1 must be selectable as a distinct task.
- v1 observation must append a body-part pain map and short memory.
- v1.0 activates only one synthetic knee side while inactive body channels stay
  structurally present.

Synthetic knee pain dynamics, reward-only pain cost, checkpoint weight surgery,
training ablations, and evaluation reports are deferred to later phases.
</domain>

<decisions>
## Implementation Decisions

### Add a subclass, not a mode flag on v0

Use `HumanoidImPainV1` as a subclass of `HumanoidImPain`. The subclass owns the
observation-dimension change, while v0 remains reproducible through the existing
`HumanoidImPain` task.

### Body map schema

Use these body-part channels in v1.0:

`left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`,
`right_ankle`, `back`, `left_foot`, `right_foot`.

Each channel exposes current pain state and short memory, for 18 appended
dimensions when enabled.

### Active synthetic side

Use `env.pain.active_knee_side` with `right`, `left`, or `none`. Phase 1 only
records metadata and keeps body-map buffers zero; Phase 2 will compute the
actual unilateral pain drive.
</decisions>

<code_context>
## Existing Code Insights

- `phc/env/tasks/humanoid_im_pain.py` contains v0 `HumanoidImPain`, with v0
  action guard/reward modes and no observation change.
- `Humanoid.__init__` sets `cfg["env"]["numObservations"] = self.get_obs_size()`
  before `BaseTask` allocates `obs_buf`, so v1 must set observation-dimension
  attributes before calling `super().__init__`.
- `HumanoidIm._compute_observations()` writes the full observation into
  `self.obs_buf`; v1 cannot call it directly because v1 `obs_buf` is wider.
- `phc/utils/parse_task.py` uses imported task class names with `eval(args.task)`,
  so registering `HumanoidImPainV1` is an import change.
</code_context>

<specifics>
## Specific Ideas

- Add `phc/data/cfg/env/env_im_pain_v1.yaml` based on `env_im_pain.yaml`.
- Set `task: HumanoidImPainV1`.
- Set `pain.append_to_obs: True`.
- Add `pain.obs.enabled`, `pain.obs.include_memory`, and `pain.obs.channels`.
- Emit observation metadata in `extras` for human inspection.
</specifics>

<deferred>
## Deferred Ideas

- Mechanical pain proxy and affected-side reward cost: Phase 2.
- Partial checkpoint loading and zero-init pain columns: Phase 3.
- Main/ablation training matrix: Phase 4.
- Mechanism evidence package: Phase 5.
</deferred>

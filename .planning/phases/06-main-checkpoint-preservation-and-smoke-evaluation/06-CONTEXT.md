# Phase 6: Main Checkpoint Preservation And Smoke Evaluation - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Evidence closure planning

<domain>
## Phase Boundary

Preserve the current `phc_shape_pnn_iccv_pain_v1` main-condition checkpoint
and verify that the trained policy can be restored through the `test=True`
player path without starting another training run.
</domain>

<decisions>
## Implementation Decisions

### Preserve before touching

The current server checkpoint has already been updated by the main smoke/short
run. Copy it to a timestamped immutable evidence filename before any further
evaluation, ablation, or resume command can overwrite `Humanoid.pth`.

### Treat smoke and short50 separately

`3967` is the actual smoke proof: v1 observation shape `963`, PNN build shape
`963`, checkpoint adaptation, one update, and checkpoint save. `3968` is an
accidental short50 run and must be recorded honestly as extra training evidence,
not as the original smoke-only state.

### Use Slurm GRES, not direct CUDA index

`idx0` and `idx2` were occupied by other jobs. Use Slurm partition/GRES `idx1`
or another verified-free index to avoid interfering with other users.
</decisions>

<code_context>
## Existing Code Insights

- `phc/learning/amp_players.py` now adapts player checkpoints with expanded
  pain observations.
- `phc/learning/amp_agent.py` and `phc/learning/amp_players.py` now include
  `get_pain_obs_size()` in the PNN input shape.
- Server `nvidia-smi` has an NVML mismatch, so PyTorch CUDA and Slurm GRES are
  the reliable readiness checks.
</code_context>

<specifics>
## Specific Run Evidence

- Smoke job: `3967`
- Short50 job: `3968`
- Main exp: `phc_shape_pnn_iccv_pain_v1`
- Server checkpoint: `~/PHC/output/HumanoidIm/phc_shape_pnn_iccv_pain_v1/Humanoid.pth`
- Relevant logs:
  - `~/PHC/logs/phc_pain_v1_short_idx1_3967.out`
  - `~/PHC/logs/phc_pain_v1_short50_idx1_3968.out`
</specifics>

<deferred>
## Deferred Ideas

Do not interpret mechanism quality in this phase. That belongs to Phase 7
after checkpoint/log preservation and eval restore are confirmed.
</deferred>

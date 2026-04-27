# Phase 3: Expanded-Observation Checkpoint Adaptation - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Autonomous auto-discuss

<domain>
## Phase Boundary

Allow pretrained PHC checkpoints to initialize policies whose observation space
has been expanded by PHC-Pain-v1 pain-map observations.
</domain>

<decisions>
## Implementation Decisions

### Adapt only when a v1 pain observation exists

Checkpoint adaptation should be conditional on the task exposing
`get_pain_obs_size() > 0`. This preserves normal PHC/v0 restore behavior.

### Copy leading columns, zero new columns

For model tensors where the target input dimension is wider than the saved
checkpoint, copy saved columns into the leading input columns and zero-fill the
new pain observation columns.

### Adapt running mean/std

The input normalizer must also expand: saved mean values copy into leading
entries, new pain means initialize to zero, and new variances initialize to one.
</decisions>

<code_context>
## Existing Code Insights

- `rl_games` restore calls `set_full_state_weights()` and expects exact tensor
  shapes.
- `phc/learning/amp_agent.py` is already PHC-specific and wraps common restore
  behavior, making it the least invasive adaptation site.
- Old optimizer state may contain buffers with old input shapes, so v1 adaptation
  should start from a fresh optimizer state.
</code_context>

<specifics>
## Specific Ideas

- Patch `AMPAgent.set_full_state_weights()`.
- Deep-copy checkpoint weights before adaptation.
- Adapt 2D model tensors whose target shape has the same output dimension and a
  wider input dimension.
- Add checkpoint adaptation metadata to the adapted in-memory state for logging.
</specifics>

<deferred>
## Deferred Ideas

- Running an actual pretrained checkpoint load requires the full PHC/IsaacGym
  runtime and is outside this static code phase.
</deferred>

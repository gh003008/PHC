# Phase 8: Minimal Ablation Runs - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Ablation planning

<domain>
## Phase Boundary

Run the minimal condition matrix required to determine whether the main
condition behavior depends on pain observation and pain reward pressure.
</domain>

<decisions>
## Implementation Decisions

### Match the main run budget unless blocked

Use the same env count, horizon, minibatch, motion file, checkpoint strategy,
seed, and Slurm GPU isolation as the main condition unless Phase 7 finds that
logging instrumentation must be fixed first.

### Preserve every condition

Each ablation must use a distinct `exp_name` and a copied pretrained/adapted
starting checkpoint so that outputs are not overwritten across conditions.

### Run only the minimal causal matrix first

Start with three conditions: main, `no_obs`, and `no_reward`. Left/right
side-specificity can be a follow-up after these prove the evidence path works.
</decisions>

<code_context>
## Existing Code Insights

- `docs/superpowers/phc_pain_v1_ablation_commands.md` already defines the
  condition commands.
- Existing main command needed `no_log=True`,
  `learning.params.config.minibatch_size=4096`, and an absolute
  `max_epochs` above the loaded checkpoint epoch.
- The current server uses `/usr/local/slurm/bin/sbatch` and Slurm GRES names
  `gpu:idx0..idx3`.
</code_context>

<specifics>
## Specific Ideas

- Main checkpoint currently reached around epoch `63802`.
- For a 50-epoch continuation from a copied checkpoint, set `max_epochs` to the
  loaded epoch plus about 50, not literal `50`.
- Verify free GPU index with:
  `/usr/local/slurm/bin/squeue -h -o "%i %P %b %T %M %R %u %j"`.
</specifics>

<deferred>
## Deferred Ideas

Left-side impairment and walking-distribution training are deferred until the
minimal main/no_obs/no_reward path produces usable metrics.
</deferred>

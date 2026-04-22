# Next session kickoff — Spec #3 (PHC-Pain-v0 short fine-tune)

Paste this into the next Claude Code session:

---

PHC repo, branch `jinsu-pain_baseline_phc_v0`. Spec #1 and Spec #2 are
complete. Summary of current state:

- Spec #1 (env bring-up + pretrained sanity, no code): ✅ done. Final
  commit `81fe0da`. Setup log `docs/superpowers/specs/phc_v0_setup_log.md`.
- Spec #2 (PHC-Pain-v0 impl — `HumanoidImPain` task + pain helpers + env
  config): ✅ done. Commits `f0ad076` (design+log skeleton), `8febe32`
  (plan), `6b6ac45` (4-file impl), `a72f5b0` (populated log). V1–V4
  verification gates all green. Impl log at
  `docs/superpowers/specs/phc_v0_spec2_impl_log.md`.

Key numbers from Spec #2 verification:
- V2 (mode=off): reward 941.05 = Spec #1 baseline exactly.
- V3 (mode=log_only): steady-state pain_scalar ~0.522 under natural
  imitation (no external push needed on the standing motion).
- V4 (mode=guard_only): collapse at step 38 (vs 999), reward 27.15.
  Guard fires correctly but **defaults are too aggressive** for the
  pretrained checkpoint.

## Now start Spec #3: short fine-tune with pain-aware training

Upstream guide references:
- `docs/phc_latest_commit_pain_baseline_quickstart_for_claude_code.md`
  §15 (fine-tune workflow), §16–18 (logging / wandb / acceptance), §20
  (acceptance condition 5).

### Before anything

1. Read the upstream guide §§15–20.
2. Read `docs/superpowers/specs/phc_v0_spec2_impl_log.md` — particularly
   the "Tuning note for Spec #3 kickoff" and "Deviations from the spec"
   sections. They capture everything Spec #3 needs to inherit.
3. Read `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md`
   §5.2 (saturation analysis) and §20 for what was left out of v0.
4. Skim `phc/env/tasks/humanoid_im_pain.py` (the v0 implementation) and
   `phc/env/util/pain_baseline.py` (the pain math). Nothing in Spec #3
   should break obs dim / network / checkpoint compatibility v0
   guarantees.

Then use `superpowers:brainstorming` to design Spec #3. Save the design
as `docs/superpowers/specs/YYYY-MM-DD-phc-pain-v0-spec3-finetune-design.md`,
commit on branch `jinsu-pain_baseline_phc_v0` (or a new branch — see
"Open questions" below), then invoke `superpowers:writing-plans`.

### Open questions the brainstorm must resolve

1. **Retune first, or train through it?**
   - Option A — retune pain params (`pain_threshold` up to ~0.6, and/or
     `guard_gain` down to ~0.2, `max_guard` down to ~0.3) so guard only
     engages in clearly-harmful regions; keep `guard_only` mode; no
     training needed beyond sanity; then optionally fine-tune.
   - Option B — keep current defaults; fine-tune with
     `pain.mode=guard_and_reward` so the policy learns to avoid pain
     rather than fight the guard. Upstream guide §15 seems to prefer
     this.
   - Option C — both: retune slightly, then fine-tune with
     `guard_and_reward`.
2. **Training scale.** Guide §15 is ambiguous on epoch count /
   environments / motion file. Decide:
   - `num_envs`: 3072 (default) vs. smaller for faster iteration?
   - Motion set: single-motion (`amass_isaac_standing_upright_slim.pkl`)
     vs. broader? Starting narrow is safer for demo.
   - Epoch count: guide suggests "short" — pick a concrete target
     (e.g. 500 epochs or 30 min wall clock, whichever first).
3. **Checkpoint hygiene.** Copy `phc_shape_pnn_iccv/Humanoid.pth` into a
   new `exp_name` dir (e.g. `phc_pain_v0_finetune`) so fine-tune doesn't
   overwrite the pristine checkpoint. Verify the copy loads before
   starting training.
4. **Acceptance for V5 (§20 condition 5).** Restate the numeric bar:
   - Pain reduction: post-finetune V3 (log_only on same motion) should
     show `pain_max` at least X% lower than Spec #2's 0.522?
   - Stability recovery: post-finetune V4 (guard_only) should survive
     at least Y steps (vs Spec #2's 38)?
   - Track-error vs baseline: MPJPE should not regress beyond Z mm?
5. **Logging.** Enable wandb / tensorboard? `im_pnn.yaml` already has
   log settings — decide whether to override or keep defaults.
6. **Branch strategy.** Continue on `jinsu-pain_baseline_phc_v0` (keeps
   all Spec #1–#3 work together) or split to
   `jinsu-pain_v0_finetune_spec3`? The spec-per-branch story has been
   "one branch covers all three specs" so far — continuing is natural.
7. **Hard constraints inherited from Spec #2 that still hold:**
   - Obs dim stays 945. No changes to `_compute_humanoid_obs`.
   - No network / action-space / MCP changes.
   - `env.num_prim=4` still required whenever loading `phc_shape_pnn_iccv`
     derivatives.
   - `append_to_obs` stays False.

### Deliverable shape (rough — brainstorm will refine)

- Possibly new learning config `phc/data/cfg/learning/im_pnn_pain.yaml`
  (or just Hydra overrides — decide in brainstorm).
- Possibly a short helper script for checkpoint copy + validation.
- `env_im_pain.yaml` may need a retune (can just be a second config
  file `env_im_pain_retuned.yaml` if we want both side-by-side).
- New implementation log `docs/superpowers/specs/phc_v0_spec3_impl_log.md`
  (mirror Spec #2's format).
- Possibly no code changes at all if Spec #3 is pure config + training.

### Execution mode

Mirror Spec #2: brainstorm → design → plan → subagent-driven execution
→ verification log. User will confirm execution mode choice after the
plan is written. Training runs will need a longer-running task pattern
(monitor / background / etc.); plan for that.

Branch HEAD at start of Spec #3: `a72f5b0`.

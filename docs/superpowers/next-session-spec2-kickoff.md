# Next session kickoff — Spec #2 (PHC-Pain-v0 implementation)

Paste this into the next Claude Code session:

---

PHC repo, branch `jinsu-pain_baseline_phc_v0`. Spec #1 (env bring-up + pretrained sanity) is done — `phc` conda env works, `phc_shape_pnn_iccv` checkpoint runs in viewer.

Now start **Spec #2: PHC-Pain-v0 implementation** — upstream guide §§6–18.

Before anything:
1. Read `docs/phc_latest_commit_pain_baseline_quickstart_for_claude_code.md` §§6–18.
2. Read `docs/superpowers/specs/phc_v0_setup_log.md` — carries four machine-specific gotchas and the `env.num_prim=4` fix.
3. Read `docs/superpowers/specs/2026-04-22-phc-pain-baseline-v0-design.md` for Spec #1 context.

Then use `superpowers:brainstorming` to design Spec #2. Deliverables defined in the guide:
- `phc/env/tasks/humanoid_im_pain.py` (new — subclass of `HumanoidIm`)
- `phc/env/util/pain_baseline.py` (new — pain math helpers)
- `phc/data/cfg/env/env_im_pain.yaml` (new — copy of `env_im_pnn.yaml` + pain block)
- `phc/utils/parse_task.py` (edit — register `HumanoidImPain`)

Hard constraints (v0):
- Do **not** append pain to obs (keep obs dim = 945 for checkpoint compat).
- Do **not** touch network builders, action space, or MCP/composer paths.
- Any eval command with `phc_shape_pnn_iccv` must include `env.num_prim=4`.
- Zero-training demo first (`pain.mode=guard_only`), then consider short fine-tune (Spec #3).

Save the new design doc as `docs/superpowers/specs/YYYY-MM-DD-phc-pain-v0-spec2-impl-design.md`, commit on the same branch, then invoke `superpowers:writing-plans` for the implementation plan.

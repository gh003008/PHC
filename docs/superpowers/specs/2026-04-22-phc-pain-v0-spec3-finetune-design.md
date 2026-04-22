# PHC-Pain-v0 — Spec #3: mild retune + short fine-tune

**Date:** 2026-04-22
**Author:** jinsu (Claude Code brainstorming + one Codex review round)
**Upstream guide:** `docs/phc_latest_commit_pain_baseline_quickstart_for_claude_code.md` §15 (fine-tune), §16–18 (logging), §20 (acceptance)
**Repo:** `ZhengyiLuo/PHC`
**Base commit:** `e2baa0c` (Spec #2 populated log + kickoff prompt stacked on `a72f5b0`).
**Target branch:** `jinsu-pain_baseline_phc_v0` (continue — do NOT split to a new branch).
**Prerequisite:** Spec #2 complete. See `docs/superpowers/specs/phc_v0_spec2_impl_log.md`.

---

## 0. Series placement

Three-spec decomposition of the upstream guide:

- Spec #1 — environment bring-up + pretrained sanity. Done.
- Spec #2 — PHC-Pain-v0 in-environment implementation (guide §§6–18). Done.
- **Spec #3 (this doc)** — mild retune then short fine-tune with `pain.mode=guard_and_reward` (guide §15, §20 condition 5).

Spec #3 closes the v0 acceptance set by retraining the pretrained policy to cooperate with the guard, using pain constants tuned so that the pre-training environment is not catastrophically unstable (which it is with Spec #2 defaults — 999→38 step collapse).

---

## 1. Goal & acceptance

### Goal

Produce a fine-tuned `Humanoid.pth` under `output/HumanoidIm/phc_shape_pnn_iccv_pain/` that:
- loads without shape / key mismatch from the official pretrained checkpoint,
- trains for a short budget (target ≥ 1000 PPO updates, per guide §15.3),
- demonstrates measurable post-training behavior change on the V3/V4 probes,
- does not regress the baseline `mode=off` behavior.

### Acceptance (closing §20)

1. Pretrained checkpoint copy verifies unchanged behavior in the new exp directory (reward within ~1% of Spec #1 baseline 941.05). *[V5.1]*
2. Retuned `guard_only` probe survives ≥ 300 steps with average reward ≥ 470 (50% of baseline). This is the **pre-training survival gate** — proves the retune is not as destructive as Spec #2's defaults before PPO is started. *[V5.2]*
3. Fine-tuning launches, completes the planned budget without reward collapse to 0, and saves the resulting checkpoint. *[V5.3]*
4. Post-training `pain.mode=log_only` probe on the same motion shows steady-state `pain_max` ≤ 0.35 (from 0.522 in Spec #2 V3). *[V6]*
5. Post-training `pain.mode=guard_only` probe shows episode length ≥ 500 steps (from 38) AND reward ≥ 470 (from 27.15). *[V7]*

Each gate is recorded in `docs/superpowers/specs/phc_v0_spec3_impl_log.md` with the command, exit code, last ~30 lines of stdout, and a one-line interpretation note — mirrors Spec #2's log format.

MPJPE is deliberately *not* part of this acceptance. It requires a separate pre-training baseline measurement that Spec #2 did not collect. Deferred to v1 (obs-appended pain).

---

## 2. Hard constraints (inherited from Spec #2)

All of these held throughout Spec #2 and must hold throughout Spec #3:

- Observation dimension stays at **945**. `_compute_observations` and its callees are untouched.
- `num_prim: 4` — required by the pretrained `phc_shape_pnn_iccv` checkpoint. Both the new yaml and all CLI commands set this explicitly.
- No edits to `phc/env/tasks/humanoid_im.py`, `phc/env/tasks/humanoid.py`, or any network builder. Extension is by subclass only — the only existing class `HumanoidImPain` is edited in place.
- No new top-level dependencies. `pain_baseline.py` stays pure torch; no change to it.
- No MCP / getup / IsaacLab paths.
- `append_to_obs: False` contract flag remains. No code path reads it.

---

## 3. Deliverables

Three files touched; one of them is a pure edit, two are new.

| # | Path | Kind | Role |
|---|---|---|---|
| 1 | `phc/data/cfg/env/env_im_pain_finetune.yaml` | **new** | Copy of `env_im_pain.yaml` with a retuned `pain:` block + `mode: "guard_and_reward"`. |
| 2 | `phc/env/tasks/humanoid_im_pain.py` | **edit** | Add body-group pain logging to `_compute_reward`; build body→group index table in `__init__`. |
| 3 | `docs/superpowers/specs/phc_v0_spec3_impl_log.md` | **new** | Verification log (V5.0 through V7). |

**No new learning config.** All training-time batch knobs (`horizon_length`, `minibatch_size`, `max_epochs`) are set via Hydra CLI overrides on `im_pnn.yaml`. Rationale: Spec #3 doesn't change the PPO structure itself; only the schedule. A separate `im_pnn_pain.yaml` would be a YAGNI clone of `im_pnn.yaml` with 2–3 value changes.

**No checkpoint-copy script.** The two shell commands to `mkdir -p` and `cp` live in the impl log as copy-paste steps — simpler and more transparent than a wrapper.

Target line budgets: (1) ~80 lines (clone + retuned values); (2) +20 lines to existing file; (3) ~200 lines (full V-sequence).

---

## 4. Architecture changes (minimal)

Spec #2 already installed the end-to-end pain pipeline:
```
pre_physics_step → _action_to_pd_targets (guard) → sim → post_physics_step → _refresh_sim_tensors → _compute_reward (pain update + penalty) → extras
```

Spec #3 only changes:
- **Parameter values** (yaml, config-only).
- **What gets logged** to `self.extras` — adds 7 new scalar floats per call. No structural changes to the reward or guard.

No changes to `pain_baseline.py`. No changes to task hooks (`__init__` additions are index table and cached group slices; `_compute_reward` additions are 5–8 lines at the end).

### 4.1 Body-group logging path

In `HumanoidImPain.__init__`:

```python
self._group_dof_idx: dict[str, torch.Tensor] = self._build_body_group_idx()
```

`_build_body_group_idx()` reads `self._body_names` (exposed by the Humanoid base) and maps SMPL body names to five groups, then converts each group's body indices (1-indexed relative to non-root) into a DOF-index tensor via `3*(body_idx-1) + {0,1,2}`. Groups:

| Group | SMPL body names (non-root only) |
|---|---|
| `legs`   | L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle, L_Toe, R_Toe |
| `arms`   | L_Thorax, R_Thorax, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist |
| `torso`  | Torso, Spine, Chest |
| `head`   | Neck, Head |
| `hands`  | L_Hand, R_Hand |

(`Pelvis` is the root body — it has no DOFs and does not appear in any group.)

In `_compute_reward` after the existing scalar extras block:

```python
for name, idx in self._group_dof_idx.items():
    self.extras[f"pain_{name}"] = float(self.pain_state[:, idx].mean().item())

self.extras["pain_p90"] = float(torch.quantile(self.pain_scalar, 0.90).item())
self.extras["pain_p99"] = float(torch.quantile(self.pain_scalar, 0.99).item())
self.extras["pain_saturated_frac"] = float((self.pain_state >= 2.5).float().mean().item())
```

All additions are scalar floats. Total extras overhead per step: 8 extra `.item()` syncs (~10–50 μs on 3070 Ti). Negligible compared to Isaac Gym sim step cost.

### 4.2 Why no per-DOF / per-body wandb logging

Per-DOF (69-dim) and per-body (24-dim) arrays are already gated behind `flags.im_eval` (Spec #2 design §6.4) and that gate remains off during training. wandb would otherwise receive ~100 scalars per step × thousands of steps = storage pressure and unreadable dashboards. Body-group aggregates + tail quantiles give enough signal to diagnose "where is pain localized" and "is any joint saturating" without the clutter.

---

## 5. Retuned pain parameters (Option C from brainstorm, Codex-validated)

### 5.1 Values

```yaml
# phc/data/cfg/env/env_im_pain_finetune.yaml  (pain: block only; rest copies env_im_pain.yaml)
pain:
  enabled: True
  mode: "guard_and_reward"       # Spec #3 is the first guard_and_reward run
  append_to_obs: False           # unchanged; v0 contract flag
  use_external_contact: True

  lambda_p: 0.02                 # was 0.05 — softer reward penalty

  w_limit: 1.0                   # unchanged
  w_torque: 0.35                 # unchanged
  w_power: 0.15                  # unchanged
  w_contact: 0.50                # unchanged

  joint_limit_margin_ratio: 0.15 # unchanged
  torque_ref_scale: 0.50         # unchanged
  power_ref: 5.0                 # unchanged
  contact_force_ref: 150.0       # unchanged

  pain_threshold: 0.30           # was 0.10 — suppress most natural standing "pain"
  pain_cap: 3.0                  # unchanged

  rise_alpha: 0.25               # unchanged
  decay_alpha: 0.02              # unchanged

  guard_gain: 0.25               # was 0.60
  max_guard: 0.30                # was 0.75 — cap guard coefficient harder

  log_joint_pain: True           # unchanged; gated on flags.im_eval
  log_contact_pain: True         # unchanged; gated on flags.im_eval
```

Six values change vs `env_im_pain.yaml`: `mode`, `lambda_p`, `pain_threshold`, `guard_gain`, `max_guard`, plus the file's `notes:` field.

### 5.2 Why these numbers

From Spec #2 V3: under `mode=log_only` on the standing motion, steady-state `pain_scalar ≈ 0.522` with `pain_threshold=0.10`. The accumulator steady state is `rise_alpha / decay_alpha · relu(inst − threshold) = 12.5 · relu(inst − threshold)`. Inverting the 0.522 observation gives a rough mean `inst ≈ 0.142` sustained across DOFs.

- **`pain_threshold: 0.30`** — above `inst ≈ 0.142`, so most natural standing is below threshold and yields zero drive. Spikes under actual stress (push, joint-limit breach) still propagate. Codex recommended 0.30 over 0.35 after arguing 0.35 erases high-tail joint signal too aggressively.
- **`guard_gain: 0.25, max_guard: 0.30`** — at `pain_state = 1.0`, guard coefficient is 0.25; at `pain_state = 1.2+`, it saturates at 0.30. That is, even a fully-saturated pain state pulls the PD target back by only 30% (vs 75% in Spec #2 defaults). This is the primary reason Spec #2 V4 collapsed at step 38 and the retune's job is to eliminate that specific failure mode.
- **`lambda_p: 0.02`** — Spec #2's 0.05 was never tested under reward_only / guard_and_reward in practice. 0.02 is a conservative starting point: with `pain_scalar ≈ 0.1–0.4` expected under retuned steady state, the penalty per step is 0.002–0.008, vs imitation reward per step of ~1. So pain accounts for ~0.2–0.8% of reward magnitude — measurable but not dominant.
- **`rise_alpha, decay_alpha, pain_cap` unchanged** — the accumulator shape is already reasonable; the retune focuses on the thresholds and guard gain, not the dynamics.

### 5.3 Saturation behavior after retune

With new defaults, update-rule steady state for constant `inst` is `12.5 · relu(inst − 0.30)`. Sustained `inst ≥ 0.54` drives `pain_state` to cap=3.0. Guard saturates at `pain_state ≥ 1.20`, reaching `max_guard=0.30`. So the pain-state range `[1.2, 3.0]` still collapses under the guard — but that collapse now pulls PD targets only 30% back, not 75%. The policy has room to operate even near saturation.

---

## 6. Training scale and GPU calibration

### 6.1 Target

- **Sample product per PPO update**: `num_envs × horizon_length ≈ 32,768` (= 1024 × 32 — the β target from brainstorm).
- **Budget**: `max_epochs=2000` OR wall-clock 30 min, whichever hits first.
- **Motion set**: `sample_data/amass_isaac_standing_upright_slim.pkl` only — matches Spec #2 V3/V4 so V6/V7 are direct before/after comparisons.
- **`headless=True`** — no viewer VRAM overhead during training.

### 6.2 GPU calibration ladder (V5.0)

Hardware: RTX 3070 Ti, 8 GB VRAM. `num_prim=4` adds ~30% memory vs default num_prim=3. PHC reference num_envs=3072 is designed for A100-class hardware; 8 GB will cap somewhere lower.

Calibration: at each rung, run 10 PPO epochs with `env.pain.enabled=False` (so we're not also debugging the pain pipeline) and record peak VRAM from a background `nvidia-smi` poll. Accept the *largest* rung where peak VRAM stays ≤ 85%.

| Rung | num_envs | horizon_length | Sample product | Note |
|---|---|---|---|---|
| 1 | 1024 | 32  | 32,768 | First try. Likely OOM on 8 GB with num_prim=4. |
| 2 | 512  | 64  | 32,768 | More probable sweet spot. |
| 3 | 256  | 128 | 32,768 | Likely final value if rung 2 OOMs. |
| 4 | 128  | 256 | 32,768 | Fallback. Higher on-policy staleness. |

If rung 4 still OOMs, drop `minibatch_size` from 16384 to 8192 (and then 4096) — this reduces per-update gradient memory specifically, without changing sample count.

### 6.3 Why ladder-halve envs × double horizon

rl_games does not natively support cross-rollout gradient accumulation, but Isaac Gym's fixed `num_envs` allocation means "run 1024 envs 4 times and concat" is bit-identical to "run 1024 envs for horizon=128" (same env instances, same on-policy data, same memory profile). Cross-rollout resets between mini-rollouts would add starting-state diversity but waste steps and require rl_games subclassing — out of scope for v0. The ladder therefore tunes `horizon_length` as the compensating lever.

Tradeoff: larger `horizon_length` → more policy staleness between the rollout start and the PPO update. For a short fine-tune from a stable pretrained policy, this is acceptable; if we were from-scratch training, rung 4's horizon=256 would be risky.

### 6.4 AMP batch knobs

`amp_batch_size: 512, amp_minibatch_size: 4096` in `im_pnn.yaml` are replay-buffer driven and independent of `num_envs`. Keep defaults; downscale only if seen in OOM signatures.

### 6.5 Escape hatch (documented, not implemented)

If calibration forces rung 4 *and* env-state diversity turns out to matter (e.g., policy overfits to 128 starting states), subclass `CommonAgent` / rl_games `A2CAgent` to do multi-rollout accumulation with `sim.reset()` between sub-rollouts. ~100 lines + validation. **Not part of v0**; noted here so a future spec can pick it up without re-discovering the idea.

---

## 7. Checkpoint hygiene

### 7.1 Copy pristine checkpoint to new exp directory

```bash
mkdir -p output/HumanoidIm/phc_shape_pnn_iccv_pain
cp output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth \
   output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth
```

Rationale (guide §15.1): `epoch=-1` loads `${output_path}/Humanoid.pth`, and training rewrites that file with each checkpoint save. Keeping the pristine copy untouched under `phc_shape_pnn_iccv/` means Spec #2's V2/V3/V4 commands stay reproducible for the lifetime of the branch.

### 7.2 V5.1 — load/reproduce sanity

Before any training, prove the new directory's checkpoint is loadable and produces baseline behavior:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

Accept iff reward within 1% of 941.05 (Spec #2 V2's number). If it diverges, the copy is corrupted — fix before proceeding.

---

## 8. Logging (Q5 from brainstorm)

### 8.1 wandb

PHC `run_hydra.py` auto-initializes wandb when `test=False AND no_log=False AND debug=False`. Training path meets all three. No config change needed — just ensure `wandb login` has been run once on this machine so runs land in the right account.

- `project=PHC` (from `env_im_pain_finetune.yaml:project_name`).
- `run.name=phc_shape_pnn_iccv_pain` (from `cfg.exp_name`).
- `notes` field set to `"pain v0 Spec #3 short fine-tune (guard_and_reward)"` via `env.notes=...` CLI override on the training command, for searchability.
- wandb **offline mode** allowed if user prefers: `WANDB_MODE=offline` env var. Impl log records which mode was used.

### 8.2 Pain scalars logged per step (via `self.extras`)

Already present from Spec #2:
- `pain_mean`, `pain_max`, `pain_internal_mean`, `pain_external_mean`

Added in Spec #3:
- `pain_p90`, `pain_p99` — env-dim quantiles of `pain_scalar`. Tail visibility.
- `pain_saturated_frac` — fraction of DOF entries with `pain_state ≥ 2.5`. Cap-saturation alarm.
- `pain_legs`, `pain_arms`, `pain_torso`, `pain_head`, `pain_hands` — body-group means. Localization without 69-dim clutter.

Per-DOF (`pain_state`) and per-body (`pain_contact_body`) full arrays remain gated behind `flags.im_eval` — not logged during training.

### 8.3 Histograms

Not logged per-step. If needed for post-hoc analysis, a one-shot `wandb.Histogram(pain_state.cpu().numpy())` at the end of training can be added in a follow-up. Out of Spec #3 scope.

---

## 9. Verification procedure

Seven gates. V5.0–V5.3 are pre-training; V6 and V7 are post-training. Each recorded in `phc_v0_spec3_impl_log.md`.

### V5.0 — GPU calibration

Run `env.pain.enabled=False` at each rung in §6.2, 10 PPO epochs each, with a background `nvidia-smi` monitoring process. Record peak VRAM per rung. Accept the largest rung ≤ 85% peak.

```bash
# Background monitor
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 1 \
  > /tmp/phc_spec3_v5.0_mem_<rung>.csv &
MONITOR_PID=$!

# Training probe (10 epochs) — throwaway exp_name, no pretrained load.
# epoch is intentionally omitted so rl_games starts from random init
# (memory profile is dominated by env sim state and network size, both
# unchanged vs loading the pretrained — calibration is about
# allocation-time VRAM, not policy quality).
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_spec3_calib_<rung> test=False no_log=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=<N> headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.horizon_length=<H> \
  learning.params.config.max_epochs=10

kill $MONITOR_PID
awk -F, 'NR>1 {if ($1>max) max=$1} END {print "peak_used_MB=" max}' /tmp/phc_spec3_v5.0_mem_<rung>.csv

# Cleanup after each rung so throwaway checkpoints don't accumulate:
rm -rf output/HumanoidIm/phc_spec3_calib_<rung>
```

Notes:
- `no_log=True` — calibration probes must NOT create wandb runs.
- `exp_name=phc_spec3_calib_<rung>` is disposable. It isolates the
  calibration output from the pristine `phc_shape_pnn_iccv_pain/` dir that
  V5.3 will write into.
- If rl_games errors out because `max_epochs=10` is below some minimum
  needed to complete a save cycle, raise to `max_epochs=50` — still
  bounded enough to be a throwaway probe.

### V5.1 — Checkpoint copy sanity

Exact command in §7.2. Accept iff `reward: ~941 steps: 999.0` (within 1% of 941.05).

### V5.2 — Retuned guard_only survival gate

Probe the retuned config *before* training to make sure it doesn't collapse:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=guard_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

Accept iff `steps ≥ 300` AND `reward ≥ 470`. If fails: retune further (raise threshold, lower guard_gain more) before attempting training. Codex's note: "prevents wasting PPO runs on another immediately destructive guard setting."

### V5.3 — Fine-tuning run

Using calibrated `num_envs=N` and `horizon_length=H` from V5.0:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=False \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=<N> headless=True \
  env.pain.enabled=True env.pain.mode=guard_and_reward \
  env.notes="pain v0 Spec #3 short fine-tune (guard_and_reward)" \
  learning.params.config.horizon_length=<H> \
  learning.params.config.max_epochs=2000
```

Accept iff training completes 2000 epochs (or is stopped at 30 min wall-clock — whichever hits first) without reward collapse to 0, and a new `Humanoid.pth` is saved in `output/HumanoidIm/phc_shape_pnn_iccv_pain/`.

Launched as a Monitor background job with a filter covering both progress and failure signatures (`elapsed_steps|Traceback|CUDA|OOM|assert|reward:`). Long-running (~30 min) — Monitor's 1-hour timeout is appropriate.

### V6 — Post-training log_only probe

Same command as Spec #2 V3 but pointing at the retrained checkpoint:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=log_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

Accept iff steady-state `pain_max ≤ 0.35` (Spec #2 V3 was 0.522). A temporary `print(self.pain_scalar.max().item())` inside `_compute_reward` can be used during bring-up, as in Spec #2 V3, and reverted before the commit.

### V7 — Post-training guard_only probe

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain epoch=-1 test=True \
  env=env_im_pain_finetune env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=guard_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

Accept iff `steps ≥ 500` AND `reward ≥ 470`. (Spec #2 V4 was 38 steps / 27.15 reward.)

---

## 10. Commit strategy

Four commits on `jinsu-pain_baseline_phc_v0`, mirroring Spec #2's pattern:

1. **Design + log skeleton.** This document + `phc_v0_spec3_impl_log.md` with section headers V5.0–V7 and no body yet.
2. **Implementation.** `env_im_pain_finetune.yaml` (new) + `humanoid_im_pain.py` edit (body-group table + 8 extras lines). Gate on `flags.im_eval=False` training path preserving obs dim / checkpoint compatibility — smoke-test by running Spec #2 V2 again unchanged.
3. **Populated log (part 1).** V5.0/V5.1/V5.2 results filled in.
4. **Populated log (part 2).** V5.3 / V6 / V7 results + exit-gate signoff.

Why a two-part log commit? V5.3 is the ~30-minute fine-tune run; committing V5.0–V5.2 first locks in the pre-training state so a mid-training failure doesn't require re-running calibration.

---

## 11. Out of scope

Explicit non-deliverables for Spec #3:

- MPJPE measurement and regression gate (v1).
- Appending pain to the observation vector (v1 branch).
- Multi-motion training sets.
- `amp_batch_size` / `amp_minibatch_size` tuning (left at defaults).
- Cross-rollout gradient-accumulation subclass (escape hatch, documented only).
- Histogram logging in wandb.
- New learning config (`im_pnn_pain.yaml` — Hydra overrides used instead).
- Any change to `pain_baseline.py` (stays pure torch, frozen).
- Retune of `rise_alpha`, `decay_alpha`, `pain_cap`, or the `w_*` weights.

---

## 12. Risks and mitigations

1. **Retune still too aggressive — V5.2 fails.**
   Mitigation: iterate on `pain_threshold` up to 0.45 and `guard_gain` down to 0.15 before declaring retune insufficient. Survival gate blocks PPO until this clears.

2. **V5.0 rung 4 OOMs even with `minibatch_size=4096`.**
   Mitigation: reduce `amp_minibatch_size` from 4096 to 2048 (first time AMP knobs enter play). If still OOM: reduce `env.num_envs=64` and push `horizon_length=512`. Record in impl log.

3. **Fine-tune diverges (reward collapse within first 100 epochs).**
   Mitigation: lower `lambda_p` to 0.005 and re-run. If still diverges, temporarily set `pain.mode=reward_only` (no guard) to disentangle whether the guard or the reward penalty is the destabilizer.

4. **V6 shows `pain_max` *higher* than 0.522 post-training.**
   Unusual but possible — would indicate the policy found a higher-reward-minus-penalty region that happens to incur more pain. Mitigation: confirm by re-measuring V6 with a different random seed; if confirmed, raise `lambda_p` and retrain.

5. **V7 episode length improves but reward stays low.**
   Policy learned to survive but tracking quality collapsed. Mitigation: raise `lambda_p` OR decrease `guard_gain` further — this redistributes pressure from "stay alive" to "track reference." Impl log's V7 note explicitly compares reward to episode length to disambiguate.

6. **Bunch of wandb runs pollute the project because of V5.0 calibration probes.**
   Mitigation: pass `no_log=True` on every V5.0 rung. Only V5.3 creates a wandb run.

7. **Training overwrites the pristine pretrained checkpoint despite §7.**
   Mitigation: V5.1 validates the copy first; `rl_games` saves into `${output_path}` which is set from `exp_name`, so `exp_name=phc_shape_pnn_iccv_pain` (never `phc_shape_pnn_iccv` during training) keeps the original untouched. Impl log's V5.1 proves the dir isolation works.

8. **`horizon_length=128+` at rung 3/4 produces NaNs in advantages.**
   rl_games' GAE is numerically fine at long horizons, but value-function bootstrap errors compound. Mitigation: log `train_info["losses/v_loss"]` (already in `common_agent.py:632`) per epoch and watch for spikes. Reduce horizon and accept smaller sample product if seen.

---

## 13. Debug hints

- **`pain_p99` stays near `pain_cap=3.0` for extended periods** → some joint is saturating. Check which body-group log (`pain_legs`, etc.) rises first; the per-DOF array is also available via `flags.im_eval=True` for a one-off measurement.
- **V5.1 reward is 941 but V5.2 reward is 470-ish** → retune is on the edge of the survival gate. Acceptable for Spec #3 start, but note that training may not have much room to improve.
- **wandb run name shows `phc_shape_pnn_iccv_pain` but logs nothing** → `test=True` or `no_log=True` got into the training command by accident. Check for Hydra override contamination.
- **V5.0 rung 1 fits in 8 GB** → 3070 Ti is over-performing expectations; re-run the ladder with `num_envs=2048, horizon_length=16` to verify, and use the larger scale if also ≤ 85%.
- **V7 passes reward bar but fails step bar** → `min_episode_steps=...` may be implicitly capping; check `env.episode_length` and `terminationHeight` in `env_im_pain_finetune.yaml` weren't accidentally changed.

---

## 14. Exit gate to done

Spec #3 — and v0 as a whole — is complete when:

- Four commits landed on `jinsu-pain_baseline_phc_v0`.
- `git diff phc/` from Spec #2's last impl commit touches exactly:
  - 1 new file under `phc/data/cfg/env/`,
  - 1 edit in `phc/env/tasks/humanoid_im_pain.py` (< 30 lines inserted),
  - 0 other files under `phc/`.
- V5.0 records calibrated `num_envs`/`horizon_length` in impl log.
- V5.1/V5.2 pass with recorded numbers.
- V5.3 produces a new `Humanoid.pth` under `output/HumanoidIm/phc_shape_pnn_iccv_pain/` AND a wandb run.
- V6 and V7 pass with recorded numbers.
- User has reviewed the impl log and approved v0 as shipped.

On approval, v0 is frozen. Any further work (obs-appended pain, MPJPE gating, multi-motion fine-tune) starts on a new `pain_baseline_phc_v1_*` branch.

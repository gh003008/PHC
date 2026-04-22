# PHC-Pain-v0 — Spec #2: Pain-aware implementation

**Date:** 2026-04-22
**Author:** jinsu (via Claude Code brainstorming, two Codex review rounds)
**Upstream guide:** `docs/phc_latest_commit_pain_baseline_quickstart_for_claude_code.md` §§6–18
**Repo:** `ZhengyiLuo/PHC`
**Base commit:** `846988d` (Spec #1 commits stacked on top)
**Target branch:** `jinsu-pain_baseline_phc_v0`
**Prerequisite:** Spec #1 (env bring-up) complete — see `docs/superpowers/specs/phc_v0_setup_log.md`.

---

## 0. Series placement

Three-spec decomposition of the upstream guide:

- Spec #1 — environment bring-up + pretrained sanity (done).
- **Spec #2 (this doc)** — PHC-Pain-v0 in-environment implementation (guide §§6–18).
- Spec #3 (future) — `guard_and_reward` short fine-tune (guide §15).

Spec #2 delivers a **zero-training pain-aware baseline** that runs on the official
single-primitive PHC checkpoint (`phc_shape_pnn_iccv`, 4 PNN primitives). Fine-tuning
is explicitly out of scope.

---

## 1. Goal & acceptance

### Goal

Add environment-side pain estimation, reward penalty, and action governor to PHC
without touching observation dimension, network builders, action space, or
checkpoint format. The pretrained checkpoint must still load and run unchanged
with `env.pain.enabled=False`.

### Acceptance (subset of upstream §20 minus fine-tune)

1. Official PHC single-primitive checkpoint still runs unchanged. *(inherited from Spec #1.)*
2. `HumanoidImPain` with `pain.enabled=False` (or `mode=off`) matches baseline behavior — reward magnitude and step count within noise of the Spec #1 sanity result.
3. `pain.mode=log_only` produces nonzero `pain_mean` / `pain_max` under external push (viewer `j` key) with no reward change.
4. `pain.mode=guard_only` visibly alters motion (reduced ROM / slower movement) on the same motion file and seed, without retraining or loading a modified checkpoint.
5. *Condition 5 of §20 (short fine-tune with `guard_and_reward`) is deferred to Spec #3.*

Each of 2/3/4 is recorded in `docs/superpowers/specs/phc_v0_spec2_impl_log.md`
as the command that was run, a stdout excerpt (~30 lines), and a one-line viewer
observation note.

---

## 2. Hard constraints (v0)

These are load-bearing — violating any of them breaks checkpoint compatibility
or scope:

- Observation dimension stays at **945**. `_compute_observations` and its callees are untouched. `append_to_obs: False` in the config is a contract flag only; v0 code does not read it.
- `num_prim: 4` in `env_im_pain.yaml` — matches the pretrained `phc_shape_pnn_iccv/Humanoid.pth` (4 PNN columns). *(Spec #1 log §4 recorded the shape-mismatch bug when `num_prim=3`.)*
- No edits to `phc/env/tasks/humanoid_im.py`, `phc/env/tasks/humanoid.py`, or the network builders. Extension is by subclass only. The only upstream edit is a one-line import in `phc/utils/parse_task.py`.
- No new top-level dependencies. `pain_baseline.py` is pure torch; the task subclass pulls only from the existing PHC inheritance chain.
- No MCP / getup composer paths (`env.models=[...]`, `HumanoidImMCP`, etc).
- No IsaacLab eval path (`scripts/eval_in_isaaclab.py`).

---

## 3. Deliverables (four files)

| # | Path | Kind | Role |
|---|---|---|---|
| 1 | `phc/env/util/pain_baseline.py` | new, pure torch | 8 helper functions; no Isaac Gym import; CPU-runnable for REPL debug. |
| 2 | `phc/env/tasks/humanoid_im_pain.py` | new, `HumanoidIm` subclass | Buffer allocation, `_update_pain_buffers`, `_compute_reward` override, `_action_to_pd_targets` override, `_reset_env_tensors` override, extras logging. |
| 3 | `phc/data/cfg/env/env_im_pain.yaml` | new | Copy of `env_im_pnn.yaml` with three field edits + a `pain:` block. |
| 4 | `phc/utils/parse_task.py` | edit, one line | Import `HumanoidImPain` so the existing `eval(args.task)` resolves it. |

Target line budgets: (1) ≤ 130 lines; (2) ≤ 180 lines; (3) ≤ 90 lines; (4) +1 line.

---

## 4. Architecture & data flow

### 4.1 One-frame flow

```
pre_physics_step(actions)   [Humanoid.pre_physics_step]
  └─ _action_to_pd_targets(actions)            ← HumanoidImPain OVERRIDE
       ├─ pd_tar = super()._action_to_pd_targets(actions)      # offset + scale
       └─ if pain_mode in {guard_only, guard_and_reward}:
            guard = clamp(guard_gain * pain_state, 0, max_guard)
            pd_tar = dof_pos + (1 - guard) * (pd_tar - dof_pos)     # rad DOF space
  ├─ freeze_hand / freeze_toe masking  (super)
  └─ set_dof_position_target_tensor    (super)

_physics_step()

post_physics_step()          [Humanoid.post_physics_step]
  ├─ _refresh_sim_tensors()   # dof_force, contact, dof_state all fresh
  └─ _compute_reward(actions)                  ← HumanoidImPain OVERRIDE
       ├─ super()._compute_reward(actions)     # imitation + power reward
       ├─ _update_pain_buffers()               # inst → state → scalar
       ├─ if pain_mode in {reward_only, guard_and_reward}:
       │    rew_buf -= lambda_p * pain_scalar
       └─ extras["pain_*"] scalar always; per-DOF arrays only when flags.im_eval

_reset_env_tensors(env_ids)  [Humanoid._reset_env_tensors]
  └─ after super(), zero pain_{inst,state,internal,external,contact_body,scalar}
     for env_ids  (prevents carry-over across episode boundaries)
```

### 4.2 Why these hook points

- **`_action_to_pd_targets` (not `pre_physics_step`) for the guard.** The guide's
  formula `q + (1-guard)*(target - q)` is only meaningful when `target` and `q`
  share the same space. In PHC, raw `actions` are scaled/offset into radian DOF
  space inside `_action_to_pd_targets` (`humanoid_im.py:1094`). Overriding that
  single method keeps the guard in the correct space and lets `super()`'s
  freeze_hand/freeze_toe masking run on the guarded target unchanged. Codex
  review round 1 confirmed this over the `pre_physics_step` alternative.

- **Pain update at the start of `_compute_reward`.** `humanoid.post_physics_step`
  (`humanoid.py:1634`) calls `_refresh_sim_tensors()` immediately before
  `_compute_reward(self.actions)`. So `self._dof_pos`, `self._dof_vel`,
  `self.dof_force_tensor`, and `self._contact_forces` are all freshly valid at
  the start of `_compute_reward`. The super call computes the imitation +
  power reward; pain update runs right after and subtracts `lambda_p *
  pain_scalar` on top when appropriate.

- **One-step pain-state lag is accepted.** `pain_state` is updated in
  `post_physics_step` of frame N and consumed in `pre_physics_step` of frame
  N+1. Both options considered in review had the same lag; v0 treats this as
  acceptable.

### 4.3 Why subclass, not edit

Upstream `HumanoidIm` is the backbone for many other tasks (`HumanoidImMCP`,
`HumanoidImGetup`, …). Editing it in place would entangle the pain code with
paths we explicitly exclude from v0. Subclassing keeps those paths bit-for-bit
identical and lets us A/B v0 by toggling `task: HumanoidIm` vs `task:
HumanoidImPain`.

---

## 5. `phc/env/util/pain_baseline.py` — helper module

Pure torch. No numpy. No Isaac Gym. No device/dtype arguments (caller
responsibility). `eps = 1e-6` default.

Shape legend: `B = num_envs`, `D = num_dof` (SMPL=69), `Nb = num_bodies`
(SMPL=24).

### 5.1 Function list (8)

**B1. `compute_joint_limit_pain(q, q_lo, q_hi, margin_ratio=0.15, eps=1e-6) -> (B, D)`**

```python
q_mid  = 0.5 * (q_lo + q_hi)
q_half = 0.5 * (q_hi - q_lo)
q_safe = q_half * (1 - margin_ratio)
excess = relu(abs(q - q_mid) - q_safe)
pain   = excess / (q_half - q_safe + eps)
pain   = pain * (q_half > eps).to(pain.dtype)    # fixed-joint mask
return pain
```

Fixed-joint mask (`q_half > eps`) prevents amplification when
`q_hi == q_lo`. In the safe zone (`|q - q_mid| ≤ q_safe`), output is 0. At the
hard limit, output ≈ `1 / margin_ratio`.

**B2. `compute_torque_pain(tau, tau_limits, ref_scale=0.5, eps=1e-6) -> (B, D)`**

```python
tau_ref = clamp(ref_scale * tau_limits, min=1.0)
return abs(tau) / (tau_ref + eps)
```

The floor at 1.0 N·m protects against small `tau_limits` entries (cf. upstream
guide §9.1).

**B3. `compute_power_pain(tau, dq, power_ref=5.0, eps=1e-6) -> (B, D)`**

```python
return abs(tau * dq) / max(power_ref, eps)
```

**B4. `compute_contact_pain(contact_forces, ref_force=150.0, eps=1e-6) -> (B, Nb)`**

```python
return norm(contact_forces, dim=-1) / max(ref_force, eps)
```

**B5. `broadcast_body_pain_to_dof(body_pain, num_dof) -> (B, D)`** *(SMPL-specific)*

```python
# assert num_dof == 3 * (body_pain.shape[1] - 1)
non_root = body_pain[:, 1:]
return non_root.unsqueeze(-1).expand(B, Nb - 1, 3).reshape(B, -1)
```

Assumes the SMPL layout `_dof_names = _body_names[1:]` with 3 consecutive DOFs
per non-root body. Root body (index 0) has no DOFs and is dropped; its contact
pain is logged separately at the task level if needed.

**B6. `combine_internal_pain(limit_pain, torque_pain, power_pain, w_limit=1.0, w_torque=0.35, w_power=0.15) -> (B, D)`**

```python
return w_limit * limit_pain + w_torque * torque_pain + w_power * power_pain
```

Kept as a helper so the task class does not reimplement the weighted sum inline
(Codex review round 1 suggestion). External contact is added at the call site
with a single scalar multiply (`w_contact * broadcast_body_pain_to_dof(...)`).

**B7. `update_pain_state(prev, inst, rise_alpha=0.25, decay_alpha=0.02, threshold=0.1, cap=3.0) -> (B, D)`**

```python
drive = relu(inst - threshold)
nxt   = prev * (1 - decay_alpha) + rise_alpha * drive
return clamp(nxt, 0.0, cap)
```

**B8. `apply_pain_action_guard(pd_tar, dof_pos, pain_state, guard_gain=0.6, max_guard=0.75) -> (B, D)`**

```python
guard = clamp(guard_gain * pain_state, 0.0, max_guard)
return dof_pos + (1.0 - guard) * (pd_tar - dof_pos)
```

### 5.2 Saturation behavior (documented, not a bug)

Given the defaults above:

- Unclamped steady state of B7 for constant `inst` is `12.5 * relu(inst - threshold)`. So sustained `inst ≥ 0.34` drives `pain_state` to `cap=3.0`.
- B8 saturates: `max_guard=0.75` is reached at `pain_state ≥ 1.25`. The pain range `[1.25, 3.0]` collapses under B8.

These are intended for v0's binary-ish "guarding visible / not visible" demo.
Spec #3 may retune.

### 5.3 What lives outside this module

- Body-index lookups, torso/contact-body id caching — task-class concern.
- Per-env state buffers — task-class concern.
- `self.extras` plumbing — task-class concern.

---

## 6. `phc/env/tasks/humanoid_im_pain.py` — task class

### 6.1 File header

```python
import torch
from phc.env.tasks.humanoid_im import HumanoidIm
from phc.utils.flags import flags
from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard,
)
```

`isaacgym` is transitively imported through `HumanoidIm`; no direct import
needed.

### 6.2 `__init__`

- Call `super().__init__` first. Codex review round 2 confirmed no construction-time call to `_compute_reward` or `_action_to_pd_targets`, so buffer allocation after super is safe.
- Read `cfg["env"].get("pain", {})` into cached per-attribute floats/bools — avoids dict lookup in the hot path.
- Validate `pain_mode ∈ {off, log_only, reward_only, guard_only, guard_and_reward}`; `assert` else raise.
- Allocate six buffers on `self.device`:
  - `pain_inst`, `pain_state`, `pain_internal`, `pain_external` — `(B, D)`
  - `pain_contact_body` — `(B, Nb)` (debug/logging of pre-broadcast body-level contact pain)
  - `pain_scalar` — `(B,)`
- Assert SMPL layout: `self.num_dof == 3 * (self.num_bodies - 1)`. Fail loudly otherwise (v0 is SMPL-only).

### 6.3 `_update_pain_buffers(self)`

Early-exit when `not self.pain_enabled or self.pain_mode == "off"`. Otherwise:

1. `q, dq, tau = self._dof_pos, self._dof_vel, self.dof_force_tensor`
2. `limit = compute_joint_limit_pain(q, self.dof_limits_lower, self.dof_limits_upper, margin)`
3. `torque = compute_torque_pain(tau, self.torque_limits, tau_scale)`
4. `power = compute_power_pain(tau, dq, power_ref)`
5. `internal = combine_internal_pain(limit, torque, power, w_limit, w_torque, w_power)`
6. If `use_external_contact`:
   `body = compute_contact_pain(self._contact_forces[:, :self.num_bodies, :], contact_ref)`
   `external = w_contact * broadcast_body_pain_to_dof(body, self.num_dof)`
   (store `body` in `self.pain_contact_body` for logging.)
   else: `external = zeros_like(internal)` and zero the body buffer.
7. `inst = internal + external`
8. Write `self.pain_{internal, external, inst}[:] = ...`
9. `self.pain_state[:] = update_pain_state(self.pain_state, inst, rise, decay, thr, cap)`
10. `self.pain_scalar[:] = self.pain_state.mean(dim=-1)`

### 6.4 `_compute_reward(self, actions)` — override

```python
super()._compute_reward(actions)           # imitation + power reward (unchanged)
if not self.pain_enabled:
    return
self._update_pain_buffers()

if self.pain_mode in ("reward_only", "guard_and_reward"):
    self.rew_buf[:] = self.rew_buf[:] - self._p_lambda * self.pain_scalar

self.extras["pain_mean"]          = float(self.pain_scalar.mean().item())
self.extras["pain_max"]           = float(self.pain_scalar.max().item())
self.extras["pain_internal_mean"] = float(self.pain_internal.mean().item())
self.extras["pain_external_mean"] = float(self.pain_external.mean().item())

if flags.im_eval:
    if self._p_log_joint:
        self.extras["pain_state"] = self.pain_state.detach().cpu().numpy()
    if self._p_log_contact:
        self.extras["pain_contact_body"] = self.pain_contact_body.detach().cpu().numpy()
```

The large per-DOF / per-body arrays are gated behind `flags.im_eval` — the same
gate `HumanoidIm.post_physics_step` uses at `humanoid_im.py:674` (Codex review
round 2). Training runs (Spec #3) must not flood wandb with 69-dim and 24-dim
arrays every step.

### 6.5 `_action_to_pd_targets(self, action)` — override

```python
pd_tar = super()._action_to_pd_targets(action)
if self.pain_enabled and self.pain_mode in ("guard_only", "guard_and_reward"):
    pd_tar = apply_pain_action_guard(
        pd_tar, self._dof_pos, self.pain_state,
        self._p_guard_gain, self._p_max_guard,
    )
return pd_tar
```

Super runs first so res_action's `[q - π/2, q + π/2]` clamp is applied before
the guard. The guard then contracts within that clamp. Codex review round 1
flagged that high pain can reduce tracking authority more than expected — this
is the intended behavior of a guard, not a bug.

### 6.6 `_reset_env_tensors(self, env_ids)` — override

```python
super()._reset_env_tensors(env_ids)
if self.pain_enabled:
    for buf in (self.pain_inst, self.pain_state, self.pain_internal,
                self.pain_external, self.pain_contact_body):
        buf[env_ids] = 0
    self.pain_scalar[env_ids] = 0
```

Prevents pain state from carrying over across episode boundaries. Codex review
round 2 called this out as a correctness issue, not just polish — without it,
a respawned env's first frame can fire reward penalties or guards based on the
previous episode's final pain.

### 6.7 Mode dispatch matrix

| `pain_mode`        | `_update_pain_buffers` | reward penalty | action guard |
|--------------------|:---------------------:|:--------------:|:------------:|
| `off`              | skip                  | no             | no           |
| `log_only`         | run                   | no             | no           |
| `reward_only`      | run                   | yes            | no           |
| `guard_only`       | run                   | no             | yes          |
| `guard_and_reward` | run                   | yes            | yes          |

---

## 7. `phc/data/cfg/env/env_im_pain.yaml`

Copy `phc/data/cfg/env/env_im_pnn.yaml` verbatim, then:

- `task: HumanoidIm` → `task: HumanoidImPain`
- `num_prim: 3` → `num_prim: 4` (matches pretrained checkpoint; rationale §2)
- `notes: " "` → `notes: "pain baseline v0 (guide §6-18)"`
- Append a `pain:` block above `plane:`:

```yaml
pain:
  enabled: True
  mode: "guard_only"          # off | log_only | reward_only | guard_only | guard_and_reward
  append_to_obs: False        # v0: contract flag only; code does NOT read this
  use_external_contact: True
  lambda_p: 0.05

  w_limit: 1.0
  w_torque: 0.35
  w_power: 0.15
  w_contact: 0.50

  joint_limit_margin_ratio: 0.15
  torque_ref_scale: 0.50
  power_ref: 5.0
  contact_force_ref: 150.0
  pain_threshold: 0.10
  pain_cap: 3.0

  rise_alpha: 0.25
  decay_alpha: 0.02

  guard_gain: 0.60
  max_guard: 0.75

  log_joint_pain: True
  log_contact_pain: True
```

Default `mode: guard_only` because §20 conditions 3 and 4 are both zero-training
viewer demos — the config is tuned for demo ergonomics, not for training.
Every mode is selectable via Hydra CLI (`env.pain.mode=...`).

---

## 8. `phc/utils/parse_task.py` edit

One-line insertion after the existing `HumanoidImMCPGetup` import:

```diff
 from phc.env.tasks.humanoid_im_mcp import HumanoidImMCP
 from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup
+from phc.env.tasks.humanoid_im_pain import HumanoidImPain
 from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
```

`parse_task`'s `eval(args.task)` (line 60) then resolves `HumanoidImPain` in the
module's local namespace. No other change.

---

## 9. Verification procedure

Automated tests are deliberately omitted for v0 (Isaac Gym + GPU + viewer make
CI-style tests heavy). Verification is manual, recorded in
`docs/superpowers/specs/phc_v0_spec2_impl_log.md` following the Spec #1 log
style.

### V1. Import check (no Isaac Gym needed)

```bash
conda run -n phc python -c "from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard); print('ok')"
```

Acceptance: prints `ok`. Verifies the pure-torch contract — if this step pulls
in Isaac Gym or numpy, the helper module has been contaminated.

### V2. `mode=off` — baseline regression absence (§20 condition 2)

```bash
python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=False env.pain.mode=off
```

Acceptance: viewer shows humanoid balancing; final stdout line matches Spec #1
sanity reward within ~1% (`reward: ~941 steps: 999`). `env.num_prim=4` is
redundant with `env_im_pain.yaml`'s default but kept on the CLI for explicit
parity with Spec #1.

### V3. `mode=log_only` — pain becomes nonzero (§20 condition 3)

```bash
python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=log_only
```

Acceptance: reward within ~1% of V2 (no penalty applied); viewer `j` key
delivers large push; subsequent frames show `pain_mean`/`pain_max` scalar
extras non-zero. Scalar extras are always populated (not gated by
`flags.im_eval`) but are not printed to stdout in the viewer test path — so
confirmation requires a **temporary `print(self.pain_scalar.max().item())`
inside `_compute_reward` during bring-up**. Remove the print before
committing implementation commit (commit 2 in §10).

### V4. `mode=guard_only` — visible motion change (§20 condition 4)

```bash
python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=guard_only
```

Acceptance: under the same `j` input as V3, motion visibly differs from V3 in
at least one of: reduced ROM, slower return, or audible/visible stiffening on
painful DOFs. A side-by-side recording of V3 vs V4 (same seed, same motion
file) is sufficient.

### V5. §20 condition 5 (fine-tune)

Deferred to Spec #3.

### Logging

Each of V1/V2/V3/V4 is captured in `docs/superpowers/specs/phc_v0_spec2_impl_log.md`:

- Command (copy-paste).
- Exit code.
- Last ~30 lines of stdout (`/tmp/phc_spec2_v?.log` or similar).
- One-line viewer observation.
- Any deviation from the command above.

---

## 10. Commit strategy

Three commits on `jinsu-pain_baseline_phc_v0`, matching the Spec #1 pattern:

1. **Design + log skeleton.** This document at
   `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md`
   plus a short skeleton at
   `docs/superpowers/specs/phc_v0_spec2_impl_log.md` with section headers
   V1–V5 and no body yet.
2. **Implementation.** Four files: `pain_baseline.py`, `humanoid_im_pain.py`,
   `env_im_pain.yaml`, `parse_task.py`. Gate on V1 import check passing before
   committing. `git diff phc/` reviewed to confirm no unintended edits.
3. **Populated log.** Fill `phc_v0_spec2_impl_log.md` with V2/V3/V4 results
   after viewer runs. Commit when all three acceptance bullets are green.

---

## 11. Out of scope

Explicit non-deliverables for Spec #2:

- Fine-tuning from the copied checkpoint (Spec #3).
- Appending pain to the observation vector (v1 branch).
- IsaacLab eval path.
- MCP / getup composer integration.
- Retuning the default pain constants beyond what sits in the yaml.
- Automated unit tests for `pain_baseline.py` (Q4 decision: manual verification
  only; helpers are REPL-accessible).
- `h1` / `g1` / SMPL-X humanoid support — SMPL only for v0. `__init__` asserts
  the SMPL layout and fails loudly otherwise.

---

## 12. Risks and mitigations

1. **pain_state carry-over across episodes.** Resolved by the
   `_reset_env_tensors` override (§6.6). Codex review round 2 flagged this.
2. **Guide's guard formula in wrong space.** Resolved by overriding
   `_action_to_pd_targets` instead of `pre_physics_step` (§4.2). Codex review
   round 1.
3. **Hot-path dict lookup.** Mitigated by caching config values into
   per-attribute floats in `__init__` (§6.2).
4. **wandb bloat during Spec #3 fine-tune.** Per-DOF arrays gated behind
   `flags.im_eval` (§6.4). Codex review round 2.
5. **Fixed-joint noise amplification in B1.** Masked by `q_half > eps` check
   (§5.1 B1). Codex review round 2 on the helper module.
6. **Default constants chosen for visible zero-training demo, not for
   fine-tuning.** Acknowledged in §5.2; Spec #3 may retune `guard_gain`,
   `max_guard`, `lambda_p`, `rise_alpha`, `decay_alpha`, `threshold`.
7. **`append_to_obs` flag is documented but unused.** Enforced by code review
   during commit 2; grep `append_to_obs` should return only the yaml line.

---

## 13. Debug hints

- **"Unexpected key(s) ...pnn.actors.3..."** → `num_prim < 4` somewhere. Check
  `env_im_pain.yaml`, then CLI overrides.
- **`pain_mean` always 0 in `log_only`** → verify `flags.im_eval` is True in
  the test path; or drop a temporary print in `_update_pain_buffers`. Also
  verify `self.dof_force_tensor` is populated — it should be, via
  `_refresh_sim_tensors` in `post_physics_step`.
- **Guard freezes motion entirely** → reduce `guard_gain` to 0.3 and
  `max_guard` to 0.4. Check that `pain_state` isn't saturated at 3.0 from a
  single extreme push.
- **Reward collapses instantly under `reward_only`** → lower `lambda_p` to
  0.01; or verify `pain_scalar` isn't capped at 3.0.
- **`AttributeError: pain_enabled`** during Isaac Gym load → super's
  `__init__` is calling into an overridden method before attributes exist.
  Codex round 2 did not find such a call, but if it appears after upstream
  churn, guard overrides with `if getattr(self, "pain_enabled", False): ...`.

---

## 14. Exit gate to Spec #3

Spec #2 is done when:

- All four deliverable files exist on `jinsu-pain_baseline_phc_v0`.
- `git diff phc/` between Spec #1's final commit and Spec #2's commit 2 touches
  exactly: 1 new file under `phc/env/util/`, 1 new file under
  `phc/env/tasks/`, 1 new file under `phc/data/cfg/env/`, and 1 line
  inserted in `phc/utils/parse_task.py`. Nothing else.
- V1 import check prints `ok` in a fresh shell.
- V2, V3, V4 acceptance bullets are green and recorded in the impl log.
- User has read the impl log and given the go-signal for Spec #3.

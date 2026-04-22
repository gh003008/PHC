# PHC-Pain-v0 Spec #2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an environment-side pain estimator, reward penalty, and action governor to PHC without changing observation dim (945), network builders, or checkpoint format — so the pretrained `phc_shape_pnn_iccv` checkpoint runs unmodified through a new `HumanoidImPain` task.

**Architecture:** Subclass `HumanoidIm` → `HumanoidImPain`, override four methods (`__init__`, `_compute_reward`, `_action_to_pd_targets`, `_reset_env_tensors`). Place all pain math in a pure-torch helper module (`phc/env/util/pain_baseline.py`) with no Isaac Gym deps. New config `env_im_pain.yaml` holds a `pain:` block. Register via one-line import in `phc/utils/parse_task.py`.

**Tech Stack:** Python 3.8, PyTorch 1.13.1 (CUDA 11.6), Isaac Gym preview4, Hydra configs, SMPL humanoid. `phc` conda env already set up per Spec #1.

---

## Spec reference

Design document: `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md`
Implementation log: `docs/superpowers/specs/phc_v0_spec2_impl_log.md` (populated in Phase C)

## Assumed working state (before plan starts)

- Branch: `jinsu-pain_baseline_phc_v0`
- HEAD: commit `f0ad076` ("Spec #2: add pain-aware implementation design + log skeleton") — this is Phase A already committed.
- `phc` conda env functional (per Spec #1 setup log).
- All 4 SMPL pickles in `data/smpl/`, sample data in `sample_data/`, pretrained `output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth` present.
- `git status` clean aside from untracked `reference/`.

## File structure

### Files to create

| Path | Responsibility | Target size |
|---|---|---|
| `phc/env/util/pain_baseline.py` | 8 pure-torch functions for pain math. No Isaac Gym, no numpy. | ~130 lines |
| `phc/env/tasks/humanoid_im_pain.py` | `HumanoidImPain(HumanoidIm)` subclass: buffer allocation, pain update, reward penalty, action guard, env reset. | ~180 lines |
| `phc/data/cfg/env/env_im_pain.yaml` | New env config — copy of `env_im_pnn.yaml` with 3 field edits + `pain:` block. | ~90 lines |

### Files to modify

| Path | Change |
|---|---|
| `phc/utils/parse_task.py` | Insert one import line so `eval(args.task)` resolves `HumanoidImPain`. |

### Files NOT to touch

Per spec §2 / §11 hard constraints:
- `phc/env/tasks/humanoid_im.py`
- `phc/env/tasks/humanoid.py`
- `phc/env/tasks/base_task.py`
- `phc/learning/amp_network_pnn_builder.py` (or any network file)
- `phc/data/cfg/env/env_im_pnn.yaml` (unchanged; v0 is additive)
- `phc/data/cfg/config.yaml`

If during implementation you find yourself editing any of these, stop and reread spec §2.

---

## Phase A — design doc (already done)

Commit `f0ad076` on `jinsu-pain_baseline_phc_v0` adds:
- `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md`
- `docs/superpowers/specs/phc_v0_spec2_impl_log.md` (skeleton)

Nothing to do in this phase.

---

## Phase B — implementation (commit 2)

Total 6 tasks. Sequential: B1 → B2 → B3 → B4 → B5 → B6.

### Task B1: Create `phc/env/util/pain_baseline.py`

**Files:**
- Create: `phc/env/util/pain_baseline.py`

- [ ] **Step 1: Write the full helper module.**

Content (exact — copy verbatim):

```python
"""Pure-torch helpers for PHC-Pain-v0. No Isaac Gym, no numpy.

Shape legend: B = num_envs, D = num_dof (SMPL=69), Nb = num_bodies (SMPL=24).

Saturation notes (Spec #2 design §5.2):
- update_pain_state unclamped steady state for constant `inst` is
  12.5 * relu(inst - threshold) under rise=0.25, decay=0.02 defaults;
  sustained inst >= 0.34 drives pain_state to cap=3.0.
- apply_pain_action_guard saturates at pain_state >= 1.25 under
  guard_gain=0.6, max_guard=0.75 defaults; pain range [1.25, 3.0]
  collapses under the guard. Retune in Spec #3 if needed.
"""

import torch


def compute_joint_limit_pain(q, q_lo, q_hi, margin_ratio=0.15, eps=1e-6):
    """Pain rises when |q - q_mid| exceeds a safe band inside the joint ROM.

    Args:
        q:        (B, D) current DOF positions.
        q_lo:     (D,) lower limits.
        q_hi:     (D,) upper limits.
    Returns:
        (B, D) in [0, ~1/margin_ratio]. 0 inside the safe zone.
        Fixed joints (q_half <= eps) are masked to zero.
    """
    q_mid = 0.5 * (q_lo + q_hi)
    q_half = 0.5 * (q_hi - q_lo)
    q_safe = q_half * (1.0 - margin_ratio)
    excess = torch.relu(torch.abs(q - q_mid) - q_safe)
    pain = excess / (q_half - q_safe + eps)
    fixed_mask = (q_half > eps).to(pain.dtype)
    return pain * fixed_mask


def compute_torque_pain(tau, tau_limits, ref_scale=0.5, eps=1e-6):
    """|tau| normalized by a safe fraction of tau_limits, with a 1 Nm floor."""
    tau_ref = torch.clamp(ref_scale * tau_limits, min=1.0)
    return torch.abs(tau) / (tau_ref + eps)


def compute_power_pain(tau, dq, power_ref=5.0, eps=1e-6):
    """|tau * dq| normalized by a scalar power reference."""
    return torch.abs(tau * dq) / max(power_ref, eps)


def compute_contact_pain(contact_forces, ref_force=150.0, eps=1e-6):
    """Body-level contact force magnitude normalized by a scalar reference.

    Args:
        contact_forces: (B, Nb, 3).
    Returns:
        (B, Nb).
    """
    return torch.norm(contact_forces, dim=-1) / max(ref_force, eps)


def broadcast_body_pain_to_dof(body_pain, num_dof):
    """Map per-body scalars to per-DOF values assuming SMPL layout.

    SMPL: _dof_names == _body_names[1:], 3 consecutive DOFs per non-root body.
    Root body (index 0) has no DOFs and is dropped.

    Args:
        body_pain: (B, Nb).
        num_dof:   expected to equal 3 * (Nb - 1).
    Returns:
        (B, num_dof).
    """
    B, Nb = body_pain.shape
    assert num_dof == 3 * (Nb - 1), (
        f"broadcast_body_pain_to_dof expects SMPL layout num_dof==3*(Nb-1); "
        f"got num_dof={num_dof}, Nb={Nb}"
    )
    non_root = body_pain[:, 1:]
    return non_root.unsqueeze(-1).expand(B, Nb - 1, 3).reshape(B, -1)


def combine_internal_pain(
    limit_pain, torque_pain, power_pain,
    w_limit=1.0, w_torque=0.35, w_power=0.15,
):
    """Weighted sum of the three internal pain channels."""
    return w_limit * limit_pain + w_torque * torque_pain + w_power * power_pain


def update_pain_state(
    prev, inst,
    rise_alpha=0.25, decay_alpha=0.02, threshold=0.1, cap=3.0,
):
    """Leaky accumulator with threshold-gated drive and hard cap."""
    drive = torch.relu(inst - threshold)
    nxt = prev * (1.0 - decay_alpha) + rise_alpha * drive
    return torch.clamp(nxt, 0.0, cap)


def apply_pain_action_guard(
    pd_tar, dof_pos, pain_state,
    guard_gain=0.6, max_guard=0.75,
):
    """Contract PD target toward current dof_pos in proportion to pain.

    pd_tar and dof_pos must be in the same (radian DOF) space.
    """
    guard = torch.clamp(guard_gain * pain_state, 0.0, max_guard)
    return dof_pos + (1.0 - guard) * (pd_tar - dof_pos)
```

- [ ] **Step 2: Import-check the module (no Isaac Gym).**

Run:
```bash
conda run -n phc python -c "from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard); print('ok')"
```

Expected stdout: `ok` (single line, exit 0).

If it fails with `ModuleNotFoundError: No module named 'isaacgym'`, the module has been contaminated — investigate which import pulled Isaac Gym in.

- [ ] **Step 3: Quick CPU sanity via REPL one-liner.**

Run:
```bash
conda run -n phc python -c "
import torch
from phc.env.util.pain_baseline import compute_joint_limit_pain, update_pain_state, apply_pain_action_guard
q = torch.tensor([[0.0, 0.95]]); lo = torch.tensor([-1.0, -1.0]); hi = torch.tensor([1.0, 1.0])
p = compute_joint_limit_pain(q, lo, hi, margin_ratio=0.15)
print('limit_pain', p.tolist())
state = torch.zeros(1, 2)
for _ in range(100):
    state = update_pain_state(state, torch.tensor([[0.0, 0.5]]))
print('steady_state', state.tolist())
pd = torch.tensor([[0.5, 0.5]]); dp = torch.tensor([[0.0, 0.0]])
print('guard_out', apply_pain_action_guard(pd, dp, torch.tensor([[0.0, 2.0]])).tolist())
"
```

Expected (approximate): `limit_pain [[0.0, ~4.33]]`, `steady_state` with element 0 = 0.0 and element 1 near cap=3.0, `guard_out` first element 0.5 (no guard), second element near 0.125 (guard saturated at 0.75).

Interpretation:
- `q=0.0` is dead-center → safe zone → pain 0.
- `q=0.95` is past `q_safe=0.85` by 0.10; denominator `q_half - q_safe = 0.15`; `eps` small → pain ≈ 0.667 / 0.15 = 4.44 (exact value: `(0.95 - 0.85) / 0.15 ≈ 4.33` with eps=1e-6).
- Sustained `inst=0.5` drives state to `(0.25 * (0.5 - 0.1)) / 0.02 = 5.0` unclamped → capped at 3.0.
- `guard = clamp(0.6 * 2.0, 0, 0.75) = 0.75` → `0.0 + (1 - 0.75) * (0.5 - 0.0) = 0.125`.

- [ ] **Step 4: No commit yet.** This file is part of commit 2 alongside B2/B3/B4.

### Task B2: Create `phc/env/tasks/humanoid_im_pain.py`

**Files:**
- Create: `phc/env/tasks/humanoid_im_pain.py`

- [ ] **Step 1: Write the task subclass.**

Content (exact — copy verbatim):

```python
"""HumanoidImPain — PHC-Pain-v0 task subclass.

Spec: docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md.
Extends HumanoidIm with an environment-side pain estimator, an optional
action guard (pd_tar pullback toward current dof_pos), and an optional
reward penalty proportional to mean per-env pain_state.

Obs dim, network, action space, and checkpoint format are unchanged by design.
"""

import torch

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.util.pain_baseline import (
    apply_pain_action_guard,
    broadcast_body_pain_to_dof,
    combine_internal_pain,
    compute_contact_pain,
    compute_joint_limit_pain,
    compute_power_pain,
    compute_torque_pain,
    update_pain_state,
)
from phc.utils.flags import flags

_VALID_MODES = {"off", "log_only", "reward_only", "guard_only", "guard_and_reward"}


class HumanoidImPain(HumanoidIm):
    """HumanoidIm with an environment-side pain proxy."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        pc = cfg["env"].get("pain", {})
        self.pain_cfg = pc
        self.pain_enabled = bool(pc.get("enabled", False))
        self.pain_mode = str(pc.get("mode", "off"))
        assert self.pain_mode in _VALID_MODES, (
            f"env.pain.mode must be one of {_VALID_MODES}; got {self.pain_mode!r}"
        )

        # Cache config into attributes to avoid dict lookup on the hot path.
        self._p_lambda = float(pc.get("lambda_p", 0.05))
        self._p_w_limit = float(pc.get("w_limit", 1.0))
        self._p_w_torque = float(pc.get("w_torque", 0.35))
        self._p_w_power = float(pc.get("w_power", 0.15))
        self._p_w_contact = float(pc.get("w_contact", 0.50))
        self._p_margin = float(pc.get("joint_limit_margin_ratio", 0.15))
        self._p_tau_scale = float(pc.get("torque_ref_scale", 0.50))
        self._p_power_ref = float(pc.get("power_ref", 5.0))
        self._p_contact_ref = float(pc.get("contact_force_ref", 150.0))
        self._p_thr = float(pc.get("pain_threshold", 0.10))
        self._p_cap = float(pc.get("pain_cap", 3.0))
        self._p_rise = float(pc.get("rise_alpha", 0.25))
        self._p_decay = float(pc.get("decay_alpha", 0.02))
        self._p_guard_gain = float(pc.get("guard_gain", 0.60))
        self._p_max_guard = float(pc.get("max_guard", 0.75))
        self._p_use_contact = bool(pc.get("use_external_contact", True))
        self._p_log_joint = bool(pc.get("log_joint_pain", True))
        self._p_log_contact = bool(pc.get("log_contact_pain", True))

        # SMPL layout assertion — v0 is SMPL-only.
        assert self.num_dof == 3 * (self.num_bodies - 1), (
            f"HumanoidImPain v0 supports SMPL layout only "
            f"(num_dof == 3*(num_bodies-1)); got num_dof={self.num_dof}, "
            f"num_bodies={self.num_bodies}."
        )

        # Per-env buffers. Allocate unconditionally so the hot path is branch-free.
        z = lambda: torch.zeros((self.num_envs, self.num_dof), device=self.device)
        zb = lambda: torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self.pain_inst = z()
        self.pain_state = z()
        self.pain_internal = z()
        self.pain_external = z()
        self.pain_contact_body = zb()
        self.pain_scalar = torch.zeros((self.num_envs,), device=self.device)

    def _update_pain_buffers(self):
        if not self.pain_enabled or self.pain_mode == "off":
            return

        q = self._dof_pos
        dq = self._dof_vel
        tau = self.dof_force_tensor

        limit_pain = compute_joint_limit_pain(
            q, self.dof_limits_lower, self.dof_limits_upper, self._p_margin
        )
        torque_pain = compute_torque_pain(tau, self.torque_limits, self._p_tau_scale)
        power_pain = compute_power_pain(tau, dq, self._p_power_ref)

        internal = combine_internal_pain(
            limit_pain, torque_pain, power_pain,
            self._p_w_limit, self._p_w_torque, self._p_w_power,
        )

        if self._p_use_contact:
            body = compute_contact_pain(
                self._contact_forces[:, : self.num_bodies, :], self._p_contact_ref
            )
            external_dof = self._p_w_contact * broadcast_body_pain_to_dof(body, self.num_dof)
            self.pain_contact_body[:] = body
        else:
            external_dof = torch.zeros_like(internal)
            self.pain_contact_body.zero_()

        inst = internal + external_dof
        self.pain_internal[:] = internal
        self.pain_external[:] = external_dof
        self.pain_inst[:] = inst
        self.pain_state[:] = update_pain_state(
            self.pain_state, inst,
            self._p_rise, self._p_decay, self._p_thr, self._p_cap,
        )
        self.pain_scalar[:] = self.pain_state.mean(dim=-1)

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        if not self.pain_enabled:
            return
        self._update_pain_buffers()

        if self.pain_mode in ("reward_only", "guard_and_reward"):
            self.rew_buf[:] = self.rew_buf[:] - self._p_lambda * self.pain_scalar

        self.extras["pain_mean"] = float(self.pain_scalar.mean().item())
        self.extras["pain_max"] = float(self.pain_scalar.max().item())
        self.extras["pain_internal_mean"] = float(self.pain_internal.mean().item())
        self.extras["pain_external_mean"] = float(self.pain_external.mean().item())

        if flags.im_eval:
            if self._p_log_joint:
                self.extras["pain_state"] = self.pain_state.detach().cpu().numpy()
            if self._p_log_contact:
                self.extras["pain_contact_body"] = self.pain_contact_body.detach().cpu().numpy()

    def _action_to_pd_targets(self, action):
        pd_tar = super()._action_to_pd_targets(action)
        if self.pain_enabled and self.pain_mode in ("guard_only", "guard_and_reward"):
            pd_tar = apply_pain_action_guard(
                pd_tar, self._dof_pos, self.pain_state,
                self._p_guard_gain, self._p_max_guard,
            )
        return pd_tar

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self.pain_enabled:
            for buf in (
                self.pain_inst, self.pain_state,
                self.pain_internal, self.pain_external,
                self.pain_contact_body,
            ):
                buf[env_ids] = 0
            self.pain_scalar[env_ids] = 0
```

- [ ] **Step 2: Verify the file parses (syntax-only check, no isaacgym needed).**

Run:
```bash
conda run -n phc python -c "import ast, pathlib; ast.parse(pathlib.Path('phc/env/tasks/humanoid_im_pain.py').read_text()); print('ok')"
```

Expected stdout: `ok`. Syntax errors block here before we need to import Isaac Gym.

- [ ] **Step 3: No commit yet.**

### Task B3: Create `phc/data/cfg/env/env_im_pain.yaml`

**Files:**
- Create: `phc/data/cfg/env/env_im_pain.yaml`
- Read: `phc/data/cfg/env/env_im_pnn.yaml` (source for the base fields)

- [ ] **Step 1: Read the base config to confirm the current field set.**

Run:
```bash
cat phc/data/cfg/env/env_im_pnn.yaml
```

Expected: 65 lines. Confirm `task: HumanoidIm`, `num_prim: 3`, and `notes: " "` are present — those are the three lines we will change.

- [ ] **Step 2: Write the new config.**

Content (exact — copy verbatim):

```yaml
# PHC-Pain-v0 env config. Derived from env_im_pnn.yaml with:
#   task:   HumanoidIm   -> HumanoidImPain
#   num_prim: 3          -> 4   (matches pretrained phc_shape_pnn_iccv)
#   notes:  " "          -> "pain baseline v0 (guide §6-18)"
# plus a new pain: block.
task: HumanoidImPain
project_name: "PHC"
notes: "pain baseline v0 (guide §6-18)"
motion_file: ""
num_envs: 3072
env_spacing: 5
episode_length: 300
is_flag_run: False
enable_debug_vis: False

fut_tracks: False
self_obs_v: 1
obs_v: 6
auto_pmcp: False
auto_pmcp_soft: True


has_pnn: True
fitting: False
num_prim: 4
training_prim: 0
actors_to_load: 0
has_lateral: False
models: []

######## Getup Configs ########
zero_out_far: False
zero_out_far_train: False
getup_udpate_epoch: 78750
cycle_motion: False
hard_negative: False
min_length: 5

kp_scale: 1
power_reward: True

shape_resampling_interval: 500

control_mode: "isaac_pd"
power_scale: 1.0
controlFrequencyInv: 2 # 30 Hz
stateInit: "Random"
hybridInitProb: 0.5
numAMPObsSteps: 10

local_root_obs: True
root_height_obs: True
key_bodies: ["R_Ankle", "L_Ankle", "R_Wrist",  "L_Wrist"]
contact_bodies: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
reset_bodies: ['Pelvis', 'L_Hip', 'L_Knee', 'R_Hip', 'R_Knee', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
terminationHeight: 0.15
enableEarlyTermination: True
terminationDistance: 0.25

### Fut config
numTrajSamples: 3
trajSampleTimestepInv: 3
enableTaskObs: True


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


plane:
  staticFriction: 1.0
  dynamicFriction: 1.0
  restitution: 0.0
```

- [ ] **Step 3: Verify the yaml parses.**

Run:
```bash
conda run -n phc python -c "import yaml, pathlib; cfg = yaml.safe_load(pathlib.Path('phc/data/cfg/env/env_im_pain.yaml').read_text()); assert cfg['task'] == 'HumanoidImPain'; assert cfg['num_prim'] == 4; assert cfg['pain']['mode'] == 'guard_only'; assert cfg['pain']['append_to_obs'] is False; print('ok')"
```

Expected stdout: `ok`. Asserts guard the three decisions that must survive any future edit: task name, num_prim, default mode.

- [ ] **Step 4: Confirm `append_to_obs` is not read anywhere in code.**

Run:
```bash
grep -rn "append_to_obs" phc/ docs/superpowers/
```

Expected: matches only in `docs/superpowers/specs/...` and in `phc/data/cfg/env/env_im_pain.yaml`. NO matches under `phc/env/` or `phc/learning/`. If code reads it, spec §2 constraint is violated — stop and reconcile.

- [ ] **Step 5: No commit yet.**

### Task B4: Edit `phc/utils/parse_task.py`

**Files:**
- Modify: `phc/utils/parse_task.py:35`

- [ ] **Step 1: Confirm current state of the import block.**

Run:
```bash
sed -n '29,40p' phc/utils/parse_task.py
```

Expected (lines 29-40):
```
from phc.env.tasks.humanoid import Humanoid
from phc.env.tasks.humanoid_amp import HumanoidAMP
from phc.env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.tasks.humanoid_im_getup import HumanoidImGetup
from phc.env.tasks.humanoid_im_mcp import HumanoidImMCP
from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup
from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from phc.env.tasks.humanoid_im_demo import HumanoidImDemo
from phc.env.tasks.humanoid_im_mcp_demo import HumanoidImMCPDemo
```

- [ ] **Step 2: Insert the new import after the `HumanoidImMCPGetup` line.**

Use the Edit tool on `phc/utils/parse_task.py` with:
- old_string:
  ```
  from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup
  from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
  ```
- new_string:
  ```
  from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup
  from phc.env.tasks.humanoid_im_pain import HumanoidImPain
  from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
  ```

- [ ] **Step 3: Verify the insertion and parse the file.**

Run:
```bash
grep -n "HumanoidImPain" phc/utils/parse_task.py && conda run -n phc python -c "import ast, pathlib; ast.parse(pathlib.Path('phc/utils/parse_task.py').read_text()); print('ok')"
```

Expected:
- grep: one line match (the import).
- python: `ok`.

- [ ] **Step 4: Confirm the diff is one line only.**

Run:
```bash
git diff phc/utils/parse_task.py
```

Expected: exactly one `+` line (the new import). Any other change is out of scope — revert with `git checkout -- phc/utils/parse_task.py` and redo Step 2.

- [ ] **Step 5: No commit yet.**

### Task B5: V1 import check (end-to-end contract)

**Files:** none (verification only)

- [ ] **Step 1: Run the V1 pure-torch import check from spec §9.**

Run:
```bash
conda run -n phc python -c "from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard); print('ok')"
```

Expected stdout: `ok`, exit 0.

If this now fails after Task B4 (it shouldn't — B4 doesn't touch `pain_baseline.py`), investigate: was the helper module corrupted? Re-read the file.

- [ ] **Step 2: Run the full import chain including the task class.**

Run:
```bash
conda run -n phc python -c "from phc.env.tasks.humanoid_im_pain import HumanoidImPain; print(HumanoidImPain.__name__)"
```

Expected stdout: `HumanoidImPain`. This touches Isaac Gym (via `HumanoidIm`) so it requires `LD_LIBRARY_PATH` set per Spec #1 setup log note 2 — the conda env already has it via `conda env config vars`.

If it fails with `ImportError: libpython3.8.so.1.0`, LD_LIBRARY_PATH is not set on the env. Run:
```bash
conda env config vars set LD_LIBRARY_PATH=/home/jinsu/miniconda3/envs/phc/lib -n phc
```
Then reactivate the env and retry.

- [ ] **Step 3: Confirm `parse_task` resolves the new task name.**

Run:
```bash
conda run -n phc python -c "
from phc.utils import parse_task as pt
print('HumanoidImPain' in dir(pt))
print(pt.HumanoidImPain.__name__)
"
```

Expected stdout: `True` then `HumanoidImPain`.

### Task B6: Commit implementation (commit 2)

**Files:** all four deliverables from B1-B4.

- [ ] **Step 1: Review the full diff one more time.**

Run:
```bash
git status --short && git diff --stat
```

Expected: 3 new files under `phc/env/util/`, `phc/env/tasks/`, `phc/data/cfg/env/`, and 1 modified file `phc/utils/parse_task.py` (+1 line).

- [ ] **Step 2: Confirm no unexpected files touched.**

Run:
```bash
git diff --name-only && git ls-files --others --exclude-standard
```

Expected tracked changes: exactly
```
phc/utils/parse_task.py
```
Expected untracked additions: exactly
```
phc/env/util/pain_baseline.py
phc/env/tasks/humanoid_im_pain.py
phc/data/cfg/env/env_im_pain.yaml
```
(plus `reference/` already-untracked from Spec #1; ignore.)

If any other file appears, stop and reconcile before committing.

- [ ] **Step 3: Stage exactly the four files.**

Run:
```bash
git add \
  phc/env/util/pain_baseline.py \
  phc/env/tasks/humanoid_im_pain.py \
  phc/data/cfg/env/env_im_pain.yaml \
  phc/utils/parse_task.py
git status --short
```

Expected: four `A` / `M` entries, nothing else staged.

- [ ] **Step 4: Commit.**

Run:
```bash
git commit -m "$(cat <<'EOF'
Spec #2: add PHC-Pain-v0 implementation (4 files)

- phc/env/util/pain_baseline.py (new, pure torch):
  8 helpers — joint-limit / torque / power / contact pain, body→DOF
  broadcast, internal combiner, leaky accumulator, action-guard
  formula in radian DOF space.
- phc/env/tasks/humanoid_im_pain.py (new, HumanoidIm subclass):
  __init__ (config caching + buffer alloc + SMPL assertion),
  _update_pain_buffers, _compute_reward override (imitation reward +
  optional pain penalty + scalar/array extras gated on flags.im_eval),
  _action_to_pd_targets override (pd_tar contracted toward dof_pos
  when mode ∈ {guard_only, guard_and_reward}),
  _reset_env_tensors override (per-env pain buffer zeroing).
- phc/data/cfg/env/env_im_pain.yaml (new):
  Derived from env_im_pnn.yaml with task=HumanoidImPain, num_prim=4
  (matches pretrained phc_shape_pnn_iccv), and a pain: block.
  Default mode is guard_only for zero-training demo ergonomics.
- phc/utils/parse_task.py (1 line):
  import HumanoidImPain so eval(args.task) resolves it.

Design: docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md
Plan:   docs/superpowers/plans/2026-04-22-phc-pain-v0-spec2-impl.md

Obs dim unchanged (945). No network/action-space/PD changes. Pretrained
checkpoint loads unmodified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify the commit landed.**

Run:
```bash
git log --oneline -3 && git show --stat HEAD | head -20
```

Expected: newest line says "Spec #2: add PHC-Pain-v0 implementation (4 files)", 4 files changed, roughly +350–400 insertions total.

---

## Phase C — verification & log (commit 3)

Total 5 tasks. Run sequentially: C1 → C2 → C3 → C4 → C5.

> **Note on print-debug:** V3 confirmation requires observing `pain_scalar` values, which are not surfaced to stdout by default. Tasks C2 and C3 temporarily add a `print(...)` inside `_update_pain_buffers`, run the verification, and **revert** the print before C5's commit. The revert is enforced by `git checkout --` in Task C4.

### Task C1: V2 — `mode=off` baseline regression absence

**Files:** none (verification only)

- [ ] **Step 1: Run V2 command.**

Run (from repo root):
```bash
mkdir -p /tmp
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=False env.pain.mode=off \
  2>&1 | tee /tmp/phc_spec2_v2.log
```

Expected behavior:
- Viewer opens with humanoid standing.
- Final line approximately: `reward: 9XX.XX steps: 999.0` (within ~1% of Spec #1's `941.05`).
- No `Traceback`, no `Unexpected key(s)` in the log.
- Exit 0 after the user closes the viewer.

Acceptance: reward within ~1% of Spec #1 log (941.05). If significantly off, the pain code is side-effecting even when disabled — inspect `_compute_reward` and `_action_to_pd_targets` paths for unconditional writes.

- [ ] **Step 2: Capture the last 30 lines.**

Run:
```bash
tail -30 /tmp/phc_spec2_v2.log > /tmp/phc_spec2_v2_tail.txt && wc -l /tmp/phc_spec2_v2_tail.txt
```

Expected: `30 /tmp/phc_spec2_v2_tail.txt`. Keep this snippet handy for C5.

### Task C2: V3 — `mode=log_only` pain nonzero under push

**Files:**
- Temporarily modify: `phc/env/tasks/humanoid_im_pain.py` (print-debug, reverted in C4).

- [ ] **Step 1: Add a temporary print inside `_update_pain_buffers`.**

Use the Edit tool on `phc/env/tasks/humanoid_im_pain.py` with:
- old_string:
  ```
          self.pain_scalar[:] = self.pain_state.mean(dim=-1)
  ```
- new_string:
  ```
          self.pain_scalar[:] = self.pain_state.mean(dim=-1)
          if self.pain_scalar.max().item() > 0.0:
              print(f"[pain-debug] max={self.pain_scalar.max().item():.3f} mean={self.pain_scalar.mean().item():.3f}")
  ```

Verify with:
```bash
git diff phc/env/tasks/humanoid_im_pain.py | head -10
```
Expected: 2 added lines inside `_update_pain_buffers`.

- [ ] **Step 2: Run V3 command.**

Run:
```bash
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=log_only \
  2>&1 | tee /tmp/phc_spec2_v3.log
```

In the viewer, once the humanoid is standing:
- Press `j` a few times in rapid succession to apply large external forces.
- Watch the terminal for `[pain-debug] max=... mean=...` lines.

Acceptance:
- At least a handful of `[pain-debug]` lines with `max > 0.1`.
- Viewer still shows humanoid moving (no guard applied in log_only).
- Final `reward:` line within ~1% of V2 reward (no penalty applied).

If `max` stays at 0.0 despite multiple `j` presses, debug:
1. Confirm `self._contact_forces` is populated: briefly add `print(self._contact_forces.abs().max().item())` to the same block and re-run.
2. Confirm `self.dof_force_tensor` nonzero after the push.
3. Confirm `pain_enabled` is actually True — print `self.pain_enabled, self.pain_mode` once in `__init__`.

- [ ] **Step 3: Capture the last 30 lines and grep debug lines.**

Run:
```bash
tail -30 /tmp/phc_spec2_v3.log > /tmp/phc_spec2_v3_tail.txt
grep -c '\[pain-debug\]' /tmp/phc_spec2_v3.log
```

Expected: second command prints a positive integer (≥ 5 lines is a typical "observed some nonzero pain" signal).

### Task C3: V4 — `mode=guard_only` visible motion change

**Files:** none this task (print-debug from C2 remains in place; still reverted in C4).

- [ ] **Step 1: Run V4 command.**

Run:
```bash
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=guard_only \
  2>&1 | tee /tmp/phc_spec2_v4.log
```

In the viewer, same protocol as V3: apply `j` pushes and observe.

Acceptance:
- `[pain-debug]` lines in stdout as in V3 (pain is still being computed).
- **Visible** difference vs V3 when pushing: reduced ROM OR slower return motion OR obvious stiffening on the painful side.
- Final `reward:` line can be noisier than V2/V3 (guard changes the resulting trajectory), but run should complete without crashing.

If motion looks identical to V3, the guard is not firing:
1. Check `pain_state` is actually reaching the `guard_gain * pain_state >= max_guard` saturation range. Add temporary print of `guard.max().item()` inside `_action_to_pd_targets`.
2. Confirm `pain_mode == "guard_only"` in the live env (can print at init).

If motion freezes entirely:
1. Reduce `guard_gain` and `max_guard` at the command line: `env.pain.guard_gain=0.3 env.pain.max_guard=0.4`.

- [ ] **Step 2: Capture the last 30 lines.**

Run:
```bash
tail -30 /tmp/phc_spec2_v4.log > /tmp/phc_spec2_v4_tail.txt && wc -l /tmp/phc_spec2_v4_tail.txt
```

Expected: `30 /tmp/phc_spec2_v4_tail.txt`.

### Task C4: Remove the print-debug

**Files:**
- Revert: `phc/env/tasks/humanoid_im_pain.py` (undo the print added in C2).

- [ ] **Step 1: Revert with git.**

Run:
```bash
git checkout -- phc/env/tasks/humanoid_im_pain.py
git diff phc/env/tasks/humanoid_im_pain.py
```

Expected: second command prints nothing (no diff).

- [ ] **Step 2: Re-run V1 import check to confirm the revert is clean.**

Run:
```bash
conda run -n phc python -c "from phc.env.tasks.humanoid_im_pain import HumanoidImPain; print(HumanoidImPain.__name__)"
```

Expected: `HumanoidImPain`.

### Task C5: Populate the implementation log and commit

**Files:**
- Modify: `docs/superpowers/specs/phc_v0_spec2_impl_log.md`

- [ ] **Step 1: Fill in V1-V4 sections of the log.**

Open `docs/superpowers/specs/phc_v0_spec2_impl_log.md` and replace each `_TBD_` block with:

```markdown
## V1 — Pure-torch import check

**Command:**
```bash
conda run -n phc python -c "from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard); print('ok')"
```

**Result:** stdout `ok`, exit 0. Confirms pain_baseline.py is pure torch
(no Isaac Gym / numpy contamination).

---

## V2 — `mode=off` baseline regression

**Command:**
```bash
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=False env.pain.mode=off
```

**Exit code:** 0. **Viewer:** humanoid balanced throughout.

**Stdout tail (last 30 lines):** <paste /tmp/phc_spec2_v2_tail.txt here>

**Observation:** reward <FILL IN> vs Spec #1's 941.05 (within ~1%). No
Traceback. Pain code path not entered (pain.enabled=False short-circuits
_compute_reward before _update_pain_buffers).

---

## V3 — `mode=log_only` nonzero pain under push

**Command:**
```bash
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=log_only
```

(Temporary print-debug was added inside `_update_pain_buffers` for the
duration of this run; reverted in commit 3 prep with `git checkout --`.)

**Exit code:** 0. **Viewer:** humanoid responds to `j` pushes; motion
unchanged from V2 baseline (log_only does not apply action guard).

**Stdout tail (last 30 lines):** <paste /tmp/phc_spec2_v3_tail.txt here>

**[pain-debug] line count:** <FILL IN from grep>. Max observed
`pain_scalar` during pushes: <FILL IN>.

**Observation:** reward <FILL IN> vs V2 (within ~1%). Pain nonzero under
external push, confirming the pipeline end-to-end: contact forces ->
compute_contact_pain -> broadcast_body_pain_to_dof -> combine_internal
-> update_pain_state.

---

## V4 — `mode=guard_only` visible motion change

**Command:**
```bash
conda run -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=False \
  env.pain.enabled=True env.pain.mode=guard_only
```

**Exit code:** 0. **Viewer observation:** compared to V3 under identical
`j` push sequence, motion was visibly <FILL IN ONE OF: slower to recover /
reduced ROM / visibly stiffer on pushed side>.

**Stdout tail (last 30 lines):** <paste /tmp/phc_spec2_v4_tail.txt here>

**Observation:** Action guard fires as intended. Pretrained checkpoint
still produced valid imitation behavior in unpained regions.

---

## V5 — §20 condition 5 (fine-tune)

Deferred to Spec #3 as planned.
```

- [ ] **Step 2: Also update the "Exit gate" checklist and handoff.**

Replace the Exit gate section:
```markdown
## Exit gate

- [x] Four deliverable files created / edited.
- [x] `git diff phc/` between Spec #1's final commit and Spec #2's implementation commit touches only the files listed in the spec §14.
- [x] V1 import check `ok`.
- [x] V2 baseline reward matches Spec #1 within ~1%.
- [x] V3 produces nonzero `pain_mean` / `pain_max`.
- [x] V4 shows visible motion change vs V3 on same seed.
- [ ] User has reviewed this log and given go-signal for Spec #3.
```

Replace the Handoff section:
```markdown
## Handoff

Spec #2 complete. Entering Spec #3 (guard_and_reward short fine-tune from
copied checkpoint, guide §15).
```

- [ ] **Step 3: Verify the log has no remaining `_TBD_` placeholders.**

Run:
```bash
grep -n "_TBD_\|<FILL IN>\|<paste" docs/superpowers/specs/phc_v0_spec2_impl_log.md || echo "clean"
```

Expected: `clean`. If any placeholder remains, fix it before committing.

- [ ] **Step 4: Stage and commit.**

Run:
```bash
git diff --stat
git add docs/superpowers/specs/phc_v0_spec2_impl_log.md
git commit -m "$(cat <<'EOF'
Spec #2: populate implementation log with V1-V4 results

All four verification gates green:
- V1 pure-torch import check: ok
- V2 (mode=off) reward within ~1% of Spec #1 baseline (941.05)
- V3 (mode=log_only) produced nonzero pain under viewer j push
- V4 (mode=guard_only) visibly reduced painful-direction motion vs V3
  on identical motion file / seed.

V5 (fine-tune) deferred to Spec #3 per plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -4
```

Expected: top commit is the new one; the previous three commits are commit 2 (implementation), commit 1 (design+log skeleton), and the Spec #1 kickoff.

---

## Exit summary

After all phases complete, the branch `jinsu-pain_baseline_phc_v0` contains (on top of Spec #1):

| Commit | Content |
|---|---|
| `f0ad076` | Spec #2 design doc + log skeleton |
| *(new)* | Four-file implementation |
| *(new)* | Populated implementation log |

`git diff master...jinsu-pain_baseline_phc_v0 -- phc/` should show exactly:
- 1 new file: `phc/env/util/pain_baseline.py`
- 1 new file: `phc/env/tasks/humanoid_im_pain.py`
- 1 new file: `phc/data/cfg/env/env_im_pain.yaml`
- 1 modified file: `phc/utils/parse_task.py` (+1 line)

Spec §14 exit gate is satisfied; next session can invoke Spec #3.

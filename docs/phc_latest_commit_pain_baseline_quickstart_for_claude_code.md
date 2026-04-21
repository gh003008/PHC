# PHC latest-commit quick baseline/prototype guide for pain-aware control
**Target repo:** `ZhengyiLuo/PHC`  
**Pinned commit:** `846988d433ce1f341e85ac6fbd2cd51911bb3341` (`phc in isaaclab eval`, 2025-08-21)  
**Goal:** build the **fastest possible pain-aware baseline/prototype** on top of PHC, with minimal code surgery, so that we can:
1. run the official pretrained PHC baseline,
2. compute/log simple pain proxies,
3. make pain visibly change motion,
4. optionally fine-tune from the pretrained checkpoint,
5. avoid refactoring PHC into a hierarchical planner for now.

---

# 0. Non-negotiable scope

This guide is **not** for building the final scientific architecture.

This guide is for a **quick baseline/prototype** that answers only:

> “Can PHC produce visibly pain-aware motion if we add a simple pain estimator and a small amount of control logic?”

This guide intentionally **does not** attempt:
- full muscle-layer integration,
- exoskeleton interaction modeling,
- patient-specific physiology,
- planner-level impedance output,
- PULSE migration,
- SONIC-style large-scale re-architecture.

If you (Claude Code) find yourself trying to:
- redesign the action space,
- change PHC into a full hierarchical planner,
- add large new config systems,
- or rewrite the policy/network builder,

**stop**. That is out of scope for this baseline.

---

# 1. Why this particular baseline strategy

## 1.1 Use the latest repo, but stay on the Isaac Gym path
The latest PHC commit adds IsaacLab eval support, but the repo’s core evaluation/training flow is still centered on:
- `phc/run_hydra.py`
- Hydra configs under `phc/data/cfg/`
- task classes under `phc/env/tasks/`

For the pain baseline, **do not** start from the IsaacLab eval script.  
Use the **native Isaac Gym + Hydra** training/eval path.

## 1.2 Use a single-primitive PHC checkpoint, not MCP/getup composer
For the quickest baseline, use a **single primitive** pretrained checkpoint, not the full MCP/getup stack.

Reason:
- fewer moving parts,
- fewer code paths,
- no composer logic,
- no `env.models=[...]` juggling,
- easier checkpoint reuse,
- easier debugging.

We do **not** need fail-state recovery for the first pain prototype.

## 1.3 Do NOT change observation size in v0
This is the most important engineering choice.

If you append pain values to the observation vector, the network input size changes and the pretrained checkpoint will not load cleanly.

For the **first prototype**, keep the policy input shape unchanged.

That means:
- pain is computed inside the environment,
- pain is logged,
- pain enters the reward,
- pain optionally modulates actions through an **action governor**,
- but pain is **not** appended to policy observation in v0.

This lets you:
- reuse pretrained PHC weights immediately,
- verify the whole pipeline fast,
- fine-tune from the pretrained model without architecture mismatch.

Later, if needed, do a v1 with pain appended to observations.

---

# 2. What to clone and how to pin it

Run exactly:

```bash
git clone https://github.com/ZhengyiLuo/PHC.git
cd PHC
git checkout 846988d433ce1f341e85ac6fbd2cd51911bb3341
git switch -c pain_baseline_phc_v0
```

Do **not** work on floating `master` for this prototype.  
Pin to the commit above so results are reproducible.

---

# 3. Repo landmarks Claude Code must inspect first

Before writing code, inspect these files locally:

## Entrypoint / config
- `phc/run_hydra.py`
- `phc/data/cfg/config.yaml`

## Environment configs
- `phc/data/cfg/env/env_im.yaml`
- `phc/data/cfg/env/env_im_pnn.yaml`

## Learning configs
- `phc/data/cfg/learning/im.yaml`
- `phc/data/cfg/learning/im_big.yaml`
- `phc/data/cfg/learning/im_pnn.yaml`

## Task implementations
- `phc/env/tasks/humanoid.py`
- `phc/env/tasks/humanoid_im.py`
- `phc/utils/parse_task.py`

## Data/model downloader
- `download_data.sh`

### What you are looking for
- where the task class is selected,
- where observations are built,
- where reward is computed,
- where actions are available before being applied to Isaac PD,
- where contact forces / DOF torques / DOF limits are stored.

---

# 4. Environment setup

Follow the repo’s own path first.

```bash
conda create -n isaac python=3.8 -y
conda activate isaac
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
pip install -r requirement.txt
```

Then install/setup Isaac Gym and download SMPL / SMPLX models exactly as the repo README asks.

Required local structure:

```text
data/
  smpl/
    SMPL_FEMALE.pkl
    SMPL_NEUTRAL.pkl
    SMPL_MALE.pkl
    SMPLX_FEMALE.pkl
    SMPLX_NEUTRAL.pkl
    SMPLX_MALE.pkl
```

Then fetch sample data and checkpoints:

```bash
bash download_data.sh
```

---

# 5. Sanity check before touching any code

Do **not** start implementing pain until the repo runs locally.

Run these in order.

## 5.1 SMPL sanity check
```bash
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

If these fail, stop and fix environment/data first.

## 5.2 Official single-primitive checkpoint sanity eval
Use the provided pretrained **single primitive** model as the base.

Recommended starting command:

```bash
python phc/run_hydra.py   learning=im_pnn   exp_name=phc_shape_pnn_iccv   epoch=-1   test=True   env=env_im_pnn   robot=smpl_humanoid_shape   robot.freeze_hand=True   robot.box_body=False   env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl   env.num_envs=1   headless=False
```

This command must run **before** you add pain code.

### Acceptance criterion
- humanoid loads,
- simulation runs,
- no config errors,
- no checkpoint loading errors,
- no SMPL asset errors.

Only after this works do you implement the pain baseline.

---

# 6. Exact baseline we want to implement

We want **PHC-Pain-v0**, defined as:

## 6.1 What stays unchanged
- policy architecture,
- observation dimension,
- Isaac PD control mode,
- task core structure,
- training runner,
- checkpoint format.

## 6.2 What gets added
1. **pain estimator** (pure environment-side torch code)
2. **pain buffers/logging**
3. **reward penalty**
4. **optional action governor** (inference-time and train-time)
5. **new env config**
6. **new task class**

## 6.3 What this baseline should demonstrate
- pain spikes when the humanoid is pushed / stressed / near joint limits,
- action amplitude is reduced in painful joints,
- motion changes visibly (guarding / reduced range / slower movement),
- no need to retrain from scratch.

---

# 7. Implementation strategy

## 7.1 Create a new task class, do NOT edit official task in place
Create:

```text
phc/env/tasks/humanoid_im_pain.py
```

Implement:

```python
class HumanoidImPain(HumanoidIm):
    ...
```

Do **not** overwrite `HumanoidIm` directly.  
Keep the upstream code path intact.

## 7.2 Register the task
In:

```text
phc/utils/parse_task.py
```

add the import and make sure the class name is available to the existing `eval(args.task)` path.

You should be able to select the task through Hydra using:

```yaml
task: HumanoidImPain
```

---

# 8. Create a new environment config

Create:

```text
phc/data/cfg/env/env_im_pain.yaml
```

Start by copying `env_im_pnn.yaml`, then modify only what is necessary.

### Minimum contents
- same base fields as `env_im_pnn.yaml`
- `task: HumanoidImPain`
- nested pain config block

Recommended initial config:

```yaml
task: HumanoidImPain
project_name: "PHC"
notes: "pain baseline"
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
num_prim: 3
training_prim: 0
actors_to_load: 0
has_lateral: False
models: []
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
controlFrequencyInv: 2
stateInit: "Random"
hybridInitProb: 0.5
numAMPObsSteps: 10
local_root_obs: True
root_height_obs: True
key_bodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]
contact_bodies: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
reset_bodies: ['Pelvis', 'L_Hip', 'L_Knee', 'R_Hip', 'R_Knee', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
terminationHeight: 0.15
enableEarlyTermination: True
terminationDistance: 0.25
numTrajSamples: 3
trajSampleTimestepInv: 3
enableTaskObs: True

pain:
  enabled: True
  mode: "guard_and_reward"   # choices: off, log_only, reward_only, guard_only, guard_and_reward
  append_to_obs: False       # keep False in v0 to preserve checkpoint compatibility
  use_internal_load: True
  use_external_contact: True
  lambda_p: 0.05

  # instantaneous proxy weights
  w_limit: 1.0
  w_torque: 0.35
  w_power: 0.15
  w_contact: 0.50

  # normalizers / thresholds
  joint_limit_margin_ratio: 0.15
  torque_ref_scale: 0.50
  power_ref: 5.0
  contact_force_ref: 150.0
  pain_threshold: 0.10
  pain_cap: 3.0

  # temporal filtering
  rise_alpha: 0.25
  decay_alpha: 0.02

  # action governor
  use_action_guard: True
  max_guard: 0.75
  guard_gain: 0.60

  # logging
  log_joint_pain: True
  log_contact_pain: True

plane:
  staticFriction: 1.0
  dynamicFriction: 1.0
  restitution: 0.0
```

---

# 9. Create a standalone pain helper module

Create:

```text
phc/env/util/pain_baseline.py
```

Put all pain math here so task code stays clean.

## 9.1 Functions to implement

### (A) joint-limit pain
```python
def compute_joint_limit_pain(q, q_lo, q_hi, margin_ratio=0.15, eps=1e-6):
    """
    q:        (B, D)
    q_lo/hi:  (D,)
    returns:  (B, D) in [0, +inf)
    """
```

Suggested formula:
- midpoint: `q_mid = 0.5 * (q_lo + q_hi)`
- half-range: `q_half = 0.5 * (q_hi - q_lo)`
- safe half-range: `q_safe = q_half * (1 - margin_ratio)`
- excess: `relu(abs(q - q_mid) - q_safe)`
- normalize by `(q_half - q_safe + eps)`

This produces pain near ROM boundaries without exploding in the center.

### (B) torque pain
```python
def compute_torque_pain(tau, tau_limits, ref_scale=0.50, eps=1e-6):
    """
    tau:       (B, D)
    tau_limits:(D,)
    """
```

Suggested formula:
- `tau_ref = clamp(ref_scale * tau_limits, min=1.0)`
- `pain = abs(tau) / tau_ref`

### (C) power pain
```python
def compute_power_pain(tau, dq, power_ref=5.0, eps=1e-6):
    power = abs(tau * dq)
    return power / max(power_ref, eps)
```

### (D) external contact pain
```python
def compute_contact_pain(contact_forces, ref_force=150.0, eps=1e-6):
    """
    contact_forces: (B, Nb, 3)
    return:         (B, Nb)
    """
```

Suggested:
- `force_mag = norm(contact_forces, dim=-1)`
- `pain = force_mag / ref_force`

### (E) per-joint aggregation
We need to map body-level or DOF-level signals into **joint-group pain**.

Implement a function that:
- aggregates 3 DOFs belonging to the same SMPL joint,
- broadcasts the joint scalar back to those 3 DOFs when needed.

Use:
- `self._dof_offsets`
- `self._dof_names`

Do **not** hardcode 69 except for debugging assertions.

### (F) temporal integration
```python
def update_pain_state(prev_state, inst_pain, rise_alpha=0.25, decay_alpha=0.02, threshold=0.1, cap=3.0):
    """
    leaky accumulation with recovery
    """
```

Suggested formula:
```python
drive = torch.relu(inst_pain - threshold)
next_state = prev_state * (1.0 - decay_alpha) + rise_alpha * drive
next_state = torch.clamp(next_state, 0.0, cap)
```

### (G) action guard
```python
def apply_pain_action_guard(actions, q, joint_pain_dof, guard_gain=0.6, max_guard=0.75):
    """
    actions:        raw PHC actions / PD targets
    q:              current dof pos
    joint_pain_dof: (B, D), already broadcast to dofs
    """
```

Suggested formula:
```python
guard = torch.clamp(guard_gain * joint_pain_dof, 0.0, max_guard)
return q + (1.0 - guard) * (actions - q)
```

This is the simplest and safest “protective stiffness/guarding” proxy for PHC:
- if pain is low, action unchanged,
- if pain is high, target is pulled back toward current joint angle.

This mimics reduced willingness to move through painful directions.

---

# 10. Modify the task class

## 10.1 In `HumanoidImPain.__init__`
After calling `super().__init__`, read `cfg["env"]["pain"]` and allocate buffers:

```python
self.pain_cfg = cfg["env"].get("pain", {})
self.pain_enabled = self.pain_cfg.get("enabled", False)
self.pain_mode = self.pain_cfg.get("mode", "off")

self.pain_inst = torch.zeros((self.num_envs, self.num_dof), device=self.device)
self.pain_state = torch.zeros((self.num_envs, self.num_dof), device=self.device)
self.pain_internal = torch.zeros((self.num_envs, self.num_dof), device=self.device)
self.pain_external = torch.zeros((self.num_envs, self.num_dof), device=self.device)
self.pain_scalar = torch.zeros((self.num_envs,), device=self.device)
```

Also build any indices you need:
- torso body ids,
- contact body ids,
- left/right lower-body indices,
- etc.

Reuse existing buffers if available:
- `self._contact_forces`
- `self.dof_force_tensor`
- `self.dof_limits_lower`
- `self.dof_limits_upper`
- `self.torque_limits`

## 10.2 Add a helper method inside the task
Implement:

```python
def _update_pain_buffers(self):
    ...
```

This method should:
1. read current `q`, `dq`, `tau`, `contact`,
2. compute instantaneous internal and external pain,
3. aggregate to joint / dof level,
4. update `self.pain_state`,
5. set `self.pain_scalar = self.pain_state.mean(dim=-1)`.

## 10.3 When to call `_update_pain_buffers`
Call it **after physics state is refreshed** and before reward/logging is finalized.

The safest practical approach:
- grep locally for the step/update flow in `humanoid.py`, `base_task.py`, `vec_task.py`,
- find the point where:
  - current `self._dof_pos`, `self._dof_vel`, `self.dof_force_tensor`, `self._contact_forces`
    are valid for the current post-step state,
  - but before `_compute_reward()` finalizes logs.

If that is messy, then the fallback is:
- call `_update_pain_buffers()` at the start of `_compute_reward()` in `HumanoidImPain`.

That is acceptable for v0.

## 10.4 Reward hook
Override `_compute_reward(self, actions)`.

Pseudo-structure:

```python
def _compute_reward(self, actions):
    super()._compute_reward(actions)

    if not self.pain_enabled:
        return

    self._update_pain_buffers()

    if self.pain_mode in ["reward_only", "guard_and_reward"]:
        lambda_p = self.pain_cfg.get("lambda_p", 0.05)
        self.rew_buf[:] = self.rew_buf[:] - lambda_p * self.pain_scalar
```

### IMPORTANT
Do not replace PHC’s imitation reward.  
Only subtract the pain penalty on top.

## 10.5 Logging hook
Populate `self.extras` with pain diagnostics:

```python
self.extras["pain_scalar"] = self.pain_scalar.detach().cpu().numpy()
self.extras["pain_mean"] = float(self.pain_scalar.mean().item())
self.extras["pain_max"] = float(self.pain_scalar.max().item())
self.extras["pain_internal_mean"] = float(self.pain_internal.mean().item())
self.extras["pain_external_mean"] = float(self.pain_external.mean().item())
```

If joint-level logging is enabled:
```python
self.extras["pain_joint"] = self.pain_state.detach().cpu().numpy()
```

Do not log huge arrays to wandb every step during training unless needed.  
For training, scalar summaries are enough.

---

# 11. Add the action governor

## 11.1 Find the action hook locally
After cloning, grep locally:

```bash
grep -R "pre_physics_step" -n phc/env/tasks phc/env
grep -R "set_dof_position_target_tensor" -n phc/env/tasks phc/env
grep -R "actions" -n phc/env/tasks/humanoid.py phc/env/tasks/humanoid_im.py
```

You must locate the point where:
- the policy action exists,
- but before the PD target is applied.

If a clean `pre_physics_step(self, actions)` exists in the inheritance chain, override it in `HumanoidImPain`.
If not, intercept the earliest point where `self.actions` is assigned or transformed.

## 11.2 Guard logic
When `pain_mode` is:
- `guard_only`
- or `guard_and_reward`

apply:

```python
actions = apply_pain_action_guard(
    actions=actions,
    q=self._dof_pos,
    joint_pain_dof=self.pain_state,
    guard_gain=self.pain_cfg.get("guard_gain", 0.6),
    max_guard=self.pain_cfg.get("max_guard", 0.75),
)
```

Then pass the modified action downstream exactly as the original code expects.

### Why this is the right v0 choice
It requires:
- no network change,
- no PD reconfiguration,
- no robot config edits,
- no control-mode refactor.

And it is enough to show:
- reduced ROM under pain,
- slower painful movement,
- visible guarding.

---

# 12. Absolutely do NOT do these in v0

- Do **not** append pain to obs in v0.
- Do **not** change network builders.
- Do **not** touch MCP / composer / getup code.
- Do **not** rewrite action space.
- Do **not** re-implement impedance inside PHC.
- Do **not** try to add exoskeleton interaction yet.
- Do **not** retrain from scratch.

If you want to explore more after v0 works, do it in a separate `pain_v1` branch.

---

# 13. Minimal runnable modes

Implement these modes in config:

## Mode 1: `off`
- baseline PHC behavior
- used to verify no regression

## Mode 2: `log_only`
- compute pain
- no reward penalty
- no action guard
- just log and visualize

## Mode 3: `reward_only`
- compute pain
- subtract `lambda_p * pain_scalar`
- no action guard
- good for fine-tuning from checkpoint

## Mode 4: `guard_only`
- compute pain
- no reward change
- apply action governor
- best for zero-training visual demo

## Mode 5: `guard_and_reward`
- both enabled
- best candidate for short fine-tuning after the guard-only demo works

---

# 14. Zero-training demo path (must be done first)

This is the fastest path to a visible prototype.

## 14.1 Add the new task/config but keep observation size unchanged
Use `append_to_obs: False`.

## 14.2 Run with pretrained checkpoint
Recommended first command:

```bash
python phc/run_hydra.py   learning=im_pnn   exp_name=phc_shape_pnn_iccv   epoch=-1   test=True   env=env_im_pain   robot=smpl_humanoid_shape   robot.freeze_hand=True   robot.box_body=False   env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl   env.num_envs=1   headless=False   env.pain.enabled=True   env.pain.mode=log_only
```

If that works, run:

```bash
python phc/run_hydra.py   learning=im_pnn   exp_name=phc_shape_pnn_iccv   epoch=-1   test=True   env=env_im_pain   robot=smpl_humanoid_shape   robot.freeze_hand=True   robot.box_body=False   env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl   env.num_envs=1   headless=False   env.pain.enabled=True   env.pain.mode=guard_only
```

## 14.3 Demo procedure
While viewer is open:
- press `m` to cancel termination if needed,
- press `j` to apply a large force,
- verify `pain_scalar` spikes,
- verify action-guard mode visibly reduces/aggressively dampens motion.

### Acceptance criteria
- no checkpoint mismatch,
- no obs-size mismatch,
- pain logs nonzero,
- visible motion difference between `off` and `guard_only`.

---

# 15. Fine-tuning path (after zero-training demo works)

Only do this after the zero-training demo is stable.

## 15.1 Copy checkpoint into a new experiment directory
Do **not** overwrite the official downloaded checkpoint directory.

```bash
mkdir -p output/HumanoidIm/phc_shape_pnn_iccv_pain
cp output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth    output/HumanoidIm/phc_shape_pnn_iccv_pain/Humanoid.pth
```

## 15.2 Fine-tune from copied checkpoint
Because `epoch=-1` loads `${output_path}/Humanoid.pth`, this works if the new experiment dir contains the copied checkpoint.

Run:

```bash
python phc/run_hydra.py   learning=im_pnn   exp_name=phc_shape_pnn_iccv_pain   epoch=-1   test=False   env=env_im_pain   robot=smpl_humanoid_shape   robot.freeze_hand=True   robot.box_body=False   env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl   env.num_envs=256   headless=True   env.pain.enabled=True   env.pain.mode=guard_and_reward   env.pain.append_to_obs=False
```

Use fewer envs first (`256` or `512`) to debug quickly.

## 15.3 Short-run goal
Do **not** launch a full training job immediately.

First target:
- can resume from checkpoint,
- runs 1k–5k PPO updates without crashing,
- reward does not instantly collapse,
- pain metrics are logged.

Only after that should you scale up env count / duration.

---

# 16. Optional v1 after v0 succeeds

Only do this in a new branch, e.g. `pain_baseline_phc_v1_obs`.

## v1 goals
- append pain scalars to observations,
- possibly add body-part pain maps,
- initialize first-layer weights carefully,
- fine-tune from checkpoint with partial weight loading.

### But this is NOT part of v0.
v0 must remain checkpoint-compatible.

---

# 17. Concrete code checklist for Claude Code

## Files to create
- `phc/env/tasks/humanoid_im_pain.py`
- `phc/env/util/pain_baseline.py`
- `phc/data/cfg/env/env_im_pain.yaml`

## Files to edit
- `phc/utils/parse_task.py`

## Files to inspect but preferably not edit in v0
- `phc/env/tasks/humanoid.py`
- `phc/env/tasks/humanoid_im.py`
- `phc/run_hydra.py`

---

# 18. Suggested class skeleton

```python
# phc/env/tasks/humanoid_im_pain.py

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.util.pain_baseline import (
    compute_joint_limit_pain,
    compute_torque_pain,
    compute_power_pain,
    compute_contact_pain,
    update_pain_state,
    apply_pain_action_guard,
)

class HumanoidImPain(HumanoidIm):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.pain_cfg = cfg["env"].get("pain", {})
        self.pain_enabled = self.pain_cfg.get("enabled", False)
        self.pain_mode = self.pain_cfg.get("mode", "off")

        self.pain_inst = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.pain_state = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.pain_internal = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.pain_external = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.pain_scalar = torch.zeros((self.num_envs,), device=self.device)

    def _update_pain_buffers(self):
        if not self.pain_enabled:
            return

        q = self._dof_pos
        dq = self._dof_vel
        tau = self.dof_force_tensor

        limit_pain = compute_joint_limit_pain(
            q, self.dof_limits_lower, self.dof_limits_upper,
            margin_ratio=self.pain_cfg.get("joint_limit_margin_ratio", 0.15),
        )

        torque_pain = compute_torque_pain(
            tau, self.torque_limits,
            ref_scale=self.pain_cfg.get("torque_ref_scale", 0.5),
        )

        power_pain = compute_power_pain(
            tau, dq,
            power_ref=self.pain_cfg.get("power_ref", 5.0),
        )

        internal = (
            self.pain_cfg.get("w_limit", 1.0) * limit_pain +
            self.pain_cfg.get("w_torque", 0.35) * torque_pain +
            self.pain_cfg.get("w_power", 0.15) * power_pain
        )

        # optional external contact -> dof broadcast
        external = torch.zeros_like(internal)
        if self.pain_cfg.get("use_external_contact", True):
            contact_body = compute_contact_pain(
                self._contact_forces[:, :self.num_bodies, :],
                ref_force=self.pain_cfg.get("contact_force_ref", 150.0),
            )
            # TODO: implement simple body->joint/dof broadcast

        inst = internal + self.pain_cfg.get("w_contact", 0.5) * external

        self.pain_internal[:] = internal
        self.pain_external[:] = external
        self.pain_inst[:] = inst
        self.pain_state[:] = update_pain_state(
            self.pain_state,
            self.pain_inst,
            rise_alpha=self.pain_cfg.get("rise_alpha", 0.25),
            decay_alpha=self.pain_cfg.get("decay_alpha", 0.02),
            threshold=self.pain_cfg.get("pain_threshold", 0.1),
            cap=self.pain_cfg.get("pain_cap", 3.0),
        )
        self.pain_scalar[:] = self.pain_state.mean(dim=-1)

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        if not self.pain_enabled:
            return
        self._update_pain_buffers()
        if self.pain_mode in ["reward_only", "guard_and_reward"]:
            self.rew_buf[:] = self.rew_buf[:] - self.pain_cfg.get("lambda_p", 0.05) * self.pain_scalar

        self.extras["pain_mean"] = float(self.pain_scalar.mean().item())
        self.extras["pain_max"] = float(self.pain_scalar.max().item())

    # if a clean action hook exists in inheritance chain, override it here
    # def pre_physics_step(self, actions):
    #     if self.pain_enabled and self.pain_mode in ["guard_only", "guard_and_reward"]:
    #         self._update_pain_buffers()
    #         actions = apply_pain_action_guard(actions, self._dof_pos, self.pain_state, ...)
    #     return super().pre_physics_step(actions)
```

This is just the shape. Claude Code must adapt it to the actual inherited method names after local inspection.

---

# 19. How to debug if things go wrong

## Problem: checkpoint won’t load
Cause:
- observation dim changed,
- model config changed,
- exp directory wrong.

Fix:
- confirm `append_to_obs=False`,
- confirm using same learning config as checkpoint,
- confirm copied `Humanoid.pth` is in the new `${output_path}` if fine-tuning.

## Problem: pain always zero
Check:
- `self.dof_force_tensor` is being refreshed,
- `_update_pain_buffers()` is called after physics tensors are valid,
- force/contact tensors are nonzero when pushing with `j`,
- joint limit formula not saturating to zero because wrong limits.

## Problem: action guard freezes motion completely
Reduce:
- `guard_gain`
- `max_guard`
- `lambda_p`
- `w_contact`

Start conservative:
- `guard_gain=0.3`
- `max_guard=0.4`
- `lambda_p=0.01`

## Problem: reward collapse during fine-tuning
Switch temporarily to:
- `mode=guard_only` for inference demo
or
- `mode=reward_only` with smaller `lambda_p`
and keep checkpoint initialization.

---

# 20. Minimal success criteria for the prototype

The baseline is considered successful if all 5 conditions hold:

1. **Official PHC single-primitive checkpoint runs unchanged.**
2. **New task `HumanoidImPain` runs with `pain.mode=off` and matches baseline behavior.**
3. **`pain.mode=log_only` produces nonzero pain metrics under push / stress.**
4. **`pain.mode=guard_only` visibly changes motion without retraining.**
5. **A short fine-tuning run from copied checkpoint starts successfully with `guard_and_reward`.**

If all 5 are satisfied, stop.  
That is enough for the baseline/prototype.

---

# 21. What to report back after implementation

Claude Code should report back with:

## A. Files changed
List every file created/edited.

## B. Commands that worked
Copy-paste exact shell commands.

## C. One short demo recipe
Example:
- run baseline,
- press `j`,
- observe `pain_mean`,
- run guard mode,
- compare motion.

## D. One screenshot or saved rollout path
Preferably save one rollout for:
- `pain.mode=off`
- `pain.mode=guard_only`

## E. Known limitations
Must explicitly state:
- no exoskeleton,
- no physiological pain model,
- no planner-level impedance,
- no appended pain obs,
- this is an environment-side proxy baseline only.

---

# 22. Final instruction to Claude Code

Implement **only PHC-Pain-v0**:

- latest commit pinned,
- single primitive checkpoint,
- no observation-size change,
- pain helper module,
- new task subclass,
- new env config,
- reward penalty,
- action governor,
- logging,
- zero-training demo first,
- short fine-tuning second.

Do not expand scope unless the zero-training demo works.

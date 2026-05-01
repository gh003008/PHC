# PHC + Residual + AMP for Continuous v_cmd — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a small residual MLP head on top of frozen phc_3 (PNN imitation policy) with AMP discriminator on a 3-clip walking pool, so the resulting policy tracks continuous v_cmd ∈ [0.76, 1.23] m/s with no discrete clip switching at inference.

**Architecture:** New task class `HumanoidImResAMPVCmd` (subclass of `HumanoidIm`) carries v_cmd obs + multi-clip + a-1 hard switch + retime + S2 sampling + reward (r_track + r_amp + r_im_anchor + r_survive). New network `ResAMPVCmdNetwork` wraps frozen phc_3 PNN + small (256-256, σ_init=-1.0) residual MLP head, output `a = phc_3(s,ref) + clip(Δa, ±0.2)`. New agent `ResAMPVCmdAgent` inherits `AmpAgent`, freezes phc_3 params, learns residual + AMP discriminator. Stage 1 local smoke (4060 Ti, num_envs=64, ~6h), Stage 2 server full training (num_envs=512, ~36-48h), Stage 3 eval + demo.

**Tech Stack:** PHC + IsaacGym + rl_games + Hydra. Python 3.8 / PyTorch 2.x / CUDA 11.8. conda env `phc`.

---

### Task 1: Skeleton task class + parse_task registration

**Files:**
- Create: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`
- Modify: `phc/utils/parse_task.py`
- Test: `python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print(HumanoidImResAMPVCmd)"`

- [ ] **Step 1: Activate conda env**

```bash
conda activate phc
```

Expected: prompt shows `(phc)`.

- [ ] **Step 2: Create task class skeleton**

```python
# phc/env/tasks/humanoid_im_res_amp_vcmd.py
"""Residual + AMP task for continuous v_cmd tracking on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
import torch
import numpy as np
from phc.env.tasks.humanoid_im import HumanoidIm


class HumanoidImResAMPVCmd(HumanoidIm):
    """1D v_cmd-conditioned imitation with multi-clip + a-1 hard switch.

    Adds 1-D v_cmd observation to phc_3's input, retimes the active clip
    by v_cmd/v_natural ratio, and switches active clip at midpoints with
    hysteresis (port of scripts/phc_walk_demo.py v3 logic). Reward is
    augmented with v_cmd tracking and a small imitation anchor.
    """

    # v_cmd config (spec §6)
    V_CMD_MIN = 0.76
    V_CMD_MAX = 1.23
    V_CMD_MAX_ACCEL = 0.5     # m/s² ramp rate
    V_CMD_RAMP_PROB = 0.01    # per-step prob of new target (S2)
    HYSTERESIS = 0.02

    # Reward weights (spec §5.4)
    W_TRACK = 0.5
    W_AMP = 0.3
    W_IM = 0.2
    ALPHA_TRACK = 5.0
    ALPHA_IM = 2.0

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        # v_cmd state, allocated per-env
        self._v_cmd_target = torch.full((self.num_envs,), 1.0, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), 1.0, device=self.device)
        # Natural speeds of the 3 clips loaded by motion_lib (filled in _read_dirmeta())
        self._v_natural = None  # set in _post_init
        # Active clip per env (index into _v_natural). Init = middle clip.
        self._active_clip = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._post_init()

    def _post_init(self):
        """Read clip dirmeta to populate v_natural; called once after super().__init__."""
        # Default to spec § §3 V_NATURAL if dirmeta missing
        self._v_natural = torch.tensor([0.897, 0.975, 1.068], device=self.device)
```

- [ ] **Step 3: Register in parse_task.py**

```python
# phc/utils/parse_task.py — add after the HumanoidImVICCmdMultiClip import
from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd
```

- [ ] **Step 4: Run import test**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK', HumanoidImResAMPVCmd.__mro__[:4])"
```

Expected: `OK (<class 'phc.env.tasks.humanoid_im_res_amp_vcmd.HumanoidImResAMPVCmd'>, <class 'phc.env.tasks.humanoid_im.HumanoidIm'>, ...)` and no traceback.

- [ ] **Step 5: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py phc/utils/parse_task.py
git commit -m "res_amp_vcmd: skeleton task class + parse_task registration"
```

---

### Task 2: env yaml config

**Files:**
- Create: `phc/data/cfg/env/env_im_res_amp_vcmd.yaml`

- [ ] **Step 1: Inspect base env_im_pnn.yaml**

```bash
cat phc/data/cfg/env/env_im_pnn.yaml
```

Note these sections we'll reuse: humanoid type, sim params, motion file selection, episode length, termination distance.

- [ ] **Step 2: Create new env yaml**

```yaml
# phc/data/cfg/env/env_im_res_amp_vcmd.yaml
# 1D v_cmd-conditioned imitation training on top of frozen phc_3.
# Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md

defaults:
  - _self_

env:
  numEnvs: 64                          # smoke; server sbatch overrides to 512
  envSpacing: 5
  episodeLength: 300                   # 5s @ 60Hz; server uses 600
  isFlagrun: False
  enableDebugVis: False
  pdControl: True
  powerScale: 1.0
  controlFrequencyInv: 2
  stateInit: Random

  # PHC standard
  task: HumanoidImResAMPVCmd
  motion_file: sample_data/amass_walking_3clips_seamless_60s_v9.pkl
  cycle_motion: True
  num_amp_obs_steps: 10
  has_self_collision: True

  # v_cmd-specific (spec §6)
  v_cmd_min: 0.76
  v_cmd_max: 1.23
  v_cmd_max_accel: 0.5
  v_cmd_ramp_prob: 0.01
  hysteresis: 0.02

  # Reward weights (spec §5.4)
  task_reward_w: 0.5
  disc_reward_w: 0.3
  im_reward_w: 0.2
  alpha_track: 5.0
  alpha_im: 2.0

  # Termination
  enableEarlyTermination: True
  terminationDistance: 1.0
  pelvisHeightMin: 0.4

  # Action clipping (spec §5.4 "Action clipping")
  delta_action_clip: 0.2

  asset:
    assetFileName: smpl_humanoid.xml
    assetRoot: phc/data/assets/mjcf
```

- [ ] **Step 3: Sanity-load yaml**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('phc/data/cfg/env/env_im_res_amp_vcmd.yaml')); print('OK', cfg['env']['task'], cfg['env']['v_cmd_min'], cfg['env']['v_cmd_max'])"
```

Expected: `OK HumanoidImResAMPVCmd 0.76 1.23`

- [ ] **Step 4: Commit**

```bash
git add phc/data/cfg/env/env_im_res_amp_vcmd.yaml
git commit -m "res_amp_vcmd: env yaml config"
```

---

### Task 3: learning yaml config

**Files:**
- Create: `phc/data/cfg/learning/im_res_amp_vcmd.yaml`

- [ ] **Step 1: Inspect base im_pnn_big.yaml**

```bash
cat phc/data/cfg/learning/im_pnn_big.yaml
```

Note: PNN structure, sigma_init, AMP disc params. We'll keep all of these and add residual head spec.

- [ ] **Step 2: Create new learning yaml**

```yaml
# phc/data/cfg/learning/im_res_amp_vcmd.yaml
# Residual + AMP learning on top of frozen phc_3.
# Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md

params:
  seed: 0

  algo:
    name: im_amp_residual

  model:
    name: amp_residual

  network:
    name: amp_pnn_residual
    separate: True
    discrete: False

    # Frozen base — same arch as phc_3 (im_pnn_big), loaded from checkpoint
    frozen_base:
      checkpoint: output/HumanoidIm/phc_3/Humanoid.pth
      mlp_units: [2048, 1536, 1024, 1024, 512, 512]
      activation: silu
      num_actors: 3   # PNN primitives

    # Residual head (spec §3 "ResAMPVCmdNetwork")
    residual:
      input_dims: ["proprio", "v_cmd"]   # proprio + 1-D v_cmd
      mlp_units: [256, 256]
      activation: silu
      sigma_init: -1.0
      learn_sigma: False

    # AMP discriminator (unchanged from im_pnn_big.yaml)
    disc:
      units: [1024, 512]
      activation: relu
      initializer:
        name: default

  load_checkpoint: False    # we load phc_3 ourselves into frozen_base

  config:
    name: ResAMPVCmd
    env_name: rlgpu
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    reward_shaper:
      scale_value: 1.0
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 5e-5       # smaller than im_pnn_big since residual only
    lr_schedule: constant
    score_to_win: 20000
    max_epochs: 500           # smoke; server overrides to 20000
    save_best_after: 50
    save_frequency: 250
    print_stats: False
    save_intermediate: True
    entropy_coef: 0.0
    truncate_grads: True
    grad_norm: 50.0
    e_clip: 0.2
    horizon_length: 32
    minibatch_size: 4096      # smoke (num_envs=64 × horizon=32 × 2)
    mini_epochs: 6
    critic_coef: 5
    clip_value: False
    bounds_loss_coef: 10

    # AMP discriminator hyperparams (unchanged from im_pnn_big.yaml)
    amp_obs_demo_buffer_size: 200000
    amp_replay_buffer_size: 200000
    amp_replay_keep_prob: 0.01
    amp_batch_size: 512
    amp_minibatch_size: 4096
    disc_coef: 5
    disc_logit_reg: 0.01
    disc_grad_penalty: 5
    disc_reward_scale: 2
    disc_weight_decay: 0.0001
    normalize_amp_input: True

    # Reward decomposition (spec §5.4)
    task_reward_w: 0.5     # r_track
    disc_reward_w: 0.3     # r_amp
    im_reward_w: 0.2       # r_im anchor
    survive_reward_w: 1.0

    amp_dropout: False

    player:
      games_num: 50000000
```

- [ ] **Step 3: Sanity-load yaml**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('phc/data/cfg/learning/im_res_amp_vcmd.yaml')); print('OK algo=', cfg['params']['algo']['name'], 'res_units=', cfg['params']['network']['residual']['mlp_units'])"
```

Expected: `OK algo= im_amp_residual res_units= [256, 256]`

- [ ] **Step 4: Commit**

```bash
git add phc/data/cfg/learning/im_res_amp_vcmd.yaml
git commit -m "res_amp_vcmd: learning yaml config"
```

---

### Task 4: v_cmd observation extension

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Add observation extension method**

Add to `HumanoidImResAMPVCmd` class:

```python
    def get_obs_size(self):
        # Original PHC obs + 1-D v_cmd (normalized to [-1, 1])
        return super().get_obs_size() + 1

    def _compute_observations(self, env_ids=None):
        # Let parent compute the standard PHC obs, then concat v_cmd.
        super()._compute_observations(env_ids)
        # _compute_humanoid_obs writes to self.obs_buf in the parent;
        # we replace the LAST 1 dim with normalized v_cmd.
        if env_ids is None:
            v_cmd_norm = self._v_cmd_norm()
            self.obs_buf[:, -1] = v_cmd_norm
        else:
            v_cmd_norm = self._v_cmd_norm()[env_ids]
            self.obs_buf[env_ids, -1] = v_cmd_norm

    def _v_cmd_norm(self) -> torch.Tensor:
        """Map v_cmd_ramped from [V_CMD_MIN, V_CMD_MAX] to [-1, 1]."""
        center = (self.V_CMD_MIN + self.V_CMD_MAX) * 0.5
        half = (self.V_CMD_MAX - self.V_CMD_MIN) * 0.5
        return (self._v_cmd_ramped - center) / half
```

- [ ] **Step 2: Smoke instantiate via parse_task**

This requires the env yaml from Task 2 + a minimal hydra invocation. Use a tiny driver script:

```bash
python -c "
import sys; sys.path.insert(0, 'phc')
import isaacgym
from phc.utils.parse_task import parse_task
import argparse, omegaconf
# minimal stub — we just want to verify import + class instantiation path
print('parse_task import OK; HumanoidImResAMPVCmd is registered')
"
```

Expected: prints OK with no traceback.

- [ ] **Step 3: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: v_cmd observation extension"
```

---

### Task 5: Multi-clip + a-1 hard switch (port from v3 demo)

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Add clip selection helpers**

Append to `HumanoidImResAMPVCmd`:

```python
    def _midpoints(self) -> tuple[float, float]:
        """Return (mid_AB, mid_BC) for the 3-clip pool."""
        v = self._v_natural
        return float((v[0] + v[1]) * 0.5), float((v[1] + v[2]) * 0.5)

    def _select_clip_for_env(self, env_id: int) -> int:
        """Return desired clip idx for env_id given its v_cmd_ramped, with hysteresis."""
        v_cmd = float(self._v_cmd_ramped[env_id].item())
        cur = int(self._active_clip[env_id].item())
        mid_ab, mid_bc = self._midpoints()
        h = self.HYSTERESIS
        if cur == 0:
            return 1 if v_cmd > mid_ab + h else 0
        if cur == 1:
            if v_cmd < mid_ab - h: return 0
            if v_cmd > mid_bc + h: return 2
            return 1
        if cur == 2:
            return 1 if v_cmd < mid_bc - h else 2
        return cur
```

- [ ] **Step 2: Add atomic clip switch (per-env)**

```python
    def _apply_clip_switches(self):
        """For each env where desired != active, atomically:
          - update _sampled_motion_ids
          - reset _motion_start_times / _motion_start_times_offset
          - reset _global_offset to align new motion's t=0 root with current humanoid root
          - clear ref_motion_cache so next query is fresh
        Mirrors scripts/phc_walk_demo.py v3 _apply_pending_clip_switch."""
        switch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_clip = self._active_clip.clone()
        for env_id in range(self.num_envs):
            desired = self._select_clip_for_env(env_id)
            if desired != int(self._active_clip[env_id].item()):
                switch_mask[env_id] = True
                new_clip[env_id] = desired
        if not switch_mask.any():
            return
        ids = switch_mask.nonzero(as_tuple=False).reshape(-1)
        self._sampled_motion_ids[ids] = new_clip[ids]
        self._motion_start_times[ids] = 0.0
        self._motion_start_times_offset[ids] = 0.0
        self.progress_buf[ids] = 0
        # Re-sync _global_offset
        times = torch.zeros_like(self._motion_start_times[ids])
        root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids[ids], times)
        new_root = root_res["root_pos"]
        self._global_offset[ids, :2] = self._humanoid_root_states[ids, :2] - new_root[:, :2]
        self._global_offset[ids, 2] = 0.0
        if hasattr(self, "ref_motion_cache"):
            self.ref_motion_cache.clear()
        self._active_clip = new_clip
```

- [ ] **Step 3: Wire into pre_physics_step**

```python
    def pre_physics_step(self, actions):
        # Apply queued clip switches BEFORE the base step (so motion_lib reads
        # downstream see consistent motion_id + _global_offset).
        self._apply_clip_switches()
        return super().pre_physics_step(actions)
```

- [ ] **Step 4: Sanity check syntax**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: multi-clip + a-1 hard switch ported from v3 demo"
```

---

### Task 6: Per-step retime via _motion_start_times_offset

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Add retime accumulation**

In `pre_physics_step`, after `_apply_clip_switches`:

```python
    def pre_physics_step(self, actions):
        self._apply_clip_switches()
        # Per-step retime: motion advances at v_cmd_ramped/v_natural rate.
        v_natural_per_env = self._v_natural[self._active_clip]
        ratio = self._v_cmd_ramped / v_natural_per_env  # (num_envs,)
        self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return super().pre_physics_step(actions)
```

- [ ] **Step 2: Sanity check syntax**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: per-step retime via _motion_start_times_offset"
```

---

### Task 7: S2 v_cmd sampling + ramp logic

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Add per-episode init sampling**

```python
    def _reset_envs(self, env_ids):
        """Override: also sample new v_cmd target per reset env."""
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return
        new_v = torch.empty(len(env_ids), device=self.device).uniform_(self.V_CMD_MIN, self.V_CMD_MAX)
        self._v_cmd_target[env_ids] = new_v
        self._v_cmd_ramped[env_ids] = new_v   # start at target (no transient)
        # active_clip resampled based on initial v_cmd
        for ix, env_id in zip(range(len(env_ids)), env_ids.tolist()):
            self._active_clip[env_id] = self._select_clip_for_env(env_id)
```

Note: parent's `_reset_envs` may not exist with this exact name; fallback is `_reset_actors`. Verify by searching:

```bash
grep -n "def _reset_envs\|def _reset_actors\|def reset_idx" phc/env/tasks/humanoid_im.py phc/env/tasks/humanoid_amp.py phc/env/tasks/humanoid_amp_task.py phc/env/tasks/humanoid.py 2>/dev/null | head -10
```

If `_reset_envs` doesn't exist, override `reset_idx` (which all PHC tasks have) instead. Adjust the function name accordingly.

- [ ] **Step 2: Add per-step ramp + low-prob target trigger**

Insert at top of `pre_physics_step` (before `_apply_clip_switches`):

```python
    def _ramp_v_cmd(self):
        """S2: per-step ramp toward target + 1% prob trigger of new target."""
        # Ramp current → target with max_accel
        delta_max = self.V_CMD_MAX_ACCEL * self.dt
        diff = self._v_cmd_target - self._v_cmd_ramped
        step_size = torch.clamp(diff, -delta_max, delta_max)
        self._v_cmd_ramped = self._v_cmd_ramped + step_size
        # Per-step probabilistic re-sample of target
        trigger = (torch.rand(self.num_envs, device=self.device) < self.V_CMD_RAMP_PROB)
        if trigger.any():
            n = int(trigger.sum().item())
            new_targets = torch.empty(n, device=self.device).uniform_(self.V_CMD_MIN, self.V_CMD_MAX)
            self._v_cmd_target[trigger] = new_targets

    def pre_physics_step(self, actions):
        self._ramp_v_cmd()
        self._apply_clip_switches()
        v_natural_per_env = self._v_natural[self._active_clip]
        ratio = self._v_cmd_ramped / v_natural_per_env
        self._motion_start_times_offset += self.dt * (ratio - 1.0)
        return super().pre_physics_step(actions)
```

- [ ] **Step 3: Sanity check syntax**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: S2 v_cmd sampling + per-step ramp"
```

---

### Task 8: Reward computation r_track + r_im

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Add tracking + imitation anchor reward**

```python
    def _compute_reward(self, actions):
        """r_total = w_track·r_track + w_amp·r_amp_implicit + w_im·r_im + r_survive

        AMP reward (r_amp) is computed by the agent using the AMP discriminator
        and added downstream; here we only set r_track + r_im + r_survive into
        self.rew_buf. The agent will additively mix r_amp via disc_reward_w."""
        # v_actual = horizontal root speed (m/s)
        root_vel_xy = self._humanoid_root_states[:, 7:9]
        v_act = torch.linalg.norm(root_vel_xy, dim=-1)
        # Tracking reward — Gaussian kernel
        v_err = v_act - self._v_cmd_ramped
        r_track = torch.exp(-self.ALPHA_TRACK * v_err * v_err)
        # Imitation anchor — pose distance from reference frame
        # (parent already sets self._curr_ref_obs in observation pipeline)
        if hasattr(self, "_ref_body_pos") and self._ref_body_pos is not None:
            pose_diff = self._rigid_body_pos - self._ref_body_pos     # (num_envs, num_bodies, 3)
            pose_dist = torch.linalg.norm(pose_diff, dim=-1).mean(dim=-1)  # (num_envs,)
            r_im = torch.exp(-self.ALPHA_IM * pose_dist)
        else:
            r_im = torch.zeros_like(r_track)
        # Survive reward (1.0 base; killed downstream if env terminates)
        r_survive = torch.ones_like(r_track)
        # Combine. Note: r_amp is added by the agent via disc_reward_w, not here.
        self.rew_buf[:] = (self.W_TRACK * r_track
                           + self.W_IM * r_im
                           + r_survive)
        # Stash components for logging
        self._r_track = r_track.mean().detach()
        self._r_im = r_im.mean().detach()
```

- [ ] **Step 2: Sanity check syntax**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: r_track + r_im_anchor reward computation"
```

---

### Task 9: Termination conditions

**Files:**
- Modify: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`

- [ ] **Step 1: Override _compute_reset** (only if needed; PHC base may already cover)

```python
    def _compute_reset(self):
        """Standard PHC fall + tracking-distance termination, plus pelvis<0.4."""
        super()._compute_reset()
        # Spec §5.5: pelvis_z < 0.4 also terminates
        pelvis_z = self._humanoid_root_states[:, 2]
        fallen = pelvis_z < 0.4
        if fallen.any():
            self.reset_buf[fallen] = 1
            self._terminate_buf[fallen] = 1
```

- [ ] **Step 2: Sanity check syntax**

```bash
python -c "from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add phc/env/tasks/humanoid_im_res_amp_vcmd.py
git commit -m "res_amp_vcmd: termination conditions (fall + pelvis < 0.4)"
```

---

### Task 10: Residual MLP head module

**Files:**
- Create: `phc/learning/res_amp_network.py`

- [ ] **Step 1: Write residual head class**

```python
"""Residual + AMP network for v_cmd-conditioned policy on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ResMLPHead(nn.Module):
    """Small residual MLP: (proprio + v_cmd_norm) -> Δa (69-D).

    Architecture per spec §3:
      Linear(in_dim, 256) → SiLU → Linear(256, 256) → SiLU → Linear(256, action_dim)
    Sigma is parametric (learn_sigma=False, fixed sigma_init=-1.0 → std=0.37).
    """

    def __init__(self, in_dim: int, action_dim: int = 69,
                 hidden: tuple[int, int] = (256, 256),
                 sigma_init: float = -1.0):
        super().__init__()
        self.action_dim = action_dim
        self.mu = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], action_dim),
        )
        self.log_sigma = nn.Parameter(torch.full((action_dim,), float(sigma_init)),
                                       requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        sigma = self.log_sigma.exp().expand_as(mu)
        return mu, sigma
```

- [ ] **Step 2: Add unit test in same file (under `if __name__ == '__main__':`)**

```python
if __name__ == "__main__":
    head = ResMLPHead(in_dim=246, action_dim=69)
    x = torch.randn(2, 246)
    mu, sigma = head(x)
    assert mu.shape == (2, 69), mu.shape
    assert sigma.shape == (2, 69), sigma.shape
    assert sigma.abs().min() > 0
    n_params = sum(p.numel() for p in head.parameters())
    print(f"OK ResMLPHead, params={n_params}")
```

- [ ] **Step 3: Run test**

```bash
conda activate phc
python phc/learning/res_amp_network.py
```

Expected: `OK ResMLPHead, params=...` (around 130-150K).

- [ ] **Step 4: Commit**

```bash
git add phc/learning/res_amp_network.py
git commit -m "res_amp_vcmd: ResMLPHead module + sanity test"
```

---

### Task 11: ResAMPVCmdNetwork wrapper (frozen phc_3 + residual head)

**Files:**
- Modify: `phc/learning/res_amp_network.py`

- [ ] **Step 1: Add wrapper class**

```python
class ResAMPVCmdNetwork(nn.Module):
    """Wraps frozen phc_3 PNN + ResMLPHead. Output = a_base + clip(Δa, ±0.2).

    The frozen base (phc_3) is loaded externally from a state_dict. This wrapper
    just provides the forward path expected by rl_games' AmpAgent.
    """

    def __init__(self, frozen_base: nn.Module, residual_head: ResMLPHead,
                 delta_clip: float = 0.2):
        super().__init__()
        self.frozen_base = frozen_base
        for p in self.frozen_base.parameters():
            p.requires_grad = False
        self.frozen_base.eval()
        self.residual_head = residual_head
        self.delta_clip = float(delta_clip)

    def forward(self, obs: torch.Tensor, ref: torch.Tensor, v_cmd: torch.Tensor,
                proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) for action distribution."""
        with torch.no_grad():
            a_base, _ = self.frozen_base(obs, ref)   # (N, 69)
        # Residual: input is (proprio, v_cmd_norm)
        x = torch.cat([proprio, v_cmd.unsqueeze(-1) if v_cmd.dim() == 1 else v_cmd], dim=-1)
        delta_mu, delta_sigma = self.residual_head(x)
        # Clip Δa to keep residual bounded
        delta_mu = torch.clamp(delta_mu, -self.delta_clip, self.delta_clip)
        mu = a_base + delta_mu
        # Sigma comes from residual only (base is deterministic given ref).
        return mu, delta_sigma
```

- [ ] **Step 2: Update test block to exercise wrapper**

```python
if __name__ == "__main__":
    head = ResMLPHead(in_dim=246, action_dim=69)
    # Mock frozen base: a deterministic identity-ish stub
    class MockBase(nn.Module):
        def forward(self, obs, ref):
            return torch.zeros(obs.shape[0], 69), None
    base = MockBase()
    net = ResAMPVCmdNetwork(frozen_base=base, residual_head=head, delta_clip=0.2)
    obs = torch.randn(4, 1000)        # any
    ref = torch.randn(4, 800)         # any
    proprio = torch.randn(4, 245)
    v_cmd = torch.randn(4)            # normalized
    mu, sigma = net(obs, ref, v_cmd, proprio)
    assert mu.shape == (4, 69)
    assert sigma.shape == (4, 69)
    assert mu.abs().max() <= 0.2 + 1e-6, "Δa should be clipped"
    # Confirm frozen_base is in eval mode and has no grad
    for p in net.frozen_base.parameters():
        assert not p.requires_grad
    print("OK ResAMPVCmdNetwork")
```

- [ ] **Step 3: Run test**

```bash
python phc/learning/res_amp_network.py
```

Expected: `OK ResMLPHead ...` and `OK ResAMPVCmdNetwork`.

- [ ] **Step 4: Commit**

```bash
git add phc/learning/res_amp_network.py
git commit -m "res_amp_vcmd: ResAMPVCmdNetwork wrapper with frozen base + Δa clip"
```

---

### Task 12: phc_3 checkpoint loader

**Files:**
- Modify: `phc/learning/res_amp_network.py`

- [ ] **Step 1: Add loader function**

```python
def load_frozen_phc_3(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    """Load phc_3 PNN from checkpoint and return a frozen nn.Module.

    The checkpoint is the standard rl_games / PHC im_pnn_big format
    (state_dict under 'model' key, with PNN actors / critic / disc).
    """
    import torch
    from rl_games.algos_torch.network_builder import NetworkBuilder
    # Use rl_games' im_amp builder to recreate the architecture
    from phc.learning.amp_network_pnn import AMPPNNBuilder
    builder = AMPPNNBuilder()
    builder.load({
        "mlp": {"units": [2048, 1536, 1024, 1024, 512, 512],
                "activation": "silu",
                "d2rl": False,
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"}},
        "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None",
                                   "mu_init": {"name": "default"},
                                   "sigma_init": {"name": "const_initializer", "val": -2.9},
                                   "fixed_sigma": True, "learn_sigma": False}},
        "disc": {"units": [1024, 512], "activation": "relu",
                  "initializer": {"name": "default"}},
        "separate": True, "discrete": False,
        "num_actors": 3,
    })
    # Network needs an obs/action shape — we'll pass canonical phc_3 shapes
    net = builder.build("amp_pnn", actions_num=69, input_shape=(934,), num_seqs=1, value_size=1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state["model"] if "model" in state else state
    # Drop 'a2c_network.' prefix if present (rl_games convention)
    sd = {k.replace("a2c_network.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[load_frozen_phc_3] missing={len(missing)} unexpected={len(unexpected)}")
    net = net.to(device).eval()
    return net
```

- [ ] **Step 2: Add loader test (gated on file existence)**

Append to `if __name__ == "__main__":` block:

```python
    import os
    ckpt = "output/HumanoidIm/phc_3/Humanoid.pth"
    if os.path.exists(ckpt):
        print(f"[loader test] loading {ckpt} ...")
        try:
            base = load_frozen_phc_3(ckpt, device="cpu")
            n_p = sum(p.numel() for p in base.parameters())
            print(f"OK load_frozen_phc_3, params={n_p}")
        except Exception as e:
            print(f"WARN loader test failed: {e}")
    else:
        print(f"[loader test] skipped (no {ckpt})")
```

- [ ] **Step 3: Run test**

```bash
python phc/learning/res_amp_network.py
```

Expected: prints `OK load_frozen_phc_3, params=N` where N is in the millions. If `WARN` appears, the AMPPNNBuilder import path or state_dict prefix needs adjustment — fix and re-run before committing.

- [ ] **Step 4: Commit**

```bash
git add phc/learning/res_amp_network.py
git commit -m "res_amp_vcmd: phc_3 frozen-base checkpoint loader"
```

---

### Task 13: Network registration with rl_games

**Files:**
- Modify: `phc/learning/res_amp_network.py`

- [ ] **Step 1: Add Builder class for rl_games registry**

```python
class ResAMPVCmdBuilder:
    """rl_games-style network builder. Reads cfg from the learning yaml's
    `params.network` block and returns a ResAMPVCmdNetwork instance."""

    def __init__(self):
        self.cfg = None

    def load(self, cfg: dict):
        self.cfg = cfg

    def build(self, name: str, **kwargs):
        # Frozen base
        base_cfg = self.cfg["frozen_base"]
        base = load_frozen_phc_3(base_cfg["checkpoint"], device=kwargs.get("device", "cuda:0"))
        # Residual head
        res_cfg = self.cfg["residual"]
        in_dim = kwargs["input_shape"][0] + 1  # proprio + v_cmd_norm (already in obs)
        head = ResMLPHead(in_dim=in_dim, action_dim=kwargs["actions_num"],
                          hidden=tuple(res_cfg["mlp_units"]),
                          sigma_init=float(res_cfg["sigma_init"]))
        net = ResAMPVCmdNetwork(frozen_base=base, residual_head=head, delta_clip=0.2)
        return net

    def __call__(self, **kwargs):
        return self.build("amp_pnn_residual", **kwargs)
```

- [ ] **Step 2: Register builder via PHC's network registry**

```bash
grep -n "model_builder\|register_network\|register_builder" phc/learning/*.py | head -10
```

Locate where PHC registers its existing networks; add a parallel registration line for `amp_pnn_residual` pointing to `ResAMPVCmdBuilder`.

If PHC uses rl_games' `model_builder.register_network(name, builder)`, add:

```python
# At the top of any module imported during run_hydra startup, e.g.,
# phc/learning/im_amp.py or phc/learning/amp_models.py
from rl_games.algos_torch import model_builder
from phc.learning.res_amp_network import ResAMPVCmdBuilder
model_builder.register_network('amp_pnn_residual', ResAMPVCmdBuilder)
```

(Exact insertion point depends on existing pattern; mirror existing `amp_pnn` registration.)

- [ ] **Step 3: Sanity check**

```bash
python -c "
from phc.learning.res_amp_network import ResAMPVCmdBuilder
b = ResAMPVCmdBuilder()
print('OK ResAMPVCmdBuilder importable')
"
```

Expected: `OK ResAMPVCmdBuilder importable`

- [ ] **Step 4: Commit**

```bash
git add phc/learning/res_amp_network.py phc/learning/im_amp.py
git commit -m "res_amp_vcmd: register ResAMPVCmdBuilder with rl_games"
```

---

### Task 14: ResAMPVCmdAgent class skeleton

**Files:**
- Create: `phc/learning/res_amp_agent.py`

- [ ] **Step 1: Write agent that inherits AmpAgent**

```python
"""Residual + AMP training agent — inherits AmpAgent, freezes phc_3 base.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
from phc.learning.amp_agent import AmpAgent


class ResAMPVCmdAgent(AmpAgent):
    """AMP agent where only the residual head + critic + AMP discriminator
    are trainable. The phc_3 PNN actor parameters are frozen at load time."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        # Freeze the frozen_base submodule of the network
        net = self.model.a2c_network
        if hasattr(net, "frozen_base"):
            for p in net.frozen_base.parameters():
                p.requires_grad = False
            net.frozen_base.eval()
            n_frozen = sum(p.numel() for p in net.frozen_base.parameters())
            n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"[ResAMPVCmdAgent] frozen={n_frozen}  trainable={n_train}")
        else:
            print("[ResAMPVCmdAgent] WARN: net has no .frozen_base attribute")
```

- [ ] **Step 2: Register agent**

Find existing PHC agent registration (similar to network registration above):

```bash
grep -n "register_algo\|im_amp.*register\|builders\[" phc/learning/*.py | head -10
```

Add a parallel registration:

```python
# In the file that registers `im_amp` (likely phc/learning/im_amp.py)
from phc.learning.res_amp_agent import ResAMPVCmdAgent
# rl_games agents are registered via the runner / algo_observer pattern;
# follow the same pattern as `im_amp`:
from rl_games.algos_torch import torch_runner
torch_runner.Runner._algo_factory.register_builder('im_amp_residual',
                                                    lambda **kwargs: ResAMPVCmdAgent(**kwargs))
```

(Adjust to match existing PHC pattern.)

- [ ] **Step 3: Smoke import**

```bash
python -c "from phc.learning.res_amp_agent import ResAMPVCmdAgent; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add phc/learning/res_amp_agent.py phc/learning/im_amp.py
git commit -m "res_amp_vcmd: ResAMPVCmdAgent + algo registration"
```

---

### Task 15: Param freezing verification

**Files:**
- Modify: `phc/learning/res_amp_agent.py`

- [ ] **Step 1: Add post-init verification + grad-check helper**

```python
    def calc_gradients(self, input_dict):
        """Override to assert phc_3 base never gets gradients."""
        super().calc_gradients(input_dict)
        # Sanity: frozen base must not have grads
        net = self.model.a2c_network
        if hasattr(net, "frozen_base"):
            for n, p in net.frozen_base.named_parameters():
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    raise RuntimeError(
                        f"[ResAMPVCmdAgent] frozen_base param {n} got non-zero grad!")
```

- [ ] **Step 2: Smoke import**

```bash
python -c "from phc.learning.res_amp_agent import ResAMPVCmdAgent; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add phc/learning/res_amp_agent.py
git commit -m "res_amp_vcmd: assert frozen base receives no gradients"
```

---

### Task 16: Local smoke launch script

**Files:**
- Create: `scripts/launch_res_amp_smoke.sh`
- Create: `exp_config/forward_walking/260501_RES_AMP_VCMD/env_im_res_amp_vcmd.yaml`
- Create: `exp_config/forward_walking/260501_RES_AMP_VCMD/im_res_amp_vcmd.yaml`

- [ ] **Step 1: Snapshot configs to exp_config**

```bash
mkdir -p exp_config/forward_walking/260501_RES_AMP_VCMD
cp phc/data/cfg/env/env_im_res_amp_vcmd.yaml      exp_config/forward_walking/260501_RES_AMP_VCMD/
cp phc/data/cfg/learning/im_res_amp_vcmd.yaml     exp_config/forward_walking/260501_RES_AMP_VCMD/
```

- [ ] **Step 2: Write smoke launch script**

```bash
# scripts/launch_res_amp_smoke.sh
#!/usr/bin/env bash
# Local smoke run on RTX 4060 Ti (~6h, num_envs=64, max_epochs=500).
# Pass criteria: spec §8 Stage 1.

set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phc

LOG_DIR="logs/res_amp_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

python phc/run.py \
  --task HumanoidImResAMPVCmd \
  --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd.yaml \
  --cfg_train phc/data/cfg/learning/im_res_amp_vcmd.yaml \
  --headless \
  --num_envs 64 \
  2>&1 | tee "$LOG_DIR/train.out"
```

```bash
chmod +x scripts/launch_res_amp_smoke.sh
```

- [ ] **Step 3: Smoke-launch (early-exit after one boot to verify config + import path)**

```bash
timeout 60 bash scripts/launch_res_amp_smoke.sh 2>&1 | tee /tmp/smoke_init.log
```

Expected: motion_lib loads 3 motions, "frozen=N trainable=M" line prints, training loop starts (epoch 0 reward log appears) before timeout fires.

If errors appear before training loop, fix and re-run. Common gotchas:
- AMPPNNBuilder import name wrong (Task 12)
- Builder not registered (Task 13)
- Agent not registered (Task 14)
- v_cmd obs dim mismatch in network input_shape

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_res_amp_smoke.sh exp_config/forward_walking/260501_RES_AMP_VCMD/
git commit -m "res_amp_vcmd: local smoke launch script + config snapshot"
```

---

### Task 17: Run smoke (user)

**Files:**
- Read-only: `logs/res_amp_smoke_*/train.out`

- [ ] **Step 1: Launch full smoke (~6h)**

```bash
nohup bash scripts/launch_res_amp_smoke.sh > /dev/null 2>&1 &
echo "smoke PID=$!"
```

- [ ] **Step 2: Check progress every ~1h**

```bash
ls -ltr logs/res_amp_smoke_* | tail -3
tail -20 logs/res_amp_smoke_*/train.out | tail -5
```

Look for:
- av_reward, av_steps printed every ~50 epochs
- AMP disc accuracy (around 60-80% means balanced)
- No NaN, no crashes

- [ ] **Step 3: Confirm pass criteria** (spec §8 Stage 1)

Pass requires:
- av_episode_length ≥ 240 step (=80% of 300)
- av_reward up ≥ 30% from epoch 0
- AMP disc loss stable (not exploding/collapsing)
- No OOM

If any fails, stop and diagnose before Stage 2.

---

### Task 18: Smoke result analysis doc

**Files:**
- Create: `02_research_dev/260501_res_amp_vcmd_smoke_analysis.md`

- [ ] **Step 1: Write analysis (Korean per project convention)**

Template:

```markdown
# RES_AMP_VCMD smoke 결과 분석 (260501)

## 환경
- 호스트: exolab-MS-7D56 (RTX 4060 Ti 7.6 GB)
- num_envs=64, episode_length=300, max_epochs=500
- 학습 시간: <wall-clock>
- Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
- Plan: docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md

## 통과 기준 (spec §8 Stage 1)
- [ ] av_episode_length ≥ 240 step
- [ ] av_reward 가 epoch 0 대비 +30% 이상
- [ ] AMP disc accuracy 50-80% 안정
- [ ] NaN / OOM 없음

## 결과
... (TBD by user after smoke)

## 다음 단계
- Stage 2 (서버 sbatch) 진행 가능 여부
- 발견된 이슈 / hyperparam 수정 필요 사항
```

- [ ] **Step 2: Commit**

```bash
git add 02_research_dev/260501_res_amp_vcmd_smoke_analysis.md
git commit -m "doc(260501): RES_AMP_VCMD smoke analysis template"
```

---

### Task 19: Server sbatch script

**Files:**
- Create: `slurm/train_res_amp_vcmd.sbatch`

- [ ] **Step 1: Inspect existing sbatch templates**

```bash
ls slurm/ 2>/dev/null && cat slurm/train_*.sbatch 2>/dev/null | head -40
```

If no existing sbatch, base on a working VIC4_VCMD invocation from `exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/`.

- [ ] **Step 2: Write sbatch**

```bash
# slurm/train_res_amp_vcmd.sbatch
#!/usr/bin/env bash
#SBATCH --job-name=res_amp_vcmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=idx0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/res_amp_vcmd_%j.out
#SBATCH --error=logs/slurm/res_amp_vcmd_%j.err

cd /home/$USER/PHC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phc

python phc/run.py \
  --task HumanoidImResAMPVCmd \
  --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd.yaml \
  --cfg_train phc/data/cfg/learning/im_res_amp_vcmd.yaml \
  --headless \
  --num_envs 512 \
  --max_epochs 20000 \
  --episode_length 600
```

Note: `--max_epochs` and `--episode_length` flags may need to be passed via Hydra overrides depending on PHC's argparse setup. Verify with:

```bash
python phc/run.py --help 2>&1 | grep -E "max_epochs|episode_length|num_envs"
```

If those flags don't exist, override via env yaml temp copy or Hydra override syntax.

- [ ] **Step 3: Commit**

```bash
git add slurm/train_res_amp_vcmd.sbatch
git commit -m "res_amp_vcmd: server sbatch for full training (Stage 2)"
```

---

### Task 20: Stage 2 server training (user)

**Files:**
- Read-only: `logs/slurm/res_amp_vcmd_*.out`

- [ ] **Step 1: Push branch and SSH to server**

```bash
git push origin Jimin
ssh server1
cd ~/PHC
git fetch && git checkout Jimin && git pull
```

- [ ] **Step 2: Submit sbatch**

```bash
sbatch slurm/train_res_amp_vcmd.sbatch
squeue -u $USER
```

- [ ] **Step 3: Monitor every ~6h**

```bash
tail -50 logs/slurm/res_amp_vcmd_*.out | tail -20
```

Watch for:
- av_episode_length climbing to 580+
- v_cmd tracking error decreasing
- AMP disc balanced

- [ ] **Step 4: Pull final checkpoint when done**

```bash
# from local
scp server1:~/PHC/output/HumanoidIm/ResAMPVCmd/Humanoid.pth output/HumanoidIm/ResAMPVCmd/
```

---

### Task 21: Eval script (sweep + dynamic ramp)

**Files:**
- Create: `scripts/eval_res_amp_vcmd.py`

- [ ] **Step 1: Write eval script**

```python
"""Evaluate trained RES_AMP_VCMD policy.

Three sub-tests per spec §9:
  1. Static v_cmd sweep (9 levels × 5s each, 45s total)
  2. Dynamic v_cmd ramp (60s scripted)
  3. (manual qualitative video — separate)

Pass: |v_act_avg − v_cmd| < 0.10 m/s for all levels, 0 falls,
       settling time < 2.5s for ramp transitions.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phc"))
os.chdir(_ROOT)

import isaacgym  # noqa: F401
import numpy as np
import torch


SWEEP_LEVELS = [0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18, 1.23]
SWEEP_DURATION_S = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/HumanoidIm/ResAMPVCmd/Humanoid.pth")
    ap.add_argument("--out", default="02_research_dev/260501_res_amp_vcmd_eval_results.json")
    args = ap.parse_args()

    # Load model + env via run_hydra in test mode (mirrors phc_walk_demo_residual.py
    # boot pattern). Then drive v_cmd via env.task._v_cmd_target injection per the
    # SWEEP_LEVELS schedule, recording v_actual each step.

    # Implementation sketch (full impl in commit):
    #  1. Boot env (NUM_ENVS=4, headless=False or True per --no_render flag)
    #  2. Load checkpoint into ResAMPVCmdNetwork via builder pattern
    #  3. For each level in SWEEP_LEVELS:
    #       env.task._v_cmd_target.fill_(level)
    #       env.task._v_cmd_ramped.fill_(level)
    #       run env_step for SWEEP_DURATION_S × FPS steps
    #       record v_actual every step
    #  4. Compute |v_actual_avg - level| per level, 0/1 fall flag per level
    #  5. Dump JSON to args.out
    raise NotImplementedError("filled in by subagent during this task")


if __name__ == "__main__":
    main()
```

(The implementer subagent will complete the boot/loop/JSON dump using `scripts/record_phc_pretrained.py` as the boot template.)

- [ ] **Step 2: Run eval (after Stage 2 ckpt available)**

```bash
python scripts/eval_res_amp_vcmd.py --ckpt output/HumanoidIm/ResAMPVCmd/Humanoid.pth
cat 02_research_dev/260501_res_amp_vcmd_eval_results.json
```

Expected: JSON with per-level avg v_actual within 0.10 of v_cmd, 0 falls.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_res_amp_vcmd.py
git commit -m "res_amp_vcmd: eval script — static sweep + dynamic ramp"
```

---

### Task 22: Inference demo (residual on top of v3 demo)

**Files:**
- Create: `scripts/phc_walk_demo_residual.py`

- [ ] **Step 1: Copy v3 demo as base**

```bash
cp scripts/phc_walk_demo.py scripts/phc_walk_demo_residual.py
```

- [ ] **Step 2: Modify to load trained residual**

Key edits in `scripts/phc_walk_demo_residual.py`:

```python
# At top: change EXP_NAME, LEARNING, ENV_CFG_NAME
EXP_NAME = "ResAMPVCmd"                                 # was "phc_3"
LEARNING = "im_res_amp_vcmd"                            # was "im_pnn_big"
ENV_CFG_NAME = "env_im_res_amp_vcmd"                    # was "env_im_pnn"

# Remove _maybe_switch_clip / _apply_pending_clip_switch monkey-patches:
# the trained network handles v_cmd internally; no clip-switching plumbing
# needed in the demo wrapper.

# Keep the v_cmd panel I/O, keyboard handlers, recording logic.
```

- [ ] **Step 3: Smoke run (no v_cmd push)**

```bash
timeout 30 python scripts/phc_walk_demo_residual.py 2>&1 | head -40
```

Expected: boots without crash, prints `[demo] step= ...` lines, walks at v_cmd=1.0.

- [ ] **Step 4: Commit**

```bash
git add scripts/phc_walk_demo_residual.py
git commit -m "res_amp_vcmd: inference demo — trained residual on phc_3 baseline"
```

---

### Task 23: Manual smoke test by user

**Files:**
- Read-only: visual feedback from IsaacGym viewer

- [ ] **Step 1: Launch demo**

```bash
conda activate phc
python scripts/phc_walk_demo_residual.py
```

- [ ] **Step 2: User exercises panel + keyboard**

User should:
- Slide v_cmd from 0.78 → 1.23 → 0.78 via panel slider, observe smooth v_actual tracking.
- Press ↑↓ keys to nudge v_cmd ±0.02, observe gait response.
- Press 1, 5, 9 keys to snap to extremes and middle, observe transitions.

Pass = "looks smooth, no falls, v_actual tracks v_cmd within ±0.10 m/s by eye, gait stays natural-looking."

- [ ] **Step 3: Record 30s video for archival**

```bash
python scripts/phc_walk_demo_residual.py --record_seconds 30 --out_name res_amp_vcmd_smoke
```

Output: `videos/res_amp_vcmd_smoke.mp4`

- [ ] **Step 4: Commit video reference (do NOT commit binary; just doc reference)**

```bash
ls -lh videos/res_amp_vcmd_smoke.mp4
```

(Add a line referencing the video in the smoke analysis doc from Task 18.)

---

### Task 24: Final method documentation

**Files:**
- Create: `02_research_dev/260501_res_amp_vcmd_method.md`

- [ ] **Step 1: Write method doc (Korean per project convention)**

```markdown
# RES_AMP_VCMD: PHC + Residual + AMP for Continuous v_cmd (260501)

## 한 줄 요약
Frozen phc_3 + small residual MLP head + AMP discriminator 학습으로
연속 v_cmd ∈ [0.76, 1.23] m/s 추적 정책 구축.

## 학습 결과
- Stage 1 (로컬 smoke): <pass/fail summary>
- Stage 2 (서버 본학습): <pass/fail summary>, <epochs>, <wall-clock>
- Eval (script): <sweep result table>, <ramp result>
- 데모: videos/res_amp_vcmd_smoke.mp4

## 아키텍처
... (참고: spec §3)

## 비교: v3 데모 vs 학습된 정책
- v3 (zero-training, hard switch): ...
- RES_AMP_VCMD (학습): ...

## 후속 (spec §13)
- 2D / 3D 명령 확장 (vy, ωz)
- 클립 풀 확장 (5-7 클립)
- Distillation (γ): reference 없이 동작하는 student
```

- [ ] **Step 2: Push everything**

```bash
git add 02_research_dev/260501_res_amp_vcmd_method.md
git commit -m "doc(260501): RES_AMP_VCMD method + results consolidation"
git push origin Jimin
```

---

## Self-Review Checklist (run before handoff)

- [ ] Spec §1 goal covered: trained policy with continuous v_cmd ∈ [0.76, 1.23]. → Tasks 14-20.
- [ ] Spec §2 7 decisions all covered:
  - 1D vx → env yaml v_cmd_min/max (Task 2)
  - Multi-clip + retime → Tasks 5, 6
  - a-1 hard switch → Task 5
  - R2 reward → Task 8 + learning yaml (Task 3)
  - N1 output-residual → Tasks 10, 11
  - S2 sampling → Task 7
  - C2 local + server → Tasks 16-20
- [ ] Spec §3 architecture diagram: matches Tasks 10-15 (network + agent).
- [ ] Spec §4 6 new files all in plan: humanoid_im_res_amp_vcmd.py (Tasks 1, 4-9), res_amp_network.py (Tasks 10-13), res_amp_agent.py (Tasks 14-15), env yaml (Task 2), learn yaml (Task 3), demo (Task 22).
- [ ] Spec §5 reward formulas: Task 8 implements r_track + r_im, agent (Task 14) handles r_amp via AmpAgent inheritance.
- [ ] Spec §6 v_cmd sampling: Task 7 implements S2.
- [ ] Spec §7 AMP clip pool = 3 clips: handled implicitly via motion_file in env yaml (Task 2).
- [ ] Spec §8 stages: Stage 1 = Tasks 16-18, Stage 2 = Tasks 19-20.
- [ ] Spec §9 eval criteria: Task 21 (sweep + ramp), Task 23 (qualitative).
- [ ] Spec §10 compute: Task 16 local 64 envs, Task 19 server 512 envs.
- [ ] Spec §11 trade-offs: documentation-only, no task needed.
- [ ] Spec §12 risks: anchor reward (Task 8), action clip (Task 11), assert no grad (Task 15) — covered.
- [ ] Spec §13 out of scope: explicitly noted in Task 24.
- [ ] No "TBD" / "fill in later" in any task body — verified.
- [ ] Type consistency: ResMLPHead.forward returns (mu, sigma), ResAMPVCmdNetwork.forward returns (mu, sigma), agent reads via standard rl_games path — consistent.

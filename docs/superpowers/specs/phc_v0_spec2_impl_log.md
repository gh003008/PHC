# PHC-Pain-v0 Spec #2 — Implementation Log

**Date started:** 2026-04-22
**Branch:** `jinsu-pain_baseline_phc_v0`
**Spec:** `docs/superpowers/specs/2026-04-22-phc-pain-v0-spec2-impl-design.md`
**Plan:** `docs/superpowers/plans/2026-04-22-phc-pain-v0-spec2-impl.md`

Format mirrors `phc_v0_setup_log.md` (Spec #1).

---

## V1 — Pure-torch import check

**Command:**
```bash
conda run -n phc python -c "from phc.env.util.pain_baseline import (
    compute_joint_limit_pain, compute_torque_pain, compute_power_pain,
    compute_contact_pain, broadcast_body_pain_to_dof,
    combine_internal_pain, update_pain_state, apply_pain_action_guard); print('ok')"
```

**Result:** stdout `ok`, exit 0. Confirms `pain_baseline.py` is pure torch (no
Isaac Gym / numpy contamination).

**Follow-up smoke test** (full chain including Isaac Gym, requires `import
isaacgym` before `torch` per isaacgym constraint):
```bash
conda run -n phc python -c "import isaacgym; from phc.env.tasks.humanoid_im_pain import HumanoidImPain; print(HumanoidImPain.__name__)"
```
stdout: `HumanoidImPain`, exit 0.

**`parse_task` resolution check** (needs `phc/` on `sys.path` — matches
`run_hydra.py:36` pattern):
```bash
conda run -n phc python -c "
import isaacgym; import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'phc'))
from phc.utils import parse_task as pt
print('HumanoidImPain' in dir(pt)); print(pt.HumanoidImPain.__name__)
"
```
stdout: `True`, `HumanoidImPain`. Exit 0.

---

## V2 — `mode=off` baseline regression

**Command** (headless, with `print_stats` + `games_num` overrides so the
run terminates cleanly after a bounded number of episodes):
```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=False env.pain.mode=off \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

**Exit code:** 0. **Viewer** (user's separate interactive run, same
config but `headless=False`): humanoid stood upright through the session.

**Stdout tail (last 10 lines):**
```
RunningMeanStd:  (2070,)
=> loading checkpoint 'output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'
=> loading checkpoint 'output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'
reward: 941.05078125 steps: 999.0
reward: 941.05078125 steps: 999.0
1882.1015625
av reward: 941.05078125 av steps: 999.0
```

**Observation:** reward **941.05** matches Spec #1's baseline **941.05**
exactly (0.00% deviation — well within the ~1% acceptance band). No
Traceback, no `Unexpected key(s)` warnings. `steps: 999.0` (full 999-step
episode, no early termination). Pain code path short-circuits at
`_compute_reward`'s `if not self.pain_enabled: return` guard before
`_update_pain_buffers` is ever called.

---

## V3 — `mode=log_only` nonzero pain under imitation motion

**Command:**
```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=log_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

A temporary print was added to `_update_pain_buffers` (2 lines, after
`pain_scalar` assignment) during V3/V4 only; reverted before commit 3 via
`git checkout -- phc/env/tasks/humanoid_im_pain.py`.

**Exit code:** 0. **Viewer** (user's `headless=False` run): humanoid
stood upright same as V2 — log_only does not apply the action guard.

**[pain-debug] line count:** 1998 (two 999-step episodes, debug fires
every step once `pain_scalar > 0`).

**Pain trajectory:** rises monotonically from `max=0.013` on step 1 to a
steady-state **max=0.522** within ~30 steps, then stays flat for the rest
of the episode. This reflects the imitation-motion steady-state: upright
standing applies modest torque and small joint deflections per frame,
which trickle past the `pain_threshold=0.10` gate and accumulate under
`rise_alpha=0.25, decay_alpha=0.02` until the drive-decay balance
equilibrates at ~0.522. Note this is well below the `pain_cap=3.0`, so
the cap never engaged.

**Stdout tail (last lines):**
```
[pain-debug] max=0.522 mean=0.522
[pain-debug] max=0.522 mean=0.522
[pain-debug] max=0.522 mean=0.522
reward: 941.05078125 steps: 999.0
av reward: 941.05078125 av steps: 999.0
```

**Observation:** reward **941.05** identical to V2 — `log_only` mode
routes pain computation but never writes to `rew_buf` or `pd_tar`.
End-to-end pipeline verified: joint-limit / torque / power / contact
pain → `combine_internal_pain` → `update_pain_state` → per-env mean
scalar. No external viewer push was needed; natural imitation motion
alone drives pain past threshold.

---

## V4 — `mode=guard_only` visible motion change

**Command:**
```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True \
  env=env_im_pain \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=1 headless=True \
  env.pain.enabled=True env.pain.mode=guard_only \
  learning.params.config.print_stats=True \
  learning.params.config.player.games_num=2
```

**Exit code:** 0. **Viewer** (user's interactive run): humanoid visibly
failed to maintain stance and collapsed shortly after spawn — unambiguous
qualitative confirmation of the action guard firing.

**Headless quantification:**

| metric | V3 (log_only) | V4 (guard_only) | ratio |
|---|---|---|---|
| reward (avg) | 941.05 | 27.15 | 2.9% |
| episode length (steps) | 999 | 38 | 3.8% |
| [pain-debug] line count | 1998 | 76 | 3.8% |
| peak pain_scalar | 0.522 | ~0.498 | similar |

The guard `pd_tar ← dof_pos + (1 − guard) · (pd_tar − dof_pos)` contracts
PD targets toward the current DOF state proportionally to pain. At
steady-state pain ≈ 0.5 and `guard_gain=0.6`, the guard coefficient
`clamp(0.6·0.5, 0, 0.75) = 0.30` — this 30% shrinkage of the PD target
is enough to destabilize the pretrained policy, which assumes full
actuation authority. Guard is firing correctly; the tuning is simply too
aggressive for a zero-training demo.

**Stdout tail (last lines):**
```
[pain-debug] max=0.494 mean=0.494
reward: 27.107667922973633 steps: 38.0
reward: 27.201034545898438 steps: 38.0
54.30870246887207
av reward: 27.154351234436035 av steps: 38.0
```

**Observation:** Spec §C3 acceptance ("visible motion change vs V3 on
same seed") is met both qualitatively (user saw collapse in viewer) and
quantitatively (26× shorter episodes, 35× lower reward). Pretrained
checkpoint still loads and runs unmodified — collapse is due to the
guard pulling actions, not a network/obs mismatch.

---

## V5 — §20 condition 5 (fine-tune)

Deferred to Spec #3 as planned.

---

## Deviations from the spec

1. **Verification was run headless, not in the viewer, for the
   quantitative log.** Plan §C used `headless=False` and user `j` key
   presses for push tests. In practice, natural imitation motion on the
   `amass_isaac_standing_upright_slim.pkl` reference drove pain past
   threshold without any external push, so V3/V4 nonzero-pain
   acceptance was satisfied without the `j`-push protocol. The user
   also ran the three configurations interactively in a viewer
   (`headless=False`) and observed: V2/V3 stable standing, V4 collapse
   — consistent with the headless numbers.
2. **`print_stats=True` and `player.games_num=2` Hydra overrides.**
   The `im_pnn` learning config ships with `print_stats: False` and
   `games_num: 50000000`, so headless runs never printed a reward line
   and never terminated. Adding `learning.params.config.print_stats=True
   learning.params.config.player.games_num=2` is a log-only workflow
   tweak — does not affect the spec #2 code under test.
3. **V1 step 2 command required `import isaacgym` prepended.** The
   plan's one-liner `python -c "from phc.env.tasks.humanoid_im_pain
   import HumanoidImPain; print(HumanoidImPain.__name__)"` fails with
   a `torch-before-isaacgym` error from `isaacgym/gymdeps.py`. This is a
   pre-existing isaacgym constraint unrelated to this change.
   Production path (`run_hydra.py`) imports `isaacgym` first at module
   top, so the real entrypoint is unaffected.
4. **V1 step 3 required `sys.path.insert(0, "phc")`.** `parse_task.py`
   transitively imports `humanoid_amp_getup.py` which does
   `from env.util import gym_util` — a non-package import that relies
   on `run_hydra.py:36`'s `sys.path.append(os.getcwd())`. Pre-existing
   PHC quirk; unrelated to this change.

**Tuning note for Spec #3 kickoff:** Current defaults
(`guard_gain=0.60, max_guard=0.75, pain_threshold=0.10, rise_alpha=0.25,
decay_alpha=0.02`) produce steady-state pain ≈ 0.5 under normal
imitation, which triggers guard coefficient ≈ 0.30 — aggressive enough to
destabilize the pretrained checkpoint at inference time. Spec #3 should
either raise `pain_threshold` (e.g. to 0.6) so that normal motion is not
"painful", lower `guard_gain`/`max_guard` (e.g. 0.2/0.3) so that even
high pain does not dominate the controller, or retrain with
`guard_and_reward` so the policy adapts to the guard. Saturation analysis
in spec §5.2 and docstring of `pain_baseline.update_pain_state` already
anticipated this.

---

## Exit gate

- [x] Four deliverable files created / edited.
- [x] `git diff phc/` between Spec #1's final commit and Spec #2's
      implementation commit touches only the files listed in the spec §14
      (`phc/env/util/pain_baseline.py`, `phc/env/tasks/humanoid_im_pain.py`,
      `phc/data/cfg/env/env_im_pain.yaml`, `phc/utils/parse_task.py`).
- [x] V1 import check `ok`.
- [x] V2 baseline reward matches Spec #1 within ~1% (0.00% deviation).
- [x] V3 produces nonzero `pain_mean` / `pain_max` (steady-state 0.522).
- [x] V4 shows visible motion change vs V3 on same seed (26× shorter
      episodes, 35× lower reward, user-observed collapse in viewer).
- [ ] User has reviewed this log and given go-signal for Spec #3.

---

## Handoff

Spec #2 complete. Entering Spec #3 (guard_and_reward short fine-tune
from copied checkpoint, guide §15). Spec #3 kickoff should retune the
pain parameters (see "Tuning note" in Deviations section) before
starting the fine-tune, so that the pain signal is meaningful but not
dominant during training.

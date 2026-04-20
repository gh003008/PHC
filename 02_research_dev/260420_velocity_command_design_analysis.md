# Velocity-Command VIC — Design Analysis & Next-Step Options

**Date**: 2026-04-20
**Context**: PHC + VIC (Variable Impedance Control) humanoid. Single-clip forward walking imitation (AMASS). We added a velocity command to the policy so the humanoid can be steered at test time. After two rounds of experiments, we're evaluating what's wrong with the current design and how to fix it.

---

## 1. What the velocity command is RIGHT NOW (factual)

Task class: `HumanoidImVICCmd` (extends `HumanoidImVIC`). Relevant code (`phc/env/tasks/humanoid_im_vic_cmd.py`, lines 37–74):

```python
def _resample_cmd(self, env_ids):
    ...
    self._current_cmd[env_ids, 0] = torch.rand(n) * (v_hi - v_lo) + v_lo   # U(0.8, 1.3) m/s
    self._current_cmd[env_ids, 1] = 0.0
    self._current_cmd[env_ids, 2] = torch.rand(n) * (w_hi - w_lo) + w_lo   # U(0, 0) — fixed
```

- `v_cmd_x` is sampled **uniformly from [0.8, 1.3] m/s at each env reset** and held fixed for the whole episode.
- `v_cmd_y` = 0, `w_cmd_yaw` = 0 (single forward clip — no turning).
- Appended to task obs as 3 extra dims.
- Reward blend: `rew = (1 - w) * base_imitation_rew + w * cmd_track_rew`,
  where `cmd_track_rew = exp(-k_v · ||v_pelvis_xy − v_cmd_xy||² − k_w · |ω_pelvis_yaw − ω_cmd|²)` (k_v=2, k_w=1).

**Key point: v_cmd is an arbitrary random scalar. It is NOT derived from the reference motion.**

The reference motion (`amass_isaac_walking_forward_single.pkl`, 5.43 s, single clip) has its own fixed natural pelvis velocity (walking gait). v_cmd is drawn independently, so when v_cmd ≠ motion's natural speed, the two differ.

---

## 2. The "collision" intuition — confirmed

The base imitation reward (`HumanoidIm`) scores how well the humanoid's pose **and body velocities** match the reference motion at each timestep. The motion has one natural walking speed. The cmd reward scores how well pelvis velocity matches v_cmd.

When v_cmd differs from the motion's natural speed, those two rewards pull in opposite directions:

- To maximize imitation → match motion's velocity (fixed).
- To maximize cmd_track → match commanded velocity (random).

The policy compromises between them, weighted by `cmd_tracking_w` (`w`).

---

## 3. Evidence from experiments (Round 1 test-greedy)

All runs: same motion, VIC 4-group, Stage 2 CCF, 20k epochs, PPO+AMP.

| Experiment | cmd? | cmd_tracking_w | Test-greedy av_reward | av_steps | Success rate (steps ≥ 290) |
|---|---|---|---|---|---|
| VIC4 (baseline, no cmd) | — | — | **951.2 / 299** | 299 | **100%** (15/15) |
| VIC8 (no cmd)           | — | — | 940.7 / 299 | 299 | 100% (7/7) |
| CMD_A | ✓ | 0.3 (imitation-heavy) | 527.0 / 236 | 236 | **65.5%** |
| CMD_B | ✓ | 0.5 (cmd-heavy)       | 440.4 / 279 | 279 | **86.1%** |

Best-episode back-calculation (solving `rew = (1−w)·base + w·1.0` for base):

| Experiment | Best-ep reward | Implied base (imitation) |
|---|---|---|
| CMD_A | 688.7 | **983** (above VIC4 951) |
| CMD_B | 507.2 | **1013** (above VIC4 951) |

**Interpretation**:
- When the cmd-conditioned policies **succeed**, imitation quality is as good as or better than no-cmd baseline.
- The cost of adding cmd conditioning shows up as **failure episodes** (CMD_A fails 35% of the time, CMD_B fails 14%), not as degraded quality on the successes.
- Higher cmd_tracking_w (B) → more robust (86% vs 65%) because the policy is pulled harder along a "command-consistent trajectory" and drifts less.

**We have not measured v_err directly** (how well the policy actually tracks a specific commanded speed). We know cmd_reward hits ≈1.0 in aggregate but never ran `v_cmd = [1.0, 1.0]` style fixed-command evaluation.

---

## 4. Round 2 (personalization) — partial, being discarded

V2_A / V2_B added SMPL body-shape variation (per-env different β parameters + β in obs). Disk-full crash at Ep 8500/8238 (≈40% of 20k). Partial data:

| @Ep 8k | Round 1 | Round 2 | Δ |
|---|---|---|---|
| w=0.3 rwd | 338 | 276 | **−62** |
| w=0.3 eps_len | 163 | 134 | **−29** |
| w=0.5 rwd | 194 | 204 | +10 |
| w=0.5 eps_len | 177 | 139 | **−38** |

**Decision (2026-04-20)**: Personalization direction is shelved. The user disagrees with adding body-shape variation on top of a single-clip motion — it compounds difficulty without a clear motivation. Return to velocity-command focus on a **single fixed SMPL body** and improve the cmd mechanism itself.

---

## 5. The main problem

**Reward tension between "imitation velocity" and "commanded velocity" on a single-clip reference.**

The motion has one natural speed; v_cmd is an arbitrary speed. When they disagree, the base imitation reward and the cmd reward are structurally in conflict. Symptoms:

- 14–35% of episodes fail at test time (policy drifts and terminates early).
- Success rate degrades most where v_cmd is far from the motion's natural speed.
- Aggregate reward looks much lower than no-cmd baseline, but this is ~90% the formula's structural dilution and ~10% actual quality loss.

**Secondary issues**:

- `ω_cmd` hardcoded 0 (single forward clip can't turn). Turning is outside the current capability.
- v_cmd is constant per episode — no smooth changes, no ramp-ups. Less realistic as a command interface.
- No direct v_err measurement on fixed commands (only aggregate success/reward).

---

## 6. Options to make cmd and motion cooperate

### A. Retime the reference motion to match v_cmd (phase-velocity scaling)

At each episode reset, compute the motion's natural pelvis velocity `v_motion`. If v_cmd ≠ v_motion, time-stretch the motion so that `v_motion_retimed = v_cmd`. Concretely: multiply the motion playback time index by `v_cmd / v_motion`.

- **Pro**: Zero reward conflict. Base imitation reward and cmd reward are now aligned (motion's retimed pelvis velocity == commanded velocity).
- **Con**: All joint velocities and step frequencies scale proportionally. Physically feasible only within a moderate range (~±30% of natural speed); beyond that, joint torques become unrealistic.
- **Effort**: Medium. Modify motion_lib sampling to stretch the time index.

### B. Multi-clip motion library with speed-conditioned clip selection

Build a library of walking clips at different natural speeds (e.g., AMASS walking subset: slow / normal / fast). At each reset, pick the clip whose natural v is closest to v_cmd.

- **Pro**: Motion stays natural across a wide speed range. The motion itself varies with speed (step length, frequency, arm swing), which is more physically correct than retiming.
- **Con**: Requires a data pipeline to extract and annotate an AMASS walking subset by speed. More storage. Command granularity limited by how finely the library samples the speed axis.
- **Effort**: High.

### C. Decouple velocity from the base imitation reward

Keep pose and joint-angle tracking in the base imitation reward, but remove or significantly downweight the pelvis-velocity tracking term. Let the cmd reward own velocity entirely.

- **Pro**: No direct conflict between the two reward sources. Style is anchored by pose matching; speed is a free parameter controlled by cmd.
- **Con**: Without a velocity anchor in the base reward, the policy has more room to exploit (e.g., freeze pose, just drive pelvis). Weight tuning matters. AMP discriminator still sees the motion, which provides a partial anchor.
- **Effort**: Low-medium. Modify the reward-term weighting in `humanoid_im.py`.

### D. Combined A + C

Retime the motion AND decouple velocity from base imitation. Belt and suspenders.

- **Pro**: Strongest alignment. Both structural conflict and reward-structure conflict are removed.
- **Con**: More changes at once → harder to attribute improvement to a specific fix.
- **Effort**: Medium-high.

### E. Baseline sanity check (cheap, do regardless)

Extend Round 1 B (current winner, 86% success) to 30k epochs via resume. Training log showed eps_len still growing at Ep 20k (185 → 202), so convergence wasn't complete.

- **Pro**: Zero code changes. Directly measures whether the current design can reach 95%+ success with more training.
- **Con**: Doesn't fix the root-cause conflict — just asks "is the ceiling higher than we measured?"
- **Effort**: Trivial.

---

## 7. Recommendation

1. Start with **Option A (motion retiming)**. It attacks the root cause, fits the existing single-clip setup, and works within the range we actually care about (walking speeds 0.8–1.3 m/s is within ±30% of typical natural gait).
2. Run **Option E (extend Round 1 B to 30k)** in parallel as a cheap sanity check on the current design's ceiling.
3. If we later want broader speed ranges (or turning), add **Option B (multi-clip library)** on top. Option C can optionally stack with A.
4. Skip personalization (Round 2 direction) for now.

---

## 8. Open questions (for prioritization / sanity check)

1. **What is the motion's actual natural pelvis velocity?** We chose `cmd_v_range = [0.8, 1.3]` somewhat blindly. If the motion's natural v is 1.2, then 0.8 cmd requires 33% slowdown — already at the edge of retiming feasibility.
2. **Is v_cmd tracking accuracy reasonable on successful episodes?** Run test with `cmd_v_range = [1.0, 1.0]` (fixed), measure mean pelvis velocity, compute v_err. Do this before deciding on a fix.
3. **Does AMP discriminator (which sees motion style, not velocity directly) already provide most of the stylistic anchor in Option C?** If yes, decoupling velocity from base imitation is much safer than it looks.
4. **What is the ultimate cmd interface we want?** Fixed per-episode (current), or time-varying during an episode (ramps, step changes)? Different interfaces imply different training distributions.

---

*Authored 2026-04-20 for external discussion. Internal references: `02_research_dev/260420_recent_6_experiments_summary.md`, `phc/env/tasks/humanoid_im_vic_cmd.py`.*

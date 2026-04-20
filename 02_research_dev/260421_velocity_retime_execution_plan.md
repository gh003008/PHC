# Velocity-Command Retiming — Execution Plan (2026-04-21)

**Input recommendation**: `02_research_dev/260420_velocity_command_recommendation.md` (ChatGPT)
**Input analysis**: `02_research_dev/260420_velocity_command_design_analysis.md` (prior analysis)

**Decision**: adopt the recommendation's core proposal — move command into the reference-generation layer via motion retiming — and run the two primary comparison arms in parallel on idx0 / idx1.

---

## 1. Two parallel tracks

| Track | GPU | What it tests | Code change needed? | Start cost |
|---|---|---|---|---|
| **Baseline A** — CMD_B resume → 30k | **idx0** | Is the current design undertrained? How high is the ceiling of the original absolute-v_cmd design? | **No** (resume existing ckpt with max_epochs=30000, save_frequency=2500) | ~5 min |
| **Baseline B** — Retimed R1, fresh train | **idx1** | Does moving command into the teacher (retimed reference) remove the imitation–command conflict and recover VIC4-baseline-level success (100%)? | **Yes** — new task class `HumanoidImVICCmdRetime`, new configs | ~1–2 h |

Both tracks use:
- single forward walking clip (`amass_isaac_walking_forward_single.pkl`)
- fixed SMPL body (no personalization)
- ω_cmd fixed at 0 (no turning)
- constant command per episode
- save_frequency = 2500 (storage-cap rule)

---

## 2. Baseline A — CMD_B resume → 30k (idx0)

**Motivation**: at Ep 20000 CMD_B was still growing (eps_len 185 → 202). The ChatGPT doc flags this as a "ceiling check" — separating "undertraining" from "structural conflict" as explanations for the 86% success rate.

**Config**: clone `exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_B.yaml`
- `max_epochs: 30000` (was 20000)
- `save_frequency: 2500` (was 100)
- All else unchanged.

**Resume**: `--epoch -1` loads `output/AMASS_CMD_B.pth` (canonical, epoch 20000).

**Expected output ckpts**: 22500, 25000, 27500, 30000 (× 90 MB ≈ 360 MB).

**Sbatch**: `exp_config/forward_walking/260421_AMASS_CMD_B_EXTEND/train_cmd_B_extend_gpu0.sh` (partition idx0, gres gpu:idx0:1).

---

## 3. Baseline B — Retimed R1 (idx1)

**Motivation**: the structural fix. Instead of sampling an absolute v_cmd independently of the motion, sample a **speed scale** `s`, retime the reference to play at `s·t_motion`, and make the imitation reward compare against the retimed teacher. With the retimed teacher already encoding the commanded speed, no explicit cmd-tracking reward is needed in R1.

### 3.1 Task class: `HumanoidImVICCmdRetime`

New file `phc/env/tasks/humanoid_im_vic_cmd_retime.py` (do not modify existing `HumanoidImVICCmd`, per VIC rule).

Responsibilities:

1. Store per-env scalar `s` in `self._current_scale` (shape `[num_envs]`). Sample at env reset from `U(s_lo, s_hi)` with default `[0.9, 1.1]`.
2. Derive `v_cmd_x = s · v_nat` for obs continuity (3-dim cmd obs kept so policy architecture is unchanged and we can warm-start from CMD_B if desired — though R1 trains from scratch by default).
3. Override the reference-state retrieval to use retimed motion time:
   ```
   motion_time_retimed = (episode_time · s)   (mod clip_duration)
   ```
4. Scale reference velocities by `s`:
   ```
   ref_body_vel_retimed = s · ref_body_vel_at(motion_time_retimed)
   ref_joint_ang_vel_retimed = s · ref_joint_ang_vel_at(motion_time_retimed)
   ```
   (Because if the clip is played at rate `s`, positions come from the scaled time but velocities must also be multiplied by `s` to match `d/dt[x_motion(s·t)] = s · x'_motion(s·t)`.)
5. Reward computation: base imitation uses retimed reference (automatic once the state retrieval is overridden). No explicit cmd reward in R1 (cmd_tracking_w = 0).

### 3.2 Configs

- `exp_config/forward_walking/260421_AMASS_RETIME/env_im_walk_vic_retime.yaml`:
  - task: `HumanoidImVICCmdRetime`
  - `retime_scale_range: [0.9, 1.1]`
  - `retime_v_nat: <measured>` (filled in after v_nat measurement)
  - `cmd_tracking_w: 0.0` (R1: no explicit cmd reward)
  - `cmd_v_range`, `cmd_w_range`: retained for observation backwards-compat but command is derived from scale.
- `exp_config/forward_walking/260421_AMASS_RETIME/im_walk_vic_retime.yaml`: clone of `im_walk_vic_cmd_B.yaml` with `name: AMASS_RETIME_R1`, `max_epochs: 20000`, `save_frequency: 2500`.
- Sbatch: `train_retime_R1_gpu1.sh` (partition idx1, gres gpu:idx1:1).

### 3.3 AMP check (pre-implementation)

Before training, inspect whether the AMP discriminator features include absolute velocities. If they do:
- easiest option: retime the AMP demo window sampling so positives are also scale-consistent,
- alternative: drop the velocity-sensitive features from AMP obs for this experiment.

I will document findings in a short note before launching the training.

### 3.4 Start-point

Trained from scratch (20k epochs). Warm-start from CMD_B is possible since obs shape is preserved, but the reward semantics and motion-time semantics have shifted enough that a clean run is cleaner to interpret.

---

## 4. What is deferred

Per the ChatGPT recommendation:

- **Personalization** (body-shape variation) — dropped.
- **Turning** (ω_cmd) — single forward clip cannot support.
- **Time-varying commands** — not yet.
- **Wide speed ranges** (e.g., ±30%) — only after v_nat is measured and narrow range succeeds.
- **Multi-clip library** — only once single-clip retiming plateaus.
- **Reward variant R2** (retimed + small λ·cmd) — queued as Baseline C, run only if R1 underperforms.

---

## 5. Measurement & evaluation plan

### 5.1 Measurement tasks (cheap, no GPU)

1. **v_nat measurement**: load the walking clip, compute pelvis forward velocity (whole-clip mean + cycle-wise mean). This fixes the absolute scale of the s-range.
2. **AMP feature audit**: grep discriminator obs builder. Produce a one-page note.

### 5.2 Post-training evaluation (after both jobs finish, ~16 h each)

For each trained checkpoint (CMD_B@30k, Retime_R1@20k):

- **Test-greedy at fixed commands**: v ∈ {0.90, 0.95, 1.00, 1.05, 1.10} · v_nat. Record success rate, av_steps, mean pelvis velocity, v_err, imitation reward.
- **Best-episode back-calculation**: implied base imitation reward per command bin.
- **Failure-mode distribution**: where (fast / slow / near-natural) do failures cluster?

### 5.3 Comparison write-up

→ `02_research_dev/260422_retime_vs_baseline_analysis.md` (after both jobs finish).

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| AMP discriminator not retimed → partial conflict survives | Medium | Audit first; apply fix if needed before launching Baseline B. |
| Retimed reference velocity scaling introduces a bug (missed term somewhere) | Medium | Code review; unit-check reference velocity numerically after retiming. |
| v_nat is far from 1.0 → current `[0.8, 1.3]` range becomes incorrect; Baseline A's results may need reinterpretation | Low-medium | Measure v_nat first; report alongside Baseline A analysis. |
| Disk growth mid-run | Low | save_frequency=2500 enforced; 145 G free at start. |
| Wrong partition / GRES index | Low | idx0 + idx1 both confirmed idle; sbatch headers explicit. |

---

## 7. Session deliverables

- **This doc**: execution plan, committed + pushed.
- **Baseline A** running on idx0.
- **Baseline B** implementation + configs + sbatch committed + pushed; training running on idx1.
- **v_nat note** (short, filed with plan or as comment in configs).
- **AMP audit note** (short).

---

*Authored 2026-04-21. Two parallel tracks started same session.*

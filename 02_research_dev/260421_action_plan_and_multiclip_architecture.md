# Action Plan + Multi-Clip Architecture Explainer (2026-04-21)

Context: Baseline A (CMD_B → 30k) finished; Baseline B (Retime R1) at ~68 % of 20k; next steps after both finish need to be scoped. This doc captures (1) the phased action plan, (2) the motion datasets we will use, and (3) how multi-clip training actually works architecturally — which is a natural question once we start talking about a "speed ladder" of clips.

---

## 1. Motion datasets on hand

| File | Clips | Description | Where it fits |
|---|---|---|---|
| `sample_data/amass_isaac_walking_forward_single.pkl` | 1 | Single KIT forward walk-to-stop, 5.4 s. Peak walking phase ≈ 0.85 m/s, whole-clip mean only 0.44 m/s (clip decelerates to a standstill). | **Baseline A, B (current)**; Baseline C (next); all immediate retiming work. |
| `sample_data/amass_isaac_walking_primitive.pkl` | 50 | AMASS walking subset: slow/medium/fast/backward/forward variants. 3.7–10.8 s per clip, total 333 s. Per-clip mean speeds 0.21–0.77 m/s (p10 0.30, p50 0.45, p90 0.65). | **Phase 5a multi-clip speed ladder**. |
| `sample_data/amass_isaac_gender_betas.pkl` | — | SMPL β parameters (body shape). | Only needed if personalization is re-introduced (deferred). |
| `sample_data/h5_motion_library*.pkl` | — | Personalized H5 walking variants. | **Not in use** — H5 direction paused per 260420 analysis. |
| `sample_data/debug_*_test.pkl` | — | Debug / axis-test motions. | Not used. |

Immediate runs only touch `walking_forward_single.pkl`. `walking_primitive.pkl` is already on server and becomes the main dataset from Phase 5a onward.

---

## 2. Action plan

### Phase 3 — Baseline C (Retime R2) on idx0, parallel to still-running Baseline B

- Same `HumanoidImVICCmdRetime`, same `s ∈ [0.9, 1.1]`, same `v_nat = 0.85`.
- Difference from B: `cmd_tracking_w = 0.1` (small residual cmd reward). Tests whether any explicit cmd signal helps on top of retiming, or is redundant.
- Dataset: same single walking clip.
- ETA: ~16 h.

### Phase 4 — Fixed-command test-greedy evaluation of A / B / C (2026-04-22)

- For each checkpoint: run with `cmd_v_range` pinned to a single value at v ∈ {0.90, 0.95, 1.00, 1.05, 1.10} · 0.85 m/s.
- Measure per bin: success rate, eps_len, mean pelvis v, v_err, base imitation reward.
- Write up: `02_research_dev/260422_ABC_retime_comparison_analysis.md`.
- **Decision gate**: does retiming (B or C) clearly outperform the absolute-v_cmd baseline (A) in success rate and tracking error?

### Phase 5a — Multi-clip speed ladder (if retiming wins Phase 4)

- New task class `HumanoidImVICCmdMultiClip` subclassing Retime: at env reset, pick the clip whose `v_nat` is closest to the sampled command, then apply a small ±5 % retime around that clip.
- Dataset: `amass_isaac_walking_primitive.pkl` (50 clips).
- Expected commandable speed range: roughly 0.19–0.85 m/s (covered by the 50 clips × their individual retiming windows).
- Network architecture: unchanged (see §3 below).
- Open risk: AMP with a mixed-clip positive distribution — may need per-clip style conditioning or per-clip discriminators.

### Phase 5b — Deeper diagnosis (if retiming does NOT clearly win Phase 4)

- Per-bin failure analysis on A/B/C to localize where the retimed reference still breaks down.
- Likely suspects: (1) the walk-to-stop clip structure itself, (2) residual AMP conflict, (3) Phase-obs that is not retimed in the current subclass (lookup table uses unscaled motion time).
- Possible fix: switch to a cleaner *cyclic* clip from `primitive.pkl` (e.g., `KIT_167_walking_medium03`, 6.13 s, more steady) before re-running retiming.

### Phase 6 — Turning + richer commands (contingent on 5a success)

- Re-enable ω_cmd using clips in `primitive.pkl` that contain yaw.
- Time-varying commands (ramps, step changes) after constant-command behavior is stable.
- Still single SMPL body; personalization stays deferred.

### Phase 7 — Long-term: exoskeleton assistance

- Couple the stable commandable walking policy to an exoskeleton actuator model and train the exo controller.
- Requires Isaac Lab (PhysX 5) migration for closed kinematic chains — significant infra work, gated on Phases 3–6 paying off.

---

## 3. Multi-clip architecture explainer

Natural question once we move to "50 clips": **is it one network or many? does the network branch on v_cmd?**

### Short answer

One network. The same policy π_θ is used for all clips. "Selection" happens in the data-sampling step (which clip to load for each env at reset), never inside the network. The network itself does not branch.

### What happens at runtime

At each env reset:

```
1. Sample v_cmd                 (e.g., 0.55 m/s)
2. Look up library  →  pick clip whose natural speed
                       is closest to v_cmd
                      (e.g., KIT_167_walking_medium03, v_nat ≈ 0.47 m/s)
3. Load that clip as the reference for this env for the whole episode
4. (Optionally) apply a small retime s ∈ [0.95, 1.05] around it
   to close the residual speed gap.
```

At every env step, ONE shared policy π_θ(action | obs) receives:

```
obs = [humanoid_state,                         ← current physics state
       reference_pose_at_current_motion_time,  ← pulled from whichever clip is loaded
       reference_velocity_at_current_motion_time,
       v_cmd]                                  ← command obs
```

And outputs actions. The policy **does not know which clip is loaded** — it just sees "this is my reference pose/velocity right now, track it."

### Why this works

Because the reference pose/velocity is **in the observation**, the task from the policy's perspective is always the same: *"imitate whatever reference I'm currently shown while staying upright."* When the reference happens to look like a slow walk, the policy produces slow-walking actions. When it looks like a fast walk, fast-walking actions. The network amortizes a general *conditional tracking controller*, not a bank of specialized ones.

This is exactly how PHC already works on the full AMASS dataset: one network, thousands of motions. The current VIC experiments are a degenerate case using a single clip.

### What moving from single-clip → multi-clip actually changes

| Component | Single clip (Baselines A/B/C) | Multi-clip (Phase 5a) |
|---|---|---|
| Motion library | 1 clip loaded | 50 clips loaded |
| At reset | same clip always picked | pick by `argmin \| v_nat(clip) − v_cmd \|` |
| Network architecture | one PPO network | **same one PPO network, unchanged** |
| Observation shape | as-is | as-is |
| Reward functions | as-is | as-is |
| AMP discriminator | positives from 1 clip | positives from all 50 clips (mixed distribution) |

The only real code delta is in the env-reset sampler (clip choice logic) and the motion-library config. The policy, value, and discriminator networks are untouched in structure.

### When multiple networks WOULD make sense (why we're not doing it)

Multiple networks = the **MCP** (Mixture-of-Controllers / Primitives) pattern. PHC already has this: `phc/env/tasks/humanoid_im_mcp.py`, `humanoid_im_mcp_vic.py`. Each primitive specializes on a narrow skill and a gating network blends them. Worth using when:

- Skills are **qualitatively different** (walk vs jump vs climb) so a single policy gets stuck in local optima trying to cover all of them.
- You want interpretable "which skill is active" signals.
- You want to freeze one primitive while training another.

For varying *walking speed*, all clips are essentially the same skill with different parameters — one network works and is simpler. MCP only becomes necessary if we later add categorically different skills (walk + turn + run + stop) and a single network starts to struggle.

### The bound this approach imposes

Commandable speed is bounded by the library. If the 50 clips have natural speeds in {0.21, 0.28, …, 0.77} m/s, the commandable range is effectively that set, plus the small retiming window (~±5–10 %) around each. You cannot command 1.5 m/s if no clip walks at ~1.5. This is why the progression is:

1. Single-clip retime (narrow range, Baselines B/C),
2. Multi-clip library (wider range, Phase 5a) — current end state we can aim for with available data,
3. Eventually, a continuously parameterized motion generator (generative motion model) — future work, not in scope for this phase.

---

*Authored 2026-04-21. To be updated after Phase 4 evaluation fixes the 5a vs 5b branch.*

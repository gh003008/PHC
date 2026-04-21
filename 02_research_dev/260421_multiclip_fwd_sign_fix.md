# Multi-Clip FWD Correction — Sign Convention Fix (2026-04-21 late session)

## What was wrong

`260421_multiclip_direction_contamination.md` correctly identified that the 50-clip `primitive.pkl` library mixes forward, backward, circle-walking, and other motions, and that magnitude-only retrieval (`argmin |v_mean_mid − v_cmd|`) picks the wrong teacher. It proposed filtering to a "forward-straight" subset via `scripts/data/compute_walking_direction_metadata.py`.

But the first run of that script produced a subset whose 9 clips were **all named `...WalkingStraightBackwards...`**, which looked suspicious. Cross-checking:

- The baseline `sample_data/amass_isaac_walking_forward_single.pkl` (the clip that actually trained successfully in all prior experiments) has `v_x_mean ≈ −0.44 m/s` — **negative** in world frame.
- Every `...WalkingStraightForwards...` clip in `primitive.pkl` also has `v_x < 0`.
- Every `...WalkingStraightBackwards...` clip has `v_x > 0`.
- The `walking_slow/medium/fast/run` family also has `v_x < 0`.

**Conclusion**: in the PHC world-frame convention, **forward walking is v_x < 0**. The filter script's `v_x > V_FWD_MIN → forward_straight` criterion was inverted — it was selecting the backward-walking clips as "forward."

## What was changed

### 1. `scripts/data/compute_walking_direction_metadata.py`

Flipped the forward/backward criterion:

```python
# before (wrong)
if v_x > V_FWD_MIN and |v_y| < V_LATERAL_MAX: return "forward_straight"
if v_x < -V_FWD_MIN and |v_y| < V_LATERAL_MAX: return "backward_straight"

# after (correct for PHC convention)
if v_x < -V_FWD_MIN and |v_y| < V_LATERAL_MAX: return "forward_straight"
if v_x > +V_FWD_MIN and |v_y| < V_LATERAL_MAX: return "backward_straight"
```

Also sort the fwd-only output by **ascending `|v_x|`** (slowest forward walk first) so a retrieval on positive-scalar v_cmd sees a monotone index.

### 2. Filter output after fix

| dir_class | count |
|---|---|
| forward_straight | **22** |
| backward_straight | 9 |
| turning | 10 |
| other | 9 |

22 forward clips, `|v_x|` range **[0.21, 0.99] m/s**, all direction-verified (`v_x < 0`, `|v_y| < 0.2`, `|yaw_rate| < 0.3 rad/s`). Good coverage for speed-conditioned retrieval.

### 3. `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py`

The task class now:

- Supports two JSON schemas: the new signed-v_x schema (`v_x_mean_mid` per entry) and the legacy magnitude-only schema (`v_mean_mid`) — for backward compat.
- When loading the signed schema, filters to `dir_class == "forward_straight"` keys and stores `|v_x_mean_mid|` as the retrieval key.
- Maintains a per-motion `_clip_eligible` boolean mask (True for forward-walking clips loaded into `motion_lib`, False otherwise).
- In `_sample_ref_state`, the retrieval `argmin |v_nat − v_cmd|` masks non-eligible entries with `+inf`, so only forward-walking clips can be chosen.

Startup now prints e.g.:
```
[MultiClip] schema=signed-vx. eligible clips: 22/512.
|v_x| range=[0.210, 0.989] (p50=0.507).
v_cmd_range=[0.25, 0.85], retime_enabled={False,True}, retime_scale_range=[1.0,1.0] or [0.9,1.1]
```

### 4. Configs

New dir `exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/` with four yamls + two sbatch scripts, pointing `multiclip_v_nat_path` at the corrected `amass_isaac_walking_primitive_fwd_only.json` and setting `multiclip_v_cmd_range: [0.25, 0.85]` to stay inside the library's support.

The previous `260421_AMASS_MULTICLIP/` dir (the contaminated-library setup) is retained as historical record.

## What is running now

- **3860** on idx0: `AMASS_MULTICLIP_FWD_NR` — 22 clips, no retime.
- **3861** on idx1: `AMASS_MULTICLIP_FWD_RT` — 22 clips, retime within [0.9, 1.1].

Both fresh from-scratch, 20k epochs, save_frequency 2500.

Superseded (cancelled): 3853 / 3854 — contaminated 50-clip library.

## Expected comparison

If the contamination was the real reason for the slow early convergence we saw in 3853/3854, the corrected runs should: (a) recover eps_len growth into the 100+ range by Ep ~5000 (matching single-clip CMD_B's trajectory), and (b) show NR vs RT differing only in the fine-grained v_err metric at test time, not in gross success rate.

Post-training fixed-command eval will span v_cmd ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8} m/s, with per-bin success rate + v_err as the primary comparison axes.

---

*Authored 2026-04-21, after pulling the local session's direction-contamination analysis and fixing the one-sign bug. Next check-in around Ep 5000 for both runs.*

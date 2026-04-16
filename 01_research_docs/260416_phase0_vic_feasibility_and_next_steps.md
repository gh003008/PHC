# Phase 0 VIC Feasibility: Findings and Path Forward

**Version**: 1.0 — 2026-04-16
**Scope**: Interim analysis after running 4 configurations on S001 treadmill walking data
**Related**: [260413_MPL_methodology_refined.md](./260413_MPL_methodology_refined.md)

---

## 1. Executive Summary

We have run four Phase 0 feasibility experiments on EXOLAB server using S001 H5 treadmill walking data and AMASS KIT walking subset. The purpose was to validate that a single-policy VIC architecture (from the MPL methodology, §2.2) can learn stable nominal walking. Results reveal a **counter-intuitive finding**:

> **Adding CCF as a learnable action does NOT improve — and in fact noticeably degrades — nominal walking performance relative to pure PD control on the same data.**

However, this does **not** invalidate the MPL methodology. It reframes Phase 0's role: it is a plumbing check, not a value check. The real test of VIC is in Phase 1+ where perturbations create a selection pressure for impedance modulation. This document summarizes the data, interprets it through the MPL framework, and proposes concrete next steps.

---

## 2. Experimental Matrix

All experiments use:
- **Architecture**: PHC single-policy (MLP [1024,1024,512,512])
- **Learning**: PPO + AMP discriminator
- **Motion data**: S001 treadmill walking (H5) OR AMASS KIT walking subset
- **Reward**: imitation + discriminator + metabolic (no perturbation, no balance reward, no α signal)
- **Max epochs**: 20,000
- **Same hyperparameters** where comparable (power_coefficient 5e-7, cycle_motion True, task/disc weight 0.5/0.5, reward curriculum switch at 10k)

Final (or latest) results:

| Cell | Motion | VIC | CCF groups | Final Epoch | `rwd` | `eps_len` (30Hz) | Walking duration |
|---|---|---|---|---|---|---|---|
| A | AMASS primitive | **OFF** (baseline) | — | 7,146 (manual stop) | 152 | 50 | 1.67 s |
| B | AMASS primitive | ON | 8 | 20,001 | 223 | 80 | 2.67 s |
| C | H5 S001 | ON | 8 | 20,001 | **392** | 130 | 4.33 s |
| D | H5 S001 | **OFF** | — | running (Ep ~18,593) | **605** | **192** | **6.40 s** |
| E | H5 S001 | ON | 4 (lower body only, upper fixed) | running (Ep ~12,674) | 330 | 115 | 3.83 s |

**Ep 10,000 snapshot** (most directly comparable since curriculum switch just occurred):

| Cell | `rwd` | `eps_len` |
|---|---|---|
| C (H5 + VIC 8grp) | 262.8 | 93.2 |
| D (H5 + no VIC) | **740.9** | **235.1** |
| E (H5 + VIC 4grp) | ~300 | ~100 |

**Key ordering (H5 motion)**: `no VIC ≫ VIC 4grp > VIC 8grp`. VIC costs 25–50% of reward relative to PD baseline on this task.

---

## 3. Interpretation Through the MPL Methodology

### 3.1 Phase 0 as Plumbing Check

The MPL roadmap (§9) defines Phase 0 as:

> "Single-policy VIC with walking data → Feasibility validation: stable walking with CCF"

The success criterion is *stable walking is achievable with CCF action added*, not *CCF improves nominal walking*. All three VIC configurations (B, C, E) produce stable gait with eps_len ≫ 1 step, so the plumbing works:

- ✅ Action space scaling (69 + N_g) is auto-derived
- ✅ CCF → impedance formula `τ = K · 2^ξ · ...` applied correctly
- ✅ CCF sigma override in `amp_agent.py` auto-adjusts to group count
- ✅ 4-group reduction (upper body fixed at 1.0×) works with zero code change — existing branch in `humanoid_im_vic.py:199-205` supports this exactly
- ✅ Learned CCF patterns in prior VIC_CCF_ON2 run (on AMASS single forward clip) show biomechanically plausible hierarchy: ankle ≫ hip ≳ knee ≈ upper body (matches Winter 2009 impedance literature)

### 3.2 Why CCF Hurts in Phase 0

The methodology states (§7.4):

> "When a stabilizing external force is detected → policy can reduce ξ (become more compliant)
> When a destabilizing force is detected → policy increases ξ (stiffen to resist)"

**CCF's value depends on events that don't exist in Phase 0.** With no perturbations, no exoskeleton force, no terrain variation:

1. The optimal CCF for nominal walking is approximately **flat** (close to constant across gait phase, maybe gentle ankle modulation for push-off).
2. However, the policy is forced to output a CCF every timestep with σ=0.37 (log-std −1.0). This produces a noisy 4-or-8-dim additional action that **perturbs the torque computation**, slightly destabilizing the PD tracking.
3. The reward function has no term that explicitly prefers high or low CCF in any given phase — it is dominated by imitation reward (track reference pose). Any CCF exploration translates into noisier torques without a task-relevant gradient.
4. The PPO optimizer therefore spends many samples reducing CCF variance toward zero, but the stochasticity injected by `sigma_init: -1.0` keeps it wide, slowing convergence of the *motor* policy.

This is consistent with a general principle: **a larger action space hurts sample efficiency unless extra dimensions contribute to the reward**.

### 3.3 Why Baseline AMASS (Cell A) Looks Worst

Cell A (152 rwd / 50 eps_len) is the weakest. Reasons:

- Different learning config than B/C/D/E (older `im_walk.yaml` without reward curriculum, without 0.5/0.5 task/disc weighting, older `power_coefficient: 5e-5` which is 100× the newer 5e-7)
- Manually terminated at Ep 7,146 due to visible plateau

**Cell A and Cell B are therefore not a clean A/B** for VIC's effect — they differ in learning hyperparameters as well. The clean A/B is **C vs D** (both H5, both 20k epoch target, both 0.5/0.5 + curriculum + 5e-7 power coefficient — only VIC differs).

Under that clean comparison, VIC's cost is explicit: ~50% reward penalty, ~45% episode length penalty.

---

## 4. The Co-Contraction Question

The user's conceptual motivation for VIC is to mirror **muscle-layer co-contraction**: humans stiffen joints by activating agonist and antagonist muscles simultaneously. CCF is the abstract representation of this at the joint level.

### 4.1 Is Co-Contraction Meaningful in Nominal Treadmill Walking?

EMG literature (Winter 2009, Sartori et al. 2015) shows:

1. Co-contraction in healthy adult treadmill walking is **low and phase-locked**, peaking mildly around heel strike (for impact absorption) and during push-off (for propulsion).
2. The range of stiffness modulation within a stride is ≈ 0.7× to 1.3× of mean stiffness — a factor of **1.9×** (log₂ ≈ 0.9, matching roughly our ξ ∈ [−1, 1] range).
3. Co-contraction **increases substantially** in the elderly, during perturbation recovery, on uneven terrain, with cognitive load, and when carrying loads. The *dynamic range* of co-contraction is driven by these external factors, not by nominal walking itself.

**Implication**: our measured result (no-VIC wins on nominal H5 walking) is consistent with human biomechanics: a healthy young adult on a treadmill should not need strong co-contraction modulation, so the RL policy doesn't benefit from CCF.

### 4.2 When CCF Should Pay Off

The MPL methodology provides five scenarios where CCF has real signal:

1. **Random external forces** (§5.1 Stage 2): F_ext ∼ U(−F_max, F_max) during 0.1–0.5 s bursts at pelvis.
2. **Exoskeleton torques** (§7.3): joint-level torque perturbations, phased assistance/resistance, sudden onset/offset.
3. **Belt perturbations** (§4.2 data collection): sudden treadmill speed changes triggering trip/slip reactions.
4. **Stability margin α** (§3.2): high α routes the policy to perturbation-response exemplars where impedance patterns differ from nominal walking.
5. **Biomechanical CCF reward** (`vic_bio_ccf_reward_w`, currently defaulted to 0): explicit reward shaping that encourages CCF to vary by gait phase — acts as an inductive bias toward realistic impedance curves.

In all five scenarios, CCF has a **reward gradient**: it is either tracked against a human reference or directly tied to survival (balance reward). In Phase 0, none of these are active.

### 4.3 Interpretation of 4-Group Result (Cell E)

Cell E fixes upper-body CCF at 1.0× and groups hip+knee per leg, reducing CCF action dim from 8 to 4. Early results (Ep 12,674: rwd 330) show:

- **Better than 8-group VIC at same epoch (C was 262 at Ep 10k, likely ~300 at Ep 12k)**
- **Still worse than no-VIC (D: 740 at Ep 10k)**

The 4-group version is the least-bad VIC. This is consistent with:

- Fewer exploration dimensions → less noise in torque
- Upper body fixed → removes CCF modulation in the region VIC_CCF_ON2 found to have the smallest natural variation (0.80–0.86× in upper body vs 1.28–1.49× in ankle)
- Hip+knee pooling costs some resolution (hip wants stiffer, knee wants more compliant in VIC_CCF_ON2 data) but the information loss is bounded

**If we want CCF in the policy, the 4-group version is the right form factor — but in Phase 0 without perturbations, even this costs performance relative to pure PD.**

---

## 5. Proposed Path Forward

### 5.1 Phase Transitions

The data strongly supports moving to **Phase 1 (perturbation curriculum in simulation)** as the next experiment, rather than continuing to iterate on Phase 0 VIC variants.

**Recommended experimental sequence:**

| Step | Configuration | Hypothesis |
|---|---|---|
| **1** | Reuse H5 + no-VIC (Cell D) as the **population pre-trained baseline** (θ_pop in §6.3). | Best nominal walker available. |
| **2** | Fine-tune Cell D policy with VIC 4-group added + **simulated perturbations** (§5.1 Stage 2). Apply ramped F_ext 20 → 200 N at pelvis, 0.1–0.5 s duration. | CCF should now acquire a reward gradient. Expected: VIC 4grp eventually surpasses no-VIC at perturbation-heavy evaluation. |
| **3** | Add α computation (§3): XCoM + CoP + angular momentum. Include α in observation. | Enables α-weighted reward blending and (later) motion library retrieval. |
| **4** | Add balance reward (§5.1 Stage 2) with α-blended weighting. | Should allow policy to sacrifice tracking accuracy during high-α states. |
| **5** | Phase 2 data collection (belt perturbations on S001) — the MPL "critical next step" (§9). | Unblocks true personalization. |

Steps 1–4 can be done with **existing simulation infrastructure and existing H5 walking data**. They do not require new motion capture.

### 5.2 CCF Value Verification (Phase 0 → Phase 1 Bridge)

Even within Phase 0, we can verify CCF is being learned meaningfully using the existing `analyze_phase_ccf.py` script:

1. Run Cell C (VIC 8grp + H5) policy at Ep 20k in evaluation mode with phase-CCF logging.
2. Plot CCF(φ) curves for each joint group across the gait cycle.
3. **If** ankle groups show stance vs swing variation > 0.2 and upper body groups show CCF ≈ 0, → VIC learned *something* useful even without reward pressure.
4. **If** all CCF are flat near zero, → confirmed that in Phase 0, CCF is effectively unused.

This analysis on existing checkpoints takes < 30 minutes and gives us actionable information without new training runs.

### 5.3 Immediate Action Items on Server

1. **Complete Cell D (Ep 20k)**: currently Ep 18,593, ~30 min to finish. Final checkpoint will serve as Phase 1 fine-tuning starting point.
2. **Complete Cell E (Ep 20k)**: currently Ep 12,674, ~8 h more. Compare final VIC 4grp vs 8grp vs no-VIC.
3. **Run CCF phase analysis** on C, E, D checkpoints (the last as control — a no-VIC policy should have no meaningful CCF output).
4. **Prepare Phase 1 config**: extend env yaml with `perturbation.enabled: True`, ramped F_ext, random intervals. The code path for external force application may already exist in `humanoid.py`; otherwise we add a small force injector.

### 5.4 What to Tell ChatGPT

If seeking external feedback, the key questions are:

1. **Is the Phase 0 finding (no-VIC > VIC on nominal walking) consistent with other published results**, e.g., Peng et al. ASE where richer action spaces sometimes hurt before perturbations are introduced? Are there canonical "VIC in RL" papers that had the same issue and resolved it in later phases?

2. **Is the proposed Phase 1 experiment (fine-tune no-VIC baseline with VIC + sim perturbations)** the right way to give CCF a reward gradient? Alternatives include:
   - From-scratch training with perturbations + VIC together
   - Co-training: half the batch with perturbation, half without, to preserve nominal gait while learning perturbation response
   - Distillation: extract CCF policy from a hand-designed stiffness controller as warm-start

3. **Is the 4-group (lower body + upper fixed)** a reasonable prior for co-contraction? Should left/right be symmetrized (2 groups: Hip+Knee, Ankle+Toe) to enforce bilateral symmetry? Or further reduced to a single scalar (Ankle stiffness only) — since the biomechanics literature says ankle dominates modulation during gait?

4. **The imitation reward (track S001 reference) may conflict with the CCF reward (co-contraction for robustness)** in perturbed conditions. How should they be balanced? The α-blended approach (§6) is the current answer, but it could collapse: if α is always low during training, CCF never gets a gradient.

5. **For exoskeleton deployment (§7)**, should CCF include not just joint impedance but also explicitly model **biceps/triceps-like antagonistic pairs**? Current CCF is per-joint; real muscle pairs give finer control but at the cost of more parameters.

---

## 6. Open Questions for the Research

1. **Is there an implicit CCF that emerges even without explicit action?** A pure PD policy (Cell D) implicitly uses constant K_p, K_d. Can we *observe* that the no-VIC policy stays in regions where co-contraction would not be needed (i.e., it learns to avoid the need for impedance modulation)?

2. **If VIC is only valuable under perturbation, is CCF the right abstraction?** Alternative formulations:
   - Direct torque output (action is τ, not (q_ref, ξ)) — gives the policy even more freedom but loses the PD structure.
   - Learned reflex gains — a simpler "if force detected → add stiffness" rule, not a full per-joint CCF action.

3. **Can the motion library retrieval mechanism (§4.4) replace some of what CCF provides?** If the right perturbation response exemplar is retrieved at high α, the policy is conditioned on human behavior that already contains the correct impedance — perhaps no explicit CCF is needed at all.

4. **For personalization (§6.2), is CCF the right personalization channel?** Subjects' impedance profiles differ (Horak & Nashner 1986). But they also differ in timing, amplitude, step placement. CCF is one of several personalization vectors; which is most valuable for exoskeleton prediction?

5. **Does the VIC_CCF_ON2 result (learned ankle 1.3×, knee 0.9×, upper 0.85× on AMASS forward_single)** generalize to S001 H5 data? That was a different subject's data. Running Cell E to completion and extracting CCF patterns will answer this.

---

## Appendix: File Locations (Server)

### Experiment code (git-tracked, Jimin branch)
- `phc/data/cfg/env/env_im_walk_vic.yaml` — Cell C config
- `phc/data/cfg/env/env_im_walk_vic_amass.yaml` — Cell B config
- `phc/data/cfg/env/env_im_walk_h5_novic.yaml` — Cell D config
- `phc/data/cfg/env/env_im_walk_vic_4grp.yaml` — Cell E config
- `phc/env/tasks/humanoid_im_vic.py:187-221` — CCF grouping (supports 4 and 8)
- `phc/learning/amp_agent.py:536-548` — CCF sigma override (auto-adjusts)

### Checkpoints (server-only)
- `~/PHC/output/VIC_CCF_ON2_H5.pth` (Cell C, Ep 20k)
- `~/PHC/output/VIC_CCF_ON2_H5_4grp.pth` (Cell E, running)
- `~/PHC/output/VIC_CCF_ON2_NoVIC_H5.pth` (Cell D, running, near complete)
- `~/PHC/output/VIC_CCF_ON2_AMASS.pth` (Cell B, Ep 20k)
- `~/PHC/output/HumanoidIm/.../Humanoid_V4_Fresh_Start_01.pth` (Cell A, Ep 7k manual stop)

### Logs
- `~/PHC/logs/phc_vic_h5_3614.out` (Cell C)
- `~/PHC/logs/phc_vic_amass_3615.out` + `3616.out` (Cell B, resumed)
- `~/PHC/logs/phc_h5_novic_3644.out` (Cell D)
- `~/PHC/logs/phc_vic_4grp_3646.out` (Cell E)
- `~/PHC/logs/progress_report_*.md` (1000-epoch milestones per cell)

# Review and Recommended Modification of the VIC / Impedance Plan
**Date:** 2026-04-16  
**Basis:** `260416_phase0_vic_feasibility_and_next_steps.md`, the broader WalkON integrated research program, and recent compliance / locomotion RL literature.

---

## 1. Executive decision

The current Phase 0 result should be interpreted as:

> **The original plan is still valid, but the role of impedance action must be narrowed and re-ordered.**

My reading is:

1. **Phase 0 did its real job already**: it showed that VIC plumbing works and that stable gait is possible with the impedance path connected.
2. **Phase 0 did not show task value for impedance**, because nominal treadmill walking gives almost no reward gradient for impedance modulation.
3. Therefore, **impedance action should not be treated as a primary gait-generation action during nominal walking**.
4. Instead, it should be treated as a **secondary, context-triggered adaptation channel** for perturbation response, interaction handling, and later personalization.

So the right next step is **not more Phase 0 architecture searching**.  
The right next step is to **lock the best nominal locomotion policy, then move the impedance module into a residual / gated / perturbation-conditioned form and test it under perturbations**.

---

## 2. What the Phase 0 result really means

The clean comparison in the note is the H5 S001 case:

- **No VIC** performs best for nominal walking.
- **VIC 4-group** is worse, but better than VIC 8-group.
- **VIC 8-group** is worst among the H5 variants.

This is not surprising once we separate **motor control** from **impedance adaptation**.

### My interpretation
In nominal treadmill walking:

- the pose-tracking policy already has a clear objective,
- the impedance action has **no strong task-aligned signal**,
- but it still injects exploration into torque production.

So the extra impedance dimensions behave like **structured noise** unless the environment contains:
- external pushes,
- sudden belt changes,
- exoskeleton interaction forces,
- uneven terrain,
- or an explicit stability / compliance objective.

That means the main lesson is not “VIC is wrong.”  
The main lesson is:

> **Impedance is useful when the task contains force uncertainty or balance recovery pressure. It is not automatically useful in nominal imitation walking.**

This actually matches the original long-term program better than forcing Phase 0 to prove too much.

---

## 3. What stays unchanged from the original plan

I would **not** rewrite the whole research direction. The broad plan still makes sense.

### Still valid
- Build a strong nominal locomotion prior first.
- Introduce perturbations and stability signals next.
- Use impedance / co-contraction ideas where robustness and interaction matter.
- Later connect this to:
  - upper-body agency,
  - hybrid human-robot simulation,
  - exoskeleton interaction,
  - and subject-specific personalization.

### Program-level reframing
The broad WalkON program already emphasizes:
- structured human-robot interaction rather than pure rigid-body control,
- perturbation-aware learning,
- hybrid human modeling,
- and personalization.

The new Phase 0 result simply says:

> **Do not spend more time trying to prove the value of impedance on nominal treadmill walking.  
> Spend time making impedance useful in the parts of the program where it naturally belongs: perturbation, interaction, and personalization.**

---

## 4. What should change in the narrow VIC roadmap

## 4.1 Change the meaning of Phase 0
**Old implicit reading:** “VIC should improve walking already.”  
**Recommended reading:** “VIC path can be connected without breaking locomotion.”

### New Phase 0 exit criteria
Phase 0 is complete if:
1. no-VIC nominal walking is stable,
2. VIC nominal walking is also stable,
3. impedance outputs are numerically sane,
4. logging / grouping / sigma / torque plumbing are verified.

That is enough.

Do **not** require VIC to outperform no-VIC in Phase 0.

---

## 4.2 Promote the no-VIC policy to the mainline baseline
The best nominal walker should become the **population locomotion prior**.

Recommended role of the current no-VIC H5 policy:
- base gait generator,
- parent checkpoint for later perturbation fine-tuning,
- reference controller for all future VIC comparisons.

This avoids wasting time relearning nominal gait with a harder action space.

---

## 4.3 Stop optimizing the 8-group version for now
The 8-group version is currently too expensive in exploration burden for the value it gives.

### Recommendation
- **Mainline:** 4-group lower-body VIC with upper body fixed.
- **Ablation only:** 8-group VIC.
- **Fallback ablation:** 2-group or ankle-dominant version if 4-group still fails under perturbation.

---

## 5. How we should deal with the impedance action

This is the main design recommendation.

## 5.1 Conceptual redefinition
Treat impedance action as:

> **a residual adaptation action, not a base locomotion action**

In other words:

- the locomotion policy decides **how to walk**,
- the impedance head decides **how stiff/compliant to be when conditions demand it**.

That means the impedance action should be:
- lower-dimensional,
- lower-frequency,
- strongly regularized around neutral,
- and activated mainly when perturbation / interaction evidence is present.

---

## 5.2 Recommended parameterization

Instead of:
```text
policy -> q_ref + xi (every timestep, same exploration style)
```

use:
```text
policy_main -> q_ref
policy_imp  -> delta_xi
gate(alpha, wrench, contact anomaly, pelvis accel, exo torque) -> g in [0, 1]

xi_eff = xi_nominal(phase) + g * LPF(delta_xi)
K_eff  = K0 * 2^(xi_eff)
```

### Meaning of each term
- `q_ref`: primary nominal motor command.
- `delta_xi`: residual impedance change.
- `g`: context gate. Closed in nominal walking, open during perturbation / force interaction.
- `LPF`: low-pass filter to prevent step-to-step torque noise.
- `xi_nominal(phase)`: optional nominal phase prior. Start with zero; only add later if analysis justifies it.

### Why this is better
It preserves the original idea of impedance modulation, but it removes the main Phase 0 failure mode:
- random impedance exploration during states where impedance is not needed.

---

## 5.3 Recommended action structure

### My recommended default
**4-group, left/right separated, upper body fixed**

Interpretation:
- left proximal (hip + knee)
- left distal (ankle + toe)
- right proximal
- right distal

### Why not force bilateral symmetry?
For nominal gait, symmetry is attractive.  
For perturbation recovery, symmetry is often wrong.

A slip, push, or trip is usually **asymmetric** in timing and load.  
So I would:
- keep left/right separate,
- use mirror augmentation or weak symmetry regularization,
- but **not** hard-tie the two sides.

---

## 5.4 Recommended exploration treatment

The current problem is not only dimensionality.  
It is also that impedance exploration is too free too early.

### Recommendations
1. **Separate actor head for impedance**
   - do not share exactly the same stochastic treatment as the motor head.

2. **Lower initial impedance sigma**
   - make it much narrower than the motor action head.
   - starting point: `sigma_init ≈ -2.5 to -3.0` rather than around `-1.0`.

3. **Start around neutral impedance**
   - initialize `delta_xi = 0`.

4. **Use stronger regularization**
   - neutral prior: `||delta_xi||^2`
   - temporal smoothness: `||delta_xi_t - delta_xi_{t-1}||^2`
   - optionally phase smoothness

5. **Use lower update rate**
   - impedance does not need to change every controller step.
   - update every 3–5 control steps, or at 5–10 Hz equivalent.

### Suggested starting ranges
These are only starting points:
- **Phase 0 / nominal:** `delta_xi ∈ [-0.25, 0.25]` or `[-0.3, 0.3]`
- **Phase 1 / perturbation:** widen gradually toward `[-0.7, 0.7]`
- only use full `[-1, 1]` if later evidence supports it

---

## 5.5 Recommended reward treatment

Impedance must receive a task-aligned gradient.

### During nominal-only walking
Use only:
- neutral prior,
- smoothness prior,
- optional phase plausibility prior.

Do **not** expect performance gain.

### During perturbation training
Now impedance can be rewarded through:
- survival / fall avoidance,
- recovery time,
- step recovery quality,
- WBAM / angular momentum control,
- foot placement recovery,
- reduced collision impulse,
- lower torque spikes,
- improved balance margin.

### Important warning
If `alpha` is almost always low, an alpha-blended design can collapse into “never use impedance.”

So training must guarantee enough high-stability-demand states via:
- explicit perturbation scheduling,
- mixed nominal/perturbed episodes,
- or a dedicated recovery stage.

---

## 5.6 Recommended nominal-vs-reactive decomposition

I think the cleanest decomposition is:

### Layer 1 — nominal gait
- no-VIC or effectively neutral-VIC
- optimize imitation quality and long-horizon stability

### Layer 2 — reactive impedance
- active only when balance/interactions demand it
- learns temporary deviation from nominal stiffness

### Layer 3 — later personalization
- subject-specific scaling of timing, gain, duration, asymmetry
- especially relevant for exoskeleton deployment

This decomposition also aligns better with the clinical / personalization part of the broader program.

---

## 6. Proposed experiment ladder

## 6.1 Immediate closure of Phase 0
Before new training, finish the minimum closure work:

### A. Complete current runs
- finish Cell D to final checkpoint
- finish Cell E to final checkpoint

### B. Run CCF phase analysis
For C and E checkpoints:
- plot group-wise CCF over gait phase
- compare amplitude, smoothness, and asymmetry

### C. Decide whether nominal CCF learned anything useful
Three possibilities:
1. **Flat near zero**  
   -> confirms that nominal walking does not need active impedance modulation.
2. **Small but smooth ankle-dominant phase pattern**  
   -> use it later as `xi_nominal(phase)` prior.
3. **Noisy or unstable pattern**  
   -> evidence that current actionization is too unconstrained.

This analysis is cheap and should happen before any more architecture branching.

---

## 6.2 Modified Phase 1 (recommended mainline)

### Stage 1 — lock nominal baseline
- freeze best no-VIC checkpoint as parent
- evaluate on fixed nominal battery

### Stage 2 — attach residual VIC head
- 4-group only
- upper body fixed
- initialize near zero
- gate closed or almost closed in nominal episodes

Goal:
- prove that VIC head can exist **without harming nominal gait too much**

### Stage 3 — mixed curriculum
Start with mixed episodes:
- **70% nominal / 30% perturbed** as a practical starting point
- later move toward **50% / 50%** if needed

Perturbations:
- pelvis pushes
- random onset
- random direction
- random duration
- random magnitude curriculum

Goal:
- keep nominal gait alive while giving impedance a real gradient

### Stage 4 — add stability signal
Introduce:
- `alpha`
- external force estimate if available
- pelvis acceleration / contact anomaly features

Use them for:
- impedance gate input
- reward blending
- later analysis

### Stage 5 — add exoskeleton-relevant disturbances
After simple pushes work:
- joint-level assistance/resistance torques
- onset/offset surprises
- belt-speed perturbations if available
- interaction loads

Goal:
- move from generic perturbation robustness to exoskeleton-relevant robustness

---

## 6.3 Evaluation criteria for Phase 1
A VIC model should **not** be accepted because its nominal reward is high.  
It should be accepted if it improves the right metrics.

### Main accept criteria
A VIC model passes if it shows:
1. **higher perturbation recovery success** than no-VIC baseline
2. **better recovery quality** (recovery time / step placement / WBAM etc.)
3. **no catastrophic nominal gait loss**

### Practical threshold
A good first success condition would be:

- better perturbation metrics than no-VIC baseline, with
- nominal reward / nominal episode length degradation kept within roughly **10–15%**

If nominal loss is much larger than the reactive benefit, the impedance design still needs work.

---

## 7. Management plan: how to manage and modify from here

## 7.1 Organize work into four branches
Instead of mixing everything into one experimental stream:

### Branch A — Nominal locomotion baseline
Owner goal:
- keep strongest no-VIC walking policy stable
- maintain nominal eval battery

### Branch B — VIC residual module
Owner goal:
- action head split
- gating
- low-pass
- sigma separation
- 4-group actionization

### Branch C — Perturbation curriculum + alpha
Owner goal:
- push generator
- recovery stage
- alpha computation
- perturbed evaluation battery

### Branch D — Analysis / tooling
Owner goal:
- phase-CCF plots
- perturbation-conditioned impedance logs
- nominal vs perturbed metric dashboard

This separation prevents endless architecture churn.

---

## 7.2 Freeze a parent-child experiment structure
For every new run:
- define exactly one parent checkpoint,
- change only one main hypothesis at a time,
- record one-line config diff,
- evaluate on the same battery.

### Suggested naming rule
Use a run name that encodes:
- parent
- VIC mode
- perturbation mode
- gate status
- group count
- sigma version

Example:
```text
H5Base_D20k__VIC4_residual_gateA_sigma3__PushCurr_v1
```

---

## 7.3 Define stop rules
Without stop rules, this topic can absorb too much time.

### Recommended stop rules
- No more than **2–3 architecture variants** before evaluating on perturbation battery.
- Do not revisit 8-group unless 4-group already shows clear reactive benefit.
- Do not add upper-body impedance until lower-body perturbation value is demonstrated.
- If VIC still shows no benefit after:
  - residual head,
  - low sigma,
  - gating,
  - perturbation curriculum,
  - alpha input,
  then pause VIC expansion and try a simpler fallback.

---

## 7.4 Recommended fallback paths if VIC still underperforms
If residual 4-group VIC still fails, use one of these fallbacks:

### Fallback A — fixed nominal phase prior + no learned impedance
- use a hand-designed or extracted ankle-dominant phase curve
- no stochastic impedance head

### Fallback B — trigger-based reflex stiffness
- impedance increases only when perturbation indicators cross threshold
- simpler than full policy output

### Fallback C — ankle-only impedance
- if data show the useful signal is almost entirely distal

### Fallback D — motion-library / retrieval-first strategy
- use perturbation-conditioned retrieval or exemplar conditioning
- let impedance stay simple while the motion prior handles recovery structure

---

## 8. Relation to the broader WalkON program

The broader program is still coherent.  
This Phase 0 result mainly changes **ordering and emphasis**, not direction.

## 8.1 Topic A (state estimation / deformation-aware control)
Later, the impedance gate should probably depend on better state signals:
- pelvis/trunk state,
- interaction effects,
- latent deformation / misalignment,
- balance margin.

So Topic A becomes even more important, not less.

## 8.2 Topic B / D (upper-body agency and hybrid human model)
Do **not** bring upper-body voluntary motion into the VIC question yet.

First show:
- lower-body impedance improves perturbation recovery in simulation.

Then later:
- add upper-body motion priors / hybrid human response,
- because reactive balance and upper-body compensation are coupled.

## 8.3 Topic E (personalization)
This result actually strengthens the personalization story.

Why?
Because nominal walking may need only weak impedance modulation, but **perturbation recovery and assistance timing are much more subject-specific**.

So personalization should target:
- timing,
- duration,
- asymmetry,
- and stiffness scaling under perturbation,

rather than only nominal kinematic tracking.

---

## 9. Concrete implementation suggestions

## 9.1 Minimal code changes I would prioritize
1. **Split policy head**
   - motor head
   - impedance head

2. **Separate impedance sigma**
   - independent init and schedule

3. **Impedance low-pass filter**
   - prevent per-step noise injection

4. **Impedance gate**
   - closed or weak in nominal episodes
   - opened by perturbation/stability evidence

5. **Unified logging**
   - mean / std of `delta_xi`
   - per-group phase curves
   - perturbation-conditioned statistics
   - gate activation histogram

6. **Evaluation battery**
   - nominal walk
   - push recovery
   - asymmetric perturbation
   - assistance onset/offset disturbance

---

## 9.2 Suggested config starting point
Illustrative only:

```yaml
vic:
  enabled: true
  mode: residual
  groups: 4
  upper_body_fixed: true

  xi_range_nominal: [-0.30, 0.30]
  xi_range_perturb: [-0.70, 0.70]

  update_every_n_steps: 4
  lpf_tau_sec: 0.15

  sigma_init: -2.8
  smoothness_weight: 0.01
  neutral_prior_weight: 0.01

  gate:
    enabled: true
    inputs: [alpha, pelvis_acc, contact_anomaly, ext_force_est, exo_torque]
    alpha_threshold: 0.15
    init_bias_closed: true

training:
  curriculum:
    nominal_episode_ratio: 0.70
    perturb_episode_ratio: 0.30
    perturb_force_N: [20, 200]
    perturb_duration_sec: [0.10, 0.50]
```

Again, these values are **starting points**, not final truths.

---

## 10. Final recommendation

If I compress everything to one sentence:

> **Keep the original research direction, but demote impedance action from a full-time gait action to a gated residual adaptation module, then test it where it should matter: perturbation, interaction, and personalization.**

### My recommended mainline from today
1. Finish D and E.
2. Run phase-CCF analysis.
3. Freeze the no-VIC nominal baseline.
4. Reintroduce VIC only as 4-group residual action with low sigma and smoothing.
5. Move immediately to mixed nominal + perturbation training.
6. Judge success on perturbation metrics, not nominal reward alone.
7. Only after positive evidence, consider richer impedance structure.

That path preserves the original plan, reduces wasted exploration, and gives the impedance idea a fair test.

---

## 11. Reference basis

### Internal documents
- `260416_phase0_vic_feasibility_and_next_steps.md`
- `walkon_integrated_research_program_detailed.md`

### External literature used for reasoning
1. Roberto Martín-Martín et al., **Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks**, IROS 2019 / arXiv:1906.08880.
2. Adrian Hartmann et al., **Deep Compliant Control for Legged Robots**, ICRA 2024.
3. Botian Xu et al., **FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots**, 2025.
4. Antonia Bronars, Younghyo Park, Pulkit Agrawal, **Tune to Learn: How Controller Gains Shape Robot Policy Learning**, arXiv:2604.02523, 2026.
5. Oliver Hausdörfer et al., **Latent Action Priors for Locomotion with Deep Reinforcement Learning**, arXiv:2410.03246, 2025.
6. Abdel-Rahman Akl et al., **Muscle Co-Activation around the Knee during Different Walking Speeds in Healthy Females**, Sensors 2021.
7. Nancy T. Nguyen et al., **Co-contraction about the ankle increases with the threat of a walking perturbation**, Journal of Electromyography and Kinesiology, 2025.
8. Stacie A. Chvatal and Lena H. Ting, **Voluntary and Reactive Recruitment of Locomotor Muscle Synergies during Perturbed Walking**, Journal of Neuroscience, 2012.
9. Maria T. Tagliaferri and Inseung Kang, **Systematic Evaluation of Hip Exoskeleton Assistance Parameters for Enhancing Gait Stability During Ground Slip Perturbations**, arXiv:2601.15056, 2026.

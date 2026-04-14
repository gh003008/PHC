# Motion Plan Layer (MPL): Refined Methodology for Personalized Humanoid Control with Perturbation Response

**Version**: v1.0 — 2026-04-13  
**Scope**: Lower-body (12 DoF), treadmill/level walking, perturbation recovery  
**Target hardware**: Exoskeleton-assisted locomotion  

---

## 1. Problem Statement

The goal of MPL is to produce a humanoid control policy that:

1. **Generates stable locomotion** from arbitrary Center-of-Mass (CoM) velocity commands without falling
2. **Preserves individual movement style** — trained on one person's data, the policy should reproduce that person's cadence, stride, impedance patterns, and idiosyncratic habits
3. **Handles perturbations as a core capability** — responses to external/robot forces (pushes, slips, exoskeleton torques) must be learned from *actual measured human reactions*, not discovered purely through RL exploration

The prior v09 architecture proposed a two-module split — Nominal Generator **G** and Residual/Impedance Policy **R** — where G handles reference trajectory generation and R corrects for physics and disturbances. This document refines that architecture, with particular attention to **how reflexive responses are captured, represented, and generalized**.

---

## 2. Architecture Overview: Unified Latent-Conditioned Policy

### 2.1 Motivation: Why Revise the G/R Split

The original G/R split assumes a clean separation:
- G produces "what the person would do in nominal conditions"
- R adds corrections for physics mismatch and disturbances

This separation is problematic for perturbation responses because:

1. **Human perturbation reactions are not residuals of nominal walking.** A compensatory step response involves a qualitatively different motor plan — altered foot placement, trunk counter-rotation, arm bracing — not a delta on top of normal gait. Forcing this through an additive residual limits expressiveness.
2. **The G/R boundary creates information bottleneck.** R only sees G's output, not the full motion library context. During recovery, R needs access to relevant perturbation-response exemplars, not just the current nominal trajectory.
3. **Efficiency concern.** Two separate networks that must coordinate adds complexity without clear benefit when the motion library already contains both walking and perturbation-response data.

### 2.2 Proposed Architecture: Single Policy π with Mode-Conditioned Motion Prior

Instead of a hard G/R split, we propose a **single policy network** π conditioned on:
- Current state $s_t$
- Stability margin $\alpha_t$ (continuous scalar, see §3)
- Motion context $m_t$ retrieved from the motion library (see §4)
- CoM velocity command $c_t$

The policy outputs:
- Target joint positions $q_{\text{ref},t} \in \mathbb{R}^{12}$
- Compliance Control Factors $\xi_t \in \mathbb{R}^{N_g}$ (per joint group)

$$
(q_{\text{ref},t},\; \xi_t) = \pi_\theta(s_t,\; \alpha_t,\; m_t,\; c_t)
$$

The motion library retrieval mechanism implicitly performs the role of the old G module — it provides trajectory context — while the policy network performs both nominal tracking and perturbation recovery in a unified computation.

**Torque computation** follows the Variable Impedance Control (VIC) law:

$$
\tau_j = K_{p,j} \cdot 2^{\xi_j} \cdot (q_{\text{ref},j} - q_j) - K_{d,j} \cdot 2^{\xi_j} \cdot \dot{q}_j
$$

where $K_{p,j}$, $K_{d,j}$ are nominal PD gains, and $\xi_j$ is the learned CCF that modulates joint impedance.

### 2.3 Network Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Observation Encoder                 │
│                                                     │
│  s_t: proprioception (q, dq, body pos/rot/vel)      │
│  α_t: stability margin [0,1]                        │
│  c_t: CoM velocity command (vx, vy, ωyaw)           │
│  φ_t: gait phase (sin θ, cos θ)                     │
│                                                     │
│  → MLP(512) → LayerNorm → SiLU                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Motion Context Fusion                   │
│                                                     │
│  m_t: retrieved motion window [t : t+K]              │
│       (joint angles + velocities, K=10 frames)       │
│  α-weighted attention:                               │
│    low α → attend to nominal walking exemplars       │
│    high α → attend to perturbation-response exemplars│
│                                                     │
│  → Cross-attention(state_enc, motion_enc)             │
│  → MLP(512) → SiLU                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              GRU Temporal Module                      │
│  hidden_dim = 256                                    │
│  Maintains temporal context across steps             │
└────────────────────┬────────────────────────────────┘
                     │
              ┌──────┴──────┐
              ▼              ▼
     ┌──────────────┐ ┌──────────────┐
     │  q_ref head  │ │   CCF head   │
     │  MLP(256,12) │ │  MLP(256,Ng) │
     │  + tanh·Δq   │ │  clamp[-1,1] │
     └──────────────┘ └──────────────┘
```

The GRU provides temporal memory critical for multi-step recovery sequences (e.g., a compensatory step that takes 300-500ms to complete).

---

## 3. Stability Margin α: The Mode-Switching Signal

### 3.1 Definition

α is a continuous scalar in [0, 1] computed from real-time physics state. It replaces discrete mode switching (normal vs. recovery) with a smooth transition signal.

$$
\alpha_t = \text{clamp}\Big(w_1 \cdot d_{\text{CoP}} + w_2 \cdot d_{\text{XCoM}} + w_3 \cdot \|\dot{L}_t\|,\;\; 0,\; 1\Big)
$$

where:

**CoP margin** $d_{\text{CoP}}$: normalized distance from Center of Pressure to the nearest edge of the support polygon.

$$
d_{\text{CoP}} = 1 - \frac{\text{dist}(p_{\text{CoP}},\; \partial \mathcal{S})}{\text{dist}(p_{\text{centroid}},\; \partial \mathcal{S})}
$$

When CoP is at the polygon centroid, $d_{\text{CoP}} = 0$ (stable). When CoP reaches the boundary, $d_{\text{CoP}} = 1$ (critical).

**Extrapolated CoM (XCoM) margin** $d_{\text{XCoM}}$: based on the inverted pendulum model:

$$
p_{\text{XCoM}} = p_{\text{CoM}} + \frac{\dot{p}_{\text{CoM}}}{\omega_0}, \quad \omega_0 = \sqrt{\frac{g}{h_{\text{CoM}}}}
$$

$$
d_{\text{XCoM}} = \max\Big(0,\; \frac{\text{signed\_dist}(p_{\text{XCoM}},\; \mathcal{S})}{\ell_{\text{foot}}}\Big)
$$

If XCoM is inside the support polygon, $d_{\text{XCoM}} = 0$. If outside, it is the normalized overshoot distance.

**Angular momentum rate** $\|\dot{L}_t\|$: rate of change of whole-body angular momentum, normalized. Rapid angular momentum changes precede falls.

**Default weights**: $w_1 = 0.3$, $w_2 = 0.5$, $w_3 = 0.2$. The XCoM term dominates because it is the most predictive of whether a compensatory step will be needed (Hof et al., 2005).

### 3.2 How α Affects the Policy

α enters the policy as part of the observation vector. Additionally, it modulates two external mechanisms:

**1. Motion library retrieval weighting** (§4):
- Low α → retrieve nominal walking clips
- High α → retrieve perturbation-response clips (weighted by motion type tag)

**2. Reward blending** (§6):
- Low α → imitation reward dominates (track reference closely)
- High α → balance reward dominates (stay upright, recover CoM)

$$
r_t = (1 - \alpha_t) \cdot r_{\text{imit}} + \alpha_t \cdot r_{\text{balance}} + r_{\text{style}} + r_{\text{power}}
$$

### 3.3 Human Balance Strategy Correspondence

The continuous α naturally maps to the three classical balance strategies:

| α Range | Strategy | Policy Behavior |
|---------|----------|----------------|
| 0.0–0.3 | Ankle strategy | Small CCF adjustments, tight reference tracking |
| 0.3–0.6 | Hip strategy | Larger hip/trunk corrections, increased stiffness |
| 0.6–1.0 | Stepping strategy | Foot placement override, maximum CCF range |

This mapping is not hard-coded — it emerges from training with the perturbation curriculum (§5).

---

## 4. Motion Library: Integrating Walking and Perturbation Responses

### 4.1 The Core Insight: Perturbation Responses Are Motion Data

The key methodological revision: **human perturbation responses must be captured as motion data and included in the motion library**, alongside normal walking. The policy cannot learn person-specific recovery strategies from RL exploration alone because:

1. Human recovery involves learned motor synergies developed over a lifetime — these are person-specific
2. RL in simulation discovers *physically valid* recovery motions, but not *human-like* ones
3. For exoskeleton control, the robot must predict and complement the human's actual response pattern, not a generic optimal one

### 4.2 Required Data Collection Protocol

For each subject S, collect:

**A. Nominal walking data** (existing):
- Treadmill walking at multiple speeds (0.75, 1.0, 1.25 m/s)
- Level and incline conditions
- Stop-and-go transitions
- MoCap (optical markers → SMPL joint angles) + force plates

**B. Perturbation response data** (new requirement):
- **Treadmill belt perturbations**: sudden speed changes (acceleration/deceleration), causing trip-like and slip-like responses
- **Lateral perturbations**: mediolateral platform translations or waist-pull cables at calibrated force levels
- **Sagittal perturbations**: anterior-posterior pushes at pelvis level
- Each perturbation at 3+ intensity levels (mild → near-fall)
- Multiple repetitions per condition (≥5 trials each)

**C. Instrumentation requirements**:
- Full-body optical MoCap (120+ Hz) → SMPL joint angles via MoSh++
- Ground reaction forces (force plates or instrumented treadmill)
- EMG on lower-limb muscles (optional but valuable for impedance estimation):
  - Tibialis anterior, gastrocnemius (medial), soleus
  - Vastus lateralis, biceps femoris
  - Gluteus medius

### 4.3 Motion Library Structure

The library $\mathcal{L}$ is indexed by two keys: **command** (CoM velocity) and **motion type** (nominal vs. perturbation category):

$$
\mathcal{L} = \{(\text{clip}_i,\; c_i,\; \text{type}_i,\; \alpha_i^{\text{peak}},\; \text{subject\_id})\}_{i=1}^{N}
$$

where:
- $c_i$: average CoM velocity during the clip
- $\text{type}_i \in \{\text{walk},\; \text{perturb\_trip},\; \text{perturb\_slip},\; \text{perturb\_lateral},\; \text{perturb\_push}\}$
- $\alpha_i^{\text{peak}}$: peak stability margin observed during the clip (computed offline from force plate + kinematics)
- subject_id: identifies whose data this is

### 4.4 Context Retrieval Mechanism

At each simulation step, the retrieval function selects the most relevant motion context:

$$
m_t = \sum_{i \in \text{top-}K} w_i \cdot \text{clip}_i[t : t+W]
$$

The weighting combines command proximity and α-conditioned type matching:

$$
w_i \propto \exp\Big(-\frac{\|c_t - c_i\|^2}{\sigma_c^2}\Big) \cdot \exp\Big(-\frac{(\alpha_t - \alpha_i^{\text{peak}})^2}{\sigma_\alpha^2}\Big)
$$

When α is low, clips with low $\alpha^{\text{peak}}$ (nominal walking) dominate. When α rises (perturbation detected), clips with matching $\alpha^{\text{peak}}$ (perturbation responses) are retrieved. This provides the policy with relevant exemplars for the current stability state.

$K = 3$ nearest clips, $W = 10$ frames (0.33s at 30Hz). $\sigma_c = 0.3$ m/s, $\sigma_\alpha = 0.2$.

---

## 5. Training: Perturbation Curriculum

### 5.1 Three-Stage Training

**Stage 1: Nominal Walking Imitation (epochs 0–5,000)**

Only walking clips are active. The policy learns to:
- Track reference joint trajectories from MoCap
- Maintain balance during normal walking
- Modulate CCF according to gait phase (stance = stiff, swing = compliant)

Reward:
$$
r_t = r_{\text{imit}} + r_{\text{amp}} + r_{\text{power}} + r_{\text{ccf\_bio}}
$$

where the imitation reward is:

$$
r_{\text{imit}} = w_p \cdot e^{-k_p \|\Delta p\|^2} + w_r \cdot e^{-k_r \|\Delta \theta\|^2} + w_v \cdot e^{-k_v \|\Delta \dot{p}\|^2} + w_\omega \cdot e^{-k_\omega \|\Delta \omega\|^2} + w_f \cdot e^{-k_f \|\Delta p_{\text{foot}}\|^2}
$$

| Component | Weight $w$ | Sharpness $k$ |
|-----------|-----------|---------------|
| Body position | 0.4 | 200 |
| Body rotation | 0.3 | 10 |
| Linear velocity | 0.2 | 1.0 |
| Angular velocity | 0.1 | 0.1 |
| Foot position | 0.2 | 40 |

**Stage 2: Perturbation Introduction (epochs 5,000–15,000)**

Perturbation clips are activated in the motion library. Simultaneously, random external forces are applied to the simulated humanoid:

$$
F_{\text{ext}}(t) \sim \mathcal{U}(-F_{\max}(e),\; F_{\max}(e))
$$

where $F_{\max}(e)$ ramps linearly with epoch $e$:

$$
F_{\max}(e) = F_{\min} + \frac{e - e_{\text{start}}}{e_{\text{end}} - e_{\text{start}}} \cdot (F_{\max,\text{final}} - F_{\min})
$$

Default: $F_{\min} = 20\text{N}$, $F_{\max,\text{final}} = 200\text{N}$, applied at pelvis, lasting 0.1–0.5s per perturbation event, with 0.5–5s intervals between events.

The reward transitions to α-weighted blending:

$$
r_t = (1 - \alpha_t) \cdot r_{\text{imit}} + \alpha_t \cdot r_{\text{balance}} + \lambda_{\text{style}} \cdot r_{\text{amp}} + r_{\text{power}}
$$

The balance reward encourages upright posture and CoM recovery without prescribing a specific strategy:

$$
r_{\text{balance}} = w_h \cdot e^{-k_h (h_{\text{CoM}} - h_{\text{target}})^2} + w_u \cdot e^{-k_u \|\theta_{\text{trunk}}\|^2} + w_c \cdot e^{-k_c \|v_{\text{CoM}} - v_{\text{cmd}}\|^2}
$$

| Balance Component | Weight | Description |
|------------------|--------|-------------|
| CoM height | 0.3 | Penalizes crouching/falling |
| Trunk uprightness | 0.4 | Penalizes excessive lean |
| CoM velocity recovery | 0.3 | Reward for returning to commanded velocity |

**Stage 3: Full Perturbation with Style Enforcement (epochs 15,000–30,000)**

AMP discriminator weight increases to enforce that recovery motions look *like the subject's actual recovery patterns*, not generic optimal solutions:

$$
r_t = (1 - \alpha_t) \cdot r_{\text{imit}} + \alpha_t \cdot r_{\text{balance}} + \lambda_{\text{style}}(e) \cdot r_{\text{amp}} + r_{\text{power}}
$$

$$
\lambda_{\text{style}}(e) = \lambda_{\min} + \frac{e - e_2}{e_3 - e_2} \cdot (\lambda_{\max} - \lambda_{\min})
$$

This ensures that even during recovery, the policy matches the subject's *measured* perturbation response patterns.

### 5.2 AMP Discriminator with Mixed Motion Types

The AMP (Adversarial Motion Priors) discriminator $D_\phi$ is trained on the full motion library (walking + perturbation responses):

$$
r_{\text{amp}} = \max\Big(0,\; 1 - 0.25 \cdot (D_\phi(s_t, s_{t+1}) - 1)^2\Big)
$$

The discriminator learns to distinguish "human-like" transitions from "non-human-like" ones. Because the training data includes both smooth walking and rapid corrective movements, the discriminator learns a *unified distribution* that considers both as valid human motion.

**Key insight**: The discriminator does not need to explicitly know whether a transition is "walking" or "recovering." A perturbation response from the real human is just as "human" as a normal gait cycle. The discriminator naturally learns this composite distribution, providing style reward for both nominal and reactive behaviors.

This is analogous to how ASE (Adversarial Skill Embeddings, Peng et al. 2022) learns a latent space over diverse skills — the discriminator acts as a unified "humanness" score across all motion types.

---

## 6. Personalization

### 6.1 Subject-Specific Motion Library

For personalization, the entire pipeline uses data from a single subject S:

$$
\mathcal{L}_S = \{\text{clips from subject } S \text{ only}\}
$$

The AMP discriminator is also trained exclusively on S's data:

$$
D_S: \text{transitions} \rightarrow [0, 1]
$$

This ensures that the style reward enforces S's specific movement patterns — their characteristic step length, arm swing (when upper body is included), push-off timing, and critically, their specific perturbation response strategies.

### 6.2 What Makes Perturbation Responses Person-Specific

Biomechanics literature documents significant inter-individual variability in balance recovery:

1. **Response latency**: Time from perturbation onset to corrective muscle activation (50–150ms for stretch reflexes, 80–200ms for triggered responses). Varies with age, pathology, and training.
2. **Strategy preference**: Some individuals preferentially use hip strategy even at perturbation magnitudes where others switch to stepping (Horak & Nashner, 1986).
3. **Step characteristics**: Compensatory step length, direction, and timing vary by ~30% across healthy adults (Maki & McIlroy, 1997).
4. **Impedance modulation**: Co-contraction patterns during and after perturbation differ between individuals, reflecting different trade-offs between stiffness (stability) and compliance (efficiency).

By training the policy on subject S's actual perturbation data, these individual patterns are captured in:
- The motion library (kinematic exemplars)
- The AMP discriminator (style enforcement)
- The learned CCF patterns (impedance strategy)

### 6.3 Population Pre-training + Subject Fine-tuning

For practical deployment:

1. **Pre-train** on a population dataset (multiple subjects): builds a general-purpose policy that understands basic walking and balance physics
2. **Fine-tune** on subject S's data (including S's perturbation responses): specializes the policy to S's movement patterns

$$
\theta_S = \text{fine-tune}(\theta_{\text{pop}},\; \mathcal{L}_S,\; D_S)
$$

Fine-tuning updates all network weights with a reduced learning rate, keeping the population-learned physics understanding while adapting to S's specific movement style.

### 6.4 Physiological Parameter Injection

Measurable subject-specific parameters are injected directly into the simulation, not learned:

| Parameter | Source | Effect |
|-----------|--------|--------|
| Body segment masses | DXA / anthropometry | Affects dynamics |
| Joint ROM limits | Goniometry | Constrains action space |
| Nominal PD gains ($K_p$, $K_d$) | Isometric strength tests + EMG | Scales torque capacity |
| Reflex delay | EMG onset latency measurement | Adds control latency in sim |

The CCF $\xi$ learned by the policy then represents the *relative* impedance modulation on top of these person-specific baselines.

---

## 7. Handling Robot/External Forces

### 7.1 Why This Is Critical for Exoskeleton Control

From the exoskeleton perspective, the human's response to robot-generated forces is the most important behavior to predict. An exoskeleton that cannot anticipate the human's reaction to its own torques will:
- Fight against the human (wasted energy, discomfort)
- Destabilize the human by producing unexpected forces
- Fail to assist during the moments that matter most (perturbation recovery)

### 7.2 Robot Force as Observation

The policy explicitly observes applied external forces:

$$
s_t = (\text{proprioception},\; \alpha_t,\; \phi_t,\; F_{\text{ext},t})
$$

where $F_{\text{ext},t}$ includes:
- Measured exoskeleton joint torques (in deployment)
- Simulated random perturbation forces (during training)
- External push forces

This makes force-response a **first-class input**, not a disturbance to be rejected. The policy learns: "when I feel this force pattern, the human would respond like this."

### 7.3 Training with Structured Force Profiles

Beyond random pushes, training includes structured force profiles that mimic exoskeleton interactions:

**A. Joint-level torque perturbations**: Random torques applied at hip, knee, ankle — mimicking an exoskeleton that is slightly misaligned or applying imperfect assistance.

$$
\tau_{\text{ext},j}(t) = A_j \cdot \sin(2\pi f_j t + \phi_j), \quad A_j \sim \mathcal{U}(0, \tau_{\max})
$$

**B. Phased assistance/resistance**: Torques that assist during swing and resist during stance (or vice versa), forcing the policy to adapt to exoskeleton-like force patterns.

**C. Sudden onset/offset**: Simulating an exoskeleton engaging or disengaging, which produces a sudden force change.

### 7.4 Impedance Response to External Forces

The CCF output $\xi$ provides the mechanism for impedance adaptation to forces:

- When a stabilizing external force is detected (e.g., exoskeleton assisting stance) → policy can reduce $\xi$ (become more compliant, accept assistance)
- When a destabilizing force is detected → policy increases $\xi$ (stiffen to resist)
- When force suddenly disappears → policy transiently increases $\xi$ (brace for instability), then relaxes

These patterns are learned from the actual human's measured responses, not hand-designed rules.

---

## 8. Mathematical Summary

### 8.1 State Space

$$
s_t = \big(\underbrace{q_t, \dot{q}_t}_{\text{joint state}},\; \underbrace{p_t^{\text{root}}, R_t^{\text{root}}, v_t^{\text{root}}, \omega_t^{\text{root}}}_{\text{root state}},\; \underbrace{\alpha_t}_{\text{stability}},\; \underbrace{\sin\phi_t, \cos\phi_t}_{\text{gait phase}},\; \underbrace{F_{\text{ext},t}}_{\text{external force}}\big) \in \mathbb{R}^{d_s}
$$

### 8.2 Action Space

$$
a_t = \big(\underbrace{q_{\text{ref},t}}_{\in \mathbb{R}^{12}},\; \underbrace{\xi_t}_{\in \mathbb{R}^{N_g}}\big) \in \mathbb{R}^{12 + N_g}
$$

with $N_g = 6$ joint groups for lower body: L_Hip, L_Knee, L_Ankle, R_Hip, R_Knee, R_Ankle.

### 8.3 Torque Law

$$
\tau_j = K_{p,j} \cdot 2^{\xi_j} \cdot (q_{\text{ref},j} - q_j) - K_{d,j} \cdot 2^{\xi_j} \cdot \dot{q}_j, \quad \xi_j \in [-1, 1]
$$

This gives an impedance modulation range of $[2^{-1}, 2^{1}] = [0.5\times, 2.0\times]$ of nominal stiffness.

### 8.4 Reward Function

$$
r_t = \underbrace{(1 - \alpha_t) \cdot r_{\text{imit}}}_{\text{tracking}} + \underbrace{\alpha_t \cdot r_{\text{balance}}}_{\text{recovery}} + \underbrace{\lambda_{\text{style}} \cdot r_{\text{amp}}}_{\text{style}} + \underbrace{r_{\text{power}}}_{\text{efficiency}} + \underbrace{r_{\text{ccf\_bio}}}_{\text{impedance guidance}}
$$

### 8.5 Policy Optimization

PPO (Proximal Policy Optimization) with AMP-style discriminator:

$$
\mathcal{L}_\pi = \mathbb{E}\Big[\min\big(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\big)\Big]
$$

$$
\mathcal{L}_D = -\mathbb{E}_{(s,s') \sim \mathcal{L}}\big[\log D_\phi(s,s')\big] - \mathbb{E}_{(s,s') \sim \pi}\big[\log(1 - D_\phi(s,s'))\big] + \lambda_{\text{gp}} \cdot \text{GP}
$$

where GP is the gradient penalty for training stability.

---

## 9. Implementation Roadmap

| Phase | Description | Data Required | Deliverable |
|-------|-------------|---------------|-------------|
| **Phase 0** (current) | Single-policy VIC with walking data | S001 treadmill walking | Feasibility validation: stable walking with CCF |
| **Phase 1** | Add α computation, perturbation curriculum | Same walking data + simulated pushes | Policy that recovers from sim perturbations |
| **Phase 2** | Collect perturbation response MoCap | **New: S001 perturbed walking data** | Motion library with walking + perturbation clips |
| **Phase 3** | Full MPL training with perturbation data | Walking + perturbation library | Personalized policy matching S001's recovery style |
| **Phase 4** | Exoskeleton co-simulation | Phase 3 policy + exo model | Exo controller that predicts human response |

### Current Status: Phase 0

- VIC architecture implemented (humanoid_im_vic.py)
- CCF learning (Stage 2) active with 6-group lower-body grouping
- Motion library: 98 clips × 15s from S001 walking data (level + incline + stop-and-go)
- Foot position tracking reward added
- MPL_H5_v5 training in progress (epoch ~5900, reward ~33.5)

### Critical Next Step: Phase 2 Data Collection

Without measured human perturbation response data, Phases 2–4 cannot proceed. The data collection protocol (§4.2) should be prioritized. Minimum viable dataset:
- 1 subject (S001)
- 3 perturbation types (belt acceleration, belt deceleration, lateral push)
- 3 intensity levels each
- 5 repetitions per condition
- Total: ~45 perturbation trials + existing walking data

---

## 10. Comparison with Related Work

| Approach | Strengths | Limitation for Our Goals |
|----------|-----------|------------------------|
| PHC (Luo et al. 2023) | Perpetual control, fail-state recovery | Recovery is RL-discovered, not personalized |
| AMP (Peng et al. 2021) | Natural motion from unstructured data | No explicit perturbation handling |
| ASE (Peng et al. 2022) | Skill latent space, reusable behaviors | Skills are discrete; no continuous α blending |
| CALM (Tessler et al. 2023) | Conditional motion with latent space | No impedance control, no perturbation data |
| DeepMimic (Peng et al. 2018) | Reference motion tracking | Single clip, no multi-modal library |
| **MPL (Ours)** | α-blended retrieval, perturbation data, VIC, personalized | Requires subject-specific perturbation MoCap |

**Key differentiator**: MPL is the only approach that combines (1) real measured perturbation responses as training data, (2) variable impedance control for hardware deployment, and (3) continuous stability-margin-driven mode blending — all personalized to an individual subject.

---

## References

1. Peng, X.B., et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation." ACM TOG, 2021.
2. Peng, X.B., et al. "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters." ACM TOG, 2022.
3. Tessler, C., et al. "CALM: Conditional Adversarial Latent Models for Directable Virtual Characters." ACM TOG, 2023.
4. Luo, Z., et al. "Perpetual Humanoid Control for Real-time Simulated Avatars." ICCV, 2023.
5. Peng, X.B., et al. "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills." ACM TOG, 2018.
6. Horak, F.B., Nashner, L.M. "Central programming of postural movements: adaptation to altered support-surface configurations." J. Neurophysiol., 1986.
7. Maki, B.E., McIlroy, W.E. "The role of limb movements in maintaining upright stance: the 'change-in-support' strategy." Phys. Ther., 1997.
8. Hof, A.L., et al. "The condition for dynamic stability." J. Biomech., 2005.
9. Winter, D.A. "Biomechanics and Motor Control of Human Movement." 4th ed., Wiley, 2009.
10. Peterka, R.J. "Sensorimotor integration in human postural control." J. Neurophysiol., 2002.

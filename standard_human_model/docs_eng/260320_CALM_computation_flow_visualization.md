# CALM Computation Flow Visualization

Created: 2026-03-20
Translated from Korean original: `260320_CALM_계산플로우_시각화.md`

---

## 1. Overall Pipeline Overview

```
External Inputs (IsaacGym Simulator)
  ├─ dof_pos      (num_envs, 69)   ← Current joint angles [rad]
  ├─ dof_vel      (num_envs, 69)   ← Current joint velocities [rad/s]
  ├─ descending_cmd (num_envs, 20) ← Higher controller muscle commands [0,1]
  └─ contact_forces (num_envs, 4)  ← Foot contact forces [N]
         │
         ▼
  ╔══════════════════════════════════════════════════════════╗
  ║                   HumanBody.compute_torques()            ║
  ║                                                          ║
  ║  Step 1: Muscle Kinematics      (moment_arm.py)          ║
  ║  Step 2: Reflex Layer           (reflex_controller.py)   ║
  ║  Step 3: Activation Dynamics    (activation_dynamics.py) ║
  ║  Step 4: Force Generation       (muscle_model.py)        ║
  ║  Step 5: Torque Mapping         (moment_arm.py)          ║
  ║  Step 6: Ligament Forces        (ligament_model.py)      ║
  ╚══════════════════════════════════════════════════════════╝
         │
         ▼
  tau_total (num_envs, 69) → IsaacGym apply_forces()
```

---

## 2. Step-by-Step Detail

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT                                                               │
│  dof_pos (B,69)  dof_vel (B,69)  descending_cmd (B,20)              │
└────────────┬─────────────┬─────────────┬────────────────────────────┘
             │             │             │
             ▼             ▼             │
┌─────────────────────────────────────┐ │
│  STEP 1: MUSCLE KINEMATICS          │ │
│  R (20, 69) — constant matrix       │ │
│  l_muscle = l_slack - R @ dof_pos   │ │
│  v_muscle = -(R @ dof_vel)          │ │
│  Bi-articular: hamstrings R has     │ │
│    Hip(-0.06) and Knee(+0.03)       │ │
└──────┬───────────────────┬──────────┘ │
       │                   │            │
       │            ┌──────┴────────────────────────────────────┐
       │            │  STEP 2: REFLEX LAYER                      │
       │            │  ① Stretch Reflex (with delay)             │
       │            │     if v_muscle > threshold:               │
       │            │       Δa += gain × (v - threshold)         │
       │            │     [Healthy: gain=1.0 / Spastic: gain=8.0]│
       │            │  ② GTO Inhibition                          │
       │            │     if F/f_max > gto_threshold: Δa -= ...  │
       │            │  ③ Reciprocal Inhibition                   │
       │            │     quadriceps ↔ hamstrings                │
       │            │  a_cmd = clamp(descending_cmd + Δa, 0,1)   │
       │            └──────────────────┬─────────────────────────┘
       │                               │ a_cmd (B,20)
       │                               ▼
       │            ┌──────────────────────────────────────────┐
       │            │  STEP 3: ACTIVATION DYNAMICS             │
       │            │  1st-order ODE (per muscle):             │
       │            │  if a_cmd > a_prev:                      │
       │            │    da = (a_cmd - a) / tau_act  [fast]    │
       │            │    tau_act: 15~20ms / 50ms (Parkinson)   │
       │            │  else:                                   │
       │            │    da = (a_cmd - a) / tau_deact [slow]   │
       │            │    tau_deact: 60~80ms / 200ms (Parkinson)│
       │            │  a(t+dt) = clamp(a + da×dt, 0, 1)       │
       │            └──────────────────┬───────────────────────┘
       │                               │ activation (B,20)
       ▼                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: FORCE GENERATION — Hill-type                         │
│  l_norm = l_muscle / l_opt        (B, 20)                    │
│  v_norm = v_muscle / (v_max × l_opt) (B, 20)                │
│                                                              │
│  F_active  = a × f_max × f_FL(l_norm) × f_FV(v_norm) × cos(α)│
│  f_FL(l) = exp(-((l-1)/σ)²)   ← peak: l_norm=1.0           │
│  f_FV(v) = concentric: weakens / eccentric: strengthens     │
│                                                              │
│  F_passive = f_max × k_pe × (exp(l_norm/ε₀) - 1)           │
│  F_damping = damping × f_max × v_muscle                     │
│  F_total = clamp(F_active + F_passive + F_damping, 0, ∞)    │
└──────────────────────────────┬───────────────────────────────┘
                               │ F_muscle (B,20)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: TORQUE MAPPING                                       │
│  tau_muscle = F_muscle @ R          (B,20)×(20,69) = (B,69) │
│  Example: hamstrings_L activation →                          │
│    τ_Hip_x  += F_ham × (-0.06)  ← hip extension torque      │
│    τ_Knee_x += F_ham × (+0.03)  ← knee flexion torque       │
│    [Bi-articular coupling automatic]                         │
└──────────────────────────────┬───────────────────────────────┘
                               │ tau_muscle (B,69)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 6: LIGAMENT FORCES                                      │
│  τ_lig = -k_lig × (exp(α×excess_upper) - 1)                │
│         +k_lig × (exp(α×excess_lower) - 1)                  │
│         -b_damp × dof_vel   [in boundary zone only]          │
│  [soft_limit_margin = 85%: activates at 85% of ROM]         │
└──────────────────────────────┬───────────────────────────────┘
                               │ tau_ligament (B,69)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT                                                      │
│  tau_total = tau_muscle + tau_ligament  (B, 69)              │
│  → IsaacGym: gym.set_dof_actuation_force_tensor()            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Tensor Dimension Flow

```
                    Muscle space (20-dim)          Joint space (69-dim)
descending_cmd:     (B, 20)
                       │ Reflex Layer
                    (B, 20) a_cmd
                       │ Activation ODE
                    (B, 20) activation
                       │
dof_pos ──────────► (B, 20) l_muscle ─► Hill Model ─► (B, 20) F_muscle
dof_vel ──────────► (B, 20) v_muscle     F=a×f_max              │ @ R(20,69)
                                         ×f_FL×f_FV              ▼
                                                            (B, 69) tau_muscle
                                                                 │
dof_pos ─────────────────────────────────────────────────────► (B, 69) tau_lig
dof_vel ─────────────────────────────────────────────────────►    │
                                                                 ▼
                                                            (B, 69) tau_total
```

---

## 4. Patient Parameters Injection Points

```
  STEP 2: Reflex Layer ◄──── stretch_gain (spastic:8.0 / healthy:1.0)
                             gto_gain, reciprocal_gain (Parkinson: reduced)
  STEP 3: Activation    ◄── tau_act (Parkinson:0.050s / healthy:0.015s)
                             tau_deact
  STEP 4: Force Gen     ◄── f_max (hemiplegia: 40-60% reduction)
                             l_opt (⚠️ was 1.0 dimensionless — bug!)
  STEP 5: Torque Map    ◄── R matrix (constant, angle dependency not implemented)
  STEP 6: Ligament      ◄── k_lig (contracture: increased), soft_limit_margin
```

---

## 5. Simulator Integration Loop

```
┌─────────────────────────────────────────────────────────┐
│                  IsaacGym Sim Loop (dt = 1/120 s)        │
│  ┌─ RL Policy ──────────────────────────────────────┐   │
│  │  obs = [dof_pos, dof_vel, contact_forces, ...]  │   │
│  │  descending_cmd = policy.act(obs)  (B, 20)      │   │
│  └──────────────────────┬──────────────────────────┘   │
│  ┌─ CALM Pipeline ──────▼──────────────────────────┐   │
│  │  tau_total = human_body.compute_torques(...)     │   │
│  └──────────────────────┬──────────────────────────┘   │
│  ┌─ PhysX Engine ───────▼──────────────────────────┐   │
│  │  gym.set_dof_actuation_force_tensor(tau_total)  │   │
│  │  gym.simulate() → gym.fetch_results()           │   │
│  │  → new dof_pos, dof_vel, contact_forces         │   │
│  └─────────────────────────────────────────────────┘   │
│  repeat ──────────────────────────────────────→        │
└─────────────────────────────────────────────────────────┘
```

# Observable Musculoskeletal Joint Dynamics Simulation for Wearable Robot Integration

Created: 2026-03-18
Translated from Korean original: `260318_근육역학_인간모델_프레임워크.md`

---

## 1. Problem Definition

### 1.1 Research Background

To develop wearable robots assisting diverse patient groups (stroke hemiplegia, Parkinson's, muscle weakness, spasticity), the in-simulation human model must faithfully reproduce each patient's pathological neuromuscular characteristics. However, skeleton-based simulations like Isaac Gym directly specify joint torques, making it structurally impossible to systematically represent pathological dynamics originating from muscles (spasticity, rigidity, velocity-dependent force changes). Meanwhile, musculoskeletal simulations like MyoSuite model individual muscles for high physiological fidelity, but most of the hundreds of parameters cannot be non-invasively measured, making Real-to-Sim transfer structurally difficult due to muscle redundancy.

**Conclusion**: A framework is needed that implements muscle dynamics at the joint level within the Isaac Gym environment, at a level where parameters can be identified from clinical measurements.

### 1.2 Required Muscle Dynamics for Patient-Specific Modeling

- **Force generation (F_max, f_FL, f_FV)**: Velocity-dependent force capability changes (not just strength reduction)
- **Variable impedance via co-contraction**: Joint stiffness changes independent of net torque, emerging from Hill model nonlinearity
- **Reflex dynamics (stretch reflex, reciprocal inhibition)**: Spasticity = hyperactive stretch reflex, rigidity = impaired reciprocal inhibition
- **Multi-articular coupling (R(q) matrix)**: Bi-articular muscles create inter-joint coupling
- **Passive dynamics and activation dynamics**: Passive stiffness, activation time constants

### 1.3 Research Question

> How much can musculoskeletal dynamics be reduced to joint-level representation while maintaining variable impedance, inter-joint coupling, and pathological dynamics, with all parameters identifiable through non-invasive clinical measurements?

### 1.4 Hypothesis

Inserting a virtual muscle dynamics layer (15–20 functional groups) between controller and physics engine can reproduce key joint-level manifestations of musculoskeletal dynamics at a level identifiable from standard clinical biomechanical measurements (isokinetic testing, passive ROM, perturbation testing, pendulum testing).

---

## 2. Framework Structure

### 2.1 Three-Layer Hierarchy

1. **Skeletal (URDF)**: Bone mass/inertia, joint axes, hard ROM, contact mesh. Loaded once.
2. **Muscle & Lower Neural (Environment Script)**: PyTorch GPU tensor operations in `pre_physics_step()`. **Core of the framework.**
3. **Higher Neural (Controller)**: RL, CPG, impedance controller, clinical data replay. Swappable without modifying environment layer.

### 2.2 Per-Timestep Computation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Higher Controller (RL / CPG / ...)         │
│                    → Motor command u output                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [Muscle Dynamics Layer — GPU tensor ops in pre_physics_step]│
│  Step 1: Read current state (q, dq ← Isaac Gym solver)      │
│  Step 2: Spinal reflex layer (stretch reflex, reciprocal)    │
│  Step 3: Activation dynamics (1st-order ODE: τ_act / τ_deact)│
│  Step 4: R(q) → muscle kinematics (l_m, v_m)                │
│  Step 5: Hill model force generation (F_active + F_passive)  │
│  Step 6: Joint torque mapping (τ = R(q)ᵀ × F + soft limit)  │
│  Step 7: Torque application (set_dof_actuation_force_tensor) │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Isaac Gym Physics Solver                         │
│              → Compute next q, dq → feedback to Step 1       │
└─────────────────────────────────────────────────────────────┘
```

**Key difference**: The same motor command u produces different torques τ depending on current q, dq. This is the fundamental difference from simple torque models.

### 2.3 Variable Impedance via Co-contraction

Joint impedance **emerges naturally** from Hill model nonlinearity. When agonist and antagonist co-activate, net torque ≈ 0, but external perturbation stretches one side (f_FV eccentric enhancement → force increase) and shortens the other (force decrease) → automatic restoring torque. Higher co-contraction = higher stiffness gradient.

### 2.4 Hill Model 3-Element Structure

| Element | Description | Formula |
|---------|-------------|---------|
| **CE** (Contractile) | Active force generation | `a × F_max × f_FL × f_FV` |
| **PE** (Parallel Elastic) | Passive elastic, above l_opt | `F_PE(l/l_opt)` |
| **SE** (Series Elastic) | Tendon force-length | `F_SE(l_tendon)` |

**Framework choice**: CE and PE fully implemented. SE uses **rigid tendon assumption** (except Achilles tendon where elastic tendon is recommended).

Rigid tendon rationale: (1) Computational efficiency (no NR iteration), (2) Parameter observability (k_tendon non-invasively unmeasurable), (3) Lumping difficulty across muscle groups.

**Achilles tendon exception**: 15cm long, stores 35%+ of walking energy. Elastic tendon recommended for soleus and gastrocnemius only.

### 2.5 R(q) Matrix: SMPL 23-Joint Mapping

R matrix: (num_muscles × 69 DOF). Very sparse (~21% non-zero). Each row represents one muscle group's moment arms across all joint axes.

Lower-limb muscles per side: hip_flexors, gluteus_max, hip_abductors, hip_adductors, rectus_femoris (bi), quadriceps, hamstrings (bi), gastrocnemius (bi), soleus, tibialis_ant + flexor_digitorum_longus (bi).

---

## 3. Parameter Specification

### Force generation: F_max, l_opt, l_slack, v_max, A_f, f_M_len, k_PE, ε₀
### Activation: τ_act (10–50ms), τ_deact (40–200ms)
### Reflex: g_reflex, v_threshold, t_delay, g_reciprocal
### Ligament: k_ligament, q_onset, α

---

## 4. Real-to-Sim Parameter Identification Pipeline

Sequential 5-step identification exploiting condition-specific parameter isolation:

1. **Passive dynamics** (activation=0): Passive ROM test → k_PE, ε₀, k_ligament
2. **Max force capacity** (activation=1): Multi-speed isokinetic test → F_max, v_max, A_f
3. **Activation dynamics**: Rapid voluntary contraction → τ_act, τ_deact
4. **Reflex parameters**: Pendulum/perturbation test → g_reflex, v_threshold
5. **Integrated validation**: Gait analysis → compare predicted vs measured

---

## 5. Patient Parameterization

All patient types use the **same model structure** with parameter changes only.

| Pathology | Key Parameter Changes |
|-----------|----------------------|
| Muscle weakness | F_max ↓, v_max ↓ |
| Spasticity | g_reflex ↑, v_threshold ↓ |
| Parkinson rigidity | g_reciprocal ↓↓ |
| Aging | F_max ↓, τ_act ↑ |

Complex conditions (e.g., spasticity + weakness) combine parameter subsets.

---

## 6. Validation (4 Levels)

1. **Component**: F-L, F-V, bi-articular coupling, reflex individual verification
2. **Phenomenological**: Co-contraction impedance, pendulum test reflex patterns
3. **Cross-validation**: Parameters from partial tests predict held-out conditions
4. **Integrated gait**: Joint torque RMSE, spatiotemporal parameters, GRF patterns, assistive device interaction

---

## References

- de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
- Hogan, N. (1984). Adaptive control of mechanical impedance by coactivation.
- Lance, J.W. (1980). Symposium synopsis. In Spasticity: Disordered Motor Control.
- Millard, M. et al. (2013). Flexing computational muscle.
- Rajagopal, A. et al. (2016). Full-body musculoskeletal model.
- Thelen, D.G. (2003). Adjustment of muscle mechanics model parameters.

# MyoSuite / OpenSim Computation Flow and CALM Comparison

Created: 2026-03-21 (Fri)
Translated from Korean original: `260321_MyoSuite_OpenSim_계산플로우_비교.md`

---

## 1. OpenSim Computation Flow

OpenSim: standard tool for musculoskeletal biomechanics. Simbody physics engine + Millard 2013 muscle model.

```
Inputs
  ├─ q         (N_dof,)     ← Generalized coordinates [rad/m]
  ├─ qdot      (N_dof,)     ← Generalized velocities
  ├─ excitation (N_muscles,) ← Muscle excitation [0,1] (from CMC/SO)
  └─ external_forces          ← GRF etc.
         │
         ▼
╔═══════════════════════════════════════════════════════════╗
║              OpenSim::Model.realizeAcceleration()          ║
║  Step 1: Musculotendon Geometry    (GeometryPath)         ║
║  Step 2: Activation Dynamics       (FirstOrderActivation) ║
║  Step 3: Tendon Equilibrium        (Millard2012)          ║
║  Step 4: Force Generation          (Hill + Elastic Tendon)║
║  Step 5: Generalized Forces        (MomentArm via Path)  ║
║  Step 6: Equations of Motion       (Simbody multibody)    ║
╚═══════════════════════════════════════════════════════════╝
         │
         ▼
  qddot (N_dof,) → numerical integration → next q, qdot
```

### Key Steps

**Step 1 — Musculotendon Geometry**: Each muscle path = [Origin → WrapPoints → Insertion]. `l_mtu(q) = Σ|P_{i+1} - P_i|`, `r(q) = ∂l_mtu/∂q` (analytical). Wrapping surfaces (cylinder/ellipsoid/torus) enable angle-dependent moment arms.

**Step 2 — Activation Dynamics**: Nonlinear ODE where τ depends on current activation level a. τ_act=10ms, τ_deact=40ms default.

**Step 3 — Tendon Equilibrium (key differentiator!)**: Solves `F_ce × cos(α) = F_tendon` via Newton-Raphson (3–5 iterations). This step accounts for ~60% of computation cost. Rigid tendon option also available.

**Step 4 — Force Generation**: Hill-type with Millard 2012 curves (5th-order spline f_AL, smooth f_FV).

**Step 5 — Generalized Forces**: `τ_j = Σ r_ij(q) × F_muscle_i` + ligament + contact + external forces.

**Step 6 — EoM**: `M(q)×qddot + C(q,qdot) + G(q) = τ_generalized + J^T×F_ext` via Simbody O(n) algorithm.

---

## 2. MyoSuite Computation Flow

MyoSuite (Vittorio+Vikash, 2022): MuJoCo-based musculoskeletal RL environment with built-in muscle model.

```
Inputs
  ├─ ctrl       (N_muscles,) ← RL Policy muscle excitation [0,1]
         │
         ▼
╔═══════════════════════════════════════════════════════════╗
║                MyoSuite Env.step(ctrl)                     ║
║  Step 1: Activation Dynamics       (MuJoCo built-in 1st LPF)║
║  Step 2: Musculotendon Length      (MuJoCo tendon wrapping)║
║  Step 3: Force Generation          (MuJoCo Hill-type)     ║
║  Step 4: Moment Arm → Torque       (MuJoCo path-based)    ║
║  Step 5: Forward Dynamics          (MuJoCo physics)       ║
║  Step 6: Reward & Obs              (MyoSuite Python)      ║
╚═══════════════════════════════════════════════════════════╝
```

**Activation**: Simple 1st-order LPF with single τ (unlike OpenSim's nonlinear, unlike CALM's split τ_act/τ_deact).

**Geometry**: Same principle as OpenSim (wrapping surfaces), computed in C by MuJoCo at high speed.

**Force**: Piecewise linear curves (fast but discontinuous derivatives). Rigid tendon default, elastic tendon option in MuJoCo 2.3+.

**Torque**: MuJoCo computes internally (black box — user cannot intervene at code level).

---

## 3. Tensor Flow Comparison (at a glance)

```
              OpenSim              MyoSuite (MuJoCo)         CALM (ours)
              ─────────            ─────────────────         ───────────
Input      excitation (N_m,)    ctrl (N_m,)               descending_cmd (B,20)

Activation  Nonlinear ODE          Simple 1st LPF            1st ODE (τ_act≠τ_deact)
            τ=f(u,a)              τ fixed                    τ split

Geometry    GeometryPath          MuJoCo spatial             R × dof_pos (constant R)
            wrapping surface      wrapping surface           No wrapping
            r(q) = ∂l/∂q          r(q) = ∂l/∂q             r = constant

Tendon Eq   Elastic (Millard)     Rigid (default)            Rigid (default)
            Newton-Raphson        Elastic option             Not implemented

Hill curves Spline (smooth)       Piecewise linear (fast)    Gaussian + piecewise

Torque      r(q) × F              r(q) × F                  F @ R (constant)

Dynamics    Simbody (CPU)         MuJoCo (CPU/GPU*)         PhysX (GPU, thousands)
                                  *MJX for GPU

Reflex      None (separate)       None (RL replaces)         Built-in (stretch/GTO/reciprocal)
Ligament    CoordinateLimitForce  joint constraint           Exponential soft limit
```

---

## 4. Key Differences

### 4.1 Moment Arm (Biggest Difference)

| | OpenSim | MyoSuite | CALM |
|--|---------|----------|------|
| Method | Wrapping surface + analytical ∂l/∂q | Wrapping surface + analytical ∂l/∂q | Constant R matrix |
| Angle dependency | Yes (continuous function) | Yes (continuous function) | No |
| Accuracy | Anatomically precise | Anatomically precise | ±10% flat walking, ±30% extreme |

### 4.2 Elastic Tendon

| | OpenSim | MyoSuite | CALM |
|--|---------|----------|------|
| Default | Elastic (Millard2012) | Rigid | Rigid |
| Achilles energy | Reflected | Not reflected (default) | Not reflected |

### 4.3 Reflex Controller

| | OpenSim | MyoSuite | CALM |
|--|---------|----------|------|
| Built-in | No | No | Yes (3 types: stretch/GTO/reciprocal) |
| Pathology | Separate plugin needed | RL learns implicitly | stretch_gain directly models spasticity |

---

## 5. Muscle-Dynamics Coupling (Most Fundamental Architectural Difference)

### 5.1 OpenSim — Monolithic Coupled ODE

System state: `y = [q, qdot, a₁...aₙ, l_ce₁...l_ceₙ]`

RK4 integrator computes k1→k4, calling ABA (forward dynamics) at each substep. Muscle force ↔ motion state updated 4× per integration step within a single ODE system.

→ Energy conservation excellent, numerically accurate.

### 5.2 MuJoCo — Semi-coupled

Activation computed first (confirmed), then muscle force from current state, then semi-implicit Euler for dynamics. Not fully coupled but semi-implicit provides some stability.

### 5.3 CALM — Staggered Decoupled

```
Time t:
  CALM: tau(t) = compute_torques(q(t), qdot(t), cmd(t))  ← tau confirmed
  PhysX: substep 1,2,...,N with tau(t) held constant      ← qdot changes but tau doesn't
  → 1-step delay in muscle-dynamics coupling
```

### Comparison

```
         OpenSim                MuJoCo                  CALM
         Coupled (RK4)         Semi-coupled             Staggered (decoupled)
Energy:  Well conserved        Mostly OK                Artificial injection/loss possible
Stability: Large dt OK         Medium dt               Small dt needed
Precision: Highest             Medium                  Converges if dt small
```

### Why CALM is OK at dt=8.3ms

| Condition | Value | Assessment |
|-----------|-------|-----------|
| Sim dt | 1/120s = 8.3ms | Well below muscle τ (15–80ms) |
| Motion range | Flat-ground walking (low speed) | Not high-impact |
| Purpose | RL training environment | GPU thousands of envs >> single env precision |
| Engine access | PhysX is a black box | This structure is the only option |

### Future Mitigation Strategies:
1. Increase PhysX substeps (2→4): dt_effective = 2.1ms
2. Increase implicit muscle damping β
3. Predictor-corrector: use extrapolated qdot(t+dt) for tau calculation
4. Isaac Lab migration (same fundamental structure)

---

## 6. CALM Improvement Roadmap (Derived from This Comparison)

| Priority | Improvement | Reference Model | Difficulty | Impact |
|---------|------------|----------------|-----------|--------|
| 1 | l_opt measured values | OpenSim Rajagopal 2016 | Easy | BUG-01 fix |
| 2 | Elastic Tendon (soleus, gastroc) | OpenSim Millard2012 | Hard | Push-off reproduction |
| 3 | R(q) polynomial fitting | OpenSim gait2392 export | Medium | 30%↑ torque accuracy |
| 4 | Asymmetric f_FL curve | OpenSim/De Groote 2016 | Easy | Minor force accuracy improvement |
| 5 | Nonlinear activation ODE | OpenSim FirstOrder | Easy | Dynamic accuracy improvement |
| 6 | Stretch reflex delay | Internal (physiology lit) | Easy | BUG-04 fix |

### CALM's Unique Strengths (to preserve)

1. **Built-in Reflex Controller**: Neither OpenSim nor MyoSuite has this. Critical for pathological patient modeling.
2. **GPU Massive Parallelism**: PhysX-based thousands of simultaneous environments → RL training speed.
3. **Patient Parameter YAML**: Intuitive pathology modeling via stretch_gain, f_max scaling.
4. **Explicit Tensor Operations**: `F @ R` structure makes debugging/analysis easy (MyoSuite is black box).

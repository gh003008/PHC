# 01_muscle_layer Validation Detail

Created: 2026-03-18
Target: `run_validation.py` T01~T09
Translated from Korean original: `260318_01_근육레이어_검증_상세설명.md`

---

## Validation Philosophy

This stage validates equations using pure PyTorch without IsaacGym. Two purposes: (1) verify physiological formulas are correctly implemented in code, (2) confirm each submodule independently produces correct output before assembling the full pipeline. Physics engine integration is separately verified in Step 2 (02_isaacgym_integration).

---

## T01 — Hill Active Force-Length

### Physiological Mechanism
Muscle fibers generate force through actin-myosin cross-bridge binding. The number of cross-bridges depends on filament overlap: too short → filaments interfere, optimal length → maximum overlap, too long → no binding possible. This produces a bell-shaped curve.

### Implementation
```
f_FL = exp( -((l_norm - 1.0)^2) / (2 * 0.45^2) )
```
Gaussian approximation. Center = 1.0 (optimal length), σ = 0.45 (Thelen 2003).

### Test: Sweep l_norm 0.3–2.0 (200 points). Peak at l_norm=1.0 ± 0.01 and peak ≥ 0.99 → PASS.

### Diagnostic Warning
Current `healthy_baseline.yaml` has `l_opt=1.0` (dimensionless), so actual operating point is `l_norm = 0.30/1.0 = 0.30` where f_FL ≈ 0.30 — only 30% of max force. This is a YAML parameter issue, not a code bug.

---

## T02 — Hill Force-Velocity

### Physiological Mechanism
- Concentric (shortening, v<0): faster shortening → fewer cross-bridge binding cycles → force decreases. Approaches 0 at v_max.
- Isometric (v=0): baseline force (f_FV = 1.0).
- Eccentric (lengthening, v>0): cross-bridges stretched like springs → force up to ~1.8× isometric. This is why large forces occur during downhill walking.

### Implementation: Piecewise function with concentric/eccentric branches. Numerical clamp at v_norm = -0.99 to prevent division by zero.

### Test: f_FV(0)=1.0, f_FV(-0.99)<0.05, max≤1.81 → PASS.

---

## T03 — Hill Passive Force-Length

### Physiological Mechanism
Even without activation, stretched muscle generates resistance via titin proteins and connective tissue. Key: **zero passive force below optimal length**, then rapid increase above it.

### Implementation
```
f_PE = k_pe * max(0, l_norm - 1.0)^2 / epsilon_0
```

### Test: f_PE=0 at l_norm=1.0, hand-calculated value matches at l_norm=1.6 → PASS.

---

## T04 — Activation Linearity

Active force is **linearly proportional** to activation: `F_active = activation × (F_max × f_FL × f_FV × cos_penn)`. This represents motor neuron recruitment.

### Test: active(0.5)/active(1.0) ratio = 0.5 ± 0.02 → PASS.

---

## T05/T05b — Moment Arm Coupling Heatmap (Bi-articular Verification)

### Physiological Mechanism
Bi-articular muscles span two joints:
- hamstrings: hip extension + knee flexion
- gastrocnemius: knee flexion + ankle plantarflexion
- rectus femoris: hip flexion + knee extension

The R matrix must have non-zero values at both joint columns for these muscles.

### Test: T05b verifies 6 bi-articular muscles each produce ≥0.1 Nm torque at ≥2 joints → PASS.

---

## T06 — L/R Symmetry

Human body is bilaterally symmetric. L and R muscles with same conditions must produce identical torques.

### Test: L/R muscle pairs with F=100N → torque error < 1e-4 Nm → PASS. Result: 0.00 Nm (exact match).

---

## T07 — Ligament Soft-Limit

### Physiological Mechanism
Joint capsule and ligaments provide exponential restoring torque near ROM boundaries, before hard bone-on-bone limits. Patient differences: contracture (earlier onset, steeper), flaccid (weakened).

### Implementation
```
τ_ligament = -k_lig × exp(α × max(0, q - q_soft_upper))
           + k_lig × exp(α × max(0, q_soft_lower - q))
```

### Test: Correct restoring direction at boundaries, near-zero in neutral zone → PASS.

---

## T08 — Stretch Reflex: Healthy vs Spastic

### Physiological Mechanism
Stretch reflex = spinal feedback loop (muscle spindle → Ia afferent → spinal cord → α motor neuron → muscle contraction). Spasticity (UMN damage): gain greatly increased + lower threshold. Flaccidity (LMN damage): reflex completely lost.

### Parameters: Healthy (gain=1.0, threshold=0.1), Spastic (gain=8.0, threshold=0.02), Flaccid (gain=0.0).

### Test: At v=0.5, Spastic > Healthy; at v=0, both ≈ 0 → PASS.

---

## T09 — Full Pipeline Zero Input

### Physiological Mechanism
At rest (cmd=0, vel=0, neutral posture): no active force, no reflex, no ligament stretch → zero torque expected.

### SMPL Coordinate Issue
SMPL `q=0` = full knee extension (lower boundary), not anatomical neutral. True neutral = ROM midpoint.

### Test: Scenario B (q=ROM center, cmd=0, vel=0) → max|τ| < 5.0 Nm → PASS. Result: 0.000 Nm.

---

## Summary Structure

```
T01~T04  HillMuscleModel (F-L active, F-V, F-L passive, activation linearity)
T05~T06  MomentArmMatrix (bi-articular coupling, L/R symmetry)
T07      LigamentModel (soft-limit direction and zone)
T08      ReflexController (healthy vs spastic differentiation)
T09      HumanBody (full pipeline zero-input consistency)
```

All PASS = muscle layer equation verification complete. Ready for Step 2 (IsaacGym integration).

---

## Mathematical Model Sources

| Module | Source |
|--------|--------|
| Active F-L (Gaussian, width=0.45) | Thelen 2003 |
| F-V (Hill equation, concentric/eccentric) | Thelen 2003, Hill 1938 |
| Passive F-L (quadratic) | Millard 2013 |
| Moment Arm R values | OpenSim Rajagopal 2016 |
| Activation dynamics (1st-order ODE) | Thelen 2003 |
| Stretch reflex (gain/threshold) | Clinical EMG literature |

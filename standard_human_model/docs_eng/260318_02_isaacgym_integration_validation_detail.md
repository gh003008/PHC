# 02_isaacgym_integration Validation Detail

Created: 2026-03-18
Result: 4/4 PASS
Translated from Korean original: `260318_02_isaacgym통합_검증_상세설명.md`

---

## Why This Validation is Needed

01_muscle_layer verified equations with pure PyTorch. But whether those numbers actually move joints when passed to the IsaacGym physics engine is a separate question.

Potential failure points:
- DOF index mismatch (our L_Knee index ≠ IsaacGym's L_Knee index)
- Torque sign convention mismatch
- `DOF_MODE_EFFORT` not set → torque silently ignored
- Profile differences don't translate to physical motion differences

---

## Common IsaacGym Setup

**Asset**: `smpl_humanoid_fixed.xml` (PHC asset with freejoint removed, base fixed).
**Test joint**: L_Knee set to `DOF_MODE_EFFORT` (direct torque injection), all others set to `DOF_MODE_POS` (PD-held at neutral).
**Sim parameters**: dt=1/60s, 2 substeps, gravity -9.81 m/s², PhysX PGS solver.

---

## I01: Torque Injection Direction

**Purpose**: Most basic check — does `gym.set_dof_actuation_force_tensor()` actually work?

**Design**: Apply +50 Nm constant to L_Knee for 2 seconds. Pass if final angle > initial + 0.1 rad (≈6°).

**Result**: Δangle = +100°+ (reached joint limit) → PASS.

Simultaneously verifies: (1) API connection works, (2) DOF_MODE_EFFORT is set, (3) L_Knee DOF index is correct.

---

## I02: Torque Sign Convention

**Purpose**: Determine if positive torque = flexion or extension in SMPL.

**Design**: env 0 gets +50 Nm, env 1 gets -50 Nm, both start at 45°. Pass if angle difference > 0.1 rad.

**Result**: +torque → flexion (angle increase), -torque → extension. Difference = 145°+ → PASS.

Confirmed: SMPL L_Knee positive torque = flexion direction. Bio-torque sign convention matches IsaacGym asset convention.

---

## I03: Patient Profile Differentiation

**Purpose**: Do three profiles (Healthy, Spastic, Flaccid) produce different motion patterns in physics simulation?

**Design — Knee Pendulum**: Start at 80° (1.4 rad) with -5 rad/s extension kick. Run 5 seconds with bio-torque from `HumanBody.compute_torques()`.

- Spastic: strong reflex + high ligament stiffness → motion resisted → high final angle
- Healthy: moderate → equilibrium somewhere
- Flaccid: no reflex, low resistance → free pendulum → low final angle

**Result**: Spastic 82.4° > Healthy 74.4° > Flaccid 23.0° (all gaps > 5°) → PASS.

---

## I04: Stretch Reflex Differential Resistance

**Purpose**: Verify stretch reflex strength differences appear directly in bio-torque magnitude.

**Key insight**: stretch reflex is velocity-dependent. At vel=0, no reflex fires (for any profile). Must provide extension kick velocity to trigger reflex.

**Design**: Start at 80° with -3 rad/s kick. Measure average |bio-torque| over first 20 steps.

**Result**: |Spastic| ≈ 1.95 Nm > |Healthy| ≈ 0.15 Nm > |Flaccid| ≈ 0.00 Nm → PASS.

Spastic ~13× stronger than Healthy (consistent with 8× gain + lower threshold). Flaccid at zero (gain=0).

---

## Integration Pipeline Flow

```
[IsaacGym PhysX]
    gym.refresh_dof_state_tensor()
         ↓
    dof_pos_all, dof_vel_all

[standard_human_model]
    HumanBody.compute_torques(pos, vel, cmd=0, dt=1/60)
         ↓ Internal: muscle kinematics → reflex → activation → Hill → R^T @ F → ligament
         ↓
    bio_tau[0, TEST_DOF_IDX]

[IsaacGym PhysX]
    gym.set_dof_actuation_force_tensor()
    gym.simulate() → gym.fetch_results()
    → new dof_pos, dof_vel
```

Loop repeats at 60Hz. CPU pipeline for validation.

---

## Results Summary

| ID | Test | Result | Key Numbers |
|----|------|--------|-------------|
| I01 | Torque injection direction | ✅ PASS | Δangle = +100°+ (threshold: >6°) |
| I02 | Torque sign convention | ✅ PASS | +torque=flexion, -torque=extension, diff=145°+ |
| I03 | Profile differentiation | ✅ PASS | Spastic 82.4° > Healthy 74.4° > Flaccid 23.0° |
| I04 | Stretch reflex differential | ✅ PASS | \|S\|=1.95 > \|H\|=0.15 > \|F\|=0.00 Nm |

**4/4 PASS** — IsaacGym integration verified. The musculoskeletal pipeline produces physically meaningful patient-differentiated behavior in the physics simulator.

---

## Known Limitations

- **l_opt parameter issue**: All muscles at l_opt=1.0 (dimensionless) → active force at ~30% of max. I03/I04 still PASS because they test relative differences, not absolute values.
- **Constant R matrix**: Moment arm doesn't vary with joint angle (20–30% variation ignored).

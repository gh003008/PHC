# Framework Design vs Current Implementation Gap Analysis

Created: 2026-03-19
Translated from Korean original: `260319_프레임워크_구현_갭분석.md`

Reference document: `standard_human_model/docs/260318_근육역학_인간모델_프레임워크.md`
Reference implementation: `standard_human_model/core/`, `config/`, `isaacgym_validation/`

---

## Purpose

Compare the requirements specified in the musculoskeletal human model framework design document against the current implementation, identifying matches and gaps to inform development priorities.

---

## 1. Well-Implemented Items

### 1.1 Core 7-Step Pipeline (Framework Section 2.2)

The 7-step flow specified for `pre_physics_step()` is accurately reflected in `human_body.py`'s `compute_torques()`.

```
Step 1: Read q, dq              → moment_arm.compute_muscle_length/velocity()
Step 2: Spinal reflex layer      → reflex_controller.py (stretch, GTO, reciprocal)
Step 3: Activation dynamics      → activation_dynamics.py (1st-order ODE, asymmetric tau)
Step 4: R(q) → l_m, v_m         → moment_arm.py
Step 5: Hill model force         → muscle_model.py (CE + PE + damping)
Step 6: τ = R^T @ F              → moment_arm.forces_to_torques()
Step 7: Torque injection         → gym.set_dof_actuation_force_tensor()
```

IsaacGym integration verification (I01–I04) confirmed through step 7 torque injection.

### 1.2 Hill Muscle Model (Framework Section 2.5)

Implemented with the formula lineage specified (Thelen 2003, Millard 2013). All T01–T04 PASS.

### 1.3 R Matrix and Bi-articular Coupling (Framework Section 2.6–2.7)

`MomentArmMatrix` implemented as `(num_muscles × 69_DOF)`. Three bi-articular muscles verified (T05b, T06 PASS).

### 1.4 Three Reflex Types (Framework Section 1.6, 3)

Stretch reflex, GTO reflex, and reciprocal inhibition all implemented with patient-specific parameters.

### 1.5 Ligament/Joint Capsule Model (Framework Section 3)

Exponential soft limit torque implemented and verified (T07 PASS).

### 1.6 Patient Profile System (Framework Section 5)

6 patient profiles implemented as YAML parameter files.

### 1.7 Validation Levels 1–2 (Framework Section 6.1–6.2)

Level 1 (component): 9/9 PASS. Level 2 (phenomenon): 4/4 PASS.

---

## 2. Gap Items

### 2.1 l_opt Unit Mismatch — Most Urgent Fix

**Framework**: `l_opt` is actual optimal fiber length in meters.
**Current**: All muscles have `l_opt=1.0` (dimensionless) → `l_norm = 0.30` → active force at 30% of max.
**Fix**: Replace with anatomical measured values (meters). YAML-only change, no code modification.

### 2.2 Achilles Tendon Elastic Tendon (SE) Not Implemented

**Framework**: Strongly recommends elastic tendon for soleus and gastrocnemius (35%+ of walking energy efficiency).
**Current**: Rigid tendon applied to all muscles. `k_tendon` parameter declared but unused.

### 2.3 Constant R Matrix (Known Limitation)

**Framework**: Angle-dependent R(q) mentioned as medium-term improvement.
**Current**: Constant R used. Up to 20–30% moment arm variation with joint angle not captured.

### 2.4 Missing Muscle: Flexor Digitorum Longus

Framework specifies 11 muscle groups per leg; current implementation has 10. Flexor digitorum longus (ankle-toe bi-articular) is missing.

### 2.5 Validation Levels 3–4 Incomplete

Level 3 (cross-validation) and Level 4 (integrated gait) remain unimplemented.

### 2.6 Real-to-Sim Parameter Identification Pipeline Not Implemented

The 5-step sequential identification pipeline (passive → max force → activation → reflex → gait integration) has no code implementation yet.

---

## 3. Summary Table

| Item | Framework | Current | Status |
|------|-----------|---------|--------|
| 7-step pipeline | Required | Fully implemented | ✅ Match |
| Hill CE (F-L, F-V) | Required | Fully implemented | ✅ Match |
| Hill PE (passive) | Required | Fully implemented | ✅ Match |
| Rigid tendon (default) | Recommended | Applied globally | ✅ Match |
| Achilles SE | Recommended for soleus/gastroc | Not implemented | ⚠️ Gap |
| l_opt units | Meters (measured) | Dimensionless 1.0 | ⚠️ Parameter bug |
| R matrix constant | Medium-term improvement | Constant R | Known limitation |
| Bi-articular (3 types) | Required | Fully implemented | ✅ Match |
| 3 reflex types | Required | Fully implemented | ✅ Match |
| Reflex time delay | 25–40ms | 1 step (~17ms) | ✅ Approximate |
| Ligament soft limit | Required | Fully implemented | ✅ Match |
| Patient profile YAML | Required | 6 types implemented | ✅ Match |
| Flexor digitorum longus | Included | Not implemented | ⚠️ Gap |
| Validation Levels 1–2 | Required | 13 tests PASS | ✅ Complete |
| Validation Levels 3–4 | Required | Incomplete | ❌ Incomplete |
| Real-to-Sim identification | Required | Not implemented | ❌ (Low priority now) |

---

## 4. Recommended Priority Order

1. **l_opt parameter fix** (no code change needed)
2. **Achilles tendon elastic tendon implementation**
3. **Validation Level 4** (integrated gait verification)
4. **Flexor digitorum longus addition**
5. **R(q) angle dependency, Real-to-Sim pipeline** (medium-term)

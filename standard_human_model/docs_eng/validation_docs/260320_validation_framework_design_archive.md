# CALM Validation Framework Design (Archive)

Created: 2026-03-20
Translated from Korean original: `validation_docs/260320_검증체계_설계_보관용.md`
Note: This is the earlier version. See `260320_validation_pyramid_v2.md` for the refined version.

---

## 0. Current State Diagnosis

### Current validation folder structure
```
validation/
  01_muscle_layer/          T01~T09   (9)  — Mathematical unit tests
  02_isaacgym_integration/  I01~I04   (4)  — IsaacGym integration
  03_visualization/                        — Visual demo (not tests)
  04_comprehensive_validation/ V4-1~5 (5)  — Scenario tests
  05_deep_validation/       V5-1~6    (6)  — Deep dynamics analysis
  06_integration_test/      I6-1~4   (4)  — Open-loop EMG → gait
```

### Problems
- Naming inconsistency (T, I, V series mixed)
- Unclear hierarchy (what level does 04~06 verify?)
- No GRF validation
- No EMG output comparison
- No quantitative comparison with real data (Winter, Perry)

### Current coverage
```
✅ Individual equation correctness (F-L, F-V, activation ODE, ligament, reflex)
✅ Simple torque injection → joint movement
✅ Qualitative differences between patient profiles
✅ Bi-articular coupling existence confirmed
❌ Real gait kinematics comparison
❌ GRF verification
❌ EMG timing comparison
❌ Energy/metabolic metrics
❌ Pathological gait pattern reproduction
```

---

## 1. Validation Goal

> **Ultimate goal**: Prove this simulation credibly reproduces real human gait, generating **kinematics (joint angles)**, **GRF (ground reaction forces)**, and **EMG (muscle activation timing)** simultaneously similar to real data.

Questions that must sequentially answer "YES":
```
L0: Are the math equations correct?
  ↓ YES
L1: Are modules correctly connected?
  ↓ YES
L2: Is physics engine integration accurate?
  ↓ YES
L3: Are known biomechanical phenomena reproduced?
  ↓ YES
L4: How close to real gait data (Kin+GRF+EMG)?
  ↓ YES
L5: Are patient-specific pathological patterns reproduced?
```

---

## 2. Five-Level Validation Pyramid

```
          ▲
         /L5\      Patient-specific pathological gait
        /────\
       / L4   \     Quantitative data comparison (Kin + GRF + EMG)
      /────────\
     /   L3     \   Phenomenological validity
    /────────────\
   /     L2       \ Physics engine integration
  /────────────────\
 /       L1         \ Module data flow
/────────────────────\
          L0           Unit math tests
```

Lower levels are **foundation**; upper levels approach the **final goal**. Results at higher levels cannot be trusted unless lower levels PASS.

---

## 3. Level Details

### Level 0 — Unit Math Tests ✅ Complete (T01~T09)
Existing `01_muscle_layer/` tests verify each equation independently.

### Level 1 — Module Data Flow (0/7) — New
Pure PyTorch verification of tensor shapes, value ranges, sign conventions, numerical stability.
Key test: L1-04 (muscle velocity sign convention).

### Level 2 — Physics Integration ✅ + Enhancement (I01~I04 + 2 new)
Existing tests verified. Recommended additions: bi-articular coupling in IsaacGym, passive drop GRF.

### Level 3 — Phenomenological Validity (Reorganize)
Known biomechanical phenomena: clonus, ankle push-off, co-contraction, stretch reflex delay, GTO inhibition, passive-active crossover, bi-articular energy transfer.

### Level 4 — Data Fidelity ← KEY GAP (0/4)
Three-channel comparison: kinematics vs Winter 2009, GRF vs Winter 2009, EMG timing vs Perry 1992.

### Level 5 — Patient Pathology (0/6)
Stroke stiff-knee gait, circumduction, Parkinson reduced step length, SCI activation loss.

---

*See `260320_validation_pyramid_v2.md` for the final, refined version with detailed test specifications and pass criteria.*

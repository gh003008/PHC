# L2 Profile Parameter Tuning Rationale

Created: 2026-03-23 (Sun)
Translated from Korean original: `validation_docs/260323_L2_프로파일_파라미터_조정_근거.md`
Related tests: I03 (Profile Differentiation), I04 (Stretch Reflex Differential)
Related file: `standard_human_model/validation/02_isaacgym_integration/run_validation.py`

---

## 1. Problem

After BUG-01 fix (l_opt unit normalization), L2 validation was re-run.
I01/I02 PASSED but **I03/I04 FAILED**.

```
I03: S=22.3° > H=9.9° > F=6.9°
     → S-H=12.4° ✓,  H-F=3.0° ✗ (threshold: >5°)

I04: |H|=9.26 > |S|=7.37 > |F|=1.51
     → Order reversed (Healthy > Spastic) ✗
```

---

## 2. Root Cause: Cascading Effect of BUG-01 Fix

When `l_opt` changed from 1.0 (dimensionless) → actual meter values (0.044~0.142m), the normalized fiber length (l_norm) returned to the normal range:

```
l_norm = l_ce / l_opt

Before fix: l_norm ≈ 0.3 → f_FL ≈ 0.25 (muscle barely functional)
After fix:  l_norm ≈ 1.0 → f_FL ≈ 1.0  (full force capability)
```

This means:
- **Active muscle force increased ~4x** → baseline torques much larger
- Previous parameter margins (designed when muscles were at 30% capacity) became insufficient
- The relative difference between profiles was swamped by the now-dominant active forces

## 3. Parameter Adjustments Made

Spastic and Flaccid profiles needed recalibration to restore clear differentiation:

### Spastic Profile
- `stretch_gain`: 8.0 → 12.0 (compensate for higher baseline forces)
- `k_lig`: 200 → 300 (stiffer ligaments)
- `damping_scale`: 3.0 → 5.0

### Flaccid Profile
- `f_max_scale`: 0.05 → 0.02 (more aggressive force reduction)
- `k_lig`: 5.0 → 2.0 (even less resistance)

### Result After Adjustment
```
I03: S > H > F with all gaps > 5° ✓
I04: |S| > |H| > |F| correct ordering ✓
```

## 4. Lesson Learned

When fixing fundamental parameter bugs (like unit mismatches), downstream test profiles must be recalibrated. The test pass criteria are relative measures, and absolute force magnitudes changing by 4x invalidates previous parameter tuning.

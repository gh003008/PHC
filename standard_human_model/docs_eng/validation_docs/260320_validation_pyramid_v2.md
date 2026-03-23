# CALM Validation Pyramid v2

Created: 2026-03-20
Previous version: 260320_검증체계_설계.md
Translated from Korean original: `validation_docs/260320_검증_ver2.md`

---

## Core Objective

> Prove that the simulation can simultaneously and credibly reproduce **kinematics (joint angles)**, **GRF (ground reaction forces)**, and **EMG (muscle activation timing)** of real human gait.

---

## Validation Pyramid

```
             ▲  Increasing confidence
            /L5\      Patient pathological gait reproduction
           /────\
          / L4   \     Quantitative comparison with real data  ← TOP PRIORITY
         /────────\       Kin + GRF + EMG
        /   L3     \   Phenomenological validity
       /────────────\   (known biomechanical phenomena)
      /     L2       \  Physics engine integration accuracy
     /────────────────\
    /       L1         \ Module data flow
   /────────────────────\
              L0           Unit math tests
```

---

## Level-by-Level Status and Test Lists

### L0 — Unit Math Tests ✅ Complete (9/9)

**Question**: Are individual equations mathematically correct?
**Tools**: PyTorch only, no IsaacGym needed
**Folder**: `validation/01_muscle_layer/`

| ID | Test Content | Pass Criteria |
|----|-------------|---------------|
| L0-01 | Hill F-L active curve shape | Peak at l_norm=1.0 ± 0.01 |
| L0-02 | Hill F-V curve monotonicity | FV(0)=1.0, concentric monotonically decreasing |
| L0-03 | Hill F-L passive curve | Zero for l_norm ≤ 1.0, then monotonically increasing |
| L0-04 | Activation-force linearity | F(a=0.5) ≈ 0.5 × F(a=1.0) ±5% |
| L0-05 | Ligament soft-limit direction | Restoring direction correct when ROM exceeded |
| L0-06 | Stretch reflex gain effect | gain 8 > gain 1 > gain 0 |
| L0-07 | L/R symmetry | Same conditions for L/R → torque error < 1e-4 Nm |
| L0-08 | Zero input → zero output | u=0, q=neutral → max\|τ\| < 5 Nm |
| L0-09 | Activation time constant measurement | Measured tau_act within ±20% of configured value |

---

### L1 — Module Data Flow ⬜ Not Implemented (0/7)

**Question**: Do modules connect with correct dimensions and signs?
**Tools**: PyTorch only, no IsaacGym needed
**Folder**: `validation/L1_module_flow/` (new)

| ID | Test Content | Key Check |
|----|-------------|-----------|
| L1-01 | Tensor shape check | (num_envs, num_muscles) shape consistency across all pipeline stages |
| L1-02 | Moment arm coupling | hamstrings/gastrocnemius/rectus_femoris have ≥2 non-zero joints in R |
| L1-03 | Activation pipeline range | u→reflex→activation all stages within [0, 1] |
| L1-04 | **Muscle velocity sign** | Knee flexion (dq>0) → hamstrings shorten (v<0), quadriceps lengthen (v>0) |
| L1-05 | Force-torque conversion numerical | Single muscle F×R^T = τ matches hand calculation, error < 1e-6 Nm |
| L1-06 | Numerical stability | Extreme inputs (u=1.0 all, v=v_max) no NaN/Inf in 1000 steps |
| L1-07 | Reset consistency | Partial env_ids reset does not affect other environments' activation |

**L1-04 is most critical**: Tests whether `moment_arm.py`'s `-(R @ dq)` sign convention matches `muscle_model.py`'s FV function input direction.

---

### L2 — Physics Engine Integration ✅ Complete (4/4), Enhancement Recommended (2)

**Question**: Does CALM torque produce expected motion in physics simulation?
**Tools**: IsaacGym required
**Folder**: `validation/02_isaacgym_integration/`

| ID | Test Content | Status |
|----|-------------|--------|
| L2-01 (I01) | +50 Nm → knee flexion direction confirmed | ✅ |
| L2-02 (I02) | +/-50 Nm → opposite directions confirmed | ✅ |
| L2-03 (I03) | Patient profile ordering (spastic > healthy > flaccid) | ✅ |
| L2-04 (I04) | Stretch reflex magnitude ordering | ✅ |
| **L2-05** | **Bi-articular coupling** — hamstrings activation → knee+hip simultaneous change | ⬜ Needs addition |
| **L2-06** | **Passive drop GRF** — u=0 drop, contact_forces sum equals body weight | ⬜ Needs addition |

---

### L3 — Phenomenological Validity 🔶 Partially Complete (needs reorganization)

**Question**: Are known biomechanical phenomena observed?
**Tools**: IsaacGym required
**Folder**: `validation/L3_phenomena/` (reorganize 04+05)

| ID | Phenomenon | Expected Result | Literature |
|----|-----------|----------------|------------|
| L3-01 | Spastic clonus onset | Sustained oscillation when stretch_gain > threshold | Lance 1980 |
| L3-02 | Ankle push-off power burst | Ankle torque surge in terminal stance | Neptune 2001 |
| L3-03 | Co-contraction | Agonist/antagonist simultaneous activation under instability | Winter 1984 |
| L3-04 | Stretch reflex delay | delay_steps×dt delay after stretch stimulus confirmed | Sherrington 1910 |
| L3-05 | GTO inhibition | F_muscle exceeds threshold → antagonist inhibition | Houk 1967 |
| L3-06 | Passive-active crossover | Passive force > active force above l_norm~1.3 | Thelen 2003 |
| L3-07 | **Bi-articular energy transfer** | Gastrocnemius alone → knee absorption + ankle generation | Bobbert 1986 |

---

### L4 — Data Quantitative Comparison ⬜ Not Implemented (0/4) ← KEY GAP

**Question**: How close to real gait data (Kin + GRF + EMG)?
**Tools**: IsaacGym + reference_data/ YAML
**Folder**: `validation/L4_data_fidelity/` (new)

**Reference Data** (managed as separate YAML files):
- `reference_data/winter2009_normative.yaml` — Joint angles + GRF (Winter 2009)
- `reference_data/perry1992_emg_timing.yaml` — EMG timing (Perry 1992)

| ID | Test Content | Pass Criteria (Minimum) | Pass Criteria (Target) |
|----|-------------|------------------------|----------------------|
| L4-01 | Kinematics vs Winter 2009 | Hip/knee/ankle ROM error < ±15° | RMSE < 8° per joint |
| L4-02 | GRF vs Winter 2009 | Vertical GRF double-peak pattern, peaks 0.8~1.5 BW | RMSE < 0.15 BW all phases |
| L4-03 | EMG timing vs Perry 1992 | 6 muscles ON phase within ±20% GC | Timing error < ±10% GC |
| L4-04 | Combined Fidelity Score | score > 0.5 | score > 0.7 |

**Combined Score Calculation**:
```
score_kin  = 1 - RMSE(sim, ref_angle) / ROM_ref
score_grf  = 1 - RMSE(sim_grf, ref_grf) / peak_ref
score_emg  = 1 - mean_timing_error / 15%
total_score = 0.4×score_kin + 0.3×score_grf + 0.3×score_emg
```

**"Believable Simulation" Criteria**:

| Level | Criteria | Significance |
|-------|---------|-------------|
| Minimum | Double-peak GRF + ROM qualitatively similar + EMG ±20% | Paper A submittable |
| Target | RMSE < 8°, GRF < 0.15 BW, EMG ±10% | Paper B contribution |
| Ideal | Patient-specific parameter identification and reproduction | Paper C possible |

---

### L5 — Patient Pathological Patterns ⬜ Not Implemented (0/6)

**Question**: Do patient parameter changes produce known pathological gait?
**Tools**: IsaacGym + patient profile YAML
**Folder**: `validation/L5_patient_pathology/` (new)

| ID | Patient Group | Expected Pathology | Quantitative Criteria |
|----|--------------|-------------------|----------------------|
| L5-01 | Stroke | Stiff Knee Gait (reduced swing flexion) | Affected swing peak knee flex < 35° |
| L5-02 | Stroke | Circumduction (hip abduction compensation) | Affected hip abduction > 5° during swing |
| L5-03 | Parkinson | Reduced step length | step_length < 0.7 × healthy |
| L5-04 | Parkinson | Foot drag | foot clearance < 2 cm |
| L5-05 | SCI | Voluntary muscle activation loss | EMG sum < 5% of normal |
| L5-06 | Common | Severity ↑ → functional metrics monotonically decrease | mild < severe for all metrics |

---

## Reorganized Folder Structure

```
validation/
  L0_unit_math/               ← renamed from 01_muscle_layer
  L1_module_flow/              ← new (1 week)
  L2_physics_integration/      ← renamed from 02_isaacgym + L2-05,06 added
  L3_phenomena/                ← reorganized from 04+05
  L4_data_fidelity/            ← new (key gap)
    reference_data/
      winter2009_normative.yaml
      perry1992_emg_timing.yaml
    test_kinematics_vs_normative.py
    test_grf_vs_normative.py
    test_emg_timing_vs_normative.py
    test_combined_fidelity_score.py
  L5_patient_pathology/        ← new
  run_all.py                   ← sequential execution + dashboard
  VALIDATION_STATUS.md         ← auto-update
```

---

## Current Status Summary

```
L0 ██████████ 9/9   PASS
L1 ░░░░░░░░░░ 0/7   Not implemented (1 week, no IsaacGym needed)
L2 ████████░░ 4/6   PASS (L2-05,06 addition recommended)
L3 ████████░░ 6/7   Needs reorganization
L4 ░░░░░░░░░░ 0/4   KEY GAP (start with reference data)
L5 ░░░░░░░░░░ 0/6   After L4 completion

Overall: ~40% (math/physics foundation complete, real-data comparison is the major gap)
```

**Immediately actionable tasks (no coding)**:
1. `reference_data/winter2009_normative.yaml` — ✅ Done
2. `reference_data/perry1992_emg_timing.yaml` — ✅ Done
3. L1 `test_tensor_shapes.py` — 1–2 days

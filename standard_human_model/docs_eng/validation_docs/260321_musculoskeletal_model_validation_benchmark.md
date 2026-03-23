# Musculoskeletal Model Validation Methodology — MyoSuite / OpenSim Benchmark Survey

Date: 2026-03-21
Reference: 260320_검증_ver2.md (CALM Validation Pyramid)

---

## Background and Purpose

Before finalizing the L4 criteria in the CALM Validation Pyramid, this document surveys how leading
musculoskeletal simulation frameworks (MyoSuite, OpenSim) validated their models and what
quantitative benchmarks they used. The goal is to anchor our CALM pyramid criteria in established
practice and identify gaps not yet covered by published standards.

---

## 1. MyoSuite Validation (Caggiano et al., 2022)

### Core Philosophy

MyoSuite validates the **fidelity of OpenSim → MuJoCo conversion**, not the physiological validity
of the underlying model. The original OpenSim model is assumed pre-validated; only the accuracy of
the MuJoCo port is tested.

### Three-Stage Validation (Vlt1–Vlt3)

| Stage | Target | Quantitative Criterion |
|-------|--------|----------------------|
| Vlt1 | Forward Kinematics (joint geometry, marker positions) | Visual convergence check (no numeric threshold) |
| Vlt2 | Moment Arm curves (full ROM × all muscles) | Elbow: RMS 0.044±0.09% / Hand: RMS 0.38±0.57% |
| Vlt3 | Force-Length-Activation relationship | Elbow: 2.2±1.4% Fmax / Hand: 4.1±2.0% Fmax |

MyoConverter automated pipeline pass criteria:
- Pearson r > 0.7
- Absolute RMSE < 1 mm (moment arm)
- Normalized RMSE < 0.4

### Items NOT Validated by MyoSuite at the Model Level

- Independent ROM range (vs. cadaver data)
- EMG matching at the model level
- Integrated gait kinematics vs. normative data
- Ground Reaction Forces (GRF)

→ These are classified as "task-level research" and delegated to downstream studies.

---

## 2. OpenSim Validation

### 2-1. Hicks et al. 2015 — "Is My Model Good Enough?"

The most authoritative OpenSim validation guideline.

**Multibody Dynamics Level:**
- Joint angles/moments: within **±2 SD** of published reference values
- Residual forces: **≤ 5% of peak and RMS** net external force
- Residual moments: **≤ 1%** of (COM height × net external force)

**Musculoskeletal Geometry Level:**
- Moment arms: within **±2 SD** of experimental data (across full ROM)
- Net joint moments: within **±2 SD** of experimental values

**Muscle-Tendon Dynamics Level:**
- Activation mean absolute force error: max 9% active / 16% passive
- FL / FV / tendon curves: must be C2-continuous with no negative-stiffness regions

**Required Sensitivity Analyses (priority order):**
1. Tendon slack length (most sensitive; critical for muscles with tendon/fiber ratio > 3)
2. Optimal fiber length
3. Maximum isometric force
4. Pennation angle (least sensitive)

### 2-2. Rajagopal et al. 2016 — Full-Body Musculoskeletal Model

The basis for MyoLeg. Provides the most concrete integrated gait validation numbers.

**Joint Moment RMSE (primary integrated validation metric):**

| Joint | Walking Error | Running Error |
|-------|-------------|--------------|
| Hip | < 2% peak moment | < 3% peak |
| Knee | < 2% peak moment | < 3% peak |
| Ankle | < 2% peak moment | < 3% peak |

**Kinematics Error:**
- Walking: rotations < 1° RMSE, pelvis translation < 0.1 cm
- Running: rotations < 4° RMSE, pelvis translation < 0.2 cm

**EMG Comparison:**
- No quantitative threshold. **Qualitative match** of timing and peak patterns only.

**Valid ROM (model validity range):**
- Ankle: 40° plantarflexion ~ 30° dorsiflexion
- Knee: 0° ~ 120° flexion
- Hip flex/ext: −30° ~ 120°; hip ab/adduction: −30° ~ 50°

---

## 3. State of Official Benchmarks in the Field

### Springer 2025 — Skeletal Muscle Models Benchmark

5 benchmark cases for stepwise validation of muscle contraction dynamics:
- Cases 1–3: Single-muscle FL/FV dynamics
- Cases 4–5: Multi-joint integrated dynamics

Key statement from the paper:
> *"The lack of standard reference data makes the computational validation of new models difficult."*

**Conclusion: No official benchmark for simultaneous Kin + GRF + EMG validation exists as of 2025.**

### Conventions for Integrated Gait Validation

| Metric | Common Threshold |
|--------|----------------|
| Vertical GRF Pearson ρ | > 0.94 |
| GRF normalized RMSE | ≤ 5.3% |
| Joint moment correlation | weak ≤ 0.35 / moderate 0.35–0.67 / strong 0.67–0.90 / excellent > 0.90 |
| EMG (KINESIS 2025) | Pearson r across 9 subjects × 9 lower-limb muscles |

---

## 4. Item-by-Item Comparison

| Validation Item | MyoSuite | OpenSim | Official Benchmark |
|----------------|---------|---------|-------------------|
| Joint ROM | Design range stated only (not validated) | < 1–4° RMSE within use range | None |
| Joint torque range | Not validated | RMSE < 2–3% peak | None |
| Force-Length curve | < 4.1% Fmax RMS | C2-continuous + 9% MAE | Springer 2025 |
| Force-Velocity curve | < 4.1% Fmax RMS | C2-continuous | Springer 2025 |
| Moment arm | < 0.38% RMS, r > 0.7 | ±2 SD (experimental) | Partial |
| EMG matching | Not validated | Qualitative only | None (no standard) |
| Gait kinematics | Not validated | < 1° RMSE, 6% peak deviation | Winter 2009 |
| GRF | Not validated | Used as input, not validated as output | ρ > 0.94 convention |

---

## 5. Mapping to User's Intuitive Framework

The user proposed three validation tiers; these map precisely to published practice:

```
User's Intuition                    →   Literature Equivalent

① Per-joint exhaustive check        →   Vlt1 (geometry) + joint moment RMSE (Rajagopal)
② Per-muscle exhaustive check       →   Vlt2 (moment arms) + Vlt3 (FL/FV curves)
③ Integrated data flow validation   →   Rajagopal 2016 gait validation (Kin + partial EMG)
```

Mapping to the CALM Validation Pyramid:

```
L0 — Unit Math Tests          ↔   Vlt3 (FL/FV curve numeric accuracy)
L1 — Module Data Flow         ↔   Vlt2 (moment arm propagation accuracy)
L2 — Physics Engine           ↔   Vlt1 (forward kinematics geometry)
L3 — Phenomenological Tests   ↔   OpenSim qualitative EMG timing comparison
L4 — Quantitative Data Match  ↔   Rajagopal 2016 gait RMSE criteria
L5 — Patient Pathology        ↔   No equivalent in literature (CALM-specific layer)
```

---

## 6. Assessment of CALM L4 Criteria

Current L4 pass criteria (from ver2.md):
- Kinematics: RMSE < 8° (minimum), < 8° (target)
- GRF: RMSE < 0.15 BW
- EMG timing: within ±10% gait cycle

Comparison with Rajagopal 2016:
- Kinematics: Rajagopal < 1° RMSE (walking) vs. CALM < 8° → **CALM is more lenient**
- Joint moment: Rajagopal < 2% peak → **CALM L4 has no moment criterion (gap)**
- EMG: Rajagopal qualitative only → **CALM is more rigorous**

Recommendations:
1. The kinematic threshold could be tightened in future iterations (< 5° RMSE as intermediate target).
2. Consider adding a joint moment RMSE criterion to L4 to close the gap with Rajagopal 2016.
3. Current L4 criteria are appropriate as an initial pass bar; they are justifiable against the field.

---

## References

- Caggiano et al. (2022). MyoSuite: A contact-rich simulation suite for musculoskeletal motor control. arXiv:2205.13600
- Hicks et al. (2015). Is My Model Good Enough? Best Practices for Musculoskeletal Model Development. J. Biomechanical Engineering. PMC4321112
- Rajagopal et al. (2016). Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. IEEE TBME. PMC5507211
- Arnold et al. (2010). A Model of the Lower Limb for Analysis of Human Movement. PMC2903973
- Springer (2025). Validation of skeletal muscle models in multibody dynamics: A collaborative collection of benchmark cases. DOI:10.1007/s11044-025-10096-8
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.). Wiley.
- Perry, J. (1992). Gait Analysis: Normal and Pathological Function. SLACK Inc.

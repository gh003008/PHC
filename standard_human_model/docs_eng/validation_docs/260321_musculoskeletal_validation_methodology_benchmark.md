# Musculoskeletal Model Validation Methodology — MyoSuite / OpenSim Benchmark Survey

Created: 2026-03-21
Reference: 260320_검증_ver2.md (CALM Validation Pyramid)
Translated from Korean original: `validation_docs/260321_근골격모델_검증방법론_벤치마크.md`

---

## Background and Purpose

Before designing L4 criteria for the CALM validation pyramid (ver2), we surveyed how existing representative frameworks (MyoSuite, OpenSim) validate their musculoskeletal models and what quantitative criteria they use.

---

## 1. MyoSuite Validation (Caggiano et al., 2022)

### Core Philosophy

MyoSuite validates **OpenSim → MuJoCo conversion accuracy**. The physiological validity of the original OpenSim model is assumed as already verified; only the fidelity of the converted MuJoCo model is tested.

### 3-Stage Validation (Vlt1~Vlt3)

| Stage | Target | Quantitative Criteria |
|-------|--------|----------------------|
| Vlt1 | Forward Kinematics (joint geometry, marker positions) | Visual convergence (no numerical criteria) |
| Vlt2 | Moment Arm curves (full ROM × all muscles) | Elbow: RMS 0.044±0.09% / Hand: RMS 0.38±0.57% |
| Vlt3 | Force-Length-Activation relationships | Elbow: 2.2±1.4% Fmax / Hand: 4.1±2.0% Fmax |

MyoConverter auto-pipeline pass criteria: Pearson r > 0.7, absolute RMSE < 1mm (moment arm), normalized RMSE < 0.4

### What MyoSuite does NOT validate:
- Independent ROM ranges (vs cadaver)
- Model-level EMG matching
- Integrated gait kinematics vs normative data
- GRF (ground reaction force) reproduction

→ These are classified as "task-level research" and not performed at model level.

---

## 2. OpenSim Validation

### 2-1. Hicks et al. 2015 — "Is My Model Good Enough?"

The most authoritative OpenSim validation guideline.

**Multibody Dynamics level:**
- Joint angles/moments: within **±2 SD** of comparison literature
- Residual force: **≤ 5%** of measured net external force (peak and RMS)
- Residual moment: **≤ 1%** of COM height × net external force

**Musculoskeletal Geometry level:**
- Moment arm: within **±2 SD** of experimental data (full ROM)
- Net joint moment: within **±2 SD** of experimental values

**Muscle-tendon dynamics level:**
- Activation max error: 9% / deactivation max error: 16% (mean absolute force error)
- FL / FV / tendon curves: C2 continuity required + no negative stiffness regions

**Sensitivity analysis priority (mandatory):**
1. Tendon slack length (most sensitive, critical for tendon/fiber > 3)
2. Optimal fiber length
3. Maximum isometric force
4. Pennation angle (relatively less sensitive)

### 2-2. Rajagopal et al. 2016 — Full-Body Musculoskeletal Model

Base model for MyoLeg. Most specific integrated gait validation numbers.

**Joint moment RMSE:**

| Joint | Walking error | Running error |
|-------|-------------|--------------|
| Hip | < 2% peak moment | < 3% peak |
| Knee | < 2% peak moment | < 3% peak |
| Ankle | < 2% peak moment | < 3% peak |

**Kinematics error:**
- Walking: rotation < 1° RMSE, pelvis displacement < 0.1 cm
- Running: rotation < 4° RMSE, pelvis displacement < 0.2 cm

**EMG comparison:** No quantitative criteria. Only **qualitative matching** of major features (timing, peak patterns).

---

## 3. Field-wide Official Benchmark Status

### Springer 2025 — Skeletal Muscle Models Benchmark

5 benchmark cases for staged muscle contraction dynamics verification (Cases 1–3: single muscle, Cases 4–5: multi-joint systems).

Paper's stated limitation:
> "The lack of standard reference data makes computational verification of new models difficult."
→ An official benchmark for integrated gait validation (Kin+GRF+EMG simultaneous) **does not exist as of 2025**.

### Conventionally used criteria for integrated gait validation:

- Vertical GRF correlation: ρ > 0.94, normalized RMSE ≤ 5.3%
- Joint moment correlation classification: weak ≤ 0.35 / moderate 0.35–0.67 / strong 0.67–0.90 / excellent > 0.90
- EMG: KINESIS 2025 — Pearson correlation, 9 subjects × 9 lower-limb muscles

---

## 4. Comparison Summary Table

| Validation Item | MyoSuite | OpenSim | Official Benchmark |
|----------------|---------|---------|-------------------|
| Joint ROM | Design range stated (no verification) | < 1–4° RMSE within usage range | None |
| Joint torque range | Not verified | RMSE < 2–3% peak | None |
| Force-Length curve | < 4.1% Fmax RMS | C2 continuous + 9% MAE | Springer 2025 |
| Force-Velocity curve | < 4.1% Fmax RMS | C2 continuous | Springer 2025 |
| Moment arm | < 0.38% RMS, r > 0.7 | ±2 SD (experimental) | Partial |
| EMG matching | Not verified | Qualitative only | None (no standard) |
| Gait kinematics | Not verified | < 1° RMSE, 6% peak deviation | Winter 2009 |
| GRF | Not verified | Used as input (no output verification) | ρ > 0.94 convention |

---

## 5. CALM Validation Pyramid Mapping

```
L0 — Unit math tests      ↔   Vlt3 (FL/FV curve numerical accuracy)
L1 — Module data flow      ↔   Vlt2 (moment arm transfer accuracy)
L2 — Physics integration   ↔   Vlt1 (forward kinematics geometry)
L3 — Phenomenological      ↔   OpenSim qualitative EMG timing comparison
L4 — Data fidelity         ↔   Rajagopal 2016 gait RMSE criteria
L5 — Patient pathology     ↔   No literature equivalent (CALM-unique layer)
```

---

## 6. Assessment of CALM L4 Criteria

Current L4 pass criteria (ver2.md):
- Kinematics: RMSE < 8° (minimum/target)
- GRF: RMSE < 0.15 BW
- EMG timing: ±10% GC

Compared to Rajagopal 2016:
- Kinematics: Rajagopal < 1° RMSE vs CALM < 8° → **CALM is more lenient**
- Joint moments: Rajagopal < 2% peak → CALM L4 has no moment criterion (room for addition)
- EMG: Rajagopal qualitative only → **CALM is more strict**

Conclusion: L4 kinematics criteria can be tightened in the future. Current criteria are appropriate as "first-pass" level.

---

## References

- Caggiano et al. (2022). MyoSuite: A contact-rich simulation suite. arXiv:2205.13600
- Hicks et al. (2015). Is My Model Good Enough? J. Biomechanical Engineering. PMC4321112
- Rajagopal et al. (2016). A Full-Body Musculoskeletal Model. IEEE TBME. PMC5507211
- Arnold et al. (2010). A Model of the Lower Limb. PMC2903973
- Springer (2025). Validation of skeletal muscle models: benchmark cases. DOI:10.1007/s11044-025-10096-8
- Winter (2009). Biomechanics and Motor Control of Human Movement (4th ed.)
- Perry (1992). Gait Analysis: Normal and Pathological Function

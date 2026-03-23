# CALM Framework Overview

Created: 2026-03-21 (Fri)

---

## 1. What is CALM?

**CALM** (Computational Anatomical Locomotion Model) is a GPU-parallel musculoskeletal human model framework built on **PyTorch** + **IsaacGym (PhysX 4)**. It models the human neuromusculoskeletal system as a modular pipeline and computes biologically plausible joint torques for physics simulation.

### One-line Summary

> A 5-layer musculoskeletal torque pipeline (Hill muscle → Moment arm → Activation dynamics → Spinal reflex → Ligament) that runs on GPU in parallel, designed for RL-based exoskeleton controller training.

### Ultimate Goal

Train an **exoskeleton controller via RL** in simulation, where the human model responds with realistic impedance modulation. CALM is the "human side" — modeling how a person generates joint torques through muscles, reflexes, and passive tissues.

---

## 2. Computation Pipeline

```
Inputs from Physics Engine (IsaacGym)
  ├─ dof_pos        (B, 69)   Joint angles [rad]
  ├─ dof_vel        (B, 69)   Joint velocities [rad/s]
  ├─ descending_cmd (B, 20)   Higher-level muscle commands [0, 1]
  └─ contact_forces (B, 4)    Foot contact forces [N]
         │
         ▼
  ╔═══════════════════════════════════════════════════════╗
  ║              HumanBody.compute_torques()              ║
  ║                                                       ║
  ║  Step 1: Muscle Kinematics      (moment_arm.py)       ║
  ║    l_muscle = l_slack - R(q) @ q                      ║
  ║    v_muscle = -R(q) @ dq                              ║
  ║                                                       ║
  ║  Step 2: Spinal Reflex Layer    (reflex_controller.py)║
  ║    stretch reflex + GTO inhibition                    ║
  ║    + reciprocal inhibition                            ║
  ║                                                       ║
  ║  Step 3: Activation Dynamics    (activation_dynamics)  ║
  ║    1st-order ODE: da/dt = (u - a) / tau               ║
  ║                                                       ║
  ║  Step 4: Hill Muscle Force      (muscle_model.py)     ║
  ║    F = a × f_max × f_FL × f_FV × cos(α) + F_passive  ║
  ║                                                       ║
  ║  Step 5: Torque Mapping         (moment_arm.py)       ║
  ║    tau_muscle = F @ R(q)                              ║
  ║                                                       ║
  ║  Step 6: Ligament Forces        (ligament_model.py)   ║
  ║    Exponential soft joint limits                      ║
  ╚═══════════════════════════════════════════════════════╝
         │
         ▼
  tau_total (B, 69) → IsaacGym apply_forces()
```

**Key design choice**: CALM computes torques *outside* the physics engine, then injects them as constant external forces per timestep. This is a **staggered/decoupled coupling** (as opposed to OpenSim's monolithic coupled ODE). At dt = 1/120s (8.3ms), which is well below muscle time constants (15–80ms), the approximation error is negligible for RL training purposes.

---

## 3. Architecture

### Module Map

```
standard_human_model/
├── core/                          # Biomechanics engine (all PyTorch)
│   ├── human_body.py              # Pipeline orchestrator (main entry)
│   ├── muscle_model.py            # Hill-type muscle: CE + PE + SE (elastic tendon)
│   ├── moment_arm.py              # R(q) matrix: angle-dependent polynomial
│   ├── activation_dynamics.py     # Neural cmd → muscle activation (1st-order ODE)
│   ├── reflex_controller.py       # Stretch, GTO, reciprocal inhibition (with delay)
│   ├── ligament_model.py          # Exponential soft joint limits
│   ├── skeleton.py                # SMPL 24-body, 23-joint, 69-DOF constants
│   ├── patient_profile.py         # YAML-based patient parameter loader
│   └── patient_dynamics.py        # Patient-specific torque computation
│
├── config/
│   ├── muscle_definitions.yaml    # 20 muscle groups, moment arms, antagonist pairs
│   └── healthy_baseline.yaml      # Rajagopal 2016-based muscle parameters
│
├── profiles/                      # Patient YAML profiles
│   ├── healthy/                   #   healthy adult baseline
│   ├── stroke/                    #   right hemiplegia
│   ├── parkinson/                 #   moderate (bradykinesia + rigidity)
│   ├── sci/                       #   T10 complete flaccid, incomplete spastic
│   └── cp/                        #   spastic diplegia
│
├── validation/                    # 5-level validation pyramid
│   ├── L0_unit_math/              #   Individual equation correctness
│   ├── L1_module_flow/            #   Data flow, shapes, sign conventions
│   ├── L2_physics_integration/    #   IsaacGym torque injection tests
│   ├── reference_data/            #   Winter 2009 gait + Perry 1992 EMG
│   └── ...
│
├── examples/                      # Standalone test scripts
└── docs/ (docs_eng/)              # Documentation (Korean / English)
```

### Key Dimensions

| Item | Value |
|------|-------|
| Bodies | 24 (SMPL humanoid) |
| Joints | 23 |
| DOFs | 69 (23 × 3 axes) |
| Muscle groups | 20 (10 per side, 5 bi-articular) |
| Patient parameters | 9 per joint group × 8 groups = 72-dim |
| Contact bodies | 4 (L/R Ankle + L/R Toe) |

### Core API

```python
class HumanBody:
    @classmethod
    def from_config(muscle_def_path, param_path, num_envs, device) -> HumanBody

    def compute_torques(
        dof_pos: Tensor,           # (B, 69)
        dof_vel: Tensor,           # (B, 69)
        descending_cmd: Tensor,    # (B, 20)  [0, 1]
        dt: float,
        sim_time: float = 0.0,
        contact_forces: Tensor = None,  # (B, 4)
    ) -> Tensor:                   # (B, 69) joint torques
```

---

## 4. Validation Status

We use a **5-level validation pyramid** to systematically verify the model from mathematical correctness (L0) up to patient-specific pathological gait reproduction (L5).

```
L0  Unit Math               ██████████  9/9   PASS
L1  Module Data Flow         ░░░░░░░░░░  0/7   Not implemented
L2  Physics Integration      ████████░░  4/6   PASS (2 additional recommended)
L3  Phenomenological         ████████░░  ~6/7  Partial (needs reorganization)
L4  Data Fidelity            ░░░░░░░░░░  0/4   NOT IMPLEMENTED (critical gap)
L5  Patient Pathology        ░░░░░░░░░░  0/6   Requires L4 first

Overall: ~40% (math/physics foundation done; real-data comparison is the major gap)
```

### Validation Levels

| Level | Question | Tools Needed |
|-------|----------|-------------|
| **L0** | Are individual equations mathematically correct? | PyTorch only |
| **L1** | Do modules connect with correct shapes and signs? | PyTorch only |
| **L2** | Does CALM torque produce expected motion in physics sim? | IsaacGym |
| **L3** | Do known biomechanical phenomena emerge? | IsaacGym |
| **L4** | How close to real gait data (Kin + GRF + EMG)? | IsaacGym + reference data |
| **L5** | Do patient parameters reproduce known pathological gait? | IsaacGym + patient profiles |

### "Believable Simulation" Criteria

| Level | Kinematics | GRF | EMG | Significance |
|-------|-----------|-----|-----|-------------|
| **Minimum** | ROM qualitatively similar | Double-peak pattern visible | ON timing ±20% GC | Paper A submittable |
| **Target** | RMSE < 8° per joint | RMSE < 0.15 BW | Timing ±10% GC | Paper B contribution |
| **Ideal** | Patient-specific fit | Patient-specific fit | Patient-specific fit | Paper C possible |

### Fidelity Score

```
score_kin  = 1 - RMSE(sim, ref_angle) / ROM_ref
score_grf  = 1 - RMSE(sim_grf, ref_grf) / peak_ref
score_emg  = 1 - mean_timing_error / 15%
total_score = 0.4 × score_kin + 0.3 × score_grf + 0.3 × score_emg
```

### Reference Data

- **Winter 2009** — Normative joint kinematics + GRF at 5% gait cycle intervals (`reference_data/winter2009_normative.yaml`)
- **Perry 1992** — EMG ON/OFF timing for 10 muscle groups (`reference_data/perry1992_emg_timing.yaml`)

---

## 5. Known Bugs & Status

| ID | Location | Impact | Severity | Status |
|----|----------|--------|----------|--------|
| BUG-01 | `healthy_baseline.yaml` | `l_opt: 1.0` (unitless) → active force only ~30% of max | Critical | **Fixed** (Rajagopal 2016 values) |
| BUG-02 | `moment_arm.py` | Constant R matrix → ±30% torque error at extreme angles | High | **Fixed** (polynomial R(q)) |
| BUG-03 | `muscle_model.py` | No elastic tendon → push-off power underestimated | High | **Fixed** (Newton-Raphson SE for soleus/gastroc) |
| BUG-04 | `reflex_controller.py` | Zero reflex delay → clonus timing inaccurate | Medium | **Fixed** (circular buffer, 5-step delay) |

---

## 6. Comparison with Existing Frameworks

### vs. OpenSim

| Aspect | OpenSim | CALM |
|--------|---------|------|
| Dynamics coupling | Monolithic coupled ODE (muscle + multibody solved together) | Staggered (muscle computed separately, torque injected) |
| Muscle model | Millard 2013, elastic tendon standard | Hill-type + elastic tendon (soleus/gastroc only) |
| Moment arm | Geometry-based wrapping surfaces | Polynomial R(q) approximation |
| Parallelism | Single environment | GPU-parallel (thousands of envs) |
| Use case | Clinical biomechanics analysis | RL training at scale |

### vs. MyoSuite (MuJoCo)

| Aspect | MyoSuite | CALM |
|--------|----------|------|
| Dynamics coupling | Semi-coupled (activation → force → MuJoCo step) | Staggered (CALM → PhysX step) |
| Muscle count | 39+ (upper extremity focus) | 20 (lower-limb gait focus) |
| Reflex model | Not included | Stretch + GTO + reciprocal inhibition |
| Patient profiles | Not included | 5 pathology types with 72-dim parameterization |
| RL integration | Native (Gym API) | IsaacGym VecTask API |

### Key Architectural Difference

All three frameworks use **open-chain tree dynamics** (ABA or similar O(n) algorithm). Muscles are force generators, not kinematic links — they do not create closed loops. The coupling difference is about how muscle state ODEs (activation, fiber length) are integrated relative to the multibody EoM:

```
OpenSim:   [q, qdot, a, l_ce] → single ODE system → RK4
MuJoCo:    [act] computed first → [q, qdot] stepped with muscle force
CALM:      muscle pipeline external → tau injected as constant → PhysX steps
```

At CALM's dt = 8.3ms (well below muscle τ = 15–80ms), staggered coupling error is negligible.

---

## 7. Team Structure & Roles

The framework is designed for **4–5 person** parallel development with clear interface contracts.

```
┌──────────────────────────────────────────────────────────────┐
│                      CALM Framework Stack                     │
│                                                              │
│  [Layer 4]  Higher-level Controller (RL / CPG)        Role D │
│  [Layer 3]  Human-Robot Integrated Sim (Isaac Lab)    Role C │
│  [Layer 2]  Patient Parameterization (Real-to-Sim)    Role B │
│  [Layer 1]  Musculoskeletal Core (Hill + Reflex)      Role A │
│  [Layer 0]  Body Model (SMPL/URDF + Exo assets)    Role A+C │
└──────────────────────────────────────────────────────────────┘
```

| Role | Focus | Owner Profile |
|------|-------|--------------|
| **A** — Biomechanics Core | `core/` modules, muscle math, API design, code review | PI (VIC/PHC background) |
| **B** — Clinical Parameterization | Real-to-Sim pipeline, patient profiles, literature values | Rehab engineering / biomechanics |
| **C** — Robot Integration | Exo URDF, Isaac Lab migration, closed-chain support | Robotics / simulation |
| **D** — Control & Learning | RL policy, CPG controller, benchmark experiments | RL / control theory |
| **E** — Validation & Visualization | L3–L5 test scripts, gait analysis plots, dashboards | Data analysis (optional 5th member) |

### Interface Contracts (frozen APIs)

```python
# Interface 1: CALM Core API (Role A → all)
HumanBody.compute_torques(dof_pos, dof_vel, descending_cmd, dt, ...) → tau

# Interface 2: Patient Profile Schema (Role B → Role A)
profiles/[condition]/[name].yaml  # fixed YAML schema

# Interface 3: Human-Robot Env API (Role C → Role D)
HumanRobotEnv.step(action) → (obs, reward, done, info)
```

Each role can develop independently as long as these interfaces are respected.

---

## 8. Roadmap & Priorities

| Priority | Task | Role | Estimated Effort |
|----------|------|------|-----------------|
| 1 | ~~BUG-01: Fix l_opt units~~ | A | ~~Done~~ |
| 2 | ~~BUG-02–04: Moment arm, elastic tendon, reflex delay~~ | A | ~~Done~~ |
| 3 | L1 module flow tests | E | 1 week |
| 4 | L4 data fidelity: Kin + GRF + EMG vs reference | A+E | 2 weeks |
| 5 | Real-to-Sim identification pipeline | B | 3 weeks |
| 6 | Ankle exoskeleton URDF | C | 2 weeks |
| 7 | IsaacGym → Isaac Lab migration | C | 4 weeks |
| 8 | RL controller with CALM muscle layer | D | 3 weeks |
| 9 | L5 patient pathology validation | A+E | 2 weeks |
| 10 | Cross-patient benchmark experiments | D | 2 weeks |

### Migration Path

```
Current:  IsaacGym (PhysX 4)  — open chain only
Target:   Isaac Lab (PhysX 5)  — closed kinematic chain support
Reason:   Exoskeleton straps create kinematic loops
```

---

## 9. Quick Start

```bash
# Run validation tests (no IsaacGym needed)
cd standard_human_model
python examples/test_pipeline.py

# Run L0 unit math validation
python validation/L0_unit_math/run_validation.py

# Run L2 physics integration (requires IsaacGym)
python validation/L2_physics_integration/run_validation.py
```

---

## 10. Document Index

### English (`docs_eng/`)

| Date | Document | Topic |
|------|----------|-------|
| 260316 | standard_human_model_strategy.md | Initial project strategy |
| 260316 | implementation_guide.md | Implementation guide |
| 260316 | musculoskeletal_joint_level_architecture.md | Joint-level architecture |
| 260317 | project_strategy01.md | Project strategy update |
| 260317 | musculoskeletal_pipeline_implementation.md | Pipeline implementation |
| 260317 | isaacgym_integration_plan.md | IsaacGym integration plan |
| 260317 | implementation_notes_01.md | Implementation caveats |
| 260318 | musculoskeletal_human_model_framework.md | Full framework description |
| 260318 | muscle_layer_validation_detail.md | L0 muscle layer validation |
| 260318 | isaacgym_integration_validation_detail.md | L2 IsaacGym validation |
| 260318 | framework_roadmap_team_structure.md | Roadmap & team structure |
| 260319 | CALM_team_structure_research_strategy.md | Team roles & strategy |
| 260319 | framework_implementation_gap_analysis.md | Gap analysis |
| 260320 | CALM_computation_flow_visualization.md | Computation flow diagrams |
| 260320 | validation_pyramid_v2.md | Validation pyramid design |
| 260321 | CALM_bug_list.md | Bug list |
| 260321 | CALM_bug_fix_detail_report.md | Bug fix report |
| 260321 | MyoSuite_OpenSim_computation_flow_comparison.md | Framework comparison |
| 260321 | **CALM_framework_overview.md** | **This document** |

### Validation Reference Data

| File | Source | Content |
|------|--------|---------|
| `winter2009_normative.yaml` | Winter (2009) 4th ed. | Joint kinematics + GRF at 5% GC intervals |
| `perry1992_emg_timing.yaml` | Perry (1992) + Kadaba 1989 | EMG ON/OFF timing for 10 muscle groups |

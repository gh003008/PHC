# HANDOFF: standard_human_model Validation Pipeline
**Date**: 2026-03-18
**Branch**: `human_model_v1`
**Last commit**: `7699215` (fix: 검증 플롯 한글 폰트 설정)
**Next task**: Step 2 — `02_isaacgym_integration` validation

---

## 1. What We Are Building

A **lab infrastructure framework** for human-robot integrated simulation.
Core pipeline: PHC (IsaacGym + rl-games) + VIC (Variable Impedance Control) + `standard_human_model` (musculoskeletal layer).

Final goal: Exoskeleton controller RL training on a realistic human model, with synthetic bio-signal generation (EMG, GRF) as a data factory for clinical research.

---

## 2. Current Architecture

```
PHC/
├── phc/env/tasks/humanoid_im_vic.py        ← VIC environment (RL training, active)
├── phc/env/tasks/humanoid_im_vic_msk.py    ← VIC + MSK integration task (new, unverified)
├── phc/data/cfg/env/env_im_walk_vic_msk.yaml
├── standard_human_model/                   ← Musculoskeletal layer (our current focus)
│   ├── core/
│   │   ├── human_body.py           ← Main pipeline class (compute_torques entry point)
│   │   ├── muscle_model.py         ← Hill-type muscle (F-L, F-V, passive)
│   │   ├── moment_arm.py           ← R matrix (20 muscles × 69 DOFs), torque mapping
│   │   ├── activation_dynamics.py  ← 1st order ODE (tau_act, tau_deact)
│   │   ├── reflex_controller.py    ← Spinal reflex (stretch, GTO, reciprocal inhibition)
│   │   ├── ligament_model.py       ← Soft-limit joint torque (exponential model)
│   │   ├── patient_profile.py      ← Patient profile YAML loader
│   │   └── skeleton.py             ← SMPL DOF definitions (NUM_DOFS=69, JOINT_DOF_RANGE)
│   ├── config/
│   │   ├── muscle_definitions.yaml ← 20 muscle groups + R matrix + antagonist pairs
│   │   └── healthy_baseline.yaml   ← Reference muscle parameters (f_max, l_opt, etc.)
│   ├── profiles/
│   │   ├── stroke/stroke_r_hemiplegia.yaml
│   │   └── sci/sci_t10_complete_flaccid.yaml
│   ├── isaacgym_validation/        ← OLD validation scripts (exp1~5, demo)
│   │   ├── demo_knee_pendulum.py   ← IsaacGym viewer demo (3 profiles, knee pendulum)
│   │   ├── smpl_humanoid_fixed.xml ← MJCF without freejoint (fix_base_link=True)
│   │   └── results/                ← PNG outputs from exp1~5
│   └── validation/                 ← NEW structured validation (our work today)
│       └── 01_muscle_layer/
│           ├── run_validation.py   ← 9 tests (T01~T09), 10/10 PASS
│           ├── README.md
│           └── results/            ← T01~T09 PNG plots
└── 04_Context_Bridge/              ← This file location
```

### HumanBody.compute_torques() pipeline
```
dof_pos, dof_vel, descending_cmd (RL/CPG/EMG)
  → muscle_length = l_slack - R @ q          (MomentArmMatrix)
  → muscle_velocity = -(R @ dq)              (MomentArmMatrix)
  → current activation (from prev step)
  → current force (Hill model, for reflex)
  → a_cmd = reflex(descending_cmd, vel, force)  (ReflexController)
  → activation = activation_dynamics.step(a_cmd, dt)
  → F_muscle = hill_model.compute_force(activation, length, velocity)
  → tau_muscle = F_muscle @ R                (moment_arm.forces_to_torques)
  → tau_ligament = ligament.compute_torque(dof_pos, dof_vel)
  → tau_total = tau_muscle + tau_ligament    ← returned
```

Integration with IsaacGym: torques injected via `gym.set_dof_actuation_force_tensor()`.
This is NOT deep integration — it's an external torque computation loop (Strategy C Hybrid, `blend_alpha=0.0` by default = identical to base VIC).

---

## 3. Muscle Model Details

### 20 Muscle Groups (10 per side, L/R)
```
hip_flexors, gluteus_max, hip_abductors, hip_adductors,
quadriceps (mono), rectus_femoris (bi: hip+knee),
hamstrings (bi: hip+knee), gastrocnemius (bi: knee+ankle),
soleus (mono: ankle), tibialis_ant (mono: ankle)
```
Bi-articular muscles verified: each spans exactly 2 joints in R matrix (T05b PASS).

### R Matrix Convention
- Shape: (num_muscles=20, num_dofs=69)
- `l_muscle = l_slack - R @ q`  (positive R → muscle shortens when joint extends)
- `tau = F_muscle @ R`  (equivalent to R^T @ F)
- Source: OpenSim Rajagopal 2016 approximations, constant (no q-dependence yet)

### Known Parameter Issue (NOT a code bug)
`l_opt=1.0` in YAML is dimensionless, but `l_slack` is in meters (~0.30m).
Result: `l_norm = l_slack / l_opt = 0.30` → muscles operate at 30% of optimal length.
Active force at this point: `exp(-((0.3-1)^2)/(2*0.45^2)) ≈ 0.30` of F_max.
**Passive force = 0** (requires l_norm > 1.0, never reached with current params).
Fix needed: set `l_opt` to actual optimal fiber length in meters (e.g., 0.08m for quadriceps).

### SMPL DOF Convention
- `q=0` for knee = full extension (lower limit), NOT neutral
- Knee ROM: [0°, 145°] → neutral (ROM center) ≈ 72.5°
- This caused T09 diagnostic: max|tau|=283 Nm at q=0 (ligament at lower boundary — correct behavior)
- Solution in T09: use ROM center `(lower+upper)/2` as the "neutral" test pose → max|tau|=0.000 Nm

---

## 4. Validation Status

### `standard_human_model/validation/01_muscle_layer/` — COMPLETE ✅
```
python standard_human_model/validation/01_muscle_layer/run_validation.py
```

| ID    | Test                            | Result | Key Numbers |
|-------|---------------------------------|--------|-------------|
| T01   | Hill F-L Active (gaussian)      | ✅ PASS | peak @l_norm=1.0005, F=1.0000 |
| T02   | Hill F-V (Hill curve)           | ✅ PASS | f_FV(0)=1.000, max_ecc=1.653 |
| T03   | Hill F-L Passive (exponential)  | ✅ PASS | l_norm=1.6 → 2.400 (exact match) |
| T04   | Activation linearity            | ✅ PASS | a=0.5/a=1.0 ratio=0.5000 |
| T05   | Moment arm heatmap              | ✅ PASS | visual, bi-articular confirmed |
| T05b  | Bi-articular coupling           | ✅ PASS | hamstrings→[L_Hip,L_Knee], gastrocnemius→[L_Knee,L_Ankle] |
| T06   | L/R symmetry                    | ✅ PASS | max error = 0.00e+00 Nm |
| T07   | Ligament soft-limit             | ✅ PASS | sign/shape correct |
| T08   | Stretch reflex (healthy/spastic)| ✅ PASS | Spastic>Healthy, v=0→reflex=0 |
| T09   | Full pipeline zero input        | ✅ PASS | ROM center: max|tau|=0.000 Nm |

Korean font: NotoSansCJK-Regular.ttc, set via `matplotlib.rcParams`.

### `standard_human_model/isaacgym_validation/` — OLD, partially verified
- `demo_knee_pendulum.py`: 3 humanoids (Healthy/Spastic/Flaccid), knee pendulum test
  - Ran headless successfully (previous session)
  - Results: Spastic final=82°, Healthy final=74°, Flaccid final=23° (5-second run)
  - DURATION recently changed to 8.0s but **8-second run not yet confirmed in new session**
  - `smpl_humanoid_fixed.xml`: freejoint removed, `fix_base_link=True`

---

## 5. Next Steps (Priority Order)

### IMMEDIATE — Step 2: `02_isaacgym_integration` validation
Create `standard_human_model/validation/02_isaacgym_integration/` with:

**What to validate:**
1. **Bio-torque injection test**: Does `gym.set_dof_actuation_force_tensor()` actually move the joint?
   - Apply known torque (e.g., 50 Nm to L_Knee) → observe DOF position change after N steps
   - Pass criterion: joint moves in expected direction

2. **Bio-torque vs gravity balance**: At 80° flexion, gravity pulls knee to extension. Bio-torque from spastic profile should resist this. Measure equilibrium angle.

3. **Profile differentiation in physics**: Run 3 profiles headless, record final angles, verify Spastic>Healthy>Flaccid ordering.

4. **Torque sign convention**: Positive bio-torque → joint flexes (angle increases). Verify via 1-step simulation.

**Structure to follow** (same as 01_muscle_layer):
```
02_isaacgym_integration/
├── run_validation.py   ← Tests I01~I04, headless IsaacGym
├── README.md
└── results/            ← PNG plots
```

### SHORT TERM — Parameter tuning
- Fix `l_opt` unit mismatch: change from 1.0 (dimensionless) to actual fiber length (meters)
  - quadriceps: l_opt ≈ 0.08m
  - hamstrings: l_opt ≈ 0.10m
  - gastrocnemius: l_opt ≈ 0.05m
  - soleus: l_opt ≈ 0.04m
- Knee neutral angle: redesign Exp2 using 70° instead of 0° as midpoint
- Patient profile ligament differentiation: stroke k_lig=200 vs healthy=50 vs SCI=5

### MEDIUM TERM — Step 3: `03_visualization`
- IsaacGym viewer: 3 profiles side by side, knee pendulum (live visual)
- Add on-screen text overlay: current bio-torque value, patient profile name
- GRF extraction: `gym.acquire_net_contact_force_tensor()` → CSV → compare with force plate shape

### LONG TERM
- `blend_alpha > 0`: mix bio-torque into VIC walking controller
- Isaac Lab migration (PhysX 5, closed kinematic chain for Exo)
- EMG synthesis from `activation_dynamics.a(t)`

---

## 6. Team Structure Plan (drafted 2026-03-18)
See: `01_research_docs/260318_framework_roadmap_team_structure.md`

4 tracks:
- **Track A**: Human-Robot Interface (Isaac Lab migration, Exo attachment)
- **Track B**: Patient population modeling (per-pathology YAML tuning)
- **Track C**: Synthetic data generator (EMG, GRF, IMU)
- **Track D**: Validation & real-data comparison (all-hands)

Key rule: `standard_human_model/core/` changes require **team PR review**.
Patient YAML in `profiles/` can be changed freely by each track member.

---

## 7. Key Commands

```bash
conda activate phc
cd /home/gunhee/workspace/PHC

# Run muscle layer validation (01)
python standard_human_model/validation/01_muscle_layer/run_validation.py

# Run IsaacGym knee pendulum demo (headless)
python standard_human_model/isaacgym_validation/demo_knee_pendulum.py --headless --pipeline cpu

# Run IsaacGym knee pendulum demo (with viewer)
python standard_human_model/isaacgym_validation/demo_knee_pendulum.py --pipeline cpu

# Run VIC training (main research)
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --headless --num_envs 512

# Run VIC evaluation
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display
```

---

## 8. Critical Code Locations

| What | Where | Line/Note |
|------|-------|-----------|
| Main torque pipeline | `standard_human_model/core/human_body.py:200` | `compute_torques()` |
| R matrix build | `standard_human_model/core/moment_arm.py:67` | `__init__` loop |
| SMPL DOF range dict | `standard_human_model/core/skeleton.py:61` | `JOINT_DOF_RANGE` |
| Patient profile YAML load | `standard_human_model/core/human_body.py:80` | `from_config()` |
| IsaacGym MSK task | `phc/env/tasks/humanoid_im_vic_msk.py` | unverified, blend_alpha=0.0 |
| IsaacGym pendulum demo | `standard_human_model/isaacgym_validation/demo_knee_pendulum.py` | working, 3 profiles |
| Fixed MJCF (no freejoint) | `standard_human_model/isaacgym_validation/smpl_humanoid_fixed.xml` | required for demo |
| Validation T01~T09 | `standard_human_model/validation/01_muscle_layer/run_validation.py` | 10/10 PASS |

---

## 9. Known Issues / Gotchas

| Issue | Cause | Status |
|-------|-------|--------|
| `l_opt=1.0` unit mismatch | dimensionless vs l_slack in meters | **Pending fix** — muscles at 30% optimal |
| Passive force never generated | l_norm stays < 1.0 always | Consequence of above |
| Knee q=0 = lower limit | SMPL convention: full extension = 0 | By design, use ROM center for neutral tests |
| IsaacGym import before torch | Required for CUDA initialization | Already handled in demo_knee_pendulum.py |
| Korean font in plots | Default matplotlib font missing glyphs | Fixed: NotoSansCJK via rcParams |
| `body._f_max` Long tensor | Multiplying float scale to Long tensor | Fixed: `.float()` cast in demo script |
| NaN with freejoint | freejoint + fix_base_link conflict | Fixed: use smpl_humanoid_fixed.xml |

---

## 10. File Naming Conventions (from CLAUDE.md)
- Docs: `YYMMDD_topic_name.extension`, Korean content
- `01_research_docs/`: experiment setup, implementation, strategy
- `02_research_dev/`: training results, analysis
- `04_Context_Bridge/`: handoff docs between sessions
- Validation scripts: `standard_human_model/validation/XX_name/run_validation.py`
- Results: auto-saved to `results/` subfolder as PNG

---

## 11. What Was Done This Session (2026-03-18)

1. **Team roadmap document** (`01_research_docs/260318_framework_roadmap_team_structure.md`): 4-track team structure, synthetic data roadmap (EMG/GRF feasibility), maintenance principles

2. **Comprehensive integration doc** (`01_research_docs/260318_standard_human_model_isaacgym_integration.md`): Full technical writeup of standard_human_model pipeline + IsaacGym integration + 5 validation experiments

3. **Validation folder created** (`standard_human_model/validation/01_muscle_layer/`):
   - `run_validation.py`: 9 tests covering every sub-module
   - `README.md`: test descriptions + interpretation guide
   - All 10 PASS, Korean font working

4. **Pushed** branch `human_model_v1`, commits `1ed0232` and `7699215`

**Did NOT do**: Run 8-second version of demo_knee_pendulum.py, Step 2 (IsaacGym integration validation), l_opt parameter fix

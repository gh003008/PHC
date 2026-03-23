# CALM Framework Team Structure and Research Strategy

Created: 2026-03-19
Team size: 4–5 (including PI)
Translated from Korean original: `260319_CALM_팀구성_연구전략.md`

---

## 1. Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    CALM Framework Stack                        │
│                                                             │
│  [Layer 4] Higher-level Controller (RL / CPG / Impedance)    │  ← Role D
│  [Layer 3] Human-Robot Integrated Sim (Isaac Lab)            │  ← Role C
│  [Layer 2] Patient Parameterization (Real-to-Sim Pipeline)    │  ← Role B
│  [Layer 1] Musculoskeletal Core (Hill + Reflex + Ligament)   │  ← Role A
│  [Layer 0] Human Body Model (SMPL/URDF + Exo Assets)        │  ← Role A+C
└─────────────────────────────────────────────────────────────┘
```

Each layer has clear I/O interfaces, enabling independent development.

---

## 2. Role Definitions (4 members)

### Role A — Musculoskeletal Core (Biomechanics Lead)
**Owner**: PI (current researcher, VIC + PHC background)

**Tasks**:
- `core/muscle_model.py` enhancement: l_opt fix, Achilles elastic tendon (SE)
- `core/moment_arm.py` enhancement: R(q) angle dependency (polynomial fitting)
- Framework-wide interface design and maintenance (API contract documentation)
- Validation Level 3–4 design and oversight
- PR review for all other Roles

**Deliverables**: All `core/` modules, `config/` YAML schema, technical docs & API contracts

### Role B — Clinical Parameterization
**Owner**: Rehab engineering / biomechanics background (1 person)

**Tasks**:
- Real-to-Sim identification pipeline (4 steps: passive dynamics → max force → activation → reflex)
- Patient profile YAML expansion and validation (stroke, Parkinson, SCI, elderly, weakness)
- Clinical measurement protocol documentation
- Literature-based parameter sourcing (Rajagopal 2016, Thelen 2003, etc.)

**Deliverables**: `tools/param_identification/`, `profiles/` YAML files, clinical measurement guide

### Role C — Robot Integration
**Owner**: Robotics / URDF experience (1 person)

**Tasks**:
- Exoskeleton URDF/MJCF design (knee/ankle orthosis first)
- IsaacGym → Isaac Lab (PhysX 5) migration (closed kinematic chain support)
- Human-robot attachment modeling (strap constraints, contact surfaces)
- Interaction force sensor simulation (virtual load cell)

**Deliverables**: `assets/exo/` URDF, `envs/human_robot_env.py`, robot integration guide

### Role D — Control & Learning
**Owner**: RL / control theory background (1 person)

**Tasks**:
- Gait imitation + CALM muscle layer combined RL training
- CPG (Central Pattern Generator) based gait controller
- Cross-patient assistive strategy comparison experiments
- Sim-to-Real gap analysis

**Deliverables**: `controllers/`, experiment configs, training result analysis docs

### Role E (5th member) — Validation & Visualization
**Owner**: Data analysis or early-stage graduate student

**Tasks**: L3–L5 validation scripts, gait analysis visualization, patient profile comparison plots.

---

## 3. Technical Interface Contracts

### Interface 1: CALM Core API (Role A → All)
```python
def compute_torques(
    dof_pos: Tensor,          # (num_envs, 69)
    dof_vel: Tensor,          # (num_envs, 69)
    descending_cmd: Tensor,   # (num_envs, num_muscles) [0, 1]
    dt: float,
    contact_forces: Tensor = None,  # (num_envs, 4)
) -> Tensor:                  # (num_envs, 69) joint torques
```

Role B swaps YAML parameters only. Role D outputs `descending_cmd`. Role C injects torques via `set_dof_actuation_force_tensor()`.

### Interface 2: Patient Profile Schema (Role B → Role A)
```yaml
# profiles/[condition]/[name].yaml — frozen schema
name: str
patient_type: str
muscle_params:
  [muscle_name]: {f_max, l_opt, v_max, ...}
reflex_params:
  default: {stretch_gain, stretch_threshold, gto_gain, gto_threshold, reciprocal_gain}
ligament_params:
  {k_lig, alpha, damping, soft_limit_margin}
```

### Interface 3: Human-Robot Env API (Role C → Role D)
```python
def step(action: Tensor) -> Tuple[Tensor, Tensor, Tensor, dict]:
    # obs, reward, done, info
    # obs includes dof_pos, dof_vel, bio_torque, exo_torque, contact_force
```

---

## 4. Integration Strategy

### Branch Strategy
```
main          — integrated, always working
develop       — integration in progress
feature/roleA-*  — Role A feature branches
feature/roleB-*  — Role B feature branches
...
```

### Weekly Sync: Monday 30min. Each role reports: completed, blocked, next week plan.

---

## 5. Milestone Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| Phase 1 (Month 1–2) | l_opt fix + elastic tendon + L1 tests | Core model accuracy |
| Phase 2 (Month 2–3) | Exo URDF + Isaac Lab migration start | Robot integration foundation |
| Phase 3 (Month 3–4) | L4 validation + Real-to-Sim pipeline | Data fidelity proof |
| Phase 4 (Month 4–6) | RL controller + cross-patient experiments | Paper A submission |

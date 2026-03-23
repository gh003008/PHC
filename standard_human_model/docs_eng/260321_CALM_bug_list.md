# CALM Framework Bug List

Created: 2026-03-21 (Fri)
Translated from Korean original: `260321_CALM_버그목록.md`

---

## BUG-01 — l_opt Unit Bug (Severity: Critical)

**Location**: `config/healthy_baseline.yaml` (all muscles)

**Problem**:
```yaml
# Current (incorrect)
quadriceps_L:
  l_opt: 1.0   # Dimensionless

# Internal calculation (muscle_model.py)
l_norm = muscle_length / l_opt
# muscle_length is computed in meters by moment_arm.py (~0.3m)
# → l_norm ≈ 0.3 / 1.0 = 0.3  ← 30% of optimal length
```

**Impact**: `f_FL(l_norm)` peaks at `l_norm=1.0`. At `l_norm=0.3`, the Gaussian returns near zero → **active muscle force is only ~30% of maximum**. Total gait torques are severely underestimated.

**Fix**:
```yaml
# Replace with Rajagopal 2016 Table 2 values
quadriceps_L:   l_opt: 0.099   # m
hamstrings_L:   l_opt: 0.109   # m
soleus_L:       l_opt: 0.044   # m
gastrocnemius_L: l_opt: 0.051  # m
tibialis_ant_L: l_opt: 0.068   # m
hip_flexors_L:  l_opt: 0.100   # m
gluteus_max_L:  l_opt: 0.142   # m
# Same values for right side
```

**Acceptance criterion**: Soleus `l_norm` ∈ [0.9, 1.05] at mid-stance

---

## BUG-02 — No Angle Dependency in R Matrix (Severity: High)

**Location**: `core/moment_arm.py`

**Problem**:
```python
# Current: R is a constant tensor (num_muscles, num_dofs)
self.R = torch.tensor(...)  # Set once at initialization

# Reality: moment arm varies up to ±30% with joint angle
# e.g., hamstrings: knee 0° → r=3.5cm, knee 90° → r=5.5cm
```

**Impact**: Up to 30% torque prediction error at extreme joint angles. Particularly significant for stair climbing and sit-to-stand motions with large knee flexion.

**Fix direction**: Apply polynomial fitting
```python
# r_i(q_j) = a0 + a1*q_j + a2*q_j^2
# Coefficients extracted from OpenSim gait2392
def _compute_R(self, dof_pos: Tensor) -> Tensor: ...
```

**Note**: Within flat-ground walking range (±30°), error is ~10%, so this is lower priority than BUG-01.

---

## BUG-03 — Elastic Tendon Not Implemented (Severity: High)

**Location**: `core/muscle_model.py`

**Problem**: `compute_force()` assumes rigid tendon. The Achilles tendon (soleus, gastrocnemius) stores and returns 35–40% of walking energy as an elastic spring, but this is completely ignored.

**Impact**: Terminal stance push-off power is underestimated. Winter 2009 ankle A2 power peak is 3.5 W/kg, which cannot be reproduced in simulation.

**Fix direction**: Add elastic tendon for `soleus` and `gastrocnemius` only
```python
def compute_tendon_force(self, l_tendon): ...   # Zajac 1989
def compute_force_elastic_tendon(self, activation, l_mtu, v_mtu): ...
# Newton-Raphson iteration for l_ce convergence (max 10 iterations)
```

**Acceptance criterion**: Achilles tendon energy storage 30–40 J/cycle during walking (Ker 1987)

---

## BUG-04 — No Stretch Reflex Delay (Severity: Medium)

**Location**: `core/reflex_controller.py`

**Problem**:
```python
# Current: muscle stretch detection → immediate reflex activation (delay=0)
a_stretch = stretch_gain * max(0, v_muscle - threshold)

# Reality: spinal reflex arc delay 25–60ms
# delay_steps = round(0.040 / dt)  # 40ms @ dt=1/120s → 5 steps
```

**Impact**: Clonus onset timing is inaccurate in spastic patient simulation. Negligible effect on normal walking, but causes errors in L3 validation (clonus onset test).

**Fix direction**:
```python
# Add muscle velocity history buffer
self.velocity_buffer = deque(maxlen=delay_steps)
v_delayed = self.velocity_buffer[0]  # Use value from delay_steps ago
```

---

## Summary Table

| ID | Location | Impact | Fix Difficulty | Priority |
|----|----------|--------|---------------|----------|
| BUG-01 | `healthy_baseline.yaml` | 70% active force loss | Easy (YAML edit) | **1st** |
| BUG-02 | `moment_arm.py` | ±30% torque error | Medium (OpenSim data needed) | 3rd |
| BUG-03 | `muscle_model.py` | Push-off power underestimated | Hard (NR iteration impl) | 2nd |
| BUG-04 | `reflex_controller.py` | Reflex timing inaccurate | Easy (buffer addition) | 4th |

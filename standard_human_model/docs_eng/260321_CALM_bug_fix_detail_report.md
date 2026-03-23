# CALM Framework Bug Fix Detail Report

Created: 2026-03-21
Original bug list: `260321_CALM_버그목록.md`
Translated from Korean original: `260321_CALM_버그수정_상세보고서.md`

---

## Fix Summary

| BUG | File | Key Change | Additional Fixes |
|-----|------|-----------|-----------------|
| BUG-01 | `healthy_baseline.yaml`, `muscle_definitions.yaml`, `muscle_model.py` | l_opt/l_tendon_slack replaced with measured values (m) + fiber length calculation completely rewritten | Damping unit bug fixed, v_norm calculation fixed |
| BUG-02 | `moment_arm.py` | Polynomial R(q) = a0 + a1*q + a2*q^2 added | API change: `forces_to_torques(F, dof_pos)` |
| BUG-03 | `muscle_model.py` | Elastic tendon Newton-Raphson implemented | `elastic_tendon_indices` option added |
| BUG-04 | `reflex_controller.py` | Default delay 1→5, circular buffer logic fixed | Buffer size off-by-one bug also fixed |

---

## BUG-01 — l_opt Unit Bug (Critical)

### Diagnosis

A unit mismatch cascading through the entire pipeline:

```
moment_arm.py                    muscle_model.py (before fix)
───────────────                  ──────────────────────
l_slack = 0.25~0.35 (m)         l_opt = 1.0 (dimensionless)
R @ q ≈ 0.02 (m)
                                 l_norm = muscle_length / l_opt
muscle_length                           = 0.30 / 1.0
  = l_slack - R@q                       = 0.30   ← 30% of optimal!
  ≈ 0.30 (m)
                                 f_FL(0.30) ≈ 0.25  ← near bottom of Gaussian
```

Active muscle force was only ~25% of maximum — a critical bug.
Additional unit mismatches existed in v_norm and damping.

### Fix Details

#### (1) `config/healthy_baseline.yaml` — Parameter values replaced with measured data

Based on Rajagopal 2016 Table 2. All 20 muscles (L+R) updated.

```yaml
# Before                          After
l_opt: 1.0          →              l_opt: 0.044           # m (soleus example)
l_tendon_slack: 1.0  →              l_tendon_slack: 0.260   # m
```

Full muscle l_opt values:

| Muscle Group | l_opt (m) | l_tendon_slack (m) | Source |
|-------------|-----------|-------------------|--------|
| hip_flexors | 0.100 | 0.133 | iliopsoas representative |
| gluteus_max | 0.142 | 0.125 | |
| hip_abductors | 0.054 | 0.065 | glut_med representative |
| hip_adductors | 0.138 | 0.110 | add_long representative |
| quadriceps | 0.099 | 0.136 | vasti representative |
| rectus_femoris | 0.076 | 0.346 | bi-articular |
| hamstrings | 0.109 | 0.326 | bicep_fem_lh representative |
| gastrocnemius | 0.051 | 0.408 | medial head |
| soleus | 0.044 | 0.260 | |
| tibialis_ant | 0.068 | 0.223 | |

#### (2) `config/muscle_definitions.yaml` — l_slack recalculated

l_slack is the total MTU length at joint neutral (0 degrees). Recalculated:

```
l_slack = l_opt * cos(pennation) + l_tendon_slack
```

Example (soleus):
```
l_slack = 0.044 * cos(0.44) + 0.260 = 0.044 * 0.905 + 0.260 = 0.300 m
```

#### (3) `core/muscle_model.py` — Fiber length/velocity calculation completely rewritten

**Before (incorrect):**
```python
l_norm = muscle_length / self.l_opt            # MTU length divided directly (unit mismatch)
v_norm = muscle_velocity / self.v_max          # Unit mismatch
F_damping = self.damping * self.f_max * muscle_velocity  # Raw m/s used
```

**After:**
```python
cos_penn = torch.cos(self.pennation)

# 1. Fiber length
l_ce = (muscle_length - self.l_tendon_slack) / cos_penn
l_ce = torch.clamp(l_ce, min=1e-4)
l_norm = l_ce / self.l_opt

# 2. Fiber velocity
v_ce = muscle_velocity / cos_penn
v_norm = v_ce / (self.v_max * self.l_opt)  # v_max in l_opt/s → absolute = v_max * l_opt

# 3. Damping uses normalized velocity
F_damping = self.damping * self.f_max * v_norm
```

**Post-fix data flow (soleus example):**
```
moment_arm.py:
  l_mtu = l_slack - R@q = 0.300 - (-0.05 * q_ankle) ≈ 0.300 m (neutral)

muscle_model.py:
  l_ce = (0.300 - 0.260) / cos(0.44) = 0.040 / 0.905 = 0.044 m
  l_norm = 0.044 / 0.044 = 1.0  ← Optimal length! f_FL(1.0) = 1.0

  v_mtu ≈ 0.03 m/s (typical during gait)
  v_ce = 0.03 / 0.905 = 0.033 m/s
  v_norm = 0.033 / (6.0 * 0.044) = 0.033 / 0.264 = 0.125  ← Reasonable range
```

---

## BUG-02 — No Angle Dependency in R Matrix (High)

### Diagnosis

Moment arm varies with joint angle. E.g., hamstrings knee moment arm: 3.5cm at 0°, 5.5cm at 90°. Constant R cannot capture this variation, causing up to 30% torque error at extreme angles.

### Fix: Polynomial moment arm added to `core/moment_arm.py`

```python
_POLY_COEFFS = {
    (muscle_name, joint_name, axis_idx): (a0, a1, a2),
    ...
}

class MomentArmMatrix:
    def _compute_R(self, dof_pos):
        R = self.R_const.expand(num_envs, ...).clone()
        q = dof_pos[:, poly_dof_indices]
        r_poly = a0 + a1*q + a2*q**2
        R[:, muscle_idx, dof_idx] = r_poly
        return R  # (num_envs, num_muscles, num_dofs)
```

#### Applied polynomial coefficients (18 entries)

| Muscle | Joint | a0 | a1 | a2 | Note |
|--------|-------|-----|-----|-----|------|
| hamstrings | Hip | -0.060 | -0.008 | +0.002 | Moment arm decreases with flexion |
| hamstrings | Knee | +0.030 | +0.015 | -0.005 | Moment arm increases with flexion |
| quadriceps | Knee | -0.040 | -0.012 | +0.004 | Patella anterior shift effect |
| rectus_femoris | Knee | -0.040 | -0.012 | +0.004 | Same as quadriceps |
| gastrocnemius | Knee | +0.020 | +0.008 | -0.003 | |
| gastrocnemius | Ankle | -0.050 | +0.005 | +0.002 | Decreases with dorsiflexion |
| soleus | Ankle | -0.050 | +0.005 | +0.002 | Similar to gastrocnemius |
| tibialis_ant | Ankle | +0.030 | -0.003 | -0.001 | |

L/R symmetric, so 18 entries = 9 pairs.

#### API change

```python
# Before
tau = moment_arm.forces_to_torques(F_muscle)

# After (dof_pos argument added)
tau = moment_arm.forces_to_torques(F_muscle, dof_pos)
# dof_pos=None uses constant R (backward compatible)
```

---

## BUG-03 — Elastic Tendon Not Implemented (High)

### Diagnosis

Achilles tendon (soleus, gastrocnemius) has tendon/fiber ratio of 5–9. 35–40% of walking energy comes from tendon elastic energy storage/return. Rigid tendon assumption completely ignores this, underestimating push-off power.

### Fix: Elastic tendon option added to `core/muscle_model.py`

#### Newton-Raphson fiber length convergence

```python
def _compute_fiber_state_elastic(self, activation, muscle_length, muscle_velocity):
    cos_penn = torch.cos(self.pennation)
    l_ce = (muscle_length - self.l_tendon_slack) / cos_penn  # initial guess

    for _ in range(max_iter):   # max_iter = 10
        l_tendon = muscle_length - l_ce * cos_penn
        l_tendon_norm = l_tendon / self.l_tendon_slack
        f_t = k_tendon * max(0, l_tendon_norm - 1.0)**2
        f_m = activation * f_FL(l_ce/l_opt) * cos_penn + f_PE(l_ce/l_opt)
        residual = f_t - f_m
        if |residual| < 1e-3: break
        # Newton step with analytical Jacobian
        l_ce -= residual / (df_t - df_m)
```

#### Usage

```python
model = HillMuscleModel(
    num_muscles=20, num_envs=512,
    elastic_tendon_indices=[7, 8, 17, 18],  # gastroc_L, soleus_L, gastroc_R, soleus_R
)
```

Muscles without elastic tendon designation use rigid tendon as before (mixed mode).

---

## BUG-04 — No Stretch Reflex Delay (Medium)

### Diagnosis — Two bugs:

**(a) Default value problem:**
```python
# Before: reflex_delay_steps = 1  → 8ms delay (actual: 25-60ms)
# After:  reflex_delay_steps = 5  → 5 * 8.3ms ≈ 41.7ms
```

**(b) Circular buffer off-by-one:**
```python
# Before: buffer size = delay_steps = 1
# write[0] then read[(0+1)%1] = read[0] ← just wrote! Effective delay = 0

# After: buffer size = delay_steps + 1
# Separate write/read slots, correct circular indexing
delayed_idx = (self._buffer_idx - self.reflex_delay_steps) % self._buf_size
```

---

## Additional Fixes Found During BUG-01

### Damping unit bug
```python
# Before: raw MTU velocity (m/s) used directly → non-physical
F_damping = self.damping * self.f_max * muscle_velocity

# After: normalized fiber velocity used → physically consistent
F_damping = self.damping * self.f_max * v_norm
```

---

## Modified Files

| File | Change Type | Related BUG |
|------|-----------|------------|
| `config/healthy_baseline.yaml` | l_opt, l_tendon_slack values replaced | BUG-01 |
| `config/muscle_definitions.yaml` | l_slack values recalculated | BUG-01 |
| `core/muscle_model.py` | Complete rewrite (fiber state calc, elastic tendon) | BUG-01, BUG-03 |
| `core/moment_arm.py` | Polynomial R(q) added, _compute_R() method | BUG-02 |
| `core/reflex_controller.py` | Delay default, buffer size/read logic | BUG-04 |

---

## Verification (Future)

| BUG | Acceptance Criteria | Method |
|-----|-------------------|--------|
| BUG-01 | Soleus l_norm ∈ [0.9, 1.05] at mid-stance | `compute_force_components()` l_norm check |
| BUG-02 | Hamstrings moment arm ≈ 5.5cm at 90° knee flexion | `_compute_R(dof_pos)` output inspection |
| BUG-03 | Achilles tendon energy storage 30–40 J/cycle during gait | Elastic tendon ON/OFF comparison sim |
| BUG-04 | Spastic clonus onset ≈ 40ms delay | Spastic profile stretch reflex sim |

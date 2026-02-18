# PHC Human Model Analysis for Wearable Robot Integration
Date: 2026-02-17
Author: Antigravity

## 1. Overview of Models in PHC
PHC (Perpetual Humanoid Control) provides several humanoid models primarily based on the SMPL (Skinned Multi-Person Linear) model family, as well as some humanoid robot models.

### Available Models:
1.  **SMPL**: Standard human model. Most widely used in PHC. Represents body shape and pose with 24 joints (72 parameters).
2.  **SMPL-H / SMPL-X**: Extensions of SMPL.
    *   **SMPL-H**: Adds hand articulation (fingers).
    *   **SMPL-X**: Adds hands and expressive face parameters. (More complex, higher computational cost).
3.  **Unitree H1 / G1**: Humanoid robots, not biological humans. Good for robotic sim-to-real but less suitable for "human" biomechanics simulation for wearable robots.

### Recommendation for Wearable Robot Integration:
*   **Best Choice:** **SMPL** (or SMPL-H if hand interaction is crucial).
*   **Reason:** It is data-driven from thousands of human scans, ensuring high anthropomorphism in terms of kinematics and shape. It is the standard for human-robot interaction simulation in research.

---

## 2. Anthropomorphism Analysis
Analyzing how well the SMPL-based models in PHC mimic actual human characteristics.

### 2.1. Kinematics & Shape (Dimensions)
*   **Mechanism:** The `SMPL_Robot` class (in `smpl_sim.smpllib.smpl_local_robot`) generates a Mujoco XML model dynamically based on `gender` and `betas` (shape parameters).
*   **Accuracy:** Since it constructs the skeleton and mesh from the SMPL statistical model, the **limb lengths, joint locations, and overall body shape are highly realistic** and variable. You can simulate different subjects (tall, short, heavy, light) by changing the `betas` parameter.
*   **Joint Axes:** The joints act as spherical joints (modeled as 3 orthogonal hinge joints in Mujoco). The axes are determined by the SMPL kinematic tree. Note that the coordinate system conversion (from SMPL Y-up to Isaac/Mujoco Z-up) is handled internally.

### 2.2. Dynamics (Mass, Inertia, Density)
*   **Mass Distribution:**
    *   Mass is calculated based on the volume of the `Geom` (geometry primitives like capsules, spheres, boxes) approximating the body parts, multiplied by `density`.
    *   **Density:** The default density appears to be set around water density (~1000 kg/m^3).
    *   **Realism:** PHC includes flags like `real_weight`, `real_weight_porpotion_capsules` in `humanoid.py` to adjust mass distribution closer to real human data (e.g., proper mass fraction for torso vs. limbs).
*   **Inertia:** Calculated automatically by the physics engine (PhysX/Mujoco) based on the geometry and mass. This is generally accurate enough for simulation unless precise biomechanical tissue soft-body dynamics are needed.

### 2.3. Range of Motion (ROM)
*   **Definition:** ROM is defined in `update_joint_limits` function in `smpl_sim/smpllib/smpl_local_robot.py`.
*   **Analysis:**
    *   The limits are hard-coded (e.g., Knee X: -180~180 deg).
    *   **Critique:** Some limits seem **overly generous** (e.g., knee extension) compared to biological human limits to prevent simulation instability or to allow easier exploration during RL training.
    *   **Action:** For accurate wearable robot simulation, strictly clamping these limits to biological norms (e.g., Knee 0~140 deg) in the XML generation script is recommended.

---

## 3. Stiffness & Impedance Definition
Understanding how joint stiffness and damping are defined and controlled is critical for "Co-contraction" implementation.

### 3.1. Baseline Definition (Passive)
*   In the generated XML (`smpl_humanoid.xml`), passive `stiffness` and `damping` attributes for joints are often set to **0**.
*   This means the "body" itself has no passive elasticity.

### 3.2. Active Control (PD Controller)
*   PHC uses a **Proportional-Derivative (PD) Controller** to simulate muscle stiffness/damping.
*   **Location:** `phc/env/tasks/humanoid.py` -> `_compute_torques` or Isaac Gym's internal drive.
*   **Formula:** $\tau = K_p (q_{des} - q) - K_d (\dot{q})$
*   **Key Parameters:**
    *   `kp_scale`: Global scaling factor for stiffness.
    *   `kd_scale`: Global scaling factor for damping.
    *   `pd_scale`: Scaling factor based on the humanoid's total mass (heavier body = stiffer joints).
*   **Current State:** The stiffness ($K_p$) and damping ($K_d$) gains are **fixed** values (scaled by body mass) during the simulation.

### 3.3. Path to Co-contraction (Variable Impedance)
*   To implement co-contraction (simulating simultaneous activation of agonist/antagonist muscles), you need to **modulate Stiffness ($K_p$) and Damping ($K_d$) dynamically**.
*   **Action Plan:**
    1.  Expand the Action Space to include $\Delta K_p$ and $\Delta K_d$ (or a co-contraction ratio).
    2.  Modify `_action_to_pd_targets` or `pre_physics_step` in `humanoid.py` to update the PD gains at every step based on the policy output.

---

## 4. Integration with Wearable Robots
*   **Feasibility:** High.
*   **Method:**
    *   The SMPL model generation is procedural (`smpl_local_robot.py`). You can inject code to **append additional bodies/geometries (the robot links)** to the SMPL kinematic tree (e.g., attach an exoskeleton link to the `L_Thigh` body/bone).
    *   Alternatively, attach the robot using a "Weld Joint" or "Fixed Joint" in the MJCF/URDF file after generation.
*   **Consideration:** Adding a wearable robot changes the total mass and inertia. The `pd_scale` logic in PHC might need adjustment to account for the added robot mass ensuring the controller remains stable.

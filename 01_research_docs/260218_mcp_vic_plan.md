# MCP-based Variable Impedance Control (VIC) Implementation Plan
Date: 2026-02-18
Author: Antigravity

## 1. Project Goal
Develop a **biologically plausible humanoid controller** that modulates joint stiffness (impedance) dynamically, using **SMPL** model within the **PHC (Perpetual Humanoid Control)** framework. The controller will be based on **MCP (Multi-Control Policy)** architecture to handle various motions robustly.

## 2. Methodology

### 2.1. Model & Control Architecture
*   **Model**: SMPL Humanoid (Standard PHC model). *MyoBody integration is deferred.*
*   **Controller**: **MCP (Multi-Control Policy)** using `HumanoidImMCP` task.
*   **Core Mechanism**: **Variable Impedance Control (VIC)** via Co-contraction Factor (CCF).
    *   The policy outputs both kinematic targets ($q_{target}$) and stiffness modulation factors ($\beta$).
    *   Stiffness $K_p$ and Damping $K_d$ are updated real-time based on $\beta$.

### 2.2. Action Space Expansion
*   **Original Action**: $A_t = \{ q_{target} \} \in \mathbb{R}^{n_{dof}}$ (Size: ~69)
*   **Expanded Action**: $A_t = \{ q_{target}, \beta \} \in \mathbb{R}^{2 \cdot n_{dof}}$ (Size: ~138)
    *   $\beta \in [-1, 1]$ corresponds to a stiffness scaling factor (e.g., $0.5\times$ to $2.0\times$).

### 2.3. Curriculum Learning Strategy
To ensure stability and efficient learning on a single GPU:
1.  **Reduced Dataset**: Use a subset (~1/10) of the full AMASS dataset, focusing on locomotion (walking, running, turning).
2.  **Stage 1: Kinematic Warm-up (Freeze Stiffness)**
    *   Train/Fine-tune the policy with `learn_stiffness = False`.
    *   Action output $\beta$ is ignored (masked to 0).
    *   The agent learns to track the motions with *fixed* canonical stiffness.
3.  **Stage 2: Stiffness Learning (Unfreeze)**
    *   Enable `learn_stiffness = True`.
    *   Action output $\beta$ actively modulates $K_p, K_d$.
    *   **Reward Signal**: Introduce `Metabolic Cost` (penalize high stiffness) to encourage relaxation when possible.
    $$ r_{metabolic} = - w_{stiff} \| K_p(t) \|^2 $$

### 2.4. Verification
*   **Pre-check**: Ensure the baseline MCP policy can track standard locomotion reference motions.
*   **Post-check**: Visualize stiffness profiles during gait cycle. (Expectation: High stiffness at heel-strike, low during swing).

---

## 3. Implementation Steps

### Step 1: Environment Modification (`humanoid.py`, `humanoid_im_mcp.py`)
*   [ ] Add `has_variable_stiffness` flag to Config.
*   [ ] Modify `_setup_character_props` to expand action space dimension ($2 \times$ DoF) if flag is True.
*   [ ] Update `pre_physics_step` to parse `ccf` from actions and apply to `dof_properties`.

### Step 2: Reference Motion Subset Creation
*   [ ] Create a script to filter and save a smaller `.pkl` dataset from `amass_isaac_standing_upright_slim.pkl` or similar, containing only essential locomotion clips.

### Step 3: Config & Training Setup
*   [ ] Create `phc/data/cfg/env/env_im_mcp_vic.yaml`.
*   [ ] Create `phc/data/cfg/learning/im_mcp_vic.yaml`.
*   [ ] Implement Curriculum flags in the yaml (or logical switches in code).

### Step 4: Training & Analysis
*   [ ] Run training (Stage 1 -> Stage 2).
*   [ ] Analyze tensorboard logs for stiffness values and tracking error.

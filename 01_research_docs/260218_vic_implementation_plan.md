# VIC (Variable Impedance Control) Implementation Plan
Date: 2026-02-18
Author: Antigravity
Objective: Implement Variable Impedance Control (Co-contraction) on PHC SMPL Humanoid to achieve biomechanically realistic motion.

## 1. Objective & Motivation
*   **Goal**: Create a humanoid policy that creates motion not just by kinematic targets ($q_{target}$) but by modulating joint stiffness ($K_p$) and damping ($K_d$), mimicking human muscle co-contraction.
*   **Biomechanical Realism**: Humans increase joint stiffness during high-impact phases (e.g., heel strike) and relax during swing phases to conserve energy.
*   **Approach**: **SMPL + VIC** first, defer MyoBody to a later stage.

## 2. Technical Strategy

### 2.1. Model Architecture: PNN-based Curriculum Learning
Instead of training from scratch, we leverage existing robust locomotion policies.
*   **Step 1 (Source)**: Load a pre-trained `phc_x_pnn` or `phc_kp` policy (Fixed Gains).
*   **Step 2 (Target)**: Train a new PNN Column with expanded Action Space.
    *   The new column learns the **residual** or **modulation** of impedance.
    *   It utilizes lateral features from the "Source" column to maintain stable gait.

### 2.2. Action Space Expansion
The policy output ($A_t$) will be expanded.
*   **Original**: $A_t = \{ q_{target} \} \in \mathbb{R}^{n_{dof}}$
*   **New**: $A_t = \{ q_{target}, \beta \} \in \mathbb{R}^{2 \cdot n_{dof}}$
    *   $\beta \in [-1, 1]$: Stiffness Modulation Factor (Co-contraction Factor). (Or simpler: one scalar $\beta$ for the whole body, or per-limb. **Decision: Per-joint for max fidelity**).

### 2.3. Controller Logic (Physics Step)
Modify `phc/env/tasks/humanoid.py`:
$$ K_p(t) = K_{p, base} \cdot 2^{\beta_t} $$
$$ K_d(t) = K_{d, base} \cdot 2^{\beta_t} $$ (Scaling damping proportionally to maintain critical damping ratio)
*   **Range**: If $\beta \in [-1, 1]$, stiffness scales from $0.5 \times$ to $2.0 \times$.

### 2.4. Reward Engineering (Crucial)
To prevent the agent from just maximizing stiffness for stability (which is biologically unrealistic and energy-inefficient), we introduce a **Metabolic Cost Penalty**.
$$ r_{effort} = - w_{torque} \| \tau \|^2 - w_{stiffness} \| K_p(t) - K_{p, min} \|^2 $$
*   The agent must balance **Stability (Stiffness)** vs. **Efficiency (Relaxation)**.

---

## 3. Implementation Steps

### Step 1: Environment Modification (`humanoid.py`)
1.  Add `self.has_variable_stiffness = True` config flag.
2.  Modify `get_action_space` to double the dimension if flag is on.
3.  Modify `pre_physics_step` or `apply_action` to parse $\beta$ and update `dof_prop['stiffness']`.

### Step 2: Config Update
1.  Create `phc/data/cfg/env/env_im_vic.yaml`.
2.  Define `stiffness_lower_bound` and `stiffness_upper_bound`.

### Step 3: PNN Network Adjustment
1.  Ensure PNN builder handles action space change between columns (or simply retrain a single MCP policy with weights init if PNN is too rigid).
    *   *Refinement*: PNN usually requires fixed output size. **Alternative**: simpler *Fine-tuning*. Load weights of `q_target` part, initialize `beta` part to 0, and retrain.

### Step 4: Validation
1.  Monitor `stiffness` values in Tensorboard.
2.  Check if stiffness peaks at Heel Start/Toe Off (Human-like behavior).

## 4. Next Actions
1.  Create environment config `env_im_vic.yaml`.
2.  Modify `humanoid.py` to support variable stiffness.

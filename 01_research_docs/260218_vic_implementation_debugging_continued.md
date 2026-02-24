# VIC Implementation Debugging Continued
Date: 260218

## 1. Overview
This document logs the continuation of the VIC implementation debugging session on February 18, 2026. The primary focus is resolving the `FileNotFoundError` for the motion data and refactoring the class hierarchy.

## 2. Issue: Missing Motion Data
- **Problem**: The training command failed because `phc/data/amass/pkls/amass_isaac_standing_upright_slim.pkl` was not found.
- **Investigation**: 
    - Checked file system. Found `sample_data/amass_isaac_standing_upright_slim.pkl`.
    - Checked `phc/data/cfg/env/env_im_mcp_vic.yaml` and found `motion_file: ""` which likely caused the fallback to a default non-existent path.
- **Solution**: Updated `phc/data/cfg/env/env_im_mcp_vic.yaml` to point to `sample_data/amass_isaac_standing_upright_slim.pkl`.

## 3. Refactoring: VIC Class Hierarchy
- **Goal**: Make `HumanoidVIC` inherit from `HumanoidIm` and reduce code duplication in `HumanoidImMCPVIC`.
- **Changes**:
    1.  **`phc/env/tasks/humanoid_vic.py`**:
        -   Changed inheritance from `BaseTask` to `humanoid_im.HumanoidIm`.
        -   Removed duplicated methods identical to `HumanoidIm`.
        -   Implemented VIC logic in `_setup_character_props`, `_physics_step`, etc.
    2.  **`phc/env/tasks/humanoid_im_mcp_vic.py`**:
        -   Changed inheritance to `class HumanoidImMCPVIC(humanoid_im_mcp.HumanoidImMCP, humanoid_vic.HumanoidVIC)`.
        -   Removed duplicated VIC logic methods, relying on `HumanoidVIC` via MRO.
        -   Verified that `HumanoidImMCP`'s logic (MCP/PNN) and `HumanoidVIC`'s logic (Stiffness Control) coexist correctly.

## 4. Verification
-   Ran the training command: `python phc/run.py --task HumanoidImMCPVIC --cfg_env phc/data/cfg/env/env_im_mcp_vic.yaml --cfg_train phc/data/cfg/learning/im_mcp_vic.yaml --headless --no_log`.
-   **Result**: 
    -   Training optimized successfully.
    -   Log confirmed VIC initialization: `VIC Setup: has_variable_stiffness=True, _dof_action_size=138, learn_stiffness=False`.
    -   Motion data loaded correctly.

## 5. Next Steps (Stage 2)
-   Wait for Stage 1 (Kinematics) to converge.
-   Enable stiffness learning: Set `learn_stiffness: True`.
-   Enable metabolic cost: Set `metabolic_cost_w > 0`.

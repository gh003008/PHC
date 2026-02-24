
import torch
import phc.env.tasks.humanoid_im as humanoid_im
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from phc.utils.flags import flags

class HumanoidVIC(humanoid_im.HumanoidIm):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Initialize Base Class
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        
        # VIC-specific configs are loaded in load_common_humanoid_configs override
        self.metabolic_cost_w = cfg["env"].get("metabolic_cost_w", 0.0)
        
        return

    def load_common_humanoid_configs(self, cfg):
        super().load_common_humanoid_configs(cfg)
        
        # VIC: Variable Impedance Control
        self.has_variable_stiffness = cfg["env"].get("has_variable_stiffness", False)
        self.stiffness_lower = cfg["env"].get("stiffness_lower", 0.5)
        self.stiffness_upper = cfg["env"].get("stiffness_upper", 2.0)
        self.learn_stiffness = cfg["env"].get("learn_stiffness", True)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        # VIC: Ensure p_gains exist (Crucial for SMPL which might not set them in base class)
        if not hasattr(self, 'p_gains'):
            kp = 300.0
            kd = 30.0
            self.p_gains = torch.ones(self._dof_size, device=self.device) * kp
            self.d_gains = torch.ones(self._dof_size, device=self.device) * kd
            print(f"VIC: Initialized default p_gains ({kp}) and d_gains ({kd}) for {self._dof_size} DoFs")

        if not hasattr(self, 'default_dof_pos'):
            self.default_dof_pos = torch.zeros(self._dof_size, device=self.device)
            print("VIC: Initialized default_dof_pos to zeros")
            
        self._dof_action_size = self._dof_size 
        
        if self.has_variable_stiffness:
            self._dof_action_size *= 2
            
            # If we are NOT using PNN/MCP (which sets num_prim), we might need to update _num_actions
            # HumanoidIm sets _num_actions = _dof_size * (something)
            # Here we ensure _num_actions reflects the policy output size if it's a direct policy
            if not hasattr(self, 'num_prim'): 
                 self._num_actions = self._dof_action_size
            
        print(f"VIC Setup: has_variable_stiffness={self.has_variable_stiffness}, _dof_action_size={self._dof_action_size}, learn_stiffness={self.learn_stiffness}")
        return

    def _setup_tensors(self):
        super()._setup_tensors()
        # Save base gains for VIC scaling
        self.p_gains_base = self.p_gains.clone()
        self.d_gains_base = self.d_gains.clone()
        
    def _physics_step(self):
        # VIC Logic: Modulate Gains based on Actions BEFORE physics step
        full_actions_backup = None
        
        if self.has_variable_stiffness:
             full_actions = self.actions
             num_dof = self.num_dof
             
             # Check if actions include stiffness (size check)
             # Note: self._dof_action_size is dof*2
             if full_actions.shape[-1] == self._dof_action_size:
                 kin_actions = full_actions[:, :num_dof]
                 stiff_actions = full_actions[:, num_dof:]
                 
                 if self.learn_stiffness:
                     # Map stiffness action [-1, 1] to [stiffness_lower, stiffness_upper] scale
                     scale = self.stiffness_lower + (stiff_actions + 1.0) / 2.0 * (self.stiffness_upper - self.stiffness_lower)
                     self.p_gains = self.p_gains_base * scale
                     self.d_gains = self.d_gains_base * scale
                 else:
                     # Use base gains
                     self.p_gains = self.p_gains_base
                     self.d_gains = self.d_gains_base
                 
                 # Prepare actions for torque computation (only kinematics)
                 full_actions_backup = self.actions
                 self.actions = kin_actions
             else:
                 # Fallback: validation or non-VIC actions
                 if self.learn_stiffness: 
                     print("Warning: learn_stiffness=True but action shape mismatch.")
                 self.p_gains = self.p_gains_base
                 self.d_gains = self.d_gains_base
        
        # Call Parent Physics Step (computes torques and simulates)
        super()._physics_step()
        
        # Restore actions for logging/buffers
        if full_actions_backup is not None:
            self.actions = full_actions_backup

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        
        # VIC: Add Metabolic Cost Reward
        if self.has_variable_stiffness and self.learn_stiffness and self.metabolic_cost_w > 0:
            num_dof = self.num_dof
            if actions.shape[-1] > num_dof:
                stiff_actions = actions[:, num_dof:]
                # Minimize stiffness (L2 penalty)
                metabolic_reward = - self.metabolic_cost_w * torch.mean(stiff_actions**2, dim=-1)
                
                self.rew_buf += metabolic_reward
                # Append to reward_raw for logging (assuming last slot or append new?)
                # HumanoidIm has reward_raw size 4 or 5. 
                # We should be careful about shape mismatch if we just concat.
                # Use extras or extend reward_raw if supported.
                # For now, just adding to rew_buf is critical.
        return
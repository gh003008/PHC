import time
import torch
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp
import numpy as np

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from phc.learning.mlp import MLP
from isaacgym import gymtorch

class HumanoidImMCPVIC(humanoid_im_mcp.HumanoidImMCP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.metabolic_cost_w = cfg["env"].get("metabolic_cost_w", 0.0)
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
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

        # HumanoidImMCP sets self._num_actions = self.num_prim (for Policy)
        # We need to define the Low-Level Action Size for VIC
        # self.num_dof is set by Humanoid class based on the model loaded.
        
        self._dof_action_size = self._dof_size 
        
        if self.has_variable_stiffness:
            self._dof_action_size *= 2
            
        print(f"VIC Setup: has_variable_stiffness={self.has_variable_stiffness}, _dof_action_size={self._dof_action_size}, learn_stiffness={self.learn_stiffness}")
        return

    def _setup_tensors(self):
        super()._setup_tensors()
        # Save base gains for VIC
        self.p_gains_base = self.p_gains.clone()
        self.d_gains_base = self.d_gains.clone()

    def _physics_step(self):
        self.render(i = 0) # Render outside of the step function.
        
        # VIC Logic: Modulate Gains based on Actions
        if self.has_variable_stiffness:
             # self.actions contains the output from the PNN (Low Level Actions).
             # Its shape should be (num_envs, _dof_action_size).
             # Let's verify shape compatibility.
             
             num_dof = self.num_dof
             kin_actions = self.actions[:, :num_dof]
             
             if self.learn_stiffness:
                 stiff_actions = self.actions[:, num_dof:]
                 scale = self.stiffness_lower + (stiff_actions + 1.0) / 2.0 * (self.stiffness_upper - self.stiffness_lower)
                 self.p_gains = self.p_gains_base * scale
                 self.d_gains = self.d_gains_base * scale
             else:
                 # If not learning stiffness, use Base Gains (Scale = 1.0)
                 self.p_gains = self.p_gains_base
                 self.d_gains = self.d_gains_base
             
             # Use only kinematic actions for torque computation
             actions_for_torque = kin_actions
        else:
             actions_for_torque = self.actions

        for i in range(self.control_freq_inv):
            if not self.paused and self.enable_viewer_sync:
                if self.control_mode in ["pd"]:
                    # Compute torques using kinematic actions
                    torques = self._compute_torques(actions_for_torque)
                    
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                else:
                    self.gym.simulate(self.sim)
        return

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        
        # VIC: Add Metabolic Cost Reward
        if self.has_variable_stiffness and self.learn_stiffness and self.metabolic_cost_w > 0:
            num_dof = self.num_dof
            stiff_actions = actions[:, num_dof:]
            # Minimize stiffness (L2 penalty)
            metabolic_reward = - self.metabolic_cost_w * torch.mean(stiff_actions**2, dim=-1)
            
            self.rew_buf += metabolic_reward
            self.reward_raw = torch.cat([self.reward_raw, metabolic_reward[:, None]], dim=1)
        return

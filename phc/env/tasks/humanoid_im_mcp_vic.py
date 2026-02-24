import time
import torch
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp
import phc.env.tasks.humanoid_vic as humanoid_vic # Import HumanoidVIC

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from phc.learning.mlp import MLP
from isaacgym import gymtorch

# Inherit from both. HumanoidImMCP is primary (for PNN/MCP logic), HumanoidVIC provides VIC logic.
class HumanoidImMCPVIC(humanoid_im_mcp.HumanoidImMCP, humanoid_vic.HumanoidVIC):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # We can just call HumanoidImMCP init, or super().
        # HumanoidImMCP init calls super().__init__, which follows MRO.
        # MRO: HumanoidImMCPVIC, HumanoidImMCP, HumanoidVIC, HumanoidIm, ...
        # HumanoidImMCP.__init__ sets up PNN/MCP stuff.
        # HumanoidVIC init sets up VIC configs (metabolic_cost_w).
        # We need to ensure both inits run or duplicated logic is handled.
        
        # HumanoidImMCP.__init__ calls super().__init__.
        # super(HumanoidImMCP, self) is HumanoidVIC.
        # So HumanoidVIC.__init__ will be called.
        # HumanoidVIC.__init__ calls super().__init__ (HumanoidIm).
        # So the chain is preserved.
        
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        return

    # No need to duplicate methods if they are identical or handled by MRO.
    # _setup_character_props: HumanoidImMCP calls super(), which hits HumanoidVIC, which hits HumanoidIm.
    # load_common_humanoid_configs: Same.
    # _physics_step: HumanoidImMCP doesn't implement it. HumanoidVIC implements it. So HumanoidVIC's version is used.
    # _compute_reward: HumanoidImMCP doesn't implement it (uses HumanoidIm's). HumanoidVIC implements it. HumanoidVIC's version is used.
    
    # We might need to ensure HumanoidImMCPVIC uses correct configs if they differ.
    # Current HumanoidImMCPVIC.py had only a few methods.
    # I removed them all because the logic is now in HumanoidVIC or handled by inheritance.


import torch, joblib, numpy as np
from phc.utils.motion_lib_smpl import MotionLibSMPL, FixHeightMode
from smpl_sim.smpllib.smpl_parser import SMPL_Parser

data = joblib.load('sample_data/h5_motion_library.pkl')
first_key = [k for k in data.keys() if not k.startswith('_')][0]
clip = data[first_key]

smpl_parser = SMPL_Parser(model_path='data/smpl', gender='neutral')
mesh_parsers = {0: smpl_parser}

trans = clip['root_trans_offset'].clone().float()
pose_aa = clip['pose_aa'].clone() if isinstance(clip['pose_aa'], torch.Tensor) else torch.from_numpy(clip['pose_aa'])
pose_aa = pose_aa.float()

gender_beta = torch.zeros(17)

print("pose_aa shape:", pose_aa.shape)
print("trans shape:", trans.shape)
print("trans before fix (frame 0):", trans[0].numpy())

try:
    trans_fixed, diff_fix = MotionLibSMPL.fix_trans_height(
        pose_aa, trans, gender_beta, mesh_parsers, fix_height_mode=FixHeightMode.full_fix
    )
    print("diff_fix:", diff_fix.item() if hasattr(diff_fix, 'item') else diff_fix)
    print("trans after fix (frame 0):", trans_fixed[0].numpy())
    print("Pelvis height after fix:", trans_fixed[0, 2].item())
except Exception as e:
    print("ERROR:", e)
    import traceback; traceback.print_exc()

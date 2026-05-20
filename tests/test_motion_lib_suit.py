"""Smoke test for SuitMotionLib (run with python, not pytest)."""
import torch
from phc.utils.motion_lib_suit import SuitMotionLib

# Use smoke pkl (5 clips) since full lib may not exist yet
pkl_candidates = [
    "sample_data/suit_reference_lib_v1.pkl",
    "sample_data/suit_reference_lib_smoke.pkl",
]
import os
pkl = None
for p in pkl_candidates:
    if os.path.exists(p):
        pkl = p
        break
assert pkl is not None, f"No suit reference pkl found. Tried: {pkl_candidates}"
print(f"[test] using {pkl}")

lib = SuitMotionLib(pkl, device="cuda:0")
assert lib.num_motions > 0, "no motions loaded"
ids = lib.sample_motions(8)
assert ids.shape == (8,), f"ids shape {ids.shape}"
ts = lib.sample_time(ids)
assert ts.shape == (8,), f"ts shape {ts.shape}"
state = lib.get_motion_state(ids, ts)
assert state["suit_q"].shape == (8, 12), f"suit_q shape {state['suit_q'].shape}"
assert state["suit_qvel"].shape == (8, 12)
assert state["base_xyz"].shape == (8, 3)
assert state["base_yaw"].shape == (8,)
assert state["base_xyz_vel"].shape == (8, 3)
for k, v in state.items():
    assert torch.isfinite(v).all(), f"{k} has non-finite values"
print("[ok] all sampled state shapes correct, values finite")
print("    suit_q:       ", state["suit_q"].shape)
print("    suit_qvel:    ", state["suit_qvel"].shape)
print("    base_xyz:     ", state["base_xyz"].shape)
print("    base_yaw:     ", state["base_yaw"].shape)
print("    base_xyz_vel: ", state["base_xyz_vel"].shape)

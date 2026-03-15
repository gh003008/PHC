"""
Stage-II corrected pkl을 PHC AMASS 형식으로 변환하는 스크립트.
연구실 측정 데이터(fullpose + trans)를 MotionLibSMPL이 읽을 수 있는 형식으로 변환.

Usage:
    python convert_stageii_to_amass.py \
        --input sample_data/S009_level_08mps_trial_01_stageii_corrected.pkl \
        --output sample_data/amass_S009_08mps.pkl \
        --src_fps 100 --tgt_fps 30
"""
import argparse
import pickle
import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot


def convert_stageii_to_amass(input_path, output_path, src_fps=100, tgt_fps=30, motion_key=None):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    fullpose = data['fullpose']  # (N, 72)
    trans = data['trans']        # (N, 3)
    betas_raw = data.get('betas', np.zeros(16))

    # Downsample from src_fps to tgt_fps
    ratio = src_fps / tgt_fps
    indices = np.round(np.arange(0, len(fullpose), ratio)).astype(int)
    indices = indices[indices < len(fullpose)]

    fullpose_ds = fullpose[indices]  # (M, 72)
    trans_ds = trans[indices]        # (M, 3)

    print(f"Downsampled: {len(fullpose)} frames @ {src_fps}fps -> {len(fullpose_ds)} frames @ {tgt_fps}fps")
    print(f"Duration: {len(fullpose_ds) / tgt_fps:.1f}s")

    # pose_aa: axis-angle (M, 72) - same as fullpose for SMPL
    pose_aa = fullpose_ds.astype(np.float64)

    # root_trans_offset: torch tensor (M, 3)
    root_trans_offset = torch.from_numpy(trans_ds.astype(np.float64))

    # pose_quat_global: (M, 24, 4) - convert each joint's axis-angle to quaternion
    M = len(pose_aa)
    pose_aa_reshaped = pose_aa.reshape(M, 24, 3)
    pose_quat_global = np.zeros((M, 24, 4), dtype=np.float64)
    for j in range(24):
        rot = sRot.from_rotvec(pose_aa_reshaped[:, j, :])
        pose_quat_global[:, j, :] = rot.as_quat()  # (x, y, z, w) scipy convention

    # beta: first 16 dims
    beta = betas_raw[:16].astype(np.float32)

    # Build AMASS-format dict
    if motion_key is None:
        motion_key = input_path.split('/')[-1].replace('.pkl', '')

    amass_data = {
        f"0-{motion_key}": {
            "pose_aa": pose_aa,
            "pose_quat_global": pose_quat_global,
            "root_trans_offset": root_trans_offset,
            "trans_orig": trans_ds.astype(np.float32),
            "beta": beta,
            "gender": "neutral",
            "fps": tgt_fps,
        }
    }

    joblib.dump(amass_data, output_path)
    print(f"Saved: {output_path}")
    print(f"Motion key: 0-{motion_key}")
    print(f"Frames: {M}, Duration: {M / tgt_fps:.1f}s")

    # Verify
    loaded = joblib.load(output_path)
    k = list(loaded.keys())[0]
    print(f"\nVerification:")
    for kk, vv in loaded[k].items():
        if hasattr(vv, 'shape'):
            print(f"  {kk}: shape={vv.shape}")
        else:
            print(f"  {kk}: {vv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input stage-II pkl path")
    parser.add_argument("--output", required=True, help="Output AMASS-format pkl path")
    parser.add_argument("--src_fps", type=int, default=100, help="Source FPS")
    parser.add_argument("--tgt_fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--key", default=None, help="Motion key name")
    args = parser.parse_args()

    convert_stageii_to_amass(args.input, args.output, args.src_fps, args.tgt_fps, args.key)

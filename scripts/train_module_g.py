"""Offline supervised training for Module G — Nominal Gait Generator.

Stage A of the MPL roadmap (concept doc §3.3). Loads the H5 motion library,
derives command labels from motion itself, and trains a small GRU to predict
(q_ref, Kp_latent) from (state, command, reference window).

Usage:
    conda activate phc
    cd /home/exolab/Documents/GitHub/PHC
    python scripts/train_module_g.py \
        --library sample_data/h5_motion_library.pkl \
        --epochs 100 \
        --batch_size 32 \
        --out output/module_g.pth

This script has NO IsaacGym dependency and can run alongside Phase 0 RL
training (uses < 500 MB VRAM).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make phc importable when launched from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phc.learning.module_g import (
    GConfig,
    NominalGaitGenerator,
    build_dataset_from_library,
    g_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=str,
                   default="sample_data/h5_motion_library.pkl",
                   help="Path to the motion library pkl (joblib format).")
    p.add_argument("--out", type=str, default="output/module_g.pth",
                   help="Where to save the trained G checkpoint.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ref_window_K", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of clips held out for validation.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[train_module_g] loading library from {args.library}")
    library = joblib.load(args.library)
    print(f"[train_module_g] loaded {len(library)} clips")

    pose, vel, cmd, ref, tgt, keys = build_dataset_from_library(
        library, ref_window_K=args.ref_window_K
    )
    print(f"[train_module_g] dataset: pose={tuple(pose.shape)} vel={tuple(vel.shape)} "
          f"cmd={tuple(cmd.shape)} ref={tuple(ref.shape)} tgt={tuple(tgt.shape)}")

    # Held-out validation split (by clip, not by frame, to avoid leakage).
    N = pose.shape[0]
    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    print(f"[train_module_g] train={len(trn_idx)} val={len(val_idx)}")

    trn_ds = TensorDataset(
        pose[trn_idx], vel[trn_idx], cmd[trn_idx], ref[trn_idx], tgt[trn_idx]
    )
    val_ds = TensorDataset(
        pose[val_idx], vel[val_idx], cmd[val_idx], ref[val_idx], tgt[val_idx]
    )
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = GConfig(ref_window_K=args.ref_window_K)
    model = NominalGaitGenerator(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_module_g] model params = {n_params/1e6:.3f}M, device = {args.device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        trn_logs = {"loss/total": 0.0, "loss/recon": 0.0, "loss/smooth": 0.0, "loss/kp_range": 0.0}
        trn_count = 0
        for batch in trn_loader:
            p_b, v_b, c_b, r_b, y_b = [t.to(args.device, non_blocking=True) for t in batch]
            q_pred, kp_pred, _ = model(p_b, v_b, c_b, r_b)
            total, logs = g_loss(q_pred, kp_pred, y_b)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in logs.items():
                trn_logs[k] += v
            trn_count += 1
        for k in trn_logs:
            trn_logs[k] /= max(trn_count, 1)

        # Validation
        model.eval()
        val_logs = {"loss/total": 0.0, "loss/recon": 0.0, "loss/smooth": 0.0, "loss/kp_range": 0.0}
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                p_b, v_b, c_b, r_b, y_b = [t.to(args.device, non_blocking=True) for t in batch]
                q_pred, kp_pred, _ = model(p_b, v_b, c_b, r_b)
                _, logs = g_loss(q_pred, kp_pred, y_b)
                for k, v in logs.items():
                    val_logs[k] += v
                val_count += 1
        for k in val_logs:
            val_logs[k] /= max(val_count, 1)

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            dt = time.time() - t0
            print(
                f"[epoch {epoch:4d}/{args.epochs}] "
                f"trn recon={trn_logs['loss/recon']:.5f} smooth={trn_logs['loss/smooth']:.5f} "
                f"kp={trn_logs['loss/kp_range']:.5f} | "
                f"val recon={val_logs['loss/recon']:.5f} | elapsed={dt:.1f}s"
            )

        if val_logs["loss/recon"] < best_val:
            best_val = val_logs["loss/recon"]
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_recon": best_val,
                    "ref_window_K": args.ref_window_K,
                },
                args.out,
            )

    print(f"[train_module_g] done. best val recon = {best_val:.5f}")
    print(f"[train_module_g] checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()

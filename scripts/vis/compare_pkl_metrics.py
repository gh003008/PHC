"""Compare two SMPL motion-lib pkls: metrics plot + side-by-side skeleton mp4.

Usage:
    python scripts/vis/compare_pkl_metrics.py \\
        --pkl_a sample_data/foo.pkl --label_a "foo" \\
        --pkl_b sample_data/bar.pkl --label_b "bar" \\
        --out_prefix output/h5_visual_check/foo_vs_bar
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import joblib
from scipy.stats import pearsonr
from numpy.fft import rfft, rfftfreq
from easydict import EasyDict
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

# PHC augmented SMPL body indices
PELVIS, L_HIP, L_KNEE, L_ANKLE, L_TOE = 0, 1, 2, 3, 4
R_HIP, R_KNEE, R_ANKLE, R_TOE = 5, 6, 7, 8
HEAD, L_WRIST, R_WRIST = 13, 17, 22
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(12,13),
         (11,14),(14,15),(15,16),(16,17),(17,18),
         (11,19),(19,20),(20,21),(21,22),(22,23)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl_a", required=True)
    p.add_argument("--label_a", default="A")
    p.add_argument("--pkl_b", required=True)
    p.add_argument("--label_b", default="B")
    p.add_argument("--out_prefix", required=True,
                   help="Output path prefix; writes <prefix>_metrics.png and <prefix>_compare.mp4")
    p.add_argument("--t_start_a", type=float, default=10.0, help="Window start (s) for pkl_a")
    p.add_argument("--t_end_a", type=float, default=30.0)
    p.add_argument("--t_start_b", type=float, default=None, help="Window start for pkl_b; default = 0.5s into clip")
    p.add_argument("--t_end_b", type=float, default=None, help="Default = clip length - 0.5s")
    p.add_argument("--view", choices=["side", "front", "iso", "all"], default="side",
                   help="Render mode for compare mp4. 'side'=2D sagittal, 'front'=2D frontal, 'iso'=3D isometric, 'all'=emit all three.")
    p.add_argument("--iso_elev", type=float, default=15.0, help="3D camera elevation for --view iso/both.")
    p.add_argument("--iso_azim", type=float, default=-60.0, help="3D camera azimuth for --view iso/both.")
    return p.parse_args()


def _build_skeleton_tree():
    robot_cfg = {"mesh": False, "rel_joint_lm": False, "upright_start": True,
        "remove_toe": False, "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, "model": "smpl", "big_ankle": True,
        "freeze_hand": False, "box_body": True, "body_params": {}, "joint_params": {},
        "geom_params": {}, "actuator_params": {}}
    smpl = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gb = np.zeros(17)
    smpl.load_from_skeleton(betas=torch.from_numpy(gb[None, 1:]), gender=gb[0:1], objs_info=None)
    xml = "/tmp/smpl/compare_pkl_tree.xml"
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    smpl.write_xml(xml)
    return SkeletonTree.from_mjcf(xml)


def _load_single_clip(pkl_path, sk_tree, device):
    data = joblib.load(pkl_path)
    clip_key = list(data.keys())[0]
    tmp = f"/tmp/smpl/_cmp_{os.path.basename(pkl_path)}"
    joblib.dump({clip_key: data[clip_key]}, tmp)
    cfg = EasyDict({"motion_file": tmp, "device": device,
        "fix_height": FixHeightMode.full_fix, "min_length": -1, "max_length": -1,
        "im_eval": False, "multi_thread": False, "smpl_type": "smpl", "randomrize_heading": False})
    m = MotionLibSMPL(cfg)
    m.load_motions(skeleton_trees=[sk_tree], gender_betas=[torch.zeros(17)],
                   limb_weights=[np.zeros(10)], random_sample=False, start_idx=0)
    return m, float(m.get_motion_length(0).item()), clip_key


def _compute_metrics(m, motion_len, t_start, t_end, device, fps=30):
    if t_start is None: t_start = 0.5
    if t_end is None: t_end = motion_len - 0.5
    t_end = min(t_end, motion_len)
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end:.2f}s) must exceed t_start ({t_start:.2f}s) after clamping to motion_len ({motion_len:.2f}s)")
    n = max(2, int(round((t_end - t_start) * fps)))
    times = torch.linspace(t_start, t_end - 1e-4, n, device=device).float()
    res = m.get_motion_state(torch.zeros_like(times).long(), times)
    rb = res["rg_pos"].cpu().numpy()
    root = res["root_pos"].cpu().numpy()
    t_arr = np.linspace(t_start, t_end, n)
    vxy = np.gradient(root[:, :2], 1.0 / fps, axis=0)
    vmean = vxy.mean(axis=0); speed = np.linalg.norm(vmean)
    fwd = vmean / (speed + 1e-9)
    def fproj(idx): return (rb[:, idx, :2] - rb[:, PELVIS, :2]) @ fwd
    trunk = rb[:, HEAD] - rb[:, PELVIS]
    trunk_fwd = trunk[:, :2] @ fwd
    trunk_lean = np.degrees(np.arctan2(trunk_fwd, trunk[:, 2]))
    pelvis_z = rb[:, PELVIS, 2]
    freqs = rfftfreq(n, 1.0 / fps); power = np.abs(rfft(pelvis_z - pelvis_z.mean())) ** 2
    peak = np.argmax(power[1:]) + 1
    step_freq = freqs[peak]
    gait_period = 2.0 / step_freq if step_freq > 0 else float("nan")
    l_knee_f = fproj(L_KNEE); r_knee_f = fproj(R_KNEE)
    l_wrist_f = fproj(L_WRIST); r_wrist_f = fproj(R_WRIST)
    r_LKLW = pearsonr(l_knee_f, l_wrist_f)[0] if n > 2 else float("nan")
    r_LKRW = pearsonr(l_knee_f, r_wrist_f)[0] if n > 2 else float("nan")
    # Walk-aligned forward velocity (signed, per-frame): project root xy gradient onto fwd direction
    root_vx_signed = vxy @ fwd  # [n] m/s along walk-aligned fwd
    return dict(rb=rb, root=root, t_arr=t_arr, fwd=fwd, speed=speed,
                pelvis_z=pelvis_z, head_z=rb[:, HEAD, 2],
                l_toe_z=rb[:, L_TOE, 2], r_toe_z=rb[:, R_TOE, 2],
                l_ank_z=rb[:, L_ANKLE, 2], r_ank_z=rb[:, R_ANKLE, 2],
                trunk_lean=trunk_lean, gait_period=gait_period,
                l_knee_f=l_knee_f, r_knee_f=r_knee_f,
                l_wrist_f=l_wrist_f, r_wrist_f=r_wrist_f,
                r_LKLW=r_LKLW, r_LKRW=r_LKRW,
                root_vx=root_vx_signed)


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sk_tree = _build_skeleton_tree()
    m_a, L_a, _ = _load_single_clip(args.pkl_a, sk_tree, device)
    m_b, L_b, _ = _load_single_clip(args.pkl_b, sk_tree, device)
    A = _compute_metrics(m_a, L_a, args.t_start_a, args.t_end_a, device)
    B = _compute_metrics(m_b, L_b, args.t_start_b, args.t_end_b, device)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # 8-panel metrics plot (4 rows × 2 cols)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    def comp(ax, title, xa, ya, xb, yb, ylabel):
        ax.plot(xa - xa[0], ya, label=args.label_a, color="tab:orange", lw=1.3)
        ax.plot(xb - xb[0], yb, label=args.label_b, color="tab:blue", lw=1.3, alpha=0.85)
        ax.set_title(title); ax.set_xlabel("t [s]"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
    comp(axes[0, 0], "Pelvis Z", A["t_arr"], A["pelvis_z"], B["t_arr"], B["pelvis_z"], "z [m]")
    comp(axes[0, 1], "L toe Z", A["t_arr"], A["l_toe_z"], B["t_arr"], B["l_toe_z"], "z [m]")
    # Walk-aligned forward velocity — flat ripple = continuous walking; dips/spikes = stops/discontinuity
    vx_a_mean = float(np.mean(A["root_vx"])); vx_b_mean = float(np.mean(B["root_vx"]))
    comp(axes[1, 0], f"Root v_x walk-aligned (A mean {vx_a_mean:+.3f}, B mean {vx_b_mean:+.3f} m/s)",
         A["t_arr"], A["root_vx"], B["t_arr"], B["root_vx"], "v_x [m/s]")
    axes[1, 1].plot(A["t_arr"] - A["t_arr"][0], A["l_wrist_f"], label=f"{args.label_a} L wrist", color="tab:orange")
    axes[1, 1].plot(A["t_arr"] - A["t_arr"][0], A["r_wrist_f"], label=f"{args.label_a} R wrist", color="tab:orange", ls="--")
    axes[1, 1].plot(B["t_arr"] - B["t_arr"][0], B["l_wrist_f"], label=f"{args.label_b} L wrist", color="tab:blue")
    axes[1, 1].plot(B["t_arr"] - B["t_arr"][0], B["r_wrist_f"], label=f"{args.label_b} R wrist", color="tab:blue", ls="--")
    axes[1, 1].set_title(f"Wrist fwd (arm swing)  {args.label_a} LK↔LW={A['r_LKLW']:+.2f}  {args.label_b} LK↔LW={B['r_LKLW']:+.2f}")
    axes[1, 1].legend(fontsize=7, ncol=2); axes[1, 1].grid(alpha=0.3); axes[1, 1].set_xlabel("t [s]")
    labels = ["speed", "trunk lean", "gait period", "pelvis bob", "LK↔LW", "LK↔RW"]
    va = [A["speed"], A["trunk_lean"].mean(), A["gait_period"],
          A["pelvis_z"].max() - A["pelvis_z"].min(), A["r_LKLW"], A["r_LKRW"]]
    vb = [B["speed"], B["trunk_lean"].mean(), B["gait_period"],
          B["pelvis_z"].max() - B["pelvis_z"].min(), B["r_LKLW"], B["r_LKRW"]]
    x = np.arange(len(labels))
    axes[2, 0].bar(x - 0.2, va, 0.4, label=args.label_a, color="tab:orange")
    axes[2, 0].bar(x + 0.2, vb, 0.4, label=args.label_b, color="tab:blue")
    axes[2, 0].set_xticks(x); axes[2, 0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[2, 0].legend(); axes[2, 0].grid(alpha=0.3, axis="y")
    axes[2, 0].set_title("Summary")
    axes[2, 1].axis("off")
    axes[2, 1].text(0.01, 0.95, f"A = {args.pkl_a}\nB = {args.pkl_b}\nwindow A = [{args.t_start_a}, {args.t_end_a}]s",
                    fontsize=9, verticalalignment="top", family="monospace")
    # Trunk lean (restored)
    comp(axes[3, 0], f"Trunk lean sagittal (A mean {A['trunk_lean'].mean():+.1f}°, B mean {B['trunk_lean'].mean():+.1f}°)",
         A["t_arr"], A["trunk_lean"], B["t_arr"], B["trunk_lean"], "deg")
    axes[3, 1].axis("off")
    plt.tight_layout()
    out_png = f"{args.out_prefix}_metrics.png"
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}")

    # Side-by-side 4s mp4 @ 15 fps playback (30 fps sampling)
    fps_sim = 30; fps_play = 15
    render_dur = min(4.0, L_a - args.t_start_a - 0.1, L_b - (args.t_start_b or 0.5) - 0.1)
    if render_dur <= 0:
        raise ValueError(
            f"render_dur={render_dur:.2f}s is non-positive: check t_start_a={args.t_start_a} "
            f"(clip_a={L_a:.2f}s) and t_start_b={args.t_start_b} (clip_b={L_b:.2f}s)"
        )
    n_r = int(render_dur * fps_sim)
    def pf_pz(m, t0):
        times = torch.linspace(t0, t0 + render_dur, n_r, device=device).float()
        res = m.get_motion_state(torch.zeros_like(times).long(), times)
        rb = res["rg_pos"].cpu().numpy(); root = res["root_pos"].cpu().numpy()
        vxy = np.gradient(root[:, :2], axis=0); vmean = vxy.mean(axis=0)
        fwd = vmean / (np.linalg.norm(vmean) + 1e-9)
        anchor = root[0, :2].copy()
        pf = (rb[:, :, :2] - anchor[None, None, :]) @ fwd
        pz = rb[:, :, 2]
        return pf, pz
    pf_a, pz_a = pf_pz(m_a, args.t_start_a)
    pf_b, pz_b = pf_pz(m_b, args.t_start_b if args.t_start_b is not None else 0.5)
    range_a = [pf_a.min() - 0.5, pf_a.max() + 0.5]
    range_b = [pf_b.min() - 0.5, pf_b.max() + 0.5]

    # Walk-aligned 3D sampler: returns (forward, lateral, up) per joint per frame
    def fwd_lat_up(m, t0):
        times = torch.linspace(t0, t0 + render_dur, n_r, device=device).float()
        res = m.get_motion_state(torch.zeros_like(times).long(), times)
        rb = res["rg_pos"].cpu().numpy(); root = res["root_pos"].cpu().numpy()
        vxy = np.gradient(root[:, :2], axis=0); vmean = vxy.mean(axis=0)
        fwd = vmean / (np.linalg.norm(vmean) + 1e-9)
        right = np.array([fwd[1], -fwd[0]])           # walk-aligned lateral basis
        anchor = root[0, :2].copy()
        rel_xy = rb[:, :, :2] - anchor[None, None, :]
        pf = rel_xy @ fwd
        pl = rel_xy @ right
        pz = rb[:, :, 2]
        return pf, pl, pz

    if args.view in ("side", "all"):
        out_mp4 = f"{args.out_prefix}_compare.mp4"
        w = imageio.get_writer(out_mp4, fps=fps_play, codec="libx264", quality=8, macro_block_size=1)
        for i in range(n_r):
            fig, axes = plt.subplots(2, 1, figsize=(16, 9))
            for ax, pf, pz, rng, lab in [
                (axes[0], pf_a, pz_a, range_a, args.label_a),
                (axes[1], pf_b, pz_b, range_b, args.label_b),
            ]:
                ax.axhline(0, color="#888", lw=1.0, zorder=1)
                for a, b in BONES:
                    ax.plot([pf[i, a], pf[i, b]], [pz[i, a], pz[i, b]], lw=4.0, solid_capstyle="round", zorder=5)
                ax.scatter(pf[i], pz[i], c="black", s=22, zorder=6)
                ax.scatter([pf[i, L_TOE]], [pz[i, L_TOE]], c="blue", s=140, marker="^", zorder=7)
                ax.scatter([pf[i, R_TOE]], [pz[i, R_TOE]], c="red", s=140, marker="v", zorder=7)
                ax.scatter([pf[i, L_WRIST]], [pz[i, L_WRIST]], c="cyan", s=130, marker="o", edgecolors="k", zorder=7)
                ax.scatter([pf[i, R_WRIST]], [pz[i, R_WRIST]], c="orange", s=130, marker="o", edgecolors="k", zorder=7)
                ax.set_xlim(rng[0], rng[1]); ax.set_ylim(-0.05, 2.05); ax.set_aspect("equal")
                ax.grid(alpha=0.25); ax.set_title(f"{lab}  t={i/fps_sim:.2f}s", loc="left")
            axes[1].set_xlabel("forward [m] (walk-aligned)")
            plt.tight_layout()
            canvas = FigureCanvasAgg(fig); canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            w.append_data(buf); plt.close(fig)
        w.close()
        print(f"wrote {out_mp4}")

    if args.view in ("front", "all"):
        pfA, plA, pzA = fwd_lat_up(m_a, args.t_start_a)
        pfB, plB, pzB = fwd_lat_up(m_b, args.t_start_b if args.t_start_b is not None else 0.5)
        # Frontal view: lateral on x-axis, up on y-axis. Subject "facing" the camera (forward into screen).
        # Use shared lateral range so left/right scale matches between A and B.
        lat_lo_f = min(plA.min(), plB.min()) - 0.4
        lat_hi_f = max(plA.max(), plB.max()) + 0.4
        out_mp4_front = f"{args.out_prefix}_compare_front.mp4"
        wf = imageio.get_writer(out_mp4_front, fps=fps_play, codec="libx264", quality=8, macro_block_size=1)
        for i in range(n_r):
            fig, axes = plt.subplots(1, 2, figsize=(16, 9))
            for ax, pl, pz, lab in [
                (axes[0], plA, pzA, args.label_a),
                (axes[1], plB, pzB, args.label_b),
            ]:
                ax.axhline(0, color="#888", lw=1.0, zorder=1)
                for a, b in BONES:
                    ax.plot([pl[i, a], pl[i, b]], [pz[i, a], pz[i, b]], lw=4.0, solid_capstyle="round", zorder=5)
                ax.scatter(pl[i], pz[i], c="black", s=22, zorder=6)
                ax.scatter([pl[i, L_TOE]], [pz[i, L_TOE]], c="blue", s=140, marker="^", zorder=7)
                ax.scatter([pl[i, R_TOE]], [pz[i, R_TOE]], c="red", s=140, marker="v", zorder=7)
                ax.scatter([pl[i, L_WRIST]], [pz[i, L_WRIST]], c="cyan", s=130, marker="o", edgecolors="k", zorder=7)
                ax.scatter([pl[i, R_WRIST]], [pz[i, R_WRIST]], c="orange", s=130, marker="o", edgecolors="k", zorder=7)
                ax.scatter([pl[i, HEAD]], [pz[i, HEAD]], c="magenta", s=120, marker="*", zorder=7)
                ax.set_xlim(lat_lo_f, lat_hi_f); ax.set_ylim(-0.05, 2.05); ax.set_aspect("equal")
                ax.grid(alpha=0.25); ax.set_title(f"{lab}  t={i/fps_sim:.2f}s  (frontal: ←L  R→)", loc="left")
                ax.set_xlabel("lateral [m]"); ax.set_ylabel("up [m]")
            plt.tight_layout()
            canvas = FigureCanvasAgg(fig); canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            wf.append_data(buf); plt.close(fig)
        wf.close()
        print(f"wrote {out_mp4_front}")

    if args.view in ("iso", "all"):
        pfA, plA, pzA = fwd_lat_up(m_a, args.t_start_a)
        pfB, plB, pzB = fwd_lat_up(m_b, args.t_start_b if args.t_start_b is not None else 0.5)
        # Shared lateral range for both panels (fair visual comparison)
        lat_lo = min(plA.min(), plB.min()) - 0.3
        lat_hi = max(plA.max(), plB.max()) + 0.3
        rngA3d = [pfA.min() - 0.3, pfA.max() + 0.3]
        rngB3d = [pfB.min() - 0.3, pfB.max() + 0.3]
        out_mp4_3d = f"{args.out_prefix}_compare_3d.mp4"
        w3 = imageio.get_writer(out_mp4_3d, fps=fps_play, codec="libx264", quality=8, macro_block_size=1)
        for i in range(n_r):
            fig = plt.figure(figsize=(16, 9))
            for panel, (pf, pl, pz, rng, lab) in enumerate([
                (pfA, plA, pzA, rngA3d, args.label_a),
                (pfB, plB, pzB, rngB3d, args.label_b),
            ]):
                ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
                for a, b in BONES:
                    ax.plot([pf[i, a], pf[i, b]], [pl[i, a], pl[i, b]], [pz[i, a], pz[i, b]],
                            lw=3.0, color="tab:gray")
                ax.scatter(pf[i], pl[i], pz[i], c="black", s=14, depthshade=False)
                ax.scatter([pf[i, L_TOE]], [pl[i, L_TOE]], [pz[i, L_TOE]], c="blue", s=80, marker="^", depthshade=False)
                ax.scatter([pf[i, R_TOE]], [pl[i, R_TOE]], [pz[i, R_TOE]], c="red", s=80, marker="v", depthshade=False)
                ax.scatter([pf[i, L_WRIST]], [pl[i, L_WRIST]], [pz[i, L_WRIST]], c="cyan", s=70, edgecolors="k", depthshade=False)
                ax.scatter([pf[i, R_WRIST]], [pl[i, R_WRIST]], [pz[i, R_WRIST]], c="orange", s=70, edgecolors="k", depthshade=False)
                ax.scatter([pf[i, HEAD]], [pl[i, HEAD]], [pz[i, HEAD]], c="magenta", s=90, marker="*", depthshade=False)
                # Ground plane reference
                gx = np.linspace(rng[0], rng[1], 2); gy = np.linspace(lat_lo, lat_hi, 2)
                GX, GY = np.meshgrid(gx, gy)
                ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.1, color="gray", edgecolor="none")
                ax.set_xlim(rng[0], rng[1]); ax.set_ylim(lat_lo, lat_hi); ax.set_zlim(0, 2.0)
                ax.set_xlabel("forward [m]"); ax.set_ylabel("lateral [m]"); ax.set_zlabel("up [m]")
                ax.view_init(elev=args.iso_elev, azim=args.iso_azim)
                try: ax.set_box_aspect((rng[1] - rng[0], lat_hi - lat_lo, 2.0))
                except Exception: pass
                ax.set_title(f"{lab}  t={i/fps_sim:.2f}s", loc="left", fontsize=11)
            plt.tight_layout()
            canvas = FigureCanvasAgg(fig); canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            w3.append_data(buf); plt.close(fig)
        w3.close()
        print(f"wrote {out_mp4_3d}")


if __name__ == "__main__":
    main()

"""
05_deep_validation / run_validation.py
=======================================
CALM 프레임워크 심화 검증 (IsaacGym 불필요 — pure Python).

검증 항목:
  V5-1  Hill Force Surface     — 2D (l × v) 근육 힘 지형도 + 성분 분해
  V5-2  Activation Bode        — 주파수 응답: 저역통과 필터 특성 확인
  V5-3  Reflex Bifurcation     — 경직 게인 스윕 → 클로누스 분기점 발견
  V5-4  Ligament Stiffness     — 관절 각도 vs 인대/관절낭 저항 토크 (비선형)
  V5-5  Reflex Pathway Isolation — Stretch / GTO / Reciprocal 반사 경로 개별 검증
  V5-6  Biarticular Coupling   — 이관절근 모멘트암 행렬 시각화 + 커플링 정량

출력:
  results/deep_metrics.json
  results/V5_*.png (6장)

사용법:
  conda activate phc
  cd /home/gunhee/workspace/PHC
  python standard_human_model/validation/05_deep_validation/run_validation.py
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.muscle_model import HillMuscleModel, MuscleParams
from standard_human_model.core.activation_dynamics import ActivationDynamics
from standard_human_model.core.reflex_controller import ReflexController, ReflexParams
from standard_human_model.core.ligament_model import LigamentModel
from standard_human_model.core.moment_arm import MomentArmMatrix
from standard_human_model.core.skeleton import (
    NUM_DOFS, JOINT_NAMES, JOINT_DOF_RANGE, get_joint_limits_tensors,
)

CONFIG_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
MUSCLE_DEF  = os.path.join(CONFIG_DIR, "muscle_definitions.yaml")
HEALTHY_CFG = os.path.join(CONFIG_DIR, "healthy_baseline.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT         = 1.0 / 60.0
L_KNEE_IDX = JOINT_DOF_RANGE["L_Knee"][0]
L_HIP_IDX  = JOINT_DOF_RANGE["L_Hip"][0]
L_ANK_IDX  = JOINT_DOF_RANGE["L_Ankle"][0]

MUSCLE_NAMES_FULL = [
    "hip_flexors_L", "gluteus_max_L", "hip_abductors_L", "hip_adductors_L",
    "quadriceps_L",  "rectus_femoris_L", "hamstrings_L",
    "gastrocnemius_L", "soleus_L", "tibialis_ant_L",
    "hip_flexors_R", "gluteus_max_R", "hip_abductors_R", "hip_adductors_R",
    "quadriceps_R",  "rectus_femoris_R", "hamstrings_R",
    "gastrocnemius_R", "soleus_R", "tibialis_ant_R",
]


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_human_body(mods=None, num_envs=1):
    body = HumanBody.from_config(MUSCLE_DEF, HEALTHY_CFG, num_envs=num_envs, device="cpu")
    if mods is None:
        return body
    if "reflex" in mods:
        r = mods["reflex"]
        if "stretch_gain"      in r: body.reflex.stretch_gain[:]     = r["stretch_gain"]
        if "stretch_threshold" in r: body.reflex.stretch_threshold[:] = r["stretch_threshold"]
    if "ligament" in mods:
        lg = mods["ligament"]
        if "k_lig"   in lg: body.ligament.k_lig[:]   = lg["k_lig"]
        if "damping" in lg: body.ligament.damping[:] = lg["damping"]
        if "alpha"   in lg: body.ligament.alpha[:]   = lg["alpha"]
    if "muscle" in mods:
        m = mods["muscle"]
        if "f_max_scale"   in m:
            body.muscle_model.f_max *= m["f_max_scale"]
            body._f_max = body._f_max.float() * m["f_max_scale"]
        if "damping_scale" in m:
            body.muscle_model.damping *= m["damping_scale"]
    return body


def run_pendulum(body, steps=None, initial_angle=1.4, initial_vel=-5.0):
    """Euler 진자 시뮬레이션 (무릎 진자, IsaacGym 불필요)."""
    if steps is None:
        steps = int(10.0 / DT)
    I_knee, m_leg, g, L_com = 0.5, 4.0, 9.81, 0.25

    pos = torch.zeros(1, NUM_DOFS)
    vel = torch.zeros(1, NUM_DOFS)
    body.activation_dyn.reset()
    body.reflex.reset()

    q, dq = float(initial_angle), float(initial_vel)
    angles  = np.zeros(steps)
    vels    = np.zeros(steps)
    torques = np.zeros(steps)
    acts    = np.zeros((steps, body.num_muscles))

    for step in range(steps):
        pos[0, L_KNEE_IDX] = q
        vel[0, L_KNEE_IDX] = dq
        cmd     = torch.zeros(1, body.num_muscles)
        tau_bio = body.compute_torques(pos, vel, cmd, dt=DT)
        tau_k   = float(tau_bio[0, L_KNEE_IDX].item())
        tau_grav = -m_leg * g * L_com * np.sin(q)
        ddq = (tau_k + tau_grav) / I_knee
        dq  = dq + ddq * DT
        q   = q  + dq  * DT
        angles[step]  = np.degrees(q)
        vels[step]    = dq
        torques[step] = tau_k
        acts[step]    = body.activation_dyn.get_activation()[0].cpu().numpy()

    return angles, vels, torques, acts


def fft_dominant_freq(signal, dt, fmin=0.5, fmax=25.0):
    n     = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(np.fft.rfft(signal - signal.mean()))
    mask  = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() == 0:
        return 0.0
    return float(freqs[mask][np.argmax(power[mask])])


def to_python(obj):
    if isinstance(obj, (np.bool_,)):      return bool(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, dict):             return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):    return [to_python(v) for v in obj]
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# V5-1  Hill Force Surface (2D l × v)
# ══════════════════════════════════════════════════════════════════════════════

def test_force_surface():
    print("\n[V5-1] Hill Force Surface — 2D (l × v)")

    N_L, N_V = 70, 70
    l_range  = np.linspace(0.3, 1.8, N_L)   # l/l_opt (l_opt=1.0 → same as absolute)
    v_range  = np.linspace(-1.0, 0.8, N_V)  # v/v_max

    model = HillMuscleModel(num_muscles=1, num_envs=N_L * N_V, device="cpu")
    params = [MuscleParams(
        name="test", f_max=1000.0, l_opt=1.0, v_max=10.0, pennation=0.0,
        tau_act=0.015, tau_deact=0.060,
        k_pe=4.0, epsilon_0=0.6, l_tendon_slack=1.0, k_tendon=35.0, damping=0.0,
    )]
    model.set_params(params)

    L_grid, V_grid = np.meshgrid(l_range, v_range, indexing="ij")  # (N_L, N_V)
    L_flat = torch.tensor(L_grid.flatten(), dtype=torch.float32).reshape(-1, 1)
    V_flat = torch.tensor(V_grid.flatten() * 10.0, dtype=torch.float32).reshape(-1, 1)  # actual vel

    c_full = model.compute_force_components(torch.ones(N_L * N_V, 1), L_flat, V_flat)
    c_zero = model.compute_force_components(torch.zeros(N_L * N_V, 1), L_flat, V_flat)

    F_active  = c_full["F_active"].squeeze(1).numpy().reshape(N_L, N_V)
    F_passive = c_zero["F_passive"].squeeze(1).numpy().reshape(N_L, N_V)
    F_total   = F_active + F_passive

    # ── 검증 ────────────────────────────────────────────────────────────────
    v_iso  = np.argmin(np.abs(v_range - 0.0))
    v_ecc  = np.argmin(np.abs(v_range - 0.3))
    v_vmax = np.argmin(np.abs(v_range - (-0.95)))
    l_opt  = np.argmin(np.abs(l_range - 1.0))

    F_iso  = float(F_active[l_opt, v_iso])
    F_ecc  = float(F_active[l_opt, v_ecc])
    F_vmax = float(F_active[l_opt, v_vmax])

    ecc_pass  = bool(F_ecc  > F_iso)                    # eccentric > isometric
    vmax_pass = bool(F_vmax < 0.15 * F_iso)             # near v_max → near zero
    peak_l    = float(l_range[np.argmax(F_active[:, v_iso])])
    fl_pass   = bool(abs(peak_l - 1.0) < 0.05)

    passed = bool(ecc_pass and vmax_pass and fl_pass)
    print(f"  F_isometric={F_iso:.1f}N  F_eccentric={F_ecc:.1f}N  "
          f"F_near_vmax={F_vmax:.1f}N")
    print(f"  Eccentric > Isometric: {'PASS' if ecc_pass else 'FAIL'}")
    print(f"  Near v_max → ~0: {'PASS' if vmax_pass else 'FAIL'}")
    print(f"  F-L peak at l/l_opt={peak_l:.3f}: {'PASS' if fl_pass else 'FAIL'}")

    # ── 그림 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    fig.suptitle("V5-1: Hill Muscle Force Surface", fontsize=14, y=0.98)

    # ① 2D Active Force Surface
    ax1 = fig.add_subplot(gs[0, :2])
    im  = ax1.contourf(v_range, l_range, F_active / 1000, levels=20, cmap="hot")
    ax1.contour(v_range, l_range, F_active / 1000, levels=10, colors="white",
                linewidths=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax1, label="Active Force (kN)")
    ax1.axvline(0, color="cyan", lw=1.5, ls="--", label="Isometric (v=0)")
    ax1.axhline(1.0, color="lime", lw=1.5, ls="--", label="Optimal length")
    ax1.set_xlabel("Normalized Velocity (v/v_max)  [neg=concentric]")
    ax1.set_ylabel("Normalized Length (l/l_opt)")
    ax1.set_title("Active Force Surface F(l, v)")
    ax1.text(0.35, 0.95, "ECCENTRIC\n(force > isometric)", transform=ax1.transAxes,
             color="cyan", fontsize=9, va="top", bbox=dict(fc="k", alpha=0.3))
    ax1.text(0.05, 0.95, "CONCENTRIC\n(force decreases)", transform=ax1.transAxes,
             color="yellow", fontsize=9, va="top", bbox=dict(fc="k", alpha=0.3))
    ax1.legend(fontsize=8, loc="lower right")

    # ② 2D Total Force Surface
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.contourf(v_range, l_range, F_passive / 1000, levels=15, cmap="Blues")
    plt.colorbar(im2, ax=ax2, label="Passive Force (kN)")
    ax2.axhline(1.0, color="red", lw=1.5, ls="--", label="l_opt")
    ax2.set_xlabel("v/v_max")
    ax2.set_ylabel("l/l_opt")
    ax2.set_title("Passive Force — F(l) only\n(velocity-independent)")
    ax2.legend(fontsize=8)

    # ③ F-L cross-section at v=0
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(l_range, F_active[:, v_iso] / 1000, "r-",  lw=2.5, label="Active (a=1.0)")
    ax3.plot(l_range, F_passive[:, v_iso] / 1000, "b--", lw=2, label="Passive")
    ax3.plot(l_range, F_total[:, v_iso] / 1000, "k-",  lw=2, label="Total")
    ax3.axvline(1.0, color="gray", ls=":", lw=1, label="l_opt")
    ax3.fill_between(l_range, F_passive[:, v_iso] / 1000, 0,
                     alpha=0.2, color="blue", label="Passive area")
    ax3.set_xlabel("Normalized Length (l/l_opt)")
    ax3.set_ylabel("Force (kN)")
    ax3.set_title("F-L Curve (v=0, isometric)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ④ F-V cross-section at l=l_opt
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(v_range, F_active[l_opt, :] / 1000, "r-", lw=2.5, label="Active (l=l_opt)")
    ax4.axvline(0, color="gray", ls=":", lw=1, label="Isometric")
    ax4.axhline(F_iso / 1000, color="gray", ls="--", lw=0.8, alpha=0.5,
                label=f"F_iso={F_iso:.0f}N")
    ax4.fill_between(v_range[v_range > 0],
                     F_active[l_opt, v_range > 0] / 1000, F_iso / 1000,
                     alpha=0.25, color="green", label="Eccentric gain")
    ax4.fill_between(v_range[v_range < 0],
                     F_active[l_opt, v_range < 0] / 1000, F_iso / 1000,
                     alpha=0.25, color="orange", label="Concentric deficit")
    ax4.set_xlabel("Normalized Velocity (v/v_max)")
    ax4.set_ylabel("Force (kN)")
    ax4.set_title("F-V Curve (l=l_opt)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ⑤ Force components at l_opt vs velocity
    ax5 = fig.add_subplot(gs[1, 2])
    fv_vals = model.force_velocity(
        torch.tensor(v_range, dtype=torch.float32).unsqueeze(1)
    ).squeeze(1).numpy()
    fl_vals = model.force_length_active(
        torch.tensor(l_range, dtype=torch.float32).unsqueeze(1)
    ).squeeze(1).numpy()
    ax5.plot(v_range, fv_vals, "g-", lw=2.5, label="f_FV (velocity factor)")
    ax5.plot(l_range, fl_vals, "m-", lw=2.5, label="f_FL (length factor)")
    ax5.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax5.axhline(0.707, color="red", ls=":", lw=1, alpha=0.7, label="-3dB (0.707)")
    ax5.set_xlabel("l/l_opt  or  v/v_max")
    ax5.set_ylabel("Scaling Factor [0–1]")
    ax5.set_title("F-L and F-V Scaling Factors")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)

    out = os.path.join(RESULTS_DIR, "V5_1_force_surface.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")
    return passed, {
        "F_isometric_N": F_iso,
        "F_eccentric_N": F_ecc,
        "F_near_vmax_N": F_vmax,
        "eccentric_gain_ratio": float(F_ecc / F_iso) if F_iso > 0 else 0,
        "fl_peak_l_norm": float(peak_l),
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-2  Activation Dynamics Frequency Response (Bode)
# ══════════════════════════════════════════════════════════════════════════════

def test_frequency_response():
    """활성화 동역학의 저역통과 특성 — Bode 플롯."""
    print("\n[V5-2] Activation Dynamics Frequency Response")

    DT_FINE  = 0.001
    tau_act  = 0.015
    tau_dec  = 0.060
    f_c_act  = 1.0 / (2 * np.pi * tau_act)   # ~10.6 Hz
    f_c_dec  = 1.0 / (2 * np.pi * tau_dec)   # ~2.65 Hz

    freqs_hz = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    gains    = []
    phases   = []

    for f_hz in freqs_hz:
        n_settle  = int(5.0 / f_hz / DT_FINE)
        n_measure = int(10.0 / f_hz / DT_FINE)
        total     = n_settle + n_measure

        ad = ActivationDynamics(num_muscles=1, num_envs=1, device="cpu")
        ad.set_time_constants(torch.tensor([tau_act]), torch.tensor([tau_dec]))

        t_arr = np.arange(total) * DT_FINE
        u_arr = 0.5 + 0.4 * np.sin(2 * np.pi * f_hz * t_arr)

        acts = np.zeros(total)
        for i in range(total):
            cmd    = torch.tensor([[u_arr[i]]])
            a      = ad.step(cmd, DT_FINE)
            acts[i] = float(a[0, 0])

        u_ss = u_arr[n_settle:]
        a_ss = acts[n_settle:]

        # FFT-based amplitude and phase at f_hz
        freqs_fft = np.fft.rfftfreq(len(u_ss), d=DT_FINE)
        k         = int(np.argmin(np.abs(freqs_fft - f_hz)))
        U_fft     = np.fft.rfft(u_ss - u_ss.mean())
        A_fft     = np.fft.rfft(a_ss - a_ss.mean())
        H         = A_fft[k] / (U_fft[k] + 1e-12)
        g         = np.clip(np.abs(H), 1e-6, 2.0)
        ph        = np.degrees(np.angle(H))

        gains.append(g)
        phases.append(ph)
        print(f"  {f_hz:5.1f} Hz | gain={g:.3f} ({20*np.log10(g):+.1f}dB) | phase={ph:+.1f}°")

    gains_db = [20 * np.log10(g) for g in gains]

    # 이론 곡선 (first-order LPF with tau_act)
    f_theory  = np.logspace(-1, 2, 200)
    H_act = 1.0 / np.sqrt(1 + (f_theory / f_c_act) ** 2)
    H_dec = 1.0 / np.sqrt(1 + (f_theory / f_c_dec) ** 2)

    # ── 검증 ────────────────────────────────────────────────────────────────
    g_low  = float(gains[freqs_hz.index(1.0)])
    g_high = float(gains[freqs_hz.index(50.0)])
    low_pass = bool(g_low > 0.8 and g_high < 0.4)

    # -3dB 교차점 추정
    f_arr  = np.array(freqs_hz)
    g_arr  = np.array(gains)
    above  = g_arr > 0.707
    if above.any() and not above.all():
        cross_idx = np.argmax(~above)
        f_3db_est = float(f_arr[cross_idx - 1]) if cross_idx > 0 else float(f_arr[0])
    else:
        f_3db_est = float("nan")

    print(f"  f_c (tau_act) = {f_c_act:.1f} Hz | estimated -3dB crossover ≈ {f_3db_est:.1f} Hz")
    print(f"  Low-pass behavior (g_1Hz>{0.8:.1f}, g_50Hz<{0.4:.1f}): {'PASS' if low_pass else 'FAIL'}")

    # ── 그림 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("V5-2: Activation Dynamics Frequency Response", fontsize=14)

    # ① Amplitude (dB)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogx(freqs_hz, gains_db, "bo-", lw=2, ms=8, label="Measured")
    ax1.semilogx(f_theory, 20 * np.log10(H_act), "r--", lw=1.5,
                 label=f"Theory (τ_act={tau_act*1e3:.0f}ms, fc={f_c_act:.1f}Hz)")
    ax1.semilogx(f_theory, 20 * np.log10(H_dec), "g--", lw=1.5,
                 label=f"Theory (τ_dec={tau_dec*1e3:.0f}ms, fc={f_c_dec:.1f}Hz)")
    ax1.axhline(-3.0, color="gray", ls=":", lw=1.2, label="-3 dB")
    ax1.axvline(f_c_act, color="red", ls=":", lw=1.0, alpha=0.7)
    ax1.axvline(f_c_dec, color="green", ls=":", lw=1.0, alpha=0.7)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.set_title("Bode Plot — Amplitude Response")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(-35, 3)

    # ② Phase
    ax2 = fig.add_subplot(gs[1, :2])
    ph_theory_act = -np.degrees(np.arctan(f_theory / f_c_act))
    ph_theory_dec = -np.degrees(np.arctan(f_theory / f_c_dec))
    ax2.semilogx(freqs_hz, phases, "bo-", lw=2, ms=8, label="Measured")
    ax2.semilogx(f_theory, ph_theory_act, "r--", lw=1.5, label="Theory (τ_act)")
    ax2.semilogx(f_theory, ph_theory_dec, "g--", lw=1.5, label="Theory (τ_dec)")
    ax2.axhline(-45, color="gray", ls=":", lw=1, label="-45° at fc")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Bode Plot — Phase Response")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(-110, 10)

    # ③ Time-domain example at 3 Hz vs 15 Hz
    for plot_col, f_ex in enumerate([3.0, 15.0]):
        ax = fig.add_subplot(gs[0 if plot_col == 0 else 1, 2])
        n_ex    = int(4.0 / f_ex / DT_FINE)
        n_set   = int(5.0 / f_ex / DT_FINE)
        t_ex    = np.arange(n_set + n_ex) * DT_FINE * 1000
        u_ex    = 0.5 + 0.4 * np.sin(2 * np.pi * f_ex * np.arange(n_set + n_ex) * DT_FINE)

        ad = ActivationDynamics(num_muscles=1, num_envs=1, device="cpu")
        ad.set_time_constants(torch.tensor([tau_act]), torch.tensor([tau_dec]))
        a_ex = np.zeros(n_set + n_ex)
        for i in range(n_set + n_ex):
            cmd = torch.tensor([[u_ex[i]]])
            a_ex[i] = float(ad.step(cmd, DT_FINE)[0, 0])

        t_show  = t_ex[n_set:]
        u_show  = u_ex[n_set:]
        a_show  = a_ex[n_set:]
        t_show -= t_show[0]

        ax.plot(t_show, u_show, "gray", lw=1.5, ls="--", label="Input u(t)")
        ax.plot(t_show, a_show, "b-",   lw=2.0, label="Activation a(t)")
        ax.fill_between(t_show, u_show, a_show, alpha=0.15, color="red",
                        label="Attenuation")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value (0–1)")
        ax.set_title(f"Time Domain @ {f_ex:.0f} Hz")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    out = os.path.join(RESULTS_DIR, "V5_2_freq_response.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")
    return low_pass, {
        "freqs_hz": freqs_hz,
        "gains": [float(g) for g in gains],
        "gains_db": [float(g) for g in gains_db],
        "f_c_theory_act_hz": float(f_c_act),
        "f_c_theory_dec_hz": float(f_c_dec),
        "f_3db_estimated_hz": float(f_3db_est) if not np.isnan(f_3db_est) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-3  Stretch Reflex Bifurcation (클로누스 분기점)
# ══════════════════════════════════════════════════════════════════════════════

def _make_body_with_delay(gain, stretch_threshold=0.02, delay_steps=3):
    """Reflex delay_steps를 증가시킨 HumanBody 생성.

    기본 delay_steps=1은 buffer 구현상 실질적 0ms 지연.
    delay_steps=3 → 3×16.7ms = 50ms 지연 (척수 반사 실제값 40-60ms).
    이 지연이 있어야 양성 피드백 루프(클로누스)가 형성될 수 있음.
    """
    body = make_human_body(mods={
        "reflex": {"stretch_gain": float(gain), "stretch_threshold": stretch_threshold}
    })
    old = body.reflex
    body.reflex = ReflexController(body.num_muscles, 1, "cpu",
                                   reflex_delay_steps=delay_steps)
    body.reflex.stretch_gain[:]      = old.stretch_gain.clone()
    body.reflex.stretch_threshold[:] = old.stretch_threshold.clone()
    body.reflex.gto_gain[:]          = old.gto_gain.clone()
    body.reflex.gto_threshold[:]     = old.gto_threshold.clone()
    body.reflex.reciprocal_gain[:]   = old.reciprocal_gain.clone()
    body.reflex.antagonist_map[:]    = old.antagonist_map.clone()
    return body


def test_reflex_bifurcation():
    """Stretch gain 스윕 → 반사 유발 관절 운동 제한 분기점.

    메커니즘:
      저게인: 진자가 중력 평형까지 자유 신전 (최대 신전 = 중력 평형점)
      고게인: stretch reflex가 신전 저항 → 더 굽혀진 위치에서 정지 (extension-resistant posture)
      임계 게인: 중력 토크 = reflex 토크인 지점 → 운동 특성 질적 전환 (bifurcation).

    Note:
      실제 클로누스 진동은 척수 지연(40-60ms)이 필요하나, 이 Euler 모델에서는
      반사 지연을 delay_steps=3 (~50ms)으로 설정해도 높은 초기 속도로 인해
      진자가 지연 이전에 평형에 수렴함. 대신 최소 도달 각도 지표(min_angle)로
      분기점을 정의 — 게인 증가 시 신전 범위 감소 (extension-resistant posture).
    """
    print("\n[V5-3] Stretch Reflex Gain Effect — Extension Resistance Bifurcation")

    gains        = np.linspace(0, 14, 35)
    STEPS        = int(8.0 / DT)
    ANALYZE_FROM = int(3.0 / DT)

    min_angles   = []    # 최대 신전 (낮을수록 더 뻗음)
    ham_peaks    = []    # 피크 hamstring activation
    peak_torques = []    # 피크 반사 토크
    sample_traces = {}
    showcase_gains = [0.0, 3.0, 7.0, 13.0]

    ham_idx_local = MUSCLE_NAMES_FULL.index("hamstrings_L")

    print(f"  Sweeping {len(gains)} gain values (0 → 14) ...")
    for gain in gains:
        body = _make_body_with_delay(gain, stretch_threshold=0.02, delay_steps=3)
        angles, vels, torques, acts = run_pendulum(body, steps=STEPS)

        min_a   = float(angles.min())
        ham_pk  = float(acts[:, ham_idx_local].max())
        tau_pk  = float(np.abs(torques).max())

        min_angles.append(min_a)
        ham_peaks.append(ham_pk)
        peak_torques.append(tau_pk)

        for sg in showcase_gains:
            if abs(gain - sg) < 0.25:
                sample_traces[sg] = (angles.copy(), vels.copy(), acts.copy())

    min_angles   = np.array(min_angles)
    ham_peaks    = np.array(ham_peaks)
    peak_torques = np.array(peak_torques)

    # ── 검증 ────────────────────────────────────────────────────────────────
    # 고게인 → 저게인보다 최소 각도 높음 (덜 신전)
    low_gain_ang  = float(min_angles[:5].mean())   # gains 0~3 구간 평균
    high_gain_ang = float(min_angles[-5:].mean())  # gains 11~14 구간 평균
    ext_resist    = bool(high_gain_ang > low_gain_ang + 5.0)   # 5° 이상 차이

    # Ham peak activation: 게인 증가 시 증가해야 함
    low_ham  = float(ham_peaks[:5].mean())
    high_ham = float(ham_peaks[-5:].mean())
    act_incr = bool(high_ham > low_ham)

    passed = bool(ext_resist and act_incr)
    print(f"  Min angle — low gain avg={low_gain_ang:.1f}°  high gain avg={high_gain_ang:.1f}°")
    print(f"  Extension resistance (high > low + 5°): {'PASS' if ext_resist else 'FAIL'}")
    print(f"  Ham activation — low gain={low_ham:.3f}  high gain={high_ham:.3f}")
    print(f"  Activation increases with gain: {'PASS' if act_incr else 'FAIL'}")

    # ── 그림 ────────────────────────────────────────────────────────────────
    t_full = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
    fig.suptitle("V5-3: Stretch Reflex Gain Effect — Extension Resistance", fontsize=14)

    # ① 분기도: gain vs min_angle + ham_peak
    ax1  = fig.add_subplot(gs[0, :])
    ax1r = ax1.twinx()
    ax1.plot(gains, min_angles, "b-o", lw=2, ms=5, label="Min Knee Angle (°)")
    ax1r.plot(gains, ham_peaks, "r--s", lw=1.8, ms=5, label="Peak Ham Activation")
    ax1.set_xlabel("Stretch Gain")
    ax1.set_ylabel("Min Angle Reached (°)  [higher = less extension]", color="blue")
    ax1r.set_ylabel("Peak Ham Activation (0–1)", color="red")
    ax1.set_title("Reflex Gain Effect: Extension Resistance & Ham Activation vs Gain")
    ax1.grid(True, alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="lower right")

    # 게인 구간 음영
    ax1.axvspan(0, 3, alpha=0.07, color="blue", label="Low gain zone")
    ax1.axvspan(11, 14, alpha=0.07, color="red", label="High gain zone")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1r.tick_params(axis="y", labelcolor="red")

    # ② 샘플 시간 파형 (4개 게인)
    colors_bif = ["#2196F3", "#8BC34A", "#FF9800", "#F44336"]
    for i, sg in enumerate(showcase_gains):
        row, col = 1 + i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        if sg in sample_traces:
            ang_tr, vel_tr, act_tr = sample_traces[sg]
            ax.plot(t_full, ang_tr, color=colors_bif[i], lw=1.8, label="Angle")
            ax2 = ax.twinx()
            ax2.plot(t_full, act_tr[:, ham_idx_local], color=colors_bif[i],
                     lw=1.2, ls="--", alpha=0.7, label="Ham act")
            ax2.set_ylabel("Ham activation", fontsize=7, color=colors_bif[i])
            ax2.set_ylim(0, 1.2)
            min_sg = float(ang_tr.min())
            ax.set_title(f"Gain={sg:.0f}  |  Min={min_sg:.1f}°  Ham={act_tr[:, ham_idx_local].max():.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Knee Angle (°)")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "V5_3_bifurcation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")
    return passed, {
        "min_angle_low_gain_deg":  float(low_gain_ang),
        "min_angle_high_gain_deg": float(high_gain_ang),
        "ham_peak_low_gain":       float(low_ham),
        "ham_peak_high_gain":      float(high_ham),
        "gains_swept": [float(g) for g in gains],
        "min_angles_deg": [float(a) for a in min_angles],
        "ham_peaks": [float(h) for h in ham_peaks],
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-4  Ligament Stiffness Profile
# ══════════════════════════════════════════════════════════════════════════════

def test_ligament_stiffness():
    """관절 각도 스윕 → 인대 토크 비선형 스티프니스 프로파일."""
    print("\n[V5-4] Ligament Stiffness Profile")

    hard_lower, hard_upper = get_joint_limits_tensors("cpu")
    q_lo = float(hard_lower[L_KNEE_IDX])
    q_hi = float(hard_upper[L_KNEE_IDX])

    q_sweep = np.linspace(q_lo - 0.3, q_hi + 0.3, 600)

    # Healthy soft limit zones (margin=0.85)
    center   = (q_lo + q_hi) / 2
    half     = (q_hi - q_lo) / 2
    s_lo     = center - half * 0.85
    s_hi     = center + half * 0.85

    PROFILES_LIG = [
        ("Healthy",         50.0,  10.0, 5.0,  "#2196F3"),
        ("Spastic (k=200)", 200.0, 15.0, 25.0, "#F44336"),
        ("Flaccid (k=5)",   5.0,   5.0,  0.5,  "#4CAF50"),
    ]

    results = {}
    for label, k_lig, alpha, damping, color in PROFILES_LIG:
        body = make_human_body()
        body.ligament.k_lig[:]   = k_lig
        body.ligament.alpha[:]   = alpha
        body.ligament.damping[:] = damping

        torques_arr = []
        for q in q_sweep:
            pos = torch.zeros(1, NUM_DOFS)
            vel = torch.zeros(1, NUM_DOFS)
            pos[0, L_KNEE_IDX] = float(q)
            tau = body.ligament.compute_torque(pos, vel)
            torques_arr.append(float(tau[0, L_KNEE_IDX].item()))

        results[label] = np.array(torques_arr)
        # Peak torque at boundary
        print(f"  {label:22s} | k_lig={k_lig:5.0f} | "
              f"peak_τ={max(abs(t) for t in torques_arr):8.1f} Nm  "
              f"(at ±0.3rad beyond limit)")

    # ── 검증 ────────────────────────────────────────────────────────────────
    # 정상 ROM 내부에서 토크가 거의 0이어야 함
    center_idx = np.argmin(np.abs(q_sweep - center))
    tau_center_healthy = abs(float(results["Healthy"][center_idx]))
    zero_in_rom = bool(tau_center_healthy < 1.0)

    # Spastic이 Healthy보다 강해야 함
    max_healthy = float(max(abs(results["Healthy"])))
    max_spastic = float(max(abs(results["Spastic (k=200)"])))
    stiff_order = bool(max_spastic > max_healthy)

    passed = bool(zero_in_rom and stiff_order)
    print(f"  Zero torque at center: {tau_center_healthy:.3f} Nm  {'PASS' if zero_in_rom else 'FAIL'}")
    print(f"  Spastic({max_spastic:.0f}) > Healthy({max_healthy:.0f}): {'PASS' if stiff_order else 'FAIL'}")

    # ── 그림 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("V5-4: Ligament Soft-Limit Stiffness Profile", fontsize=14)

    for ax_idx, (ax, log_y) in enumerate(zip(axes, [False, True])):
        for label, k_lig, alpha, damping, color in PROFILES_LIG:
            tau = results[label]
            if log_y:
                ax.semilogy(np.degrees(q_sweep), np.abs(tau) + 1e-3,
                            color=color, lw=2, label=label)
            else:
                ax.plot(np.degrees(q_sweep), tau, color=color, lw=2, label=label)

        # Soft limit markers (healthy)
        for sl, sl_label in [(s_lo, "soft_lower"), (s_hi, "soft_upper")]:
            ax.axvline(np.degrees(sl), color="gray", ls="--", lw=1.2, alpha=0.8)
        ax.axvspan(np.degrees(q_lo), np.degrees(q_hi), alpha=0.06,
                   color="blue", label="Hard ROM")
        ax.axvspan(np.degrees(s_lo), np.degrees(s_hi), alpha=0.06,
                   color="green", label="Soft ROM (Healthy)")

        ax.set_xlabel("L_Knee Angle (°)")
        ax.set_ylabel("Ligament Torque (Nm)")
        ax.set_title(f"{'Log scale' if log_y else 'Linear scale'} — Torque vs Angle")
        ax.legend(fontsize=8, loc="upper center")
        ax.grid(True, alpha=0.3, which="both" if log_y else "major")
        if not log_y:
            ax.axhline(0, color="black", lw=0.8)

    # 소프트 리밋 레이블
    axes[0].text(np.degrees(s_lo) - 2, axes[0].get_ylim()[1] * 0.8,
                 "soft_lower", ha="right", fontsize=8, color="gray")
    axes[0].text(np.degrees(s_hi) + 2, axes[0].get_ylim()[1] * 0.8,
                 "soft_upper", ha="left", fontsize=8, color="gray")

    out = os.path.join(RESULTS_DIR, "V5_4_ligament_stiffness.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")
    return passed, {
        "tau_center_healthy_Nm": float(tau_center_healthy),
        "max_tau_healthy_Nm": float(max_healthy),
        "max_tau_spastic_Nm": float(max_spastic),
        "soft_lower_deg": float(np.degrees(s_lo)),
        "soft_upper_deg": float(np.degrees(s_hi)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-5  Reflex Pathway Isolation
# ══════════════════════════════════════════════════════════════════════════════

def test_reflex_pathways():
    """Stretch / GTO / Reciprocal 반사 경로를 각각 독립적으로 검증."""
    print("\n[V5-5] Reflex Pathway Isolation")

    ham_idx  = MUSCLE_NAMES_FULL.index("hamstrings_L")    # 6
    quad_idx = MUSCLE_NAMES_FULL.index("quadriceps_L")    # 4
    NUM_M    = len(MUSCLE_NAMES_FULL)
    NUM_E    = 100   # velocity/force sweep 점 수

    # ── ① Stretch Reflex ─────────────────────────────────────────────────────
    # 속도 범위를 낮게 설정 (0~0.5 m/s): 포화 이전에 게인 차이가 나타남
    # threshold=0.02, gain=1 → 포화점=1.02/1.0≈1.02 m/s, gain=8 → 포화점≈0.145 m/s
    vel_sweep = np.linspace(0, 0.5, NUM_E)

    # num_envs=NUM_E로 생성해야 delay_buffer shape이 (delay_steps, NUM_E, num_muscles)
    stretch_outputs = {}
    for gain_label, s_gain in [("Healthy (g=1)", 1.0), ("Spastic (g=8)", 8.0)]:
        body = make_human_body(
            mods={"reflex": {"stretch_gain": s_gain, "stretch_threshold": 0.02}},
            num_envs=NUM_E,
        )
        desc = torch.zeros(NUM_E, NUM_M)
        mvel = torch.zeros(NUM_E, NUM_M)
        mfor = torch.zeros(NUM_E, NUM_M)
        mvel[:, ham_idx] = torch.tensor(vel_sweep, dtype=torch.float32)
        # 2번 호출: 1번째로 delay buffer를 채운 뒤, 2번째 호출이 steady-state
        body.reflex.compute(desc, mvel, mfor, None, None)
        a_out = body.reflex.compute(desc, mvel, mfor, None, None)
        stretch_outputs[gain_label] = a_out[:, ham_idx].numpy()
        v3_idx_local = int(np.argmin(np.abs(vel_sweep - 0.15)))
        print(f"  Stretch [{gain_label}]: a_cmd @ v=0.15m/s = {stretch_outputs[gain_label][v3_idx_local]:.3f}  "
              f"max = {stretch_outputs[gain_label][-1]:.3f}")

    # 검증: Spastic > Healthy at v=0.10m/s (포화 이전 구간)
    v3_idx   = int(np.argmin(np.abs(vel_sweep - 0.10)))
    spas_v3  = float(stretch_outputs["Spastic (g=8)"][v3_idx])
    heal_v3  = float(stretch_outputs["Healthy (g=1)"][v3_idx])
    stretch_pass = bool(spas_v3 > heal_v3)
    print(f"  Spastic({spas_v3:.3f}) > Healthy({heal_v3:.3f}) @ v=0.10m/s: {'PASS' if stretch_pass else 'FAIL'}")

    # ── ② GTO Reflex ─────────────────────────────────────────────────────────
    act_sweep  = np.linspace(0, 1, NUM_E)  # 근육 활성화 (force 대용)
    f_max_val  = 1000.0
    f_max_ten  = torch.ones(NUM_M) * f_max_val

    body = make_human_body(num_envs=NUM_E)  # gto_gain=0.5, gto_threshold=0.8
    desc_gto  = torch.zeros(NUM_E, NUM_M)
    mfor_gto  = torch.zeros(NUM_E, NUM_M)
    mvel_gto  = torch.zeros(NUM_E, NUM_M)

    # Hamstrings에 다양한 힘 적용
    desc_gto[:, ham_idx] = 1.0  # 풀 descending command
    mfor_gto[:, ham_idx] = torch.tensor(act_sweep * f_max_val, dtype=torch.float32)

    body.reflex.reset()
    gto_out = body.reflex.compute(desc_gto, mvel_gto, mfor_gto, None, f_max_ten)
    gto_activation = gto_out[:, ham_idx].numpy()

    # 검증: GTO가 force > threshold에서 활성화를 낮춤
    f_thresh   = float(body.reflex.gto_threshold[ham_idx]) * f_max_val
    idx_below  = int(np.argmin(np.abs(act_sweep * f_max_val - f_thresh * 0.5)))
    idx_above  = int(np.argmin(np.abs(act_sweep * f_max_val - f_thresh * 1.5)))
    gto_pass   = bool(gto_activation[idx_above] < gto_activation[idx_below])
    gto_thresh_norm = float(body.reflex.gto_threshold[ham_idx])
    print(f"  GTO threshold: {gto_thresh_norm:.1f} × F_max = {f_thresh:.0f}N")
    print(f"  a_cmd below thresh={gto_activation[idx_below]:.3f}  above={gto_activation[idx_above]:.3f}")
    print(f"  GTO inhibition works: {'PASS' if gto_pass else 'FAIL'}")

    # ── ③ Reciprocal Inhibition ───────────────────────────────────────────────
    # ham descending = 0.5 (기준 명령), quad descending을 0→1로 증가시켜 ham 억제 확인
    desc_ri = torch.zeros(NUM_E, NUM_M)
    desc_ri[:, quad_idx] = torch.tensor(act_sweep, dtype=torch.float32)  # quad sweep
    desc_ri[:, ham_idx]  = 0.5   # ham baseline: 억제되면 0.5 아래로 내려가야 함

    body_ri = make_human_body(num_envs=NUM_E)
    body_ri.reflex.reset()
    ri_out = body_ri.reflex.compute(desc_ri, torch.zeros(NUM_E, NUM_M),
                                     torch.zeros(NUM_E, NUM_M), None, None)

    ri_ham     = ri_out[:, ham_idx].numpy()     # hamstrings 출력 (억제될수록 낮아짐)
    ri_pass    = bool(ri_ham[-1] < ri_ham[0])   # 높은 quad → ham 억제 → 출력 감소
    recip_gain = float(body_ri.reflex.reciprocal_gain[quad_idx])
    print(f"  Reciprocal gain={recip_gain:.2f}: "
          f"ham@quad=0: {ri_ham[0]:.3f}  ham@quad=1: {ri_ham[-1]:.3f}  (ham_baseline=0.5)")
    print(f"  Reciprocal inhibition works: {'PASS' if ri_pass else 'FAIL'}")

    # ── 그림 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("V5-5: Reflex Pathway Isolation", fontsize=14)

    # Stretch Reflex
    ax = axes[0]
    for (lbl, _), color in zip([("Healthy (g=1)", 1.0), ("Spastic (g=8)", 8.0)],
                                ["#2196F3", "#F44336"]):
        ax.plot(vel_sweep, stretch_outputs[lbl], color=color, lw=2.5, label=lbl)
    ax.axvline(float(body.reflex.stretch_threshold[ham_idx]), color="gray",
               ls="--", lw=1.2, label="Threshold")
    ax.set_xlabel("Muscle Velocity (m/s)")
    ax.set_ylabel("Activation Command (0–1)")
    ax.set_title("① Stretch Reflex\n(Ham_L, descending=0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # GTO
    ax = axes[1]
    ax.plot(act_sweep * f_max_val, gto_activation, "purple", lw=2.5)
    ax.axvline(f_thresh, color="red", ls="--", lw=1.5,
               label=f"GTO threshold ({gto_thresh_norm:.1f}×F_max)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5, label="descending=1.0")
    ax.fill_between(act_sweep * f_max_val, gto_activation, 1.0,
                    alpha=0.2, color="red", label="GTO inhibition zone")
    ax.set_xlabel("Muscle Force (N)")
    ax.set_ylabel("Net Activation Command (0–1)")
    ax.set_title("② GTO (Golgi Tendon) Reflex\n(Ham_L, descending=1.0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.15)

    # Reciprocal Inhibition
    ax = axes[2]
    ham_baseline_line = np.full_like(act_sweep, 0.5)
    ax.plot(act_sweep, ri_ham, "#4CAF50", lw=2.5, label="Ham_L output")
    ax.axhline(0.5, color="gray", ls="--", lw=1.5, label="Ham baseline (0.5)")
    ax.fill_between(act_sweep, ri_ham, 0.5,
                    where=(ri_ham < 0.5),
                    alpha=0.3, color="red", label="Inhibition below baseline")
    ax.set_xlabel("Quad_L Descending Command (0–1)")
    ax.set_ylabel("Ham_L Output Command (0–1)")
    ax.set_title(f"③ Reciprocal Inhibition\n(Quad→Ham, gain={recip_gain:.2f}, Ham_baseline=0.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    out = os.path.join(RESULTS_DIR, "V5_5_reflex_pathways.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")

    passed = bool(stretch_pass and gto_pass and ri_pass)
    return passed, {
        "stretch_Healthy_at_v3ms": float(heal_v3),
        "stretch_Spastic_at_v3ms": float(spas_v3),
        "gto_below_thresh": float(gto_activation[idx_below]),
        "gto_above_thresh": float(gto_activation[idx_above]),
        "ri_ham_at_quad0": float(ri_ham[0]),
        "ri_ham_at_quad1": float(ri_ham[-1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-6  Bi-articular Coupling
# ══════════════════════════════════════════════════════════════════════════════

def test_biarticular_coupling():
    """이관절근 커플링: 단일 근육 1000N → 관절 토크 분배 시각화."""
    print("\n[V5-6] Bi-articular Muscle Coupling")

    body = make_human_body()
    R    = body.moment_arm.R          # (num_muscles, num_dofs)
    coupling = body.moment_arm.get_coupling_info()

    TEST_MUSCLES = [
        ("rectus_femoris_L",  ["L_Hip", "L_Knee"],  "bi"),
        ("hamstrings_L",      ["L_Hip", "L_Knee"],  "bi"),
        ("gastrocnemius_L",   ["L_Knee", "L_Ankle"], "bi"),
        ("soleus_L",          ["L_Ankle"],            "mono"),
        ("quadriceps_L",      ["L_Knee"],             "mono"),
    ]

    F_TEST = 1000.0  # N
    results = {}

    for muscle_name, expected_joints, mtype in TEST_MUSCLES:
        m_idx = body.moment_arm.get_muscle_index(muscle_name)
        F     = torch.zeros(1, body.num_muscles)
        F[0, m_idx] = F_TEST

        tau   = body.moment_arm.forces_to_torques(F)  # (1, num_dofs)
        torques_by_joint = {}
        for joint_name in JOINT_NAMES:
            s, e = JOINT_DOF_RANGE[joint_name]
            t    = float(tau[0, s:e].abs().max().item())
            if t > 0.1:
                torques_by_joint[joint_name] = t

        actual_joints = list(torques_by_joint.keys())
        biart_correct  = set(actual_joints) == set(expected_joints)
        results[muscle_name] = {
            "type": mtype,
            "expected_joints": expected_joints,
            "actual_joints":   actual_joints,
            "torques_Nm":      torques_by_joint,
            "coupling_ok":     biart_correct,
        }
        ok_str = "PASS" if biart_correct else "FAIL"
        print(f"  {muscle_name:22s} ({mtype:4s}) | joints: {actual_joints} | {ok_str}")

    all_coupling_ok = all(v["coupling_ok"] for v in results.values())

    # ── 그림 1: R matrix 히트맵 (관련 근육 행, 관련 관절 열만) ──────────────
    MUSCLES_SHOW = [name for name, _, _ in TEST_MUSCLES]
    JOINTS_SHOW  = ["L_Hip", "L_Knee", "L_Ankle"]

    R_sub  = np.zeros((len(MUSCLES_SHOW), len(JOINTS_SHOW)))
    for i, mname in enumerate(MUSCLES_SHOW):
        mi = body.moment_arm.get_muscle_index(mname)
        for j, jname in enumerate(JOINTS_SHOW):
            s, e = JOINT_DOF_RANGE[jname]
            R_sub[i, j] = float(R[mi, s].item())

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("V5-6: Bi-articular Muscle Coupling via Moment Arm Matrix", fontsize=14)

    # ① R matrix heatmap
    ax1 = fig.add_subplot(gs[:, 0])
    vabs = max(abs(R_sub.min()), abs(R_sub.max()))
    im   = ax1.imshow(R_sub, cmap="RdBu", vmin=-vabs, vmax=vabs, aspect="auto")
    plt.colorbar(im, ax=ax1, label="Moment Arm (m)")
    ax1.set_xticks(range(len(JOINTS_SHOW)))
    ax1.set_xticklabels(JOINTS_SHOW, fontsize=10)
    ax1.set_yticks(range(len(MUSCLES_SHOW)))
    ax1.set_yticklabels(MUSCLES_SHOW, fontsize=9)
    ax1.set_title("Moment Arm Matrix R\n(selected muscles × joints)")
    for i in range(len(MUSCLES_SHOW)):
        for j in range(len(JOINTS_SHOW)):
            val = R_sub[i, j]
            if abs(val) > 0.001:
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=9, color="white" if abs(val) > 0.03 else "black",
                         fontweight="bold")

    # ② 커플링 토크 바 차트
    ax2 = fig.add_subplot(gs[0, 1])
    x       = np.arange(len(MUSCLES_SHOW))
    width   = 0.25
    colors3 = ["#2196F3", "#FF9800", "#9C27B0"]
    for j, jname in enumerate(JOINTS_SHOW):
        tau_bars = []
        for mname in MUSCLES_SHOW:
            tau_bars.append(results[mname]["torques_Nm"].get(jname, 0))
        ax2.bar(x + (j - 1) * width, tau_bars, width, label=jname,
                color=colors3[j], alpha=0.85, edgecolor="k", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split("_L")[0][:12] for m in MUSCLES_SHOW],
                         rotation=20, fontsize=8)
    ax2.set_ylabel("Torque (Nm) per 1000N muscle force")
    ax2.set_title(f"Coupling: 1 kN Force → Joint Torques")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # ③ 이관절근 커플링 다이어그램 (화살표)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("Coupling Diagram")

    # 관절 위치
    joints_pos = {"L_Hip": (2, 8), "L_Knee": (5, 5), "L_Ankle": (8, 2)}
    for jname, (jx, jy) in joints_pos.items():
        ax3.scatter(jx, jy, s=500, c="navy", zorder=5)
        ax3.text(jx, jy + 0.5, jname.replace("L_", ""), ha="center",
                 fontsize=10, fontweight="bold", color="navy")

    # 근육-관절 연결
    muscle_colors = {"rectus_femoris_L": "#F44336", "hamstrings_L": "#FF9800",
                     "gastrocnemius_L": "#9C27B0"}
    offsets = {"rectus_femoris_L": -0.3, "hamstrings_L": 0.0, "gastrocnemius_L": 0.3}
    for mname, mcolor in muscle_colors.items():
        m_exp_joints = results[mname]["expected_joints"]
        for k in range(len(m_exp_joints) - 1):
            j1 = joints_pos[m_exp_joints[k]]
            j2 = joints_pos[m_exp_joints[k + 1]]
            off = offsets[mname]
            ax3.annotate("", xy=(j2[0] + off, j2[1] + off),
                         xytext=(j1[0] + off, j1[1] + off),
                         arrowprops=dict(arrowstyle="->", color=mcolor, lw=2.5))
        short_name = mname.replace("_femoris_L", "_fem").replace("_L", "").replace("gastrocnemius", "gastroc")
        mid_joint_pos = joints_pos[m_exp_joints[0]]
        ax3.text(mid_joint_pos[0] + offsets[mname] * 2 + 0.8,
                 mid_joint_pos[1] + offsets[mname] * 2,
                 short_name, color=mcolor, fontsize=8, fontweight="bold")

    # 범례
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#F44336", lw=2, label="Rectus Femoris (hip flex + knee ext)"),
        Line2D([0], [0], color="#FF9800", lw=2, label="Hamstrings (hip ext + knee flex)"),
        Line2D([0], [0], color="#9C27B0", lw=2, label="Gastrocnemius (knee flex + plantarflex)"),
    ]
    ax3.legend(handles=legend_elems, loc="lower left", fontsize=8)

    out = os.path.join(RESULTS_DIR, "V5_6_biarticular_coupling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")
    return all_coupling_ok, {m: to_python(v) for m, v in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("CALM Framework — Deep Validation Suite (V5)")
    print("=" * 68)

    all_passed  = {}
    all_metrics = {}

    for test_fn, key in [
        (test_force_surface,         "V5-1_force_surface"),
        (test_frequency_response,    "V5-2_freq_response"),
        (test_reflex_bifurcation,    "V5-3_bifurcation"),
        (test_ligament_stiffness,    "V5-4_ligament"),
        (test_reflex_pathways,       "V5-5_reflex_pathways"),
        (test_biarticular_coupling,  "V5-6_biarticular"),
    ]:
        p, m = test_fn()
        all_passed[key]  = bool(p)
        all_metrics[key] = to_python(m)

    out_json = os.path.join(RESULTS_DIR, "deep_metrics.json")
    with open(out_json, "w") as f:
        json.dump({"passed": all_passed, "metrics": all_metrics}, f, indent=2)
    print(f"\n  Metrics: {out_json}")

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("-" * 68)
    for name, ok in all_passed.items():
        print(f"  {name:35s}  {'✅ PASS' if ok else '❌ FAIL'}")
    n_pass = sum(all_passed.values())
    total  = len(all_passed)
    print("-" * 68)
    print(f"  Result: {n_pass}/{total} PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()

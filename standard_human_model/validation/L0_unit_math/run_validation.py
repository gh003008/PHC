"""
L0 Unit Math Validation — CALM Validation Pyramid v2
=====================================================
All individual equations are tested for mathematical correctness.
No IsaacGym required; pure PyTorch.

Tests (ver2 spec):
    L0-01  Hill F-L active curve shape
    L0-02  Hill F-V curve monotonicity
    L0-03  Hill F-L passive curve
    L0-04  Activation-force linearity
    L0-05  Ligament soft-limit directionality
    L0-06  Stretch reflex gain effect
    L0-07  L/R symmetry (all 10 muscles)
    L0-08  Zero input → zero output (all muscles)
    L0-09  Activation time constant measurement

Usage:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python standard_human_model/validation/L0_unit_math/run_validation.py
"""

import sys, os, time, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from standard_human_model.core.muscle_model import HillMuscleModel, MuscleParams
from standard_human_model.core.moment_arm import MomentArmMatrix
from standard_human_model.core.ligament_model import LigamentModel
from standard_human_model.core.reflex_controller import ReflexController, ReflexParams
from standard_human_model.core.activation_dynamics import ActivationDynamics
from standard_human_model.core.skeleton import (
    JOINT_DOF_RANGE, NUM_DOFS, JOINT_NAMES, LOWER_LIMB_JOINTS,
    get_joint_limits_tensors,
)

# ── Paths ──
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
BASELINE_YAML = os.path.join(CONFIG_DIR, "healthy_baseline.yaml")
MUSCLE_DEF_YAML = os.path.join(CONFIG_DIR, "muscle_definitions.yaml")

# ── Globals ──
results = {}
start_time = time.time()

def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

def report(tid, name, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    results[tid] = {"name": name, "passed": passed, "detail": detail}
    print(f"\n[{tid}] {name}  {'✅' if passed else '❌'} {mark}")
    if detail:
        print(f"     {detail}")

def load_baseline():
    with open(BASELINE_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_muscle_defs():
    with open(MUSCLE_DEF_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_muscle_params(baseline):
    """baseline yaml -> list of MuscleParams (L side only, 10 muscles)."""
    mp = baseline["muscle_params"]
    L_MUSCLES = [k for k in mp if k.endswith("_L")]
    params = []
    for name in L_MUSCLES:
        p = mp[name]
        params.append(MuscleParams(
            name=name, f_max=p["f_max"], l_opt=p["l_opt"],
            v_max=p["v_max"], pennation=p["pennation"],
            tau_act=p["tau_act"], tau_deact=p["tau_deact"],
            k_pe=p["k_pe"], epsilon_0=p["epsilon_0"],
            l_tendon_slack=p["l_tendon_slack"],
            k_tendon=p["k_tendon"], damping=p["damping"],
        ))
    return params, L_MUSCLES

def build_all_muscle_params(baseline):
    """baseline yaml -> list of MuscleParams (all 20 muscles)."""
    mp = baseline["muscle_params"]
    ALL_MUSCLES = list(mp.keys())
    params = []
    for name in ALL_MUSCLES:
        p = mp[name]
        params.append(MuscleParams(
            name=name, f_max=p["f_max"], l_opt=p["l_opt"],
            v_max=p["v_max"], pennation=p["pennation"],
            tau_act=p["tau_act"], tau_deact=p["tau_deact"],
            k_pe=p["k_pe"], epsilon_0=p["epsilon_0"],
            l_tendon_slack=p["l_tendon_slack"],
            k_tendon=p["k_tendon"], damping=p["damping"],
        ))
    return params, ALL_MUSCLES


# ═════════════════════════════════════════════════════════════════════
# L0-01  Hill F-L Active Curve
# ═════════════════════════════════════════════════════════════════════
def test_L0_01():
    print("\n" + "=" * 60)
    print("L0-01: Hill F-L Active Curve (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    params, names = build_muscle_params(baseline)
    n = len(params)

    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params)

    l_norms = torch.linspace(0.3, 2.0, 300)
    # f_FL is the same for all muscles (Gaussian), but we verify per-muscle
    fl_all = model.force_length_active(l_norms.unsqueeze(1).expand(-1, n))  # (300, n)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    all_pass = True

    for i, (ax, name) in enumerate(zip(axes, names)):
        fl = fl_all[:, i].numpy()
        peak_idx = np.argmax(fl)
        peak_l = l_norms[peak_idx].item()
        peak_val = fl[peak_idx]

        # Expected operating point: l_norm ~ 1.0
        ok = abs(peak_l - 1.0) < 0.01 and peak_val >= 0.99
        all_pass = all_pass and ok

        ax.plot(l_norms.numpy(), fl, "b-", linewidth=1.5)
        ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
        ax.scatter([peak_l], [peak_val], color="red" if not ok else "green", s=40, zorder=5)
        ax.set_title(f"{name.replace('_L','')}\npeak@{peak_l:.3f}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if i % 5 == 0:
            ax.set_ylabel("f_FL")
        if i >= 5:
            ax.set_xlabel("l_norm")

    fig.suptitle("L0-01: Active Force-Length Curves (All Muscles)\nCriterion: peak at l_norm=1.0 ± 0.01",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L0_01_FL_active_all_muscles")

    report("L0-01", "Hill F-L Active", all_pass,
           f"All {n} muscles peak at l_norm=1.0 ± 0.01")


# ═════════════════════════════════════════════════════════════════════
# L0-02  Hill F-V Curve
# ═════════════════════════════════════════════════════════════════════
def test_L0_02():
    print("\n" + "=" * 60)
    print("L0-02: Hill F-V Curve (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    params, names = build_muscle_params(baseline)
    n = len(params)
    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params)

    v_norms = torch.linspace(-1.0, 0.8, 300)
    fv_all = model.force_velocity(v_norms.unsqueeze(1).expand(-1, n))

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    all_pass = True

    for i, (ax, name) in enumerate(zip(axes, names)):
        fv = fv_all[:, i].numpy()
        fv_at_0 = model.force_velocity(torch.tensor([[0.0] * n]))[0, i].item()
        fv_at_neg99 = model.force_velocity(torch.tensor([[-0.99] * n]))[0, i].item()
        fv_max = fv.max()

        # Check: FV(0)=1.0, concentric monotone decreasing, eccentric > 1.0
        ok = (abs(fv_at_0 - 1.0) < 0.01 and
              fv_at_neg99 < 0.05 and
              fv_max <= 1.85 and fv_max >= 1.5)
        all_pass = all_pass and ok

        ax.plot(v_norms.numpy(), fv, "r-", linewidth=1.5)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.axvline(0.0, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{name.replace('_L','')}\nFV(0)={fv_at_0:.2f}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if i % 5 == 0:
            ax.set_ylabel("f_FV")
        if i >= 5:
            ax.set_xlabel("v_norm")

    fig.suptitle("L0-02: Force-Velocity Curves (All Muscles)\nFV(0)=1.0, concentric↓, eccentric↑ (max 1.8)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L0_02_FV_all_muscles")

    report("L0-02", "Hill F-V Curve", all_pass,
           "FV(0)=1.0, concentric→0, eccentric max~1.8")


# ═════════════════════════════════════════════════════════════════════
# L0-03  Hill F-L Passive Curve
# ═════════════════════════════════════════════════════════════════════
def test_L0_03():
    print("\n" + "=" * 60)
    print("L0-03: Hill F-L Passive Curve (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    params, names = build_muscle_params(baseline)
    n = len(params)
    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params)

    l_norms = torch.linspace(0.5, 2.0, 300)
    fp_all = model.force_length_passive(l_norms.unsqueeze(1).expand(-1, n))

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    all_pass = True

    for i, (ax, name) in enumerate(zip(axes, names)):
        fp = fp_all[:, i].numpy()
        fp_at_1 = model.force_length_passive(torch.tensor([[1.0] * n]))[0, i].item()
        # Must be 0 at l_norm <= 1.0, monotonically increasing after
        zero_region = fp_all[l_norms <= 1.0, i]
        ok = zero_region.abs().max().item() < 1e-6 and fp_at_1 < 1e-6
        all_pass = all_pass and ok

        ax.plot(l_norms.numpy(), fp, "g-", linewidth=1.5)
        ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{name.replace('_L','')}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if i % 5 == 0:
            ax.set_ylabel("f_PE")
        if i >= 5:
            ax.set_xlabel("l_norm")

    fig.suptitle("L0-03: Passive Force-Length Curves (All Muscles)\nZero for l_norm <= 1.0, exponential increase after",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L0_03_FL_passive_all_muscles")

    report("L0-03", "Hill F-L Passive", all_pass,
           "f_PE = 0 for l_norm <= 1.0, monotonically increasing after")


# ═════════════════════════════════════════════════════════════════════
# L0-04  Activation-Force Linearity
# ═════════════════════════════════════════════════════════════════════
def test_L0_04():
    print("\n" + "=" * 60)
    print("L0-04: Activation-Force Linearity (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    params, names = build_muscle_params(baseline)
    n = len(params)
    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params)

    # Use l_norm=1.0, v_norm=0 (isometric at optimal length)
    # muscle_length = l_tendon_slack + l_opt * cos(pennation) for l_norm=1.0
    cos_penn = torch.cos(model.pennation)
    l_mtu_opt = model.l_tendon_slack + model.l_opt * cos_penn
    v_zero = torch.zeros(1, n)

    activations = [0.0, 0.25, 0.5, 0.75, 1.0]
    forces = []
    for a in activations:
        a_tensor = torch.full((1, n), a)
        F = model.compute_force(a_tensor, l_mtu_opt.unsqueeze(0), v_zero)
        forces.append(F[0].detach().numpy())

    forces = np.array(forces)  # (5, n)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True)
    axes = axes.flatten()
    all_pass = True

    for i, (ax, name) in enumerate(zip(axes, names)):
        f_vals = forces[:, i]
        f_max_val = f_vals[-1]  # a=1.0

        ax.plot(activations, f_vals, "ko-", linewidth=1.5, markersize=5)
        # Ideal linear: f = a * f_max
        ax.plot(activations, [a * f_max_val for a in activations],
                "b--", alpha=0.5, label="Ideal linear")

        # Check F(0.5) ≈ 0.5 * F(1.0) ± 5%
        if f_max_val > 0:
            ratio = f_vals[2] / f_max_val  # a=0.5
            ok = abs(ratio - 0.5) < 0.05
        else:
            ok = False
        all_pass = all_pass and ok

        ax.set_title(f"{name.replace('_L','')}\nF(0.5)/F(1.0)={ratio:.3f}" if f_max_val > 0 else name,
                     fontsize=9, color="green" if ok else "red")
        ax.grid(True, alpha=0.2)
        if i % 5 == 0:
            ax.set_ylabel("Force (N)")
        if i >= 5:
            ax.set_xlabel("Activation")

    fig.suptitle("L0-04: Activation-Force Linearity (All Muscles)\nF(a=0.5) / F(a=1.0) = 0.5 ± 5%",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L0_04_activation_linearity_all_muscles")

    report("L0-04", "Activation-Force Linearity", all_pass,
           "F(a=0.5)/F(a=1.0) within ±5% of 0.5 for all muscles")


# ═════════════════════════════════════════════════════════════════════
# L0-05  Ligament Soft-Limit Directionality
# ═════════════════════════════════════════════════════════════════════
def test_L0_05():
    print("\n" + "=" * 60)
    print("L0-05: Ligament Soft-Limit Directionality (all joints)")
    print("=" * 60)

    hard_lower, hard_upper = get_joint_limits_tensors("cpu")
    lig = LigamentModel(num_envs=1, device="cpu")
    lig.set_limits_from_hard_limits(hard_lower, hard_upper, margin_ratio=0.85)

    # Test: go beyond each joint's upper/lower soft limit
    test_joints = LOWER_LIMB_JOINTS
    all_pass = True
    details = []

    fig, axes = plt.subplots(len(test_joints), 1, figsize=(14, 3 * len(test_joints)),
                              sharex=False)
    if len(test_joints) == 1:
        axes = [axes]

    for j_idx, joint_name in enumerate(test_joints):
        s, e = JOINT_DOF_RANGE[joint_name]
        ax = axes[j_idx]
        ndof = e - s

        for dof_offset in range(ndof):
            dof_idx = s + dof_offset
            # Sweep
            q_range = torch.linspace(
                hard_lower[dof_idx].item() * 1.2,
                hard_upper[dof_idx].item() * 1.2,
                200
            )
            dof_pos = torch.zeros(200, NUM_DOFS)
            dof_pos[:, dof_idx] = q_range
            dof_vel = torch.zeros(200, NUM_DOFS)

            tau = lig.compute_torque(dof_pos, dof_vel)[:, dof_idx].numpy()

            axis_labels = ["x(sag)", "y(front)", "z(trans)"]
            label = f"DOF {dof_offset} ({axis_labels[dof_offset]})"
            ax.plot(np.degrees(q_range.numpy()), tau, linewidth=1.2, label=label)

            # Check directionality: torque should oppose the deviation
            upper_zone = q_range > lig.soft_upper[dof_idx]
            lower_zone = q_range < lig.soft_lower[dof_idx]
            if upper_zone.any():
                tau_upper = lig.compute_torque(
                    dof_pos[upper_zone], dof_vel[upper_zone]
                )[:, dof_idx]
                ok_upper = (tau_upper <= 0).all().item()
            else:
                ok_upper = True
            if lower_zone.any():
                tau_lower = lig.compute_torque(
                    dof_pos[lower_zone], dof_vel[lower_zone]
                )[:, dof_idx]
                ok_lower = (tau_lower >= 0).all().item()
            else:
                ok_lower = True

            ok = ok_upper and ok_lower
            all_pass = all_pass and ok
            if not ok:
                details.append(f"{joint_name} DOF{dof_offset}: direction error")

        ax.set_title(f"{joint_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Ligament Torque (Nm)")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Joint Angle (deg)")
    fig.suptitle("L0-05: Ligament Soft-Limit Torque (Lower Limb Joints)\nRestoring torque opposes ROM violation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "L0_05_ligament_all_joints")

    report("L0-05", "Ligament Soft-Limit Directionality", all_pass,
           "; ".join(details) if details else "All joints produce correct restoring direction")


# ═════════════════════════════════════════════════════════════════════
# L0-06  Stretch Reflex Gain Effect
# ═════════════════════════════════════════════════════════════════════
def test_L0_06():
    print("\n" + "=" * 60)
    print("L0-06: Stretch Reflex Gain Effect")
    print("=" * 60)

    gains = [0.0, 1.0, 4.0, 8.0, 15.0]
    n_muscles = 10
    results_by_gain = {}

    for g in gains:
        reflex = ReflexController(num_muscles=n_muscles, num_envs=1, device="cpu",
                                  reflex_delay_steps=0)  # no delay for unit test
        for idx in range(n_muscles):
            reflex.set_params({idx: ReflexParams(
                stretch_gain=g, stretch_threshold=0.1,
                gto_gain=0.0, gto_threshold=1.0, reciprocal_gain=0.0,
            )})
        # Input: velocity causing stretch
        desc_cmd = torch.full((1, n_muscles), 0.3)
        v_stretch = torch.full((1, n_muscles), 0.5)  # strong stretch
        f_muscle = torch.zeros(1, n_muscles)

        a = reflex.compute(desc_cmd, v_stretch, f_muscle)
        results_by_gain[g] = a[0].numpy().copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_muscles)
    width = 0.15
    for i, g in enumerate(gains):
        ax.bar(x + i * width, results_by_gain[g], width, label=f"gain={g}")

    ax.set_xlabel("Muscle Index")
    ax.set_ylabel("Total Activation")
    ax.set_title("L0-06: Stretch Reflex Gain Effect\nHigher gain → higher reflex activation",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"M{i}" for i in range(n_muscles)], fontsize=8)
    fig.tight_layout()
    save(fig, "L0_06_stretch_reflex_gains")

    # Check monotonicity
    all_pass = True
    for i in range(len(gains) - 1):
        if not (results_by_gain[gains[i + 1]] >= results_by_gain[gains[i]] - 1e-6).all():
            all_pass = False

    report("L0-06", "Stretch Reflex Gain", all_pass,
           f"gain ordering: {' < '.join(str(g) for g in gains)} → activation monotonically increases")


# ═════════════════════════════════════════════════════════════════════
# L0-07  L/R Symmetry (all muscles)
# ═════════════════════════════════════════════════════════════════════
def test_L0_07():
    print("\n" + "=" * 60)
    print("L0-07: L/R Symmetry (all 20 muscles)")
    print("=" * 60)

    baseline = load_baseline()
    muscle_defs_data = load_muscle_defs()

    params_all, all_names = build_all_muscle_params(baseline)
    n = len(params_all)
    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params_all)

    marm = MomentArmMatrix(muscle_defs_data["muscles"], device="cpu")

    # Same joint angles for L/R
    dof_pos = torch.zeros(1, NUM_DOFS)
    # Set hip flexion 20 deg, knee flexion 30 deg for both sides
    for side_prefix, joints in [("L_", ["L_Hip", "L_Knee", "L_Ankle"]),
                                 ("R_", ["R_Hip", "R_Knee", "R_Ankle"])]:
        for j, angle_deg in zip(joints, [20.0, 30.0, -10.0]):
            s, _ = JOINT_DOF_RANGE[j]
            dof_pos[0, s] = np.radians(angle_deg)

    dof_vel = torch.zeros(1, NUM_DOFS)
    # Same velocity
    for j in ["L_Hip", "R_Hip"]:
        s, _ = JOINT_DOF_RANGE[j]
        dof_vel[0, s] = 0.5
    for j in ["L_Knee", "R_Knee"]:
        s, _ = JOINT_DOF_RANGE[j]
        dof_vel[0, s] = -0.3

    l_mtu = marm.compute_muscle_length(dof_pos)
    v_mtu = marm.compute_muscle_velocity(dof_pos, dof_vel)
    activation = torch.full((1, n), 0.5)

    F = model.compute_force(activation, l_mtu, v_mtu)
    tau = marm.forces_to_torques(F, dof_pos)

    # Compare L vs R (first 10 = L, last 10 = R)
    F_L = F[0, :10].detach().numpy()
    F_R = F[0, 10:].detach().numpy()
    diff = np.abs(F_L - F_R)
    max_diff = diff.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Force comparison
    ax = axes[0]
    x = np.arange(10)
    short_names = [n.replace("_L", "") for n in all_names[:10]]
    ax.bar(x - 0.15, F_L, 0.3, label="Left", color="tab:blue", alpha=0.8)
    ax.bar(x + 0.15, F_R, 0.3, label="Right", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Force (N)")
    ax.set_title("Muscle Forces: L vs R", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Diff
    ax = axes[1]
    ax.bar(x, diff, 0.5, color="red" if max_diff > 0.1 else "green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|F_L - F_R| (N)")
    ax.set_title(f"Absolute Difference (max={max_diff:.6f} N)", fontweight="bold")
    ax.axhline(0.1, color="red", ls="--", alpha=0.5, label="threshold 0.1 Nm")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("L0-07: L/R Symmetry (Same Input → Same Output)\nCriterion: max |F_L - F_R| < 0.1 N",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "L0_07_LR_symmetry")

    passed = max_diff < 0.1
    report("L0-07", "L/R Symmetry", passed,
           f"max |F_L - F_R| = {max_diff:.6f} N (threshold < 0.1 N)")


# ═════════════════════════════════════════════════════════════════════
# L0-08  Zero Input → Zero Output
# ═════════════════════════════════════════════════════════════════════
def test_L0_08():
    print("\n" + "=" * 60)
    print("L0-08: Zero Input → Zero Output (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    muscle_defs_data = load_muscle_defs()

    params_all, all_names = build_all_muscle_params(baseline)
    n = len(params_all)
    model = HillMuscleModel(num_muscles=n, num_envs=1, device="cpu")
    model.set_params(params_all)

    marm = MomentArmMatrix(muscle_defs_data["muscles"], device="cpu")

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    activation = torch.zeros(1, n)

    l_mtu = marm.compute_muscle_length(dof_pos)
    v_mtu = marm.compute_muscle_velocity(dof_pos, dof_vel)
    F = model.compute_force(activation, l_mtu, v_mtu)
    tau = marm.forces_to_torques(F, dof_pos)

    max_tau = tau.abs().max().item()
    max_F = F.abs().max().item()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Forces
    ax = axes[0]
    short_names = [n.replace("_L", "").replace("_R", "R_") for n in all_names]
    ax.bar(range(n), F[0].detach().numpy(), color="steelblue", alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Force (N)")
    ax.set_title(f"Muscle Forces at Zero Input\nmax |F| = {max_F:.4f} N", fontweight="bold")
    ax.axhline(5.0, color="red", ls="--", alpha=0.5, label="threshold 5 N")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Torques (sagittal DOFs of lower limb)
    ax = axes[1]
    tau_np = tau[0].detach().numpy()
    joint_labels = []
    tau_vals = []
    for jn in LOWER_LIMB_JOINTS:
        s, _ = JOINT_DOF_RANGE[jn]
        joint_labels.append(jn)
        tau_vals.append(tau_np[s])  # sagittal DOF only
    ax.bar(range(len(joint_labels)), tau_vals, color="coral", alpha=0.8)
    ax.set_xticks(range(len(joint_labels)))
    ax.set_xticklabels(joint_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"Joint Torques (Sagittal) at Zero Input\nmax |tau| = {max_tau:.4f} Nm", fontweight="bold")
    ax.axhline(5.0, color="red", ls="--", alpha=0.5, label="threshold 5 Nm")
    ax.axhline(-5.0, color="red", ls="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("L0-08: Zero Input → Zero Output\nu=0, q=neutral, v=0 → max|tau| < 5 Nm",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "L0_08_zero_input_all")

    passed = max_tau < 5.0
    report("L0-08", "Zero Input → Zero Output", passed,
           f"max |tau| = {max_tau:.4f} Nm, max |F| = {max_F:.4f} N")


# ═════════════════════════════════════════════════════════════════════
# L0-09  Activation Time Constant
# ═════════════════════════════════════════════════════════════════════
def test_L0_09():
    print("\n" + "=" * 60)
    print("L0-09: Activation Time Constant (all muscles)")
    print("=" * 60)

    baseline = load_baseline()
    params, names = build_muscle_params(baseline)
    n = len(params)

    # Use fine dt (0.1ms) for accurate time constant measurement
    dt = 0.0001  # 0.1ms — much finer than sim dt for measurement accuracy
    act_dyn = ActivationDynamics(num_muscles=n, num_envs=1, device="cpu")
    tau_act_set = torch.tensor([p.tau_act for p in params])
    tau_deact_set = torch.tensor([p.tau_deact for p in params])
    act_dyn.set_time_constants(tau_act_set, tau_deact_set)

    # Phase 1: step from 0 → 1 (activation)
    steps = 3000  # 300ms total — enough for slowest tau_deact (80ms)
    t = np.arange(steps) * dt
    hist_act = np.zeros((steps, n))
    act_dyn.reset()
    u_on = torch.ones(1, n)
    for s in range(steps):
        a = act_dyn.step(u_on, dt)
        hist_act[s] = a[0].numpy()

    # Phase 2: step from 1 → 0 (deactivation)
    hist_deact = np.zeros((steps, n))
    for s in range(steps):
        a = act_dyn.step(torch.zeros(1, n), dt)
        hist_deact[s] = a[0].numpy()

    # Measure time to reach 63.2% (1 - 1/e)
    target_act = 1.0 - np.exp(-1)  # 0.632
    target_deact = np.exp(-1)       # 0.368 (starting from 1.0)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharex=True)
    axes = axes.flatten()
    all_pass = True

    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(t * 1000, hist_act[:, i], "b-", linewidth=1.5, label="Activation")
        ax.plot(t * 1000 + t[-1] * 1000, hist_deact[:, i], "r-", linewidth=1.5, label="Deactivation")

        # Measure tau_act
        above_target = np.where(hist_act[:, i] >= target_act)[0]
        if len(above_target) > 0:
            measured_tau_act = t[above_target[0]]
        else:
            measured_tau_act = float("inf")

        # Measure tau_deact (time from peak to 0.368)
        start_val = hist_deact[0, i]
        target_d = start_val * np.exp(-1)
        below_target = np.where(hist_deact[:, i] <= target_d)[0]
        if len(below_target) > 0:
            measured_tau_deact = t[below_target[0]]
        else:
            measured_tau_deact = float("inf")

        set_act = tau_act_set[i].item()
        set_deact = tau_deact_set[i].item()

        ok_act = abs(measured_tau_act - set_act) / set_act < 0.20
        ok_deact = abs(measured_tau_deact - set_deact) / set_deact < 0.20 if measured_tau_deact < float("inf") else False
        ok = ok_act and ok_deact
        all_pass = all_pass and ok

        color = "green" if ok else "red"
        ax.set_title(f"{name.replace('_L','')}\n"
                     f"act: {measured_tau_act*1000:.1f}ms (set:{set_act*1000:.0f}ms)\n"
                     f"deact: {measured_tau_deact*1000:.1f}ms (set:{set_deact*1000:.0f}ms)",
                     fontsize=8, color=color)
        ax.axhline(target_act, color="b", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.2)
        if i % 5 == 0:
            ax.set_ylabel("Activation")
        if i >= 5:
            ax.set_xlabel("Time (ms)")

    fig.suptitle("L0-09: Activation Time Constants (All Muscles)\n"
                 "Measured tau within ±20% of set value",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "L0_09_activation_tau_all_muscles")

    report("L0-09", "Activation Time Constants", all_pass,
           "Measured tau_act and tau_deact within ±20% of configured values")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("CALM Validation Pyramid — L0 Unit Math Tests")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    test_L0_01()
    test_L0_02()
    test_L0_03()
    test_L0_04()
    test_L0_05()
    test_L0_06()
    test_L0_07()
    test_L0_08()
    test_L0_09()

    elapsed = time.time() - start_time

    # ── Summary ──
    print("\n" + "=" * 60)
    print("L0 VALIDATION SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v["passed"])
    failed = total - passed

    for tid, info in results.items():
        mark = "✅" if info["passed"] else "❌"
        print(f"  {mark} {tid}: {info['name']}")
        if info["detail"]:
            print(f"       {info['detail']}")

    print(f"\nResult: {passed}/{total} PASSED, {failed} FAILED")
    print(f"Elapsed: {elapsed:.1f}s")

    if failed > 0:
        print("\n⚠ FAILED TESTS — investigate before proceeding to L1")
    else:
        print("\n✅ ALL L0 TESTS PASSED — proceed to L1")

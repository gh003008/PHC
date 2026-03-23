"""
L1 Module Flow Validation — CALM Validation Pyramid v2
=======================================================
Modules are connected with correct dimensions and signs.
No IsaacGym required; pure PyTorch.

Tests:
    L1-01  Tensor shape check (full pipeline)
    L1-02  Moment arm coupling (bi-articular muscles)
    L1-03  Activation pipeline range [0, 1]
    L1-04  Muscle velocity sign convention (CRITICAL)
    L1-05  Force-torque conversion numerical check
    L1-06  Numerical stability (extreme inputs, 1000 steps)
    L1-07  Reset consistency (partial env reset)

Usage:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python standard_human_model/validation/L1_module_flow/run_validation.py
"""

import sys, os, time, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.muscle_model import HillMuscleModel, MuscleParams
from standard_human_model.core.moment_arm import MomentArmMatrix
from standard_human_model.core.activation_dynamics import ActivationDynamics
from standard_human_model.core.reflex_controller import ReflexController, ReflexParams
from standard_human_model.core.skeleton import (
    JOINT_DOF_RANGE, NUM_DOFS, JOINT_NAMES, LOWER_LIMB_JOINTS,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))

results = {}
start_time = time.time()

def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

def report(tid, name, passed, detail=""):
    results[tid] = {"name": name, "passed": passed, "detail": detail}
    mark = "PASS" if passed else "FAIL"
    print(f"\n[{tid}] {name}  {'✅' if passed else '❌'} {mark}")
    if detail:
        print(f"     {detail}")

def make_body(num_envs=4):
    return HumanBody.from_config(
        muscle_def_path=os.path.join(CONFIG_DIR, "muscle_definitions.yaml"),
        param_path=os.path.join(CONFIG_DIR, "healthy_baseline.yaml"),
        num_envs=num_envs, device="cpu",
    )


# ═════════════════════════════════════════════════════════════════════
# L1-01  Tensor Shape Check
# ═════════════════════════════════════════════════════════════════════
def test_L1_01():
    print("\n" + "=" * 60)
    print("L1-01: Tensor Shape Check (Full Pipeline)")
    print("=" * 60)

    num_envs = 8
    body = make_body(num_envs)
    n_muscles = body.num_muscles

    dof_pos = torch.zeros(num_envs, NUM_DOFS)
    dof_vel = torch.zeros(num_envs, NUM_DOFS)
    desc_cmd = torch.ones(num_envs, n_muscles) * 0.3

    # Step through pipeline manually
    shapes = {}
    l_mtu = body.moment_arm.compute_muscle_length(dof_pos)
    shapes["muscle_length"] = tuple(l_mtu.shape)

    v_mtu = body.moment_arm.compute_muscle_velocity(dof_pos, dof_vel)
    shapes["muscle_velocity"] = tuple(v_mtu.shape)

    a_cmd = body.reflex.compute(desc_cmd, v_mtu, torch.zeros(num_envs, n_muscles))
    shapes["reflex_output"] = tuple(a_cmd.shape)

    activation = body.activation_dyn.step(a_cmd, 1/120)
    shapes["activation"] = tuple(activation.shape)

    F = body.muscle_model.compute_force(activation, l_mtu, v_mtu)
    shapes["muscle_force"] = tuple(F.shape)

    tau = body.moment_arm.forces_to_torques(F, dof_pos)
    shapes["joint_torque"] = tuple(tau.shape)

    tau_lig = body.ligament.compute_torque(dof_pos, dof_vel)
    shapes["ligament_torque"] = tuple(tau_lig.shape)

    # Full pipeline
    tau_total = body.compute_torques(dof_pos, dof_vel, desc_cmd, dt=1/120)
    shapes["total_torque"] = tuple(tau_total.shape)

    expected = {
        "muscle_length": (num_envs, n_muscles),
        "muscle_velocity": (num_envs, n_muscles),
        "reflex_output": (num_envs, n_muscles),
        "activation": (num_envs, n_muscles),
        "muscle_force": (num_envs, n_muscles),
        "joint_torque": (num_envs, NUM_DOFS),
        "ligament_torque": (num_envs, NUM_DOFS),
        "total_torque": (num_envs, NUM_DOFS),
    }

    all_pass = True
    detail_lines = []
    for key in expected:
        ok = shapes[key] == expected[key]
        all_pass = all_pass and ok
        status = "OK" if ok else "MISMATCH"
        line = f"{key}: {shapes[key]} (expected {expected[key]}) {status}"
        detail_lines.append(line)
        print(f"  {line}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(expected.keys())
    got = [str(shapes[k]) for k in labels]
    exp = [str(expected[k]) for k in labels]
    colors = ["green" if shapes[k] == expected[k] else "red" for k in labels]

    y = np.arange(len(labels))
    ax.barh(y, [1]*len(labels), color=colors, alpha=0.6, height=0.6)
    for i, (g, e) in enumerate(zip(got, exp)):
        ax.text(0.5, i, f"{labels[i]}: {g}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"L1-01: Tensor Shape Check\nnum_envs={num_envs}, num_muscles={n_muscles}, num_dofs={NUM_DOFS}",
                 fontweight="bold")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    save(fig, "L1_01_tensor_shapes")

    report("L1-01", "Tensor Shape Check", all_pass,
           f"{sum(1 for k in expected if shapes[k]==expected[k])}/{len(expected)} shapes correct")


# ═════════════════════════════════════════════════════════════════════
# L1-02  Moment Arm Coupling
# ═════════════════════════════════════════════════════════════════════
def test_L1_02():
    print("\n" + "=" * 60)
    print("L1-02: Moment Arm Coupling (Bi-articular Muscles)")
    print("=" * 60)

    body = make_body(1)
    coupling = body.moment_arm.get_coupling_info()

    # Expected bi-articular muscles
    expected_biart = {
        "rectus_femoris_L": ["L_Hip", "L_Knee"],
        "rectus_femoris_R": ["R_Hip", "R_Knee"],
        "hamstrings_L": ["L_Hip", "L_Knee"],
        "hamstrings_R": ["R_Hip", "R_Knee"],
        "gastrocnemius_L": ["L_Knee", "L_Ankle"],
        "gastrocnemius_R": ["R_Knee", "R_Ankle"],
    }

    all_pass = True
    details = []

    for muscle, expected_joints in expected_biart.items():
        actual_joints = coupling.get(muscle, [])
        ok = all(j in actual_joints for j in expected_joints) and len(actual_joints) >= 2
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        details.append(f"{muscle}: {actual_joints} (expected {expected_joints}) {status}")
        print(f"  {status}: {muscle} -> {actual_joints}")

    # Heatmap of R matrix
    R = body.moment_arm.R_const.numpy()
    muscle_names = body.moment_arm.muscle_names

    fig, ax = plt.subplots(figsize=(16, 8))
    # Only show sagittal DOFs for clarity
    sag_dof_indices = []
    sag_labels = []
    for jn in JOINT_NAMES:
        s, _ = JOINT_DOF_RANGE[jn]
        sag_dof_indices.append(s)
        sag_labels.append(jn)

    R_sag = R[:, sag_dof_indices]
    im = ax.imshow(R_sag, aspect="auto", cmap="RdBu_r", vmin=-0.08, vmax=0.08)
    ax.set_yticks(range(len(muscle_names)))
    ax.set_yticklabels([n.replace("_L", " L").replace("_R", " R") for n in muscle_names], fontsize=8)
    ax.set_xticks(range(len(sag_labels)))
    ax.set_xticklabels(sag_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("L1-02: Moment Arm Matrix R (Sagittal DOFs)\nBi-articular muscles span ≥2 joints",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Moment Arm (m)", shrink=0.8)

    # Annotate values
    for i in range(R_sag.shape[0]):
        for j in range(R_sag.shape[1]):
            val = R_sag[i, j]
            if abs(val) > 0.001:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                       color="white" if abs(val) > 0.04 else "black")

    fig.tight_layout()
    save(fig, "L1_02_moment_arm_coupling")

    report("L1-02", "Moment Arm Coupling", all_pass,
           f"6 bi-articular muscles verified: {'; '.join(details[:3])}")


# ═════════════════════════════════════════════════════════════════════
# L1-03  Activation Pipeline Range [0, 1]
# ═════════════════════════════════════════════════════════════════════
def test_L1_03():
    print("\n" + "=" * 60)
    print("L1-03: Activation Pipeline Range [0, 1]")
    print("=" * 60)

    body = make_body(16)
    n = body.num_muscles
    dt = 1/120

    # Run 200 steps with various extreme inputs
    all_pass = True
    violations = []

    # Test scenarios: extreme descending commands
    scenarios = [
        ("zeros", torch.zeros(16, n)),
        ("ones", torch.ones(16, n)),
        ("negative (clamp test)", torch.full((16, n), -0.5)),
        ("above 1 (clamp test)", torch.full((16, n), 1.5)),
        ("random", torch.rand(16, n)),
    ]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 3 * len(scenarios)), sharex=True)
    body.reset()

    for s_idx, (name, cmd) in enumerate(scenarios):
        ax = axes[s_idx]
        body.reset()
        dof_pos = torch.randn(16, NUM_DOFS) * 0.3
        dof_vel = torch.randn(16, NUM_DOFS) * 0.5

        hist = []
        for step in range(100):
            tau = body.compute_torques(dof_pos, dof_vel, cmd, dt)
            a = body.get_activation()
            hist.append(a.numpy().copy())

            if a.min() < -1e-6 or a.max() > 1.0 + 1e-6:
                violations.append(f"step {step}: min={a.min():.4f}, max={a.max():.4f}")
                all_pass = False

        hist = np.array(hist)  # (100, 16, n)
        # Plot mean ± std across envs for each muscle
        mean_a = hist.mean(axis=1)  # (100, n)
        for m in range(n):
            ax.plot(mean_a[:, m], linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="red", ls="--", alpha=0.3)
        ax.axhline(1, color="red", ls="--", alpha=0.3)
        ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel("Activation")
        ax.set_title(f"Scenario: {name}", fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step")
    fig.suptitle("L1-03: Activation Pipeline Range Check\nAll values must stay in [0, 1]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "L1_03_activation_range")

    report("L1-03", "Activation Pipeline Range", all_pass,
           f"Violations: {len(violations)}" if violations else "All activations in [0, 1] across 5 scenarios × 100 steps")


# ═════════════════════════════════════════════════════════════════════
# L1-04  Muscle Velocity Sign Convention (CRITICAL)
# ═════════════════════════════════════════════════════════════════════
def test_L1_04():
    print("\n" + "=" * 60)
    print("L1-04: Muscle Velocity Sign Convention (CRITICAL)")
    print("=" * 60)

    body = make_body(1)
    n = body.num_muscles
    marm = body.moment_arm

    # Test: knee flexion (dq > 0 at knee sagittal DOF)
    # Expected: hamstrings shorten (v < 0), quadriceps lengthen (v > 0)
    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)

    # L_Knee sagittal DOF: positive = flexion
    s_knee, _ = JOINT_DOF_RANGE["L_Knee"]
    dof_vel[0, s_knee] = 1.0  # knee flexing at 1 rad/s

    v_mtu = marm.compute_muscle_velocity(dof_pos, dof_vel)

    # Find muscle indices
    names = marm.muscle_names
    ham_idx = names.index("hamstrings_L")
    quad_idx = names.index("quadriceps_L")
    gastroc_idx = names.index("gastrocnemius_L")
    rf_idx = names.index("rectus_femoris_L")

    v_ham = v_mtu[0, ham_idx].item()
    v_quad = v_mtu[0, quad_idx].item()
    v_gastroc = v_mtu[0, gastroc_idx].item()
    v_rf = v_mtu[0, rf_idx].item()

    print(f"  Knee flexing (dq > 0):")
    print(f"    hamstrings_L velocity: {v_ham:.4f} m/s (expect < 0 = shortening)")
    print(f"    quadriceps_L velocity: {v_quad:.4f} m/s (expect > 0 = lengthening)")
    print(f"    gastrocnemius_L velocity: {v_gastroc:.4f} m/s (expect < 0 = shortening)")
    print(f"    rectus_femoris_L velocity: {v_rf:.4f} m/s (expect > 0 = lengthening)")

    ok_ham = v_ham < 0
    ok_quad = v_quad > 0
    ok_gastroc = v_gastroc < 0
    ok_rf = v_rf > 0

    # Also test hip flexion
    dof_vel2 = torch.zeros(1, NUM_DOFS)
    s_hip, _ = JOINT_DOF_RANGE["L_Hip"]
    dof_vel2[0, s_hip] = 1.0  # hip flexing

    v_mtu2 = marm.compute_muscle_velocity(dof_pos, dof_vel2)
    v_hipflex = v_mtu2[0, names.index("hip_flexors_L")].item()
    v_glutmax = v_mtu2[0, names.index("gluteus_max_L")].item()

    print(f"\n  Hip flexing (dq > 0):")
    print(f"    hip_flexors_L velocity: {v_hipflex:.4f} m/s (expect < 0 = shortening)")
    print(f"    gluteus_max_L velocity: {v_glutmax:.4f} m/s (expect > 0 = lengthening)")

    ok_hipflex = v_hipflex < 0
    ok_glutmax = v_glutmax > 0

    # Ankle dorsiflexion
    dof_vel3 = torch.zeros(1, NUM_DOFS)
    s_ankle, _ = JOINT_DOF_RANGE["L_Ankle"]
    dof_vel3[0, s_ankle] = 1.0  # ankle dorsiflexion

    v_mtu3 = marm.compute_muscle_velocity(dof_pos, dof_vel3)
    v_sol = v_mtu3[0, names.index("soleus_L")].item()
    v_ta = v_mtu3[0, names.index("tibialis_ant_L")].item()

    print(f"\n  Ankle dorsiflexing (dq > 0):")
    print(f"    soleus_L velocity: {v_sol:.4f} m/s (expect > 0 = lengthening)")
    print(f"    tibialis_ant_L velocity: {v_ta:.4f} m/s (expect < 0 = shortening)")

    ok_sol = v_sol > 0
    ok_ta = v_ta < 0

    all_pass = ok_ham and ok_quad and ok_gastroc and ok_rf and ok_hipflex and ok_glutmax and ok_sol and ok_ta

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Knee test
    ax = axes[0]
    test_muscles_knee = ["hamstrings_L", "quadriceps_L", "gastrocnemius_L", "rectus_femoris_L"]
    v_vals = [v_mtu[0, names.index(m)].item() for m in test_muscles_knee]
    expected_signs = ["< 0 (short)", "> 0 (long)", "< 0 (short)", "> 0 (long)"]
    colors = ["green" if (v < 0 and "short" in e) or (v > 0 and "long" in e) else "red"
              for v, e in zip(v_vals, expected_signs)]
    ax.barh(range(len(test_muscles_knee)), v_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(test_muscles_knee)))
    ax.set_yticklabels([m.replace("_L","") for m in test_muscles_knee], fontsize=9)
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("Muscle Velocity (m/s)")
    ax.set_title("Knee Flexion (dq > 0)", fontweight="bold")

    # Hip test
    ax = axes[1]
    test_muscles_hip = ["hip_flexors_L", "gluteus_max_L", "hamstrings_L", "rectus_femoris_L"]
    v_vals2 = [v_mtu2[0, names.index(m)].item() for m in test_muscles_hip]
    expected2 = ["< 0", "> 0", "> 0", "< 0"]
    colors2 = ["green" if (v < 0 and e == "< 0") or (v > 0 and e == "> 0") else "red"
               for v, e in zip(v_vals2, expected2)]
    ax.barh(range(len(test_muscles_hip)), v_vals2, color=colors2, alpha=0.8)
    ax.set_yticks(range(len(test_muscles_hip)))
    ax.set_yticklabels([m.replace("_L","") for m in test_muscles_hip], fontsize=9)
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("Muscle Velocity (m/s)")
    ax.set_title("Hip Flexion (dq > 0)", fontweight="bold")

    # Ankle test
    ax = axes[2]
    test_muscles_ankle = ["soleus_L", "gastrocnemius_L", "tibialis_ant_L"]
    v_vals3 = [v_mtu3[0, names.index(m)].item() for m in test_muscles_ankle]
    expected3 = ["> 0", "> 0", "< 0"]
    colors3 = ["green" if (v < 0 and e == "< 0") or (v > 0 and e == "> 0") else "red"
               for v, e in zip(v_vals3, expected3)]
    ax.barh(range(len(test_muscles_ankle)), v_vals3, color=colors3, alpha=0.8)
    ax.set_yticks(range(len(test_muscles_ankle)))
    ax.set_yticklabels([m.replace("_L","") for m in test_muscles_ankle], fontsize=9)
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("Muscle Velocity (m/s)")
    ax.set_title("Ankle Dorsiflexion (dq > 0)", fontweight="bold")

    fig.suptitle("L1-04: Muscle Velocity Sign Convention\nGreen=correct, Red=wrong",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L1_04_velocity_sign")

    report("L1-04", "Muscle Velocity Sign", all_pass,
           "All 8 muscle-joint pairs have correct velocity sign convention")


# ═════════════════════════════════════════════════════════════════════
# L1-05  Force-Torque Conversion Numerical
# ═════════════════════════════════════════════════════════════════════
def test_L1_05():
    print("\n" + "=" * 60)
    print("L1-05: Force-Torque Conversion (Hand Calculation)")
    print("=" * 60)

    body = make_body(1)
    marm = body.moment_arm
    names = marm.muscle_names

    # Test: single muscle force → joint torque
    # hamstrings_L: R_hip=-0.06, R_knee=+0.03
    ham_idx = names.index("hamstrings_L")
    s_hip, _ = JOINT_DOF_RANGE["L_Hip"]
    s_knee, _ = JOINT_DOF_RANGE["L_Knee"]

    F = torch.zeros(1, len(names))
    F[0, ham_idx] = 1000.0  # 1000 N

    # At dof_pos=0, R is constant (polynomial a0 values)
    dof_pos = torch.zeros(1, NUM_DOFS)
    tau = marm.forces_to_torques(F, dof_pos)

    # Expected (from polynomial a0 at q=0):
    # R_hip = -0.060, R_knee = 0.030
    # tau_hip = F * R_hip = 1000 * (-0.060) = -60 Nm
    # tau_knee = F * R_knee = 1000 * 0.030 = 30 Nm
    tau_hip = tau[0, s_hip].item()
    tau_knee = tau[0, s_knee].item()

    expected_hip = 1000.0 * (-0.060)
    expected_knee = 1000.0 * 0.030

    err_hip = abs(tau_hip - expected_hip)
    err_knee = abs(tau_knee - expected_knee)

    print(f"  hamstrings_L F=1000N:")
    print(f"    tau_hip  = {tau_hip:.4f} Nm (expected {expected_hip:.1f}, error={err_hip:.6f})")
    print(f"    tau_knee = {tau_knee:.4f} Nm (expected {expected_knee:.1f}, error={err_knee:.6f})")

    # Test with non-zero joint angle (polynomial R)
    dof_pos2 = torch.zeros(1, NUM_DOFS)
    dof_pos2[0, s_knee] = 0.5  # 30 deg knee flexion
    tau2 = marm.forces_to_torques(F, dof_pos2)
    tau_knee2 = tau2[0, s_knee].item()
    # Expected: R_knee(q=0.5) = 0.030 + 0.015*0.5 + (-0.005)*0.25 = 0.030 + 0.0075 - 0.00125 = 0.03625
    expected_knee2 = 1000.0 * (0.030 + 0.015 * 0.5 + (-0.005) * 0.5**2)
    err_knee2 = abs(tau_knee2 - expected_knee2)
    print(f"\n  At knee=0.5 rad (polynomial R):")
    print(f"    tau_knee = {tau_knee2:.4f} Nm (expected {expected_knee2:.1f}, error={err_knee2:.6f})")

    all_pass = err_hip < 1e-4 and err_knee < 1e-4 and err_knee2 < 1e-4

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All muscles single force test
    ax = axes[0]
    all_muscles_tau = []
    for m_idx, m_name in enumerate(names):
        F_single = torch.zeros(1, len(names))
        F_single[0, m_idx] = 1000.0
        tau_single = marm.forces_to_torques(F_single, torch.zeros(1, NUM_DOFS))

        sag_taus = []
        for jn in LOWER_LIMB_JOINTS:
            s, _ = JOINT_DOF_RANGE[jn]
            sag_taus.append(tau_single[0, s].item())
        all_muscles_tau.append(sag_taus)

    all_muscles_tau = np.array(all_muscles_tau)  # (20, 8)
    im = ax.imshow(all_muscles_tau, aspect="auto", cmap="RdBu_r", vmin=-80, vmax=80)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_L", " L").replace("_R", " R") for n in names], fontsize=7)
    ax.set_xticks(range(len(LOWER_LIMB_JOINTS)))
    ax.set_xticklabels(LOWER_LIMB_JOINTS, rotation=45, ha="right", fontsize=8)
    ax.set_title("Joint Torque per 1000N Muscle Force\n(Sagittal DOFs)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Torque (Nm)", shrink=0.8)
    for i in range(all_muscles_tau.shape[0]):
        for j in range(all_muscles_tau.shape[1]):
            v = all_muscles_tau[i, j]
            if abs(v) > 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                       color="white" if abs(v) > 40 else "black")

    # Polynomial R effect
    ax = axes[1]
    q_range = np.linspace(-0.5, 1.5, 50)
    for m_name, j_name, label in [
        ("hamstrings_L", "L_Knee", "Hamstrings @ Knee"),
        ("quadriceps_L", "L_Knee", "Quadriceps @ Knee"),
        ("soleus_L", "L_Ankle", "Soleus @ Ankle"),
    ]:
        m_idx = names.index(m_name)
        s, _ = JOINT_DOF_RANGE[j_name]
        taus = []
        for q in q_range:
            F_t = torch.zeros(1, len(names))
            F_t[0, m_idx] = 1000.0
            dp = torch.zeros(1, NUM_DOFS)
            dp[0, s] = q
            t = marm.forces_to_torques(F_t, dp)[0, s].item()
            taus.append(t)
        ax.plot(np.degrees(q_range), taus, linewidth=2, label=label)

    ax.set_xlabel("Joint Angle (deg)")
    ax.set_ylabel("Torque (Nm) per 1000N")
    ax.set_title("Polynomial R(q) Effect\nTorque varies with joint angle", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("L1-05: Force-Torque Conversion Verification\nHand-calc error < 1e-4 Nm",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "L1_05_force_torque_conversion")

    report("L1-05", "Force-Torque Conversion", all_pass,
           f"max error = {max(err_hip, err_knee, err_knee2):.2e} Nm (threshold < 1e-4)")


# ═════════════════════════════════════════════════════════════════════
# L1-06  Numerical Stability
# ═════════════════════════════════════════════════════════════════════
def test_L1_06():
    print("\n" + "=" * 60)
    print("L1-06: Numerical Stability (Extreme Inputs, 1000 Steps)")
    print("=" * 60)

    body = make_body(8)
    n = body.num_muscles
    dt = 1/120

    body.reset()
    nan_count = 0
    inf_count = 0
    max_tau = 0

    for step in range(1000):
        # Extreme inputs
        dof_pos = torch.randn(8, NUM_DOFS) * 2.0  # wide range angles
        dof_vel = torch.randn(8, NUM_DOFS) * 10.0  # fast velocities
        cmd = torch.ones(8, n)  # full activation

        tau = body.compute_torques(dof_pos, dof_vel, cmd, dt)

        if torch.isnan(tau).any():
            nan_count += 1
        if torch.isinf(tau).any():
            inf_count += 1
        max_tau = max(max_tau, tau.abs().max().item())

    all_pass = nan_count == 0 and inf_count == 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5,
            f"1000 Steps × 8 Envs\n"
            f"dof_pos ~ N(0, 2.0)\n"
            f"dof_vel ~ N(0, 10.0)\n"
            f"cmd = 1.0 (full activation)\n\n"
            f"NaN count: {nan_count}\n"
            f"Inf count: {inf_count}\n"
            f"Max |torque|: {max_tau:.1f} Nm\n\n"
            f"{'✅ STABLE' if all_pass else '❌ UNSTABLE'}",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightgreen" if all_pass else "lightcoral", alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("L1-06: Numerical Stability Under Extreme Inputs", fontweight="bold")
    fig.tight_layout()
    save(fig, "L1_06_numerical_stability")

    report("L1-06", "Numerical Stability", all_pass,
           f"NaN={nan_count}, Inf={inf_count}, max|tau|={max_tau:.1f} Nm over 1000 steps")


# ═════════════════════════════════════════════════════════════════════
# L1-07  Reset Consistency
# ═════════════════════════════════════════════════════════════════════
def test_L1_07():
    print("\n" + "=" * 60)
    print("L1-07: Reset Consistency (Partial Env Reset)")
    print("=" * 60)

    body = make_body(8)
    n = body.num_muscles
    dt = 1/120

    body.reset()

    # Run 50 steps to build up activation state
    cmd = torch.rand(8, n)
    dof_pos = torch.randn(8, NUM_DOFS) * 0.3
    dof_vel = torch.randn(8, NUM_DOFS) * 0.5

    for _ in range(50):
        body.compute_torques(dof_pos, dof_vel, cmd, dt)

    # Record activation of env 3,5,7 before reset
    a_before = body.get_activation().clone()
    a_env2_before = a_before[2].clone()
    a_env4_before = a_before[4].clone()

    # Reset only envs 0,1,6
    reset_ids = torch.tensor([0, 1, 6])
    body.reset(reset_ids)

    a_after = body.get_activation()

    # Check: reset envs should be zero
    reset_zero = (a_after[reset_ids].abs().max().item() < 1e-8)

    # Check: non-reset envs should be unchanged
    keep_ids = [2, 3, 4, 5, 7]
    max_diff = 0
    for kid in keep_ids:
        diff = (a_after[kid] - a_before[kid]).abs().max().item()
        max_diff = max(max_diff, diff)

    keep_unchanged = max_diff < 1e-8

    all_pass = reset_zero and keep_unchanged

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before/after comparison
    ax = axes[0]
    ax.imshow(a_before.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Before Reset", fontweight="bold")
    ax.set_ylabel("Env ID")
    ax.set_xlabel("Muscle Index")
    for r_id in [0, 1, 6]:
        ax.axhline(r_id, color="red", ls="--", alpha=0.5)
    ax.text(n * 0.7, 0.5, "← reset", color="red", fontsize=9)

    ax = axes[1]
    ax.imshow(a_after.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("After Reset (envs 0,1,6)", fontweight="bold")
    ax.set_ylabel("Env ID")
    ax.set_xlabel("Muscle Index")
    for r_id in [0, 1, 6]:
        ax.axhline(r_id, color="red", ls="--", alpha=0.5)

    fig.suptitle(f"L1-07: Partial Reset Consistency\n"
                 f"Reset envs zeroed: {reset_zero}, Non-reset unchanged: {keep_unchanged} (max diff={max_diff:.2e})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "L1_07_reset_consistency")

    report("L1-07", "Reset Consistency", all_pass,
           f"Reset envs zeroed: {reset_zero}, non-reset max diff: {max_diff:.2e}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("CALM Validation Pyramid — L1 Module Flow Tests")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    test_L1_01()
    test_L1_02()
    test_L1_03()
    test_L1_04()
    test_L1_05()
    test_L1_06()
    test_L1_07()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("L1 VALIDATION SUMMARY")
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
        print("\n⚠ FAILED TESTS — investigate before proceeding to L2")
    else:
        print("\n✅ ALL L1 TESTS PASSED — proceed to L2")
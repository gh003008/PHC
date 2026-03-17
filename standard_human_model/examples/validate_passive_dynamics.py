"""
검증 실험 1~5: 근골격 파이프라인 수동/반사 역학 검증.

IsaacGym 없이 CPU에서 semi-implicit Euler 적분으로 forward dynamics 시뮬레이션.
각 테스트는 결과를 콘솔 출력 + matplotlib 플롯으로 저장.

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python -m standard_human_model.examples.validate_passive_dynamics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import NUM_DOFS, JOINT_DOF_RANGE

# 출력 디렉토리
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "validation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 유틸: 단일 DOF forward dynamics (semi-implicit Euler)
# =============================================================================
def simulate_1dof(body, joint_name, axis, init_angle, init_vel,
                  descending_cmd, dt, num_steps, inertia,
                  gravity_torque_fn=None, vel_clamp=20.0):
    """단일 DOF에 대해 forward dynamics semi-implicit Euler 적분.

    Semi-implicit Euler: v(n+1) = v(n) + a*dt, q(n+1) = q(n) + v(n+1)*dt
    → explicit Euler보다 에너지 보존이 좋아 발산 억제.

    Args:
        body: HumanBody 인스턴스
        joint_name: 관절 이름
        axis: 축 인덱스 (0=x)
        init_angle: 초기 각도 (rad)
        init_vel: 초기 각속도 (rad/s)
        descending_cmd: (1, num_muscles) tensor, 또는 callable(step) → tensor
        dt: 시간 간격 (s)
        num_steps: 시뮬레이션 스텝 수
        inertia: 등가 관성 모멘트 (kg·m²)
        gravity_torque_fn: callable(angle) → scalar, 중력 토크
        vel_clamp: 최대 각속도 제한 (rad/s)
    """
    dof_start, _ = JOINT_DOF_RANGE[joint_name]
    dof_idx = dof_start + axis

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    dof_pos[0, dof_idx] = init_angle
    dof_vel[0, dof_idx] = init_vel

    body.reset()

    times, angles, velocities, torques_log = [], [], [], []

    for step in range(num_steps):
        t = step * dt
        times.append(t)
        q = dof_pos[0, dof_idx].item()
        dq = dof_vel[0, dof_idx].item()
        angles.append(q)
        velocities.append(dq)

        # 상위 명령
        if callable(descending_cmd):
            cmd = descending_cmd(step)
        else:
            cmd = descending_cmd

        # 근골격 토크
        tau = body.compute_torques(dof_pos, dof_vel, cmd, dt=dt)
        tau_ms = tau[0, dof_idx].item()

        # 중력 토크
        tau_grav = gravity_torque_fn(q) if gravity_torque_fn is not None else 0.0

        tau_total = tau_ms + tau_grav
        torques_log.append(tau_total)

        # Semi-implicit Euler
        ddq = tau_total / inertia
        new_vel = dq + ddq * dt
        new_vel = max(-vel_clamp, min(vel_clamp, new_vel))  # clamp
        new_pos = q + new_vel * dt

        dof_pos[0, dof_idx] = new_pos
        dof_vel[0, dof_idx] = new_vel

    return {
        "time": np.array(times),
        "angle": np.array(angles),
        "velocity": np.array(velocities),
        "torque": np.array(torques_log),
    }


# =============================================================================
# Test 1: Pendulum Test — 단일 관절 자유 진동
# =============================================================================
def test_pendulum():
    """무릎 관절을 초기 각도에서 놓고 중력 하 자유 진동 관찰.

    중력 토크: τ_grav = -m*g*L_com*sin(θ)
    (하퇴+발: m≈4.5kg, L_com≈0.25m)

    검증 항목:
    - 중력 진자에 수동 역학이 더해져 감쇠 진동
    - 경직 → 더 빠른 감쇠 (velocity-dependent resistance)
    """
    print("=" * 70)
    print("Test 1: Pendulum Test — 단일 관절 자유 진동 (중력 포함)")
    print("=" * 70)

    dt = 1.0 / 240  # 240 Hz (안정성)
    num_steps = 1200  # 5초
    inertia = 0.35    # 하퇴+발 (kg·m²)
    m_leg = 4.5       # 하퇴+발 질량 (kg)
    L_com = 0.25      # 질량 중심까지 거리 (m)
    g = 9.81
    init_angle = 1.2  # ~69도 굴곡

    def gravity_torque(angle):
        # 무릎 굴곡 양의 방향 → 중력은 복원력 (음방향)
        return -m_leg * g * L_com * np.sin(angle)

    # --- 정상인 ---
    body_healthy = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )
    cmd_zero = torch.zeros(1, body_healthy.num_muscles)
    result_healthy = simulate_1dof(
        body_healthy, "L_Knee", 0, init_angle, 0.0,
        cmd_zero, dt, num_steps, inertia,
        gravity_torque_fn=gravity_torque,
    )

    # --- 경직 (stretch_gain 5배, threshold 30%) ---
    body_spastic = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )
    body_spastic.reflex.stretch_gain *= 5.0
    body_spastic.reflex.stretch_threshold *= 0.3
    cmd_zero_sp = torch.zeros(1, body_spastic.num_muscles)
    result_spastic = simulate_1dof(
        body_spastic, "L_Knee", 0, init_angle, 0.0,
        cmd_zero_sp, dt, num_steps, inertia,
        gravity_torque_fn=gravity_torque,
    )

    # --- 이완 (reflex OFF, damping만) ---
    body_flaccid = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )
    body_flaccid.reflex.stretch_gain *= 0.0
    cmd_zero_fl = torch.zeros(1, body_flaccid.num_muscles)
    result_flaccid = simulate_1dof(
        body_flaccid, "L_Knee", 0, init_angle, 0.0,
        cmd_zero_fl, dt, num_steps, inertia,
        gravity_torque_fn=gravity_torque,
    )

    # --- 플롯 ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for res, label, style in [
        (result_flaccid, "Flaccid (reflex OFF)", "g-."),
        (result_healthy, "Healthy (normal reflex)", "b-"),
        (result_spastic, "Spastic (5x stretch gain)", "r--"),
    ]:
        axes[0].plot(res["time"], np.degrees(res["angle"]), style, label=label, linewidth=2)
        axes[1].plot(res["time"], res["velocity"], style, label=label, linewidth=2)
        axes[2].plot(res["time"], res["torque"], style, label=label, linewidth=2)

    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Pendulum Test: L_Knee Free Oscillation under Gravity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_pendulum_test.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"초기 각도: {np.degrees(init_angle):.1f} deg")
    for name, res in [("Flaccid", result_flaccid), ("Healthy", result_healthy), ("Spastic", result_spastic)]:
        final_ang = np.degrees(res["angle"][-1])
        peak_vel = np.max(np.abs(res["velocity"]))
        print(f"  {name:>10} — 최종 각도: {final_ang:>7.2f} deg, 최대 속도: {peak_vel:.2f} rad/s")
    print(f"플롯 저장: {path}")


# =============================================================================
# Test 2: Passive ROM Test — 관절 수동 이동 시 저항 곡선
# =============================================================================
def test_passive_rom():
    """외력으로 무릎을 0→140도 천천히 이동, 수동 저항 토크 측정.

    정적(quasi-static) 테스트: 각 각도에서 정지 상태의 수동 토크
    + 여러 속도에서의 속도 의존 저항.

    검증 항목:
    - J-shape 수동 저항 곡선
    - ROM 끝에서 ligament 급증
    - 속도 높을수록 저항 증가 (viscous + reflex)
    """
    print("\n" + "=" * 70)
    print("Test 2: Passive ROM Test — 수동 저항 곡선")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )

    dof_start, _ = JOINT_DOF_RANGE["L_Knee"]
    dof_x = dof_start

    angles_rad = np.linspace(0, 2.5, 200)
    speeds = [0.0, 0.5, 1.0, 2.0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for speed in speeds:
        total_torques = []

        for angle in angles_rad:
            body.reset()
            dof_pos = torch.zeros(1, NUM_DOFS)
            dof_vel = torch.zeros(1, NUM_DOFS)
            dof_pos[0, dof_x] = angle
            dof_vel[0, dof_x] = speed

            cmd = torch.zeros(1, body.num_muscles)
            tau_total = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/120)
            total_torques.append(tau_total[0, dof_x].item())

        label = f"v={speed:.1f} rad/s"
        axes[0].plot(np.degrees(angles_rad), total_torques, label=label, linewidth=1.5)

    # 분리 플롯: v=0 (정적)에서 근육 수동 vs 인대
    lig_static, mus_static = [], []
    for angle in angles_rad:
        body.reset()
        dof_pos = torch.zeros(1, NUM_DOFS)
        dof_vel = torch.zeros(1, NUM_DOFS)
        dof_pos[0, dof_x] = angle
        cmd = torch.zeros(1, body.num_muscles)
        tau_lig = body.ligament.compute_torque(dof_pos, dof_vel)
        tau_total = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/120)
        lig_static.append(tau_lig[0, dof_x].item())
        mus_static.append(tau_total[0, dof_x].item() - tau_lig[0, dof_x].item())

    axes[1].plot(np.degrees(angles_rad), mus_static, "b-", label="Muscle passive", linewidth=2)
    axes[1].plot(np.degrees(angles_rad), lig_static, "r--", label="Ligament", linewidth=2)
    axes[1].plot(np.degrees(angles_rad), np.array(mus_static) + np.array(lig_static),
                 "k-", label="Total (static)", linewidth=1.5, alpha=0.7)

    axes[0].set_ylabel("Total Passive Torque (Nm)")
    axes[0].set_title("Passive ROM Test: L_Knee Resistance vs Angle at Different Speeds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].set_ylabel("Component Torques (Nm)")
    axes[1].set_xlabel("Knee Angle (deg)")
    axes[1].set_title("Static (v=0): Muscle Passive vs Ligament Contribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_passive_rom_test.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"각도 범위: 0 ~ {np.degrees(angles_rad[-1]):.0f} deg")
    print(f"속도 조건: {speeds} rad/s")
    print(f"플롯 저장: {path}")


# =============================================================================
# Test 3: Perturbation Response — 순간 외란 + 반사 반응
# =============================================================================
def test_perturbation_response():
    """정립(0도)에서 순간 속도 외란 → stretch reflex 자동 저항.

    검증 항목:
    - reflex OFF vs ON vs 경직에서 최대 변위 차이
    - 더 강한 reflex → 더 작은 변위 (더 빠른 복원)
    """
    print("\n" + "=" * 70)
    print("Test 3: Perturbation Response — 외란 + 반사")
    print("=" * 70)

    dt = 1.0 / 240
    num_steps = 720  # 3초
    perturbation_vel = 2.0

    foot_inertia = 0.08
    foot_m = 1.0
    foot_L = 0.10

    def gravity_ankle(angle):
        return -foot_m * 9.81 * foot_L * np.sin(angle)

    configs = [
        ("Reflex OFF", {"gain_mult": 0.0, "thresh_mult": 1.0}, "g-."),
        ("Normal Reflex", {"gain_mult": 1.0, "thresh_mult": 1.0}, "b-"),
        ("Spastic (5x)", {"gain_mult": 5.0, "thresh_mult": 0.3}, "r--"),
    ]

    results = {}
    for name, cfg, _ in configs:
        body = HumanBody.from_config(
            muscle_def_path="muscle_definitions.yaml",
            param_path="healthy_baseline.yaml",
            num_envs=1, device="cpu",
        )
        body.reflex.stretch_gain *= cfg["gain_mult"]
        body.reflex.stretch_threshold *= cfg["thresh_mult"]

        cmd = torch.zeros(1, body.num_muscles)
        result = simulate_1dof(
            body, "L_Ankle", 0, 0.0, perturbation_vel,
            cmd, dt, num_steps, foot_inertia,
            gravity_torque_fn=gravity_ankle, vel_clamp=15.0,
        )
        results[name] = result

    # --- 플롯 ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for (name, res), (_, _, style) in zip(results.items(), configs):
        axes[0].plot(res["time"], np.degrees(res["angle"]), style, label=name, linewidth=2)
        axes[1].plot(res["time"], res["velocity"], style, label=name, linewidth=2)
        axes[2].plot(res["time"], res["torque"], style, label=name, linewidth=2)

    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title(f"Perturbation Response: L_Ankle (v0={perturbation_vel} rad/s)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_perturbation_response.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"외란 속도: {perturbation_vel} rad/s")
    for name, res in results.items():
        peak = np.max(np.abs(res["angle"]))
        print(f"  {name:>15} — 최대 변위: {np.degrees(peak):.1f} deg")
    print(f"플롯 저장: {path}")


# =============================================================================
# Test 4: Co-contraction — 길항근 동시 활성화 → 관절 강성
# =============================================================================
def test_cocontraction():
    """길항근 동시 활성화 수준별로 외란 응답 비교.

    검증 항목:
    - CC level 증가 → 최대 변위 감소
    """
    print("\n" + "=" * 70)
    print("Test 4: Co-contraction → 관절 강성 검증")
    print("=" * 70)

    dt = 1.0 / 240
    num_steps = 720  # 3초
    init_vel = 1.5

    body_ref = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1, device="cpu",
    )
    muscle_names = body_ref.moment_arm.muscle_names
    quad_idx = muscle_names.index("quadriceps_L")
    ham_idx = muscle_names.index("hamstrings_L")

    cocontraction_levels = [0.0, 0.1, 0.3, 0.5, 0.8]
    results = {}

    for cc_level in cocontraction_levels:
        body = HumanBody.from_config(
            muscle_def_path="muscle_definitions.yaml",
            param_path="healthy_baseline.yaml",
            num_envs=1, device="cpu",
        )
        cmd = torch.zeros(1, body.num_muscles)
        cmd[0, quad_idx] = cc_level
        cmd[0, ham_idx] = cc_level

        # co-contraction 안정화 (50 steps)
        dof_pos_init = torch.zeros(1, NUM_DOFS)
        dof_vel_init = torch.zeros(1, NUM_DOFS)
        for _ in range(50):
            body.compute_torques(dof_pos_init, dof_vel_init, cmd, dt=dt)

        result = simulate_1dof(
            body, "L_Knee", 0, 0.0, init_vel,
            cmd, dt, num_steps, inertia=0.35, vel_clamp=10.0,
        )
        results[cc_level] = result

    # --- 플롯 ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cocontraction_levels)))

    for (cc, res), color in zip(results.items(), colors):
        label = f"CC={cc:.0%}"
        axes[0].plot(res["time"], np.degrees(res["angle"]),
                     color=color, label=label, linewidth=2)
        axes[1].plot(res["time"], res["torque"],
                     color=color, label=label, linewidth=2)

    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title(f"Co-contraction Test: L_Knee (perturbation v0={init_vel} rad/s)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_cocontraction_test.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"외란 속도: {init_vel} rad/s")
    print(f"{'CC Level':>10}  {'Peak Angle (deg)':>18}  {'Reduction':>12}")
    print("-" * 45)
    peak_ref = np.max(np.abs(results[0.0]["angle"]))
    for cc, res in results.items():
        peak = np.max(np.abs(res["angle"]))
        reduction = (1.0 - peak / peak_ref) * 100 if peak_ref > 0 else 0
        print(f"{cc:>10.0%}  {np.degrees(peak):>18.2f}  {reduction:>11.1f}%")
    print(f"플롯 저장: {path}")


# =============================================================================
# Test 5: Bi-articular Coupling — 이관절근 토크 비율 검증
# =============================================================================
def test_biarticular_coupling():
    """이관절근 활성화에 의한 토크 *증분*이 moment arm 비율과 일치하는지.

    방법: activation=0 토크를 baseline으로 빼서, 순수 active 기여분만 비교.

    검증 항목:
    - hamstrings: |Δτ_hip| / |Δτ_knee| ≈ 0.06/0.03 = 2.0
    - gastrocnemius: |Δτ_knee| / |Δτ_ankle| ≈ 0.02/0.05 = 0.4
    - rectus_femoris: |Δτ_hip| / |Δτ_knee| ≈ 0.03/0.04 = 0.75
    """
    print("\n" + "=" * 70)
    print("Test 5: Bi-articular Coupling — 토크 비율 검증")
    print("=" * 70)

    biarticular_tests = [
        {
            "muscle": "hamstrings_L",
            "joints": [("L_Hip", 0), ("L_Knee", 0)],
            "expected_ratio": 0.06 / 0.03,
            "description": "Hamstrings: |Hip ext| / |Knee flex|",
        },
        {
            "muscle": "gastrocnemius_L",
            "joints": [("L_Knee", 0), ("L_Ankle", 0)],
            "expected_ratio": 0.02 / 0.05,
            "description": "Gastrocnemius: |Knee flex| / |Ankle PF|",
        },
        {
            "muscle": "rectus_femoris_L",
            "joints": [("L_Hip", 0), ("L_Knee", 0)],
            "expected_ratio": 0.03 / 0.04,
            "description": "Rectus Femoris: |Hip flex| / |Knee ext|",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    print(f"{'Muscle':>22}  {'dJ1 Torque':>12}  {'dJ2 Torque':>12}  "
          f"{'Ratio':>8}  {'Expected':>8}  {'Error':>8}")
    print("-" * 80)

    for i, test in enumerate(biarticular_tests):
        body = HumanBody.from_config(
            muscle_def_path="muscle_definitions.yaml",
            param_path="healthy_baseline.yaml",
            num_envs=1, device="cpu",
        )
        muscle_names = body.moment_arm.muscle_names
        m_idx = muscle_names.index(test["muscle"])

        j1_name, j1_axis = test["joints"][0]
        j2_name, j2_axis = test["joints"][1]
        j1_dof = JOINT_DOF_RANGE[j1_name][0] + j1_axis
        j2_dof = JOINT_DOF_RANGE[j2_name][0] + j2_axis

        act_levels = np.linspace(0.0, 1.0, 21)
        delta_j1, delta_j2 = [], []

        # baseline: activation=0에서의 토크 (수동 기여분)
        body.reset()
        cmd_zero = torch.zeros(1, body.num_muscles)
        dof_pos = torch.zeros(1, NUM_DOFS)
        dof_vel = torch.zeros(1, NUM_DOFS)
        for _ in range(15):
            tau_base = body.compute_torques(dof_pos, dof_vel, cmd_zero, dt=1/120)
        base_j1 = tau_base[0, j1_dof].item()
        base_j2 = tau_base[0, j2_dof].item()

        for act in act_levels:
            body.reset()
            cmd = torch.zeros(1, body.num_muscles)
            cmd[0, m_idx] = act

            for _ in range(15):
                tau = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/120)

            delta_j1.append(abs(tau[0, j1_dof].item() - base_j1))
            delta_j2.append(abs(tau[0, j2_dof].item() - base_j2))

        delta_j1 = np.array(delta_j1)
        delta_j2 = np.array(delta_j2)

        # activation=1.0에서의 비율
        if delta_j2[-1] > 1e-6:
            actual_ratio = delta_j1[-1] / delta_j2[-1]
        else:
            actual_ratio = float("inf")
        error_pct = abs(actual_ratio - test["expected_ratio"]) / test["expected_ratio"] * 100

        print(f"{test['muscle']:>22}  {delta_j1[-1]:>12.2f}  {delta_j2[-1]:>12.2f}  "
              f"{actual_ratio:>8.3f}  {test['expected_ratio']:>8.3f}  {error_pct:>7.1f}%")

        # 플롯
        axes[i].plot(act_levels, delta_j1, "b-o", label=f"|dt| {j1_name}", markersize=3)
        axes[i].plot(act_levels, delta_j2, "r-s", label=f"|dt| {j2_name}", markersize=3)
        axes[i].set_xlabel("Muscle Activation")
        axes[i].set_ylabel("|dTorque| (Nm)")
        axes[i].set_title(f"{test['muscle']}\nratio={actual_ratio:.3f} (exp={test['expected_ratio']:.3f})")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Bi-articular Coupling: Active Torque Increment at Two Joints", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_biarticular_coupling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"플롯 저장: {path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Standard Human Model — 수동/반사 역학 검증 실험")
    print(f"결과 저장 경로: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    test_pendulum()
    test_passive_rom()
    test_perturbation_response()
    test_cocontraction()
    test_biarticular_coupling()

    print("\n" + "=" * 70)
    print("모든 검증 실험 완료!")
    print(f"결과 플롯: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

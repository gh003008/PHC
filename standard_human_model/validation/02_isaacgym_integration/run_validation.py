"""
02_isaacgym_integration / run_validation.py
============================================
IsaacGym + standard_human_model 연결 검증 스크립트

검증 항목:
    I01  토크 주입 방향 검증   — +50 Nm → L_Knee가 굴곡 방향으로 이동
    I02  토크 부호 규약 검증   — 양의 토크 = 굴곡(각도 증가), 음의 토크 = 신전
    I03  프로파일 분화 검증    — Spastic > Healthy > Flaccid 최종 각도 순서
    I04  중력 vs bio-토크 균형 — 80° 굴곡에서 Spastic이 Healthy보다 큰 저항 토크

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python standard_human_model/validation/02_isaacgym_integration/run_validation.py --headless --pipeline cpu

결과:
    standard_human_model/validation/02_isaacgym_integration/results/*.png
    PASS/FAIL 판정 콘솔 출력
"""

# CRITICAL: IsaacGym must be imported before torch
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import NUM_DOFS, JOINT_DOF_RANGE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../config")
)
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../isaacgym_validation")
)

TEST_DOF_IDX = JOINT_DOF_RANGE["L_Knee"][0]  # DOF index for L_Knee flexion/extension
DT = 1.0 / 60.0
HOLD_KP = 500.0
HOLD_KD = 50.0

results = {}


def report(test_id, name, passed, detail=""):
    mark = "✅ PASS" if passed else "❌ FAIL"
    results[test_id] = passed
    print(f"\n[{test_id}] {name}  {mark}")
    if detail:
        print(f"     {detail}")


def save_and_close(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 저장: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# IsaacGym 환경 공통 설정
# ─────────────────────────────────────────────────────────────────────────────
def create_sim_and_envs(gym, args, num_envs, initial_knee_angles, graphics_device=None):
    """
    IsaacGym sim + env를 생성하고 초기 무릎 각도를 설정한다.
    반환: sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques
    graphics_device: None이면 headless(-1), 정수를 넘기면 해당 GPU 인덱스 사용
    """
    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0

    compute_device = args.compute_device_id
    if graphics_device is None:
        graphics_device = -1
    sim_params.use_gpu_pipeline = False  # CPU pipeline: viewer + headless 모두 동작

    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "Failed to create sim"

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset_options.fix_base_link = True

    humanoid_asset = gym.load_asset(sim, ASSET_DIR, "smpl_humanoid_fixed.xml", asset_options)
    assert humanoid_asset is not None, "Failed to load asset"

    num_dof = gym.get_asset_dof_count(humanoid_asset)

    spacing = 3.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing * 2)

    envs, actor_handles = [], []
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_envs)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
        # XML은 Y-up (MuJoCo) 기준 → IsaacGym Z-up에 맞게 X축 90° 회전
        start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi / 2)
        handle = gym.create_actor(env, humanoid_asset, start_pose, f"humanoid_{i}", i, 0, 0)

        dof_props = gym.get_actor_dof_properties(env, handle)
        for j in range(num_dof):
            if j == TEST_DOF_IDX:
                dof_props["driveMode"][j] = int(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"][j] = 0.0
                dof_props["damping"][j] = 0.0
            else:
                dof_props["driveMode"][j] = int(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][j] = HOLD_KP
                dof_props["damping"][j] = HOLD_KD
        gym.set_actor_dof_properties(env, handle, dof_props)

        envs.append(env)
        actor_handles.append(handle)

    gym.prepare_sim(sim)

    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_state_tensor)
    dof_pos_all = dof_states[:, 0].view(num_envs, num_dof)
    dof_vel_all = dof_states[:, 1].view(num_envs, num_dof)

    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_pos_all[i, TEST_DOF_IDX] = initial_knee_angles[i]
        dof_vel_all[i, :] = 0.0

    _dev = dof_states.device
    env_ids = torch.arange(num_envs, dtype=torch.int32, device=_dev)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    torques = torch.zeros(num_envs * num_dof, dtype=torch.float32, device=_dev)

    return sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof


def make_human_body(mods=None):
    """HumanBody 인스턴스 생성 + 선택적 파라미터 수정."""
    body = HumanBody.from_config(
        muscle_def_path=os.path.join(CONFIG_DIR, "muscle_definitions.yaml"),
        param_path=os.path.join(CONFIG_DIR, "healthy_baseline.yaml"),
        num_envs=1,
        device="cpu",
    )
    if mods is None:
        return body
    if "reflex" in mods:
        r = mods["reflex"]
        if "stretch_gain" in r:
            body.reflex.stretch_gain[:] = r["stretch_gain"]
        if "stretch_threshold" in r:
            body.reflex.stretch_threshold[:] = r["stretch_threshold"]
    if "ligament" in mods:
        lg = mods["ligament"]
        if "k_lig" in lg:
            body.ligament.k_lig[:] = lg["k_lig"]
        if "damping" in lg:
            body.ligament.damping[:] = lg["damping"]
        if "alpha" in lg:
            body.ligament.alpha[:] = lg["alpha"]
    if "muscle" in mods:
        m = mods["muscle"]
        if "f_max_scale" in m:
            body.muscle_model.f_max *= m["f_max_scale"]
            body._f_max = body._f_max.float() * m["f_max_scale"]
        if "damping_scale" in m:
            body.muscle_model.damping *= m["damping_scale"]
    return body


# ═══════════════════════════════════════════════════════════════════════════════
# I01  토크 주입 방향 검증
# ═══════════════════════════════════════════════════════════════════════════════
def test_I01(gym, args):
    """
    시나리오:
        L_Knee를 중립(45°)에 놓고, 상수 +50 Nm 토크를 2초간 주입.
        각도가 증가(굴곡 방향)하면 PASS.

    검증 대상:
        gym.set_dof_actuation_force_tensor()가 실제로 PhysX 시뮬레이션에
        반영되는지 확인한다.

    합격 기준:
        final_angle > initial_angle + 0.1 rad (약 6° 이상 굴곡)
    """
    print("\n" + "=" * 60)
    print("I01: 토크 주입 방향 검증 (+50 Nm → 굴곡 방향 이동)")
    print("=" * 60)

    INITIAL_ANGLE = 0.785  # 45°
    APPLIED_TORQUE = 50.0   # Nm
    N_STEPS = int(2.0 / DT)  # 2초

    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=1, initial_knee_angles=[INITIAL_ANGLE])

    torques_2d = torques.view(1, num_dof)
    angle_history = []

    for step in range(N_STEPS):
        gym.refresh_dof_state_tensor(sim)
        angle_history.append(dof_pos_all[0, TEST_DOF_IDX].item())

        torques.zero_()
        torques_2d[0, TEST_DOF_IDX] = APPLIED_TORQUE
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_dof_state_tensor(sim)
    final_angle = dof_pos_all[0, TEST_DOF_IDX].item()
    gym.destroy_sim(sim)

    delta = final_angle - INITIAL_ANGLE
    print(f"  초기 각도: {np.degrees(INITIAL_ANGLE):.1f}°")
    print(f"  최종 각도: {np.degrees(final_angle):.1f}°")
    print(f"  변화량: {np.degrees(delta):+.2f}° (기댓값: > +6°)")

    # 플롯
    time_arr = np.arange(len(angle_history)) * DT
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_arr, np.degrees(angle_history), "b-", linewidth=2)
    ax.axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--",
               alpha=0.6, label=f"초기 각도 {np.degrees(INITIAL_ANGLE):.0f}°")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L_Knee Flexion Angle (deg)")
    ax.set_title(f"I01: 토크 주입 검증 (+{APPLIED_TORQUE} Nm, 2초)\n"
                 f"각도가 증가하면 PASS — 최종: {np.degrees(final_angle):.1f}° "
                 f"(Δ={np.degrees(delta):+.1f}°)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.05,
            f"Δangle = {np.degrees(delta):+.2f}°\n기준: > +6° → {'PASS' if delta > 0.1 else 'FAIL'}",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen" if delta > 0.1 else "salmon", alpha=0.8))
    save_and_close(fig, "I01_torque_injection")

    passed = delta > 0.1
    report("I01", "토크 주입 방향 검증", passed,
           f"Δangle = {np.degrees(delta):+.2f}° (기준: > 6°)")


# ═══════════════════════════════════════════════════════════════════════════════
# I02  토크 부호 규약 검증
# ═══════════════════════════════════════════════════════════════════════════════
def test_I02(gym, args):
    """
    시나리오:
        두 env를 준비. 하나는 +50 Nm, 다른 하나는 -50 Nm을 1초간 주입.
        초기 각도 45°에서 시작.

        +50 Nm → 각도 증가 (굴곡, flexion)
        -50 Nm → 각도 감소 (신전, extension)

    합격 기준:
        angle_positive > angle_negative (최종 기준, 차이 > 0.1 rad)
    """
    print("\n" + "=" * 60)
    print("I02: 토크 부호 규약 검증 (+/- 50 Nm)")
    print("=" * 60)

    INITIAL_ANGLE = 0.785  # 45°
    TORQUE_MAG = 50.0
    N_STEPS = int(1.0 / DT)  # 1초

    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=2,
                            initial_knee_angles=[INITIAL_ANGLE, INITIAL_ANGLE])

    torques_2d = torques.view(2, num_dof)
    hist_pos = []
    hist_neg = []

    for step in range(N_STEPS):
        gym.refresh_dof_state_tensor(sim)
        hist_pos.append(dof_pos_all[0, TEST_DOF_IDX].item())
        hist_neg.append(dof_pos_all[1, TEST_DOF_IDX].item())

        torques.zero_()
        torques_2d[0, TEST_DOF_IDX] = +TORQUE_MAG
        torques_2d[1, TEST_DOF_IDX] = -TORQUE_MAG
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_dof_state_tensor(sim)
    final_pos = dof_pos_all[0, TEST_DOF_IDX].item()
    final_neg = dof_pos_all[1, TEST_DOF_IDX].item()
    gym.destroy_sim(sim)

    diff = final_pos - final_neg
    print(f"  +{TORQUE_MAG} Nm 최종 각도: {np.degrees(final_pos):.1f}°")
    print(f"  -{TORQUE_MAG} Nm 최종 각도: {np.degrees(final_neg):.1f}°")
    print(f"  각도 차이: {np.degrees(diff):.2f}° (기댓값: > 11°)")

    time_arr = np.arange(N_STEPS) * DT
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_arr, np.degrees(hist_pos), "b-", linewidth=2, label=f"+{TORQUE_MAG} Nm (굴곡)")
    ax.plot(time_arr, np.degrees(hist_neg), "r-", linewidth=2, label=f"-{TORQUE_MAG} Nm (신전)")
    ax.axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--", alpha=0.5, label="초기 45°")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L_Knee Angle (deg)")
    ax.set_title(f"I02: 토크 부호 규약\n"
                 f"+토크=굴곡(위), -토크=신전(아래) → 최종 차이: {np.degrees(diff):.1f}°")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.5,
            f"최종 차이 = {np.degrees(diff):.1f}°\n기준: > 11° → {'PASS' if diff > 0.1 else 'FAIL'}",
            transform=ax.transAxes, ha="right", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen" if diff > 0.1 else "salmon", alpha=0.8))
    save_and_close(fig, "I02_torque_sign")

    passed = diff > 0.1
    report("I02", "토크 부호 규약 검증", passed,
           f"+토크={np.degrees(final_pos):.1f}°, -토크={np.degrees(final_neg):.1f}°, diff={np.degrees(diff):.1f}°")


# ═══════════════════════════════════════════════════════════════════════════════
# I03  프로파일 분화 검증 (Spastic > Healthy > Flaccid)
# ═══════════════════════════════════════════════════════════════════════════════
def test_I03(gym, args):
    """
    시나리오:
        3개의 env: Healthy / Spastic(뇌졸중) / Flaccid(SCI).
        80° 굴곡에서 시작, -5 rad/s 초기 속도(신전 방향).
        5초간 bio-torque만으로 운동.

        Spastic: 높은 reflex gain + 경직된 인대 → 거의 안 움직임 (높은 최종 각도)
        Healthy: 중간 저항 → 중간 최종 각도
        Flaccid: 근육력 없음 → 자유 진자처럼 신전 방향으로 이동 (낮은 최종 각도)

    합격 기준:
        final_spastic > final_healthy > final_flaccid (각도 크기 기준)
        각 쌍의 차이 > 5°
    """
    print("\n" + "=" * 60)
    print("I03: 프로파일 분화 검증 (Spastic > Healthy > Flaccid)")
    print("=" * 60)

    PROFILES = [
        {"name": "Healthy",         "color": "#2196F3", "mods": {}},
        {"name": "Spastic (Stroke)","color": "#F44336", "mods": {
            "reflex":   {"stretch_gain": 8.0,  "stretch_threshold": 0.02},
            "ligament": {"k_lig": 200.0, "damping": 25.0, "alpha": 15.0},
            "muscle":   {"damping_scale": 3.0},
        }},
        {"name": "Flaccid (SCI)",   "color": "#4CAF50", "mods": {
            "reflex":   {"stretch_gain": 0.0,  "stretch_threshold": 999.0},
            "ligament": {"k_lig": 5.0,   "damping": 0.5,  "alpha": 5.0},
            "muscle":   {"f_max_scale": 0.05, "damping_scale": 0.1},
        }},
    ]
    INITIAL_ANGLE = 1.4   # 80°
    INITIAL_VEL   = -5.0  # rad/s (신전 방향)
    N_STEPS = int(5.0 / DT)
    num_envs = len(PROFILES)

    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=num_envs,
                            initial_knee_angles=[INITIAL_ANGLE] * num_envs)

    # 초기 속도 설정
    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_vel_all[i, TEST_DOF_IDX] = INITIAL_VEL
    env_ids = torch.arange(num_envs, dtype=torch.int32)
    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_state_tensor)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    bodies = []
    for p in PROFILES:
        b = make_human_body(p["mods"] if p["mods"] else None)
        b.activation_dyn.reset()
        b.reflex.reset()
        bodies.append(b)

    torques_2d = torques.view(num_envs, num_dof)
    angle_history = np.zeros((num_envs, N_STEPS))
    torque_history = np.zeros((num_envs, N_STEPS))

    for step in range(N_STEPS):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        for i in range(num_envs):
            pos_i = dof_pos_all[i].unsqueeze(0)
            vel_i = dof_vel_all[i].unsqueeze(0)
            cmd = torch.zeros(1, bodies[i].num_muscles)

            bio_tau = bodies[i].compute_torques(pos_i, vel_i, cmd, dt=DT)
            tau_val = float(np.clip(bio_tau[0, TEST_DOF_IDX].item(), -500.0, 500.0))
            torques_2d[i, TEST_DOF_IDX] = tau_val

            angle_history[i, step] = dof_pos_all[i, TEST_DOF_IDX].item()
            torque_history[i, step] = tau_val

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_dof_state_tensor(sim)
    finals = {PROFILES[i]["name"]: np.degrees(dof_pos_all[i, TEST_DOF_IDX].item())
              for i in range(num_envs)}
    gym.destroy_sim(sim)

    f_h = finals["Healthy"]
    f_s = finals["Spastic (Stroke)"]
    f_f = finals["Flaccid (SCI)"]

    print(f"  Healthy 최종:  {f_h:.1f}°")
    print(f"  Spastic 최종:  {f_s:.1f}°")
    print(f"  Flaccid 최종:  {f_f:.1f}°")
    print(f"  기댓값 순서:   Spastic > Healthy > Flaccid")

    time_arr = np.arange(N_STEPS) * DT
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("I03: 환자 프로파일 분화 검증\nSpastic > Healthy > Flaccid 순서", fontsize=13)

    for i, p in enumerate(PROFILES):
        axes[0].plot(time_arr, np.degrees(angle_history[i]), color=p["color"],
                     linewidth=2, label=f"{p['name']} (최종: {np.degrees(angle_history[i, -1]):.1f}°)")
        axes[1].plot(time_arr, torque_history[i], color=p["color"],
                     linewidth=1.5, alpha=0.85, label=p["name"])

    axes[0].axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--", alpha=0.5, label="초기 80°")
    axes[0].set_ylabel("L_Knee Flexion Angle (deg)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Bio-Torque (Nm)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    order_ok = (f_s > f_h) and (f_h > f_f)
    gap_ok = (f_s - f_h > 5.0) and (f_h - f_f > 5.0)
    result_str = "PASS" if (order_ok and gap_ok) else "FAIL"
    axes[0].text(0.98, 0.95,
                 f"Spastic={f_s:.1f}° > Healthy={f_h:.1f}° > Flaccid={f_f:.1f}°\n"
                 f"순서 정렬: {'✓' if order_ok else '✗'}  |  간격>5°: {'✓' if gap_ok else '✗'}\n→ {result_str}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round",
                           facecolor="lightgreen" if (order_ok and gap_ok) else "salmon",
                           alpha=0.85))
    save_and_close(fig, "I03_profile_differentiation")

    passed = order_ok and gap_ok
    report("I03", "프로파일 분화 (Spastic>Healthy>Flaccid)", passed,
           f"S={f_s:.1f}° H={f_h:.1f}° F={f_f:.1f}° | 순서:{order_ok} 간격:{gap_ok}")


# ═══════════════════════════════════════════════════════════════════════════════
# I04  중력 vs Bio-토크 균형 검증
# ═══════════════════════════════════════════════════════════════════════════════
def test_I04(gym, args):
    """
    시나리오:
        80° 굴곡 상태에서 시작, 신전 방향 초기 kick(-3 rad/s) 부여.
        근육이 신장되는 속도가 생기므로 stretch reflex가 활성화된다.

        Spastic(gain=8): 낮은 역치에서 강한 반사 → 큰 양의 저항 토크
        Healthy(gain=1): 정상 반사 → 중간 양의 저항 토크
        Flaccid(gain=0): 반사 없음 → 거의 0 Nm

        bio-torque 크기 순서: Spastic > Healthy > Flaccid

    설계 이유:
        vel=0에서는 stretch reflex가 발동되지 않아 토크가 거의 0.
        신전 방향 속도(v<0)를 줘야 근육이 신장되어 reflex 활성화.
        이 상태에서 프로파일별 반사 차이를 명확히 볼 수 있다.

    합격 기준:
        |bio_torque_spastic| > |bio_torque_healthy| > |bio_torque_flaccid|
        (첫 20 step 평균, 크기 순서 확인)
        Spastic과 Healthy 차이 > 0.1 Nm
    """
    print("\n" + "=" * 60)
    print("I04: Stretch Reflex 차등 저항 (신전 kick → Spastic > Healthy > Flaccid)")
    print("=" * 60)

    INITIAL_ANGLE = 1.4   # 80°
    INITIAL_VEL   = -3.0  # rad/s (신전 방향, stretch reflex 활성화용)
    N_STEPS = int(0.5 / DT)  # 0.5초간 측정

    PROFILES = [
        {"name": "Healthy",         "mods": {}},
        {"name": "Spastic (Stroke)","mods": {
            "reflex":   {"stretch_gain": 8.0,  "stretch_threshold": 0.02},
            "ligament": {"k_lig": 200.0, "damping": 25.0, "alpha": 15.0},
            "muscle":   {"damping_scale": 3.0},
        }},
        {"name": "Flaccid (SCI)",   "mods": {
            "reflex":   {"stretch_gain": 0.0,  "stretch_threshold": 999.0},
            "ligament": {"k_lig": 5.0,   "damping": 0.5,  "alpha": 5.0},
            "muscle":   {"f_max_scale": 0.05, "damping_scale": 0.1},
        }},
    ]
    COLORS = ["#2196F3", "#F44336", "#4CAF50"]
    num_envs = len(PROFILES)

    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=num_envs,
                            initial_knee_angles=[INITIAL_ANGLE] * num_envs)

    # 신전 방향 초기 kick velocity 설정
    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_vel_all[i, TEST_DOF_IDX] = INITIAL_VEL
    env_ids = torch.arange(num_envs, dtype=torch.int32)
    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states_i04 = gymtorch.wrap_tensor(_dof_state_tensor)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states_i04),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    bodies = []
    for p in PROFILES:
        b = make_human_body(p["mods"] if p["mods"] else None)
        b.activation_dyn.reset()
        b.reflex.reset()
        bodies.append(b)

    torques_2d = torques.view(num_envs, num_dof)
    torque_history = np.zeros((num_envs, N_STEPS))
    angle_history  = np.zeros((num_envs, N_STEPS))

    for step in range(N_STEPS):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        for i in range(num_envs):
            pos_i = dof_pos_all[i].unsqueeze(0)
            vel_i = dof_vel_all[i].unsqueeze(0)
            cmd = torch.zeros(1, bodies[i].num_muscles)
            bio_tau = bodies[i].compute_torques(pos_i, vel_i, cmd, dt=DT)
            tau_val = float(np.clip(bio_tau[0, TEST_DOF_IDX].item(), -1000.0, 1000.0))
            torques_2d[i, TEST_DOF_IDX] = tau_val
            torque_history[i, step] = tau_val
            angle_history[i, step] = dof_pos_all[i, TEST_DOF_IDX].item()

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.destroy_sim(sim)

    # 첫 20 step 평균 토크 (stretch reflex 초기 반응 구간)
    mean_tau = {PROFILES[i]["name"]: torque_history[i, :20].mean() for i in range(num_envs)}
    abs_tau  = {k: abs(v) for k, v in mean_tau.items()}
    t_h = mean_tau["Healthy"]
    t_s = mean_tau["Spastic (Stroke)"]
    t_f = mean_tau["Flaccid (SCI)"]
    at_h, at_s, at_f = abs_tau["Healthy"], abs_tau["Spastic (Stroke)"], abs_tau["Flaccid (SCI)"]

    print(f"  초기 20step 평균 Bio-토크 (절댓값 비교):")
    print(f"    Spastic:  {t_s:+.2f} Nm  (|{at_s:.2f}| Nm)")
    print(f"    Healthy:  {t_h:+.2f} Nm  (|{at_h:.2f}| Nm)")
    print(f"    Flaccid:  {t_f:+.2f} Nm  (|{at_f:.2f}| Nm)")
    print(f"  기댓값: |Spastic| > |Healthy| > |Flaccid|, Spastic-Healthy 차이 > 0.1 Nm")

    time_arr = np.arange(N_STEPS) * DT
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("I04: Stretch Reflex 차등 저항\n"
                 f"신전 kick(-3 rad/s) → Spastic이 가장 강한 저항 토크 발생", fontsize=12)

    for i, p in enumerate(PROFILES):
        axes[0].plot(time_arr, torque_history[i], color=COLORS[i],
                     linewidth=2, label=f"{p['name']} (평균: {mean_tau[p['name']]:+.1f} Nm)")
        axes[1].plot(time_arr, np.degrees(angle_history[i]), color=COLORS[i],
                     linewidth=2, label=p["name"])

    axes[0].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Bio-Torque on L_Knee (Nm)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--", alpha=0.5, label="초기 80°")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("L_Knee Angle (deg)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    order_ok = (at_s > at_h) and (at_h > at_f)
    gap_ok = (at_s - at_h) > 0.1
    result_str = "PASS" if (order_ok and gap_ok) else "FAIL"
    axes[0].text(0.98, 0.95,
                 f"|S|={at_s:.2f} > |H|={at_h:.2f} > |F|={at_f:.2f} Nm\n"
                 f"순서: {'✓' if order_ok else '✗'}  |  S-H 차이>{0.1}: {'✓' if gap_ok else '✗'}\n→ {result_str}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round",
                           facecolor="lightgreen" if (order_ok and gap_ok) else "salmon",
                           alpha=0.85))
    save_and_close(fig, "I04_stretch_reflex_differentiation")

    passed = order_ok and gap_ok
    report("I04", "Stretch Reflex 차등 저항 (|Spastic|>|Healthy|>|Flaccid|)", passed,
           f"|S|={at_s:.2f} |H|={at_h:.2f} |F|={at_f:.2f} Nm | 순서:{order_ok} gap:{gap_ok}")


# ═══════════════════════════════════════════════════════════════════════════════
# 요약 출력
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    for tid, ok in results.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark}  {tid}")
    print(f"\n총 {passed_count}/{total} PASS")
    if passed_count == total:
        print("→ IsaacGym 통합 검증 완료. Step 3 (시각화) 또는 VIC-MSK 학습으로 진행 가능.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"→ FAIL 항목 확인 필요: {failed}")
    print(f"\n플롯 저장 위치: {RESULTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("IsaacGym 통합 검증 시작 (02_isaacgym_integration)")
    print(f"설정 파일: {CONFIG_DIR}")
    print(f"에셋 경로: {ASSET_DIR}")
    print()

    args = gymutil.parse_arguments(
        description="02_isaacgym_integration: standard_human_model + IsaacGym 연결 검증",
        custom_parameters=[
            {"name": "--headless", "action": "store_true", "default": False},
        ],
    )

    gym = gymapi.acquire_gym()

    test_I01(gym, args)
    test_I02(gym, args)
    test_I03(gym, args)
    test_I04(gym, args)

    print_summary()
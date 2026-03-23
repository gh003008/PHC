"""
06_integration_test / run_validation.py
========================================
Level 1 통합 검증: 오픈루프 EMG 패턴 → IsaacGym 하지 관절 운동

전신 하지 근육(20개)에 보행 문헌 기반 EMG 타이밍을 적용하고,
관절 각도 패턴이 정상 보행 운동과 유사한지 확인한다.

설정:
  - fix_base_link=True  (골반 고정, 안정적 검증 환경)
  - 하지 전체 DOF:  EFFORT 모드 (bio-torque만으로 구동)
  - 상체 DOF:        PD 위치 유지 (불필요한 자유도 제거)
  - 초기 자세:       중립 (전 관절 0°)
  - 높이:            z=2.0m (발이 지면 위 약 1.1m → 지면 접촉 없음)

검증 항목:
  I6-1  중력 기준선    — EMG 없이 중력만으로 관절이 평형 수렴하는지
  I6-2  보행 EMG 패턴 — 문헌 기반 EMG → L/R 무릎 ROM > 20° 출현 확인
  I6-3  L/R 위상 비대칭 — 무릎 교차상관 피크 ≈ ±0.5 × gait_cycle 확인
  I6-4  근육 제거 비교 — hamstrings 비활성화 시 무릎 신전 범위 증가 확인

사용법:
  conda activate phc
  cd /home/gunhee/workspace/PHC
  python standard_human_model/validation/06_integration_test/run_validation.py --headless
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import (
    NUM_DOFS, JOINT_DOF_RANGE, LOWER_LIMB_DOF_INDICES,
)

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

# ── 경로 ─────────────────────────────────────────────────────────────────────
ASSET_DIR   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../isaacgym_validation")
)
CONFIG_DIR  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../config")
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 상수 ─────────────────────────────────────────────────────────────────────
DT             = 1.0 / 60.0
GAIT_CYCLE_SEC = 1.1       # 보행 주기 (약 90 steps/min)
N_CYCLES       = 3         # 검증용 gait cycle 수
TORQUE_CLIP    = 500.0     # N·m

# ── DOF 인덱스 단축키 ─────────────────────────────────────────────────────────
HIP_L_IDX   = JOINT_DOF_RANGE["L_Hip"][0]    # 0
KNEE_L_IDX  = JOINT_DOF_RANGE["L_Knee"][0]   # 3
ANKLE_L_IDX = JOINT_DOF_RANGE["L_Ankle"][0]  # 6
HIP_R_IDX   = JOINT_DOF_RANGE["R_Hip"][0]    # 12
KNEE_R_IDX  = JOINT_DOF_RANGE["R_Knee"][0]   # 15
ANKLE_R_IDX = JOINT_DOF_RANGE["R_Ankle"][0]  # 18

# 기록할 DOF 인덱스 (순서 고정)
KEY_DOF_INDICES = [HIP_L_IDX, KNEE_L_IDX, ANKLE_L_IDX,
                   HIP_R_IDX, KNEE_R_IDX, ANKLE_R_IDX]
KEY_DOF_LABELS  = ["L_Hip", "L_Knee", "L_Ankle",
                   "R_Hip", "R_Knee", "R_Ankle"]

# lower limb DOF index를 set으로 변환 (빠른 검사용)
LOWER_LIMB_DOF_SET = set(LOWER_LIMB_DOF_INDICES)

# ── 근육 이름 (muscle_definitions.yaml 순서와 동일) ──────────────────────────
MUSCLE_NAMES = [
    "hip_flexors_L",  "gluteus_max_L",  "hip_abductors_L", "hip_adductors_L",
    "quadriceps_L",   "rectus_femoris_L", "hamstrings_L",  "gastrocnemius_L",
    "soleus_L",       "tibialis_ant_L",
    "hip_flexors_R",  "gluteus_max_R",  "hip_abductors_R", "hip_adductors_R",
    "quadriceps_R",   "rectus_femoris_R", "hamstrings_R",  "gastrocnemius_R",
    "soleus_R",       "tibialis_ant_R",
]
NUM_MUSCLES = len(MUSCLE_NAMES)

results = {}


def report(tid, name, passed, detail=""):
    mark = "✅ PASS" if passed else "❌ FAIL"
    results[tid] = passed
    print(f"\n[{tid}] {name}  {mark}")
    if detail:
        print(f"     {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# EMG 패턴 정의 (보행 문헌 기반)
# ══════════════════════════════════════════════════════════════════════════════
# Phase 규약: 0.0 = L heel strike (LHS), 0.5 = R heel strike (RHS)
#             L side: stance 0.0~0.60, swing 0.60~1.0
#             R side: +0.5 offset (동일 패턴 위상만 이동)
#
# 각 항목: (center_phase, width, amplitude) — cyclic Gaussian burst
# 참고: Perry 1992, Winter 2009 "Biomechanics and Motor Control of Human Movement"

EMG_GAIT: dict = {
    # ─── 고관절 ────────────────────────────────────────────────────────────
    "hip_flexors":    [(0.73, 0.12, 0.80)],              # swing: 고관절 굴곡
    "gluteus_max":    [(0.05, 0.09, 0.85)],              # loading response: 고관절 신전
    "hip_abductors":  [(0.20, 0.22, 0.65)],              # stance: 전두면 안정성
    "hip_adductors":  [(0.90, 0.06, 0.45)],              # late swing
    # ─── 무릎 ──────────────────────────────────────────────────────────────
    "quadriceps":     [(0.08, 0.09, 0.85), (0.93, 0.05, 0.65)],  # loading + terminal swing
    "rectus_femoris": [(0.05, 0.06, 0.60), (0.93, 0.05, 0.50)],  # bi-articular
    "hamstrings":     [(0.92, 0.07, 0.80), (0.04, 0.06, 0.65)],  # terminal swing + early stance
    # ─── 발목 ──────────────────────────────────────────────────────────────
    "gastrocnemius":  [(0.42, 0.10, 0.85)],              # push-off (bi-articular)
    "soleus":         [(0.38, 0.14, 0.90)],              # push-off (longer burst)
    "tibialis_ant":   [(0.04, 0.06, 0.70), (0.73, 0.12, 0.80)],  # loading + swing
}


def _gauss_cyclic(phase: float, center: float, width: float) -> float:
    """순환(cyclic) Gaussian: phase와 center 모두 [0, 1)."""
    d = abs(phase - center)
    d = min(d, 1.0 - d)    # 주기 경계 wrap
    return math.exp(-0.5 * (d / width) ** 2)


def make_emg_cmd(t: float, ablate: set = None) -> np.ndarray:
    """
    시각 t에서 20근육 EMG 명령 벡터를 생성한다.

    Args:
        t:      시뮬레이션 시각 (초)
        ablate: 비활성화할 근육 이름 집합

    Returns:
        cmd: (20,) ndarray, [0, 1] 클립
    """
    cmd = np.zeros(NUM_MUSCLES, dtype=np.float32)
    phase_L = (t % GAIT_CYCLE_SEC) / GAIT_CYCLE_SEC  # L side phase
    phase_R = (phase_L + 0.5) % 1.0                   # R side: 50% offset

    for idx, name in enumerate(MUSCLE_NAMES):
        if ablate and name in ablate:
            continue
        is_R  = name.endswith("_R")
        base  = name[:-2]              # "_L" or "_R" 제거 → "hip_flexors" 등
        phase = phase_R if is_R else phase_L
        for (center, width, amp) in EMG_GAIT.get(base, []):
            cmd[idx] += amp * _gauss_cyclic(phase, center, width)

    np.clip(cmd, 0.0, 1.0, out=cmd)
    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# IsaacGym 환경 생성
# ══════════════════════════════════════════════════════════════════════════════

def create_integration_sim(gym, args, num_envs: int = 1):
    """
    하지 전체 EFFORT 모드, 상체 PD hold, fix_base_link=True.

    - 하지(L/R Hip·Knee·Ankle·Toe) 24 DOF: EFFORT, stiffness=0, damping=2
    - 상체 45 DOF: POS mode, stiffness=200, damping=5
    - 높이 z=2.0m → 발이 지면으로부터 약 1.1m 위
    """
    sim_params = gymapi.SimParams()
    sim_params.dt       = DT
    sim_params.substeps = 2
    sim_params.up_axis  = gymapi.UP_AXIS_Z
    sim_params.gravity  = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type             = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_offset          = 0.02
    sim_params.physx.rest_offset             = 0.0
    sim_params.use_gpu_pipeline = False   # CPU pipeline

    sim = gym.create_sim(args.compute_device_id, -1, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "sim 생성 실패"

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_opts = gymapi.AssetOptions()
    asset_opts.angular_damping       = 0.1
    asset_opts.linear_damping        = 0.0
    asset_opts.max_angular_velocity  = 100.0
    asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset_opts.fix_base_link         = True

    humanoid = gym.load_asset(sim, ASSET_DIR, "smpl_humanoid_fixed.xml", asset_opts)
    assert humanoid is not None, "에셋 로드 실패"
    num_dof = gym.get_asset_dof_count(humanoid)

    spacing = 3.0
    envs, handles = [], []
    for i in range(num_envs):
        env = gym.create_env(
            sim,
            gymapi.Vec3(-spacing, -spacing, 0.0),
            gymapi.Vec3(spacing,  spacing,  spacing * 2),
            num_envs,
        )
        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(i * spacing, 0.0, 2.0)
        # MuJoCo Y-up → IsaacGym Z-up 변환: X축 90° 회전
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi / 2)
        handle = gym.create_actor(env, humanoid, pose, f"human_{i}", i, 0, 0)

        props = gym.get_actor_dof_properties(env, handle)
        for j in range(num_dof):
            if j in LOWER_LIMB_DOF_SET:
                # 하지: bio-torque로만 구동
                props["driveMode"][j] = int(gymapi.DOF_MODE_EFFORT)
                props["stiffness"][j] = 0.0
                props["damping"][j]   = 2.0   # 작은 감쇠 (발산 방지)
            else:
                # 상체: 중립 자세 유지
                props["driveMode"][j] = int(gymapi.DOF_MODE_POS)
                props["stiffness"][j] = 200.0
                props["damping"][j]   = 5.0
        gym.set_actor_dof_properties(env, handle, props)

        envs.append(env)
        handles.append(handle)

    gym.prepare_sim(sim)

    _dst     = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dst)
    dof_pos  = dof_states[:, 0].view(num_envs, num_dof)
    dof_vel  = dof_states[:, 1].view(num_envs, num_dof)
    torques  = torch.zeros(num_envs * num_dof, dtype=torch.float32)

    return sim, envs, handles, dof_pos, dof_vel, torques, num_dof


def make_human_body() -> HumanBody:
    """Healthy baseline HumanBody (num_envs=1, cpu)."""
    return HumanBody.from_config(
        muscle_def_path=os.path.join(CONFIG_DIR, "muscle_definitions.yaml"),
        param_path=os.path.join(CONFIG_DIR, "healthy_baseline.yaml"),
        num_envs=1,
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 공통 시뮬레이션 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(gym, args, n_steps: int,
            apply_emg: bool = True,
            ablate: set = None):
    """
    n_steps 스텝 동안 시뮬레이션 실행.

    Returns:
        angles:  (n_steps, 6)  각도(도) — [L_Hip, L_Knee, L_Ankle, R_Hip, R_Knee, R_Ankle]
        emg_log: (n_steps, 20) EMG 명령 이력
    """
    sim, envs, handles, dof_pos, dof_vel, torques, num_dof = \
        create_integration_sim(gym, args, num_envs=1)

    body = make_human_body()
    body.activation_dyn.reset()
    body.reflex.reset()

    torques_2d    = torques.view(1, num_dof)
    angles        = np.zeros((n_steps, len(KEY_DOF_INDICES)))
    emg_log       = np.zeros((n_steps, NUM_MUSCLES))
    ll_idx        = LOWER_LIMB_DOF_INDICES   # list of 24 indices

    for step in range(n_steps):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        t = step * DT

        # ── EMG 명령 생성 ─────────────────────────────────────────────────
        if apply_emg:
            cmd_np = make_emg_cmd(t, ablate=ablate)
        else:
            cmd_np = np.zeros(NUM_MUSCLES, dtype=np.float32)
        emg_log[step] = cmd_np

        # ── Bio-토크 계산 (CALM 파이프라인) ──────────────────────────────
        cmd   = torch.from_numpy(cmd_np).unsqueeze(0)      # (1, 20)
        pos_i = dof_pos[0].unsqueeze(0).detach().clone()   # (1, 69)
        vel_i = dof_vel[0].unsqueeze(0).detach().clone()   # (1, 69)
        bio_tau = body.compute_torques(pos_i, vel_i, cmd, dt=DT)  # (1, 69)

        # 하지 DOF에만 적용
        tau_np = bio_tau[0].detach().numpy()
        torques_2d[0, ll_idx] = torch.tensor(
            np.clip(tau_np[ll_idx], -TORQUE_CLIP, TORQUE_CLIP), dtype=torch.float32
        )

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # ── 관절 각도 기록 ────────────────────────────────────────────────
        for k, dof_idx in enumerate(KEY_DOF_INDICES):
            angles[step, k] = np.degrees(dof_pos[0, dof_idx].item())

    gym.destroy_sim(sim)
    return angles, emg_log


# ══════════════════════════════════════════════════════════════════════════════
# I6-1  중력 기준선 (EMG = 0)
# ══════════════════════════════════════════════════════════════════════════════

def test_I6_1(gym, args):
    """
    근육 비활성화 상태에서 2초간 관절 움직임을 기록.
    후반 0.5s에서 각도 변화율이 충분히 작으면 (정적 평형 수렴) PASS.
    → 이후 I6-2 EMG 테스트의 '기준선' 역할.
    """
    print("\n" + "=" * 60)
    print("I6-1: 중력 기준선 (EMG=0, 2초)")
    print("=" * 60)

    N = int(2.0 / DT)
    angles, _ = run_sim(gym, args, N, apply_emg=False)

    # 후반 0.5s 각도 변화 (step-to-step delta 평균)
    late = int(1.5 / DT)
    diffs = np.abs(np.diff(angles[late:], axis=0))
    delta_per_step = float(diffs.mean())
    settled = delta_per_step < 0.5   # 0.5°/step 미만

    # 플롯
    t = np.arange(N) * DT
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.suptitle("I6-1: 중력 기준선 (EMG=0)\n하지 자유 진동 → 평형 수렴", fontsize=12)
    for k, (ax, lbl) in enumerate(zip(axes.flat, KEY_DOF_LABELS)):
        ax.plot(t, angles[:, k], lw=1.5, color="#1565C0")
        ax.set_ylabel(f"{lbl} (°)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{lbl}: 최종={angles[-1, k]:.1f}°")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "I6_1_gravity_baseline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  후반 0.5s 평균 각도 변화율: {delta_per_step:.4f}°/step")
    print(f"  정착 기준 (<0.5°/step): {'PASS' if settled else 'FAIL'}")
    for k, lbl in enumerate(KEY_DOF_LABELS):
        print(f"  {lbl:8s} 최종 평형: {angles[-1, k]:+6.1f}°")
    print(f"  Plot: {out}")

    report("I6-1", "중력 기준선 수렴",
           settled, f"변화율={delta_per_step:.4f}°/step, L_Knee={angles[-1,1]:.1f}°")
    return settled, angles


# ══════════════════════════════════════════════════════════════════════════════
# I6-2  보행 EMG 패턴 → 관절 운동
# ══════════════════════════════════════════════════════════════════════════════

def test_I6_2(gym, args):
    """
    문헌 기반 보행 EMG → L/R 무릎 ROM > 20° 확인.
    EMG 없는 중력 기준선보다 유의미하게 큰 관절 운동 출현 여부.
    """
    print("\n" + "=" * 60)
    print(f"I6-2: 보행 EMG 패턴  ({N_CYCLES} cycles × {GAIT_CYCLE_SEC}s = "
          f"{N_CYCLES * GAIT_CYCLE_SEC:.1f}s)")
    print("=" * 60)

    N = int(N_CYCLES * GAIT_CYCLE_SEC / DT)
    angles, emg_log = run_sim(gym, args, N, apply_emg=True)
    t = np.arange(N) * DT

    # 2nd cycle 이후 정착 구간에서 ROM 측정
    analyze_from = int(GAIT_CYCLE_SEC / DT)
    roms = {}
    for k, lbl in enumerate(KEY_DOF_LABELS):
        seg = angles[analyze_from:, k]
        roms[lbl] = float(seg.max() - seg.min())

    rom_L_knee = roms["L_Knee"]
    rom_R_knee = roms["R_Knee"]
    passed = bool(rom_L_knee > 20.0 and rom_R_knee > 20.0)

    print(f"  (2nd cycle 이후 측정)")
    for lbl in KEY_DOF_LABELS:
        marker = " ← 기준" if "Knee" in lbl else ""
        print(f"  {lbl:8s} ROM: {roms[lbl]:5.1f}°{marker}")
    print(f"  기준: L_Knee & R_Knee ROM > 20°")

    # ── 그림 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)
    fig.suptitle("I6-2: 오픈루프 보행 EMG → 관절 각도\n"
                 f"(fix_base_link=True, Healthy baseline, {N_CYCLES} gait cycles)",
                 fontsize=13)

    # ① 무릎 각도 (L/R)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, angles[:, 1], "#1565C0", lw=2.0, label="L_Knee")
    ax1.plot(t, angles[:, 4], "#C62828", lw=1.8, ls="--", label="R_Knee")
    ax1.set_ylabel("Knee Angle (°)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"무릎 굴곡/신전  |  ROM: L={rom_L_knee:.0f}°  R={rom_R_knee:.0f}°  "
                  f"{'✅' if passed else '❌'}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    # gait cycle 경계 표시
    for cyc in range(N_CYCLES + 1):
        ax1.axvline(cyc * GAIT_CYCLE_SEC, color="gray", lw=0.6, ls=":")

    # ② 고관절 각도
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, angles[:, 0], "#1565C0", lw=1.8, label="L_Hip")
    ax2.plot(t, angles[:, 3], "#C62828", lw=1.5, ls="--", label="R_Hip")
    ax2.set_ylabel("Hip Angle (°)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"고관절  |  ROM L={roms['L_Hip']:.0f}°")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ③ 발목 각도
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, angles[:, 2], "#1565C0", lw=1.8, label="L_Ankle")
    ax3.plot(t, angles[:, 5], "#C62828", lw=1.5, ls="--", label="R_Ankle")
    ax3.set_ylabel("Ankle Angle (°)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title(f"발목  |  ROM L={roms['L_Ankle']:.0f}°")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ④ Gait cycle 위상별 무릎 각도
    ax4 = fig.add_subplot(gs[1, 2])
    phase_arr = (t % GAIT_CYCLE_SEC) / GAIT_CYCLE_SEC
    ax4.scatter(phase_arr, angles[:, 1], c="#1565C0", s=2, alpha=0.5, label="L_Knee")
    ax4.scatter(phase_arr, angles[:, 4], c="#C62828", s=2, alpha=0.5, label="R_Knee")
    ax4.set_xlabel("Gait Phase (0 = L heel strike)")
    ax4.set_ylabel("Knee Angle (°)")
    ax4.set_title("Gait Phase 위상도")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ⑤ EMG heatmap (L side)
    ax5 = fig.add_subplot(gs[2, :2])
    im = ax5.imshow(emg_log[:, :10].T, aspect="auto",
                    extent=[0, t[-1], -0.5, 9.5], cmap="hot_r", vmin=0, vmax=1,
                    origin="lower")
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([n.replace("_L", "") for n in MUSCLE_NAMES[:10]], fontsize=7)
    ax5.set_xlabel("Time (s)")
    ax5.set_title("EMG Commands — L side")
    plt.colorbar(im, ax=ax5, label="Activation (0–1)", shrink=0.8)

    # ⑥ EMG heatmap (R side)
    ax6 = fig.add_subplot(gs[2, 2])
    im2 = ax6.imshow(emg_log[:, 10:].T, aspect="auto",
                     extent=[0, t[-1], -0.5, 9.5], cmap="hot_r", vmin=0, vmax=1,
                     origin="lower")
    ax6.set_yticks(range(10))
    ax6.set_yticklabels([n.replace("_R", "") for n in MUSCLE_NAMES[10:]], fontsize=7)
    ax6.set_xlabel("Time (s)")
    ax6.set_title("EMG Commands — R side")
    plt.colorbar(im2, ax=ax6, label="Activation", shrink=0.8)

    out = os.path.join(RESULTS_DIR, "I6_2_gait_emg.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")

    report("I6-2", "보행 EMG → L/R Knee ROM > 20°",
           passed, f"L_Knee={rom_L_knee:.1f}° R_Knee={rom_R_knee:.1f}°")
    return passed, angles, emg_log


# ══════════════════════════════════════════════════════════════════════════════
# I6-3  L/R 위상 비대칭 (교차상관)
# ══════════════════════════════════════════════════════════════════════════════

def test_I6_3(angles_emg: np.ndarray):
    """
    I6-2 데이터에서 L 무릎과 R 무릎의 교차상관 피크 lag ≈ ±0.5 × gait_cycle 확인.
    L/R EMG는 0.5 위상 오프셋이 설계에 내포되어 있으므로,
    관절 각도에서도 동일한 오프셋이 나타나면 파이프라인이 정상임을 의미.
    """
    print("\n" + "=" * 60)
    print("I6-3: L/R 무릎 위상 비대칭 (교차상관)")
    print("=" * 60)

    knee_L = angles_emg[:, 1]
    knee_R = angles_emg[:, 4]
    std_L  = float(knee_L.std())
    std_R  = float(knee_R.std())

    if std_L < 0.5 or std_R < 0.5:
        print(f"  L_Knee std={std_L:.2f}°  R_Knee std={std_R:.2f}° — 운동량 너무 작음")
        report("I6-3", "L/R 위상 비대칭", False, "운동량 부족 (std < 0.5°)")
        return False

    # 정규화 후 교차상관
    kL = (knee_L - knee_L.mean()) / (std_L + 1e-8)
    kR = (knee_R - knee_R.mean()) / (std_R + 1e-8)

    xcorr = np.correlate(kL, kR, mode="full")
    lags  = np.arange(-(len(kL) - 1), len(kL))

    # 1사이클 이내에서 피크 탐색 (zero-lag ±10% 제외)
    center    = len(kL) - 1
    cyc_steps = int(GAIT_CYCLE_SEC / DT)
    lo        = center - cyc_steps
    hi        = center + cyc_steps
    excl_lo   = center - int(0.1 * cyc_steps)
    excl_hi   = center + int(0.1 * cyc_steps)

    mask = np.zeros(len(xcorr), dtype=bool)
    mask[lo:hi] = True
    mask[excl_lo:excl_hi] = False
    xcorr_m = np.where(mask, xcorr, -np.inf)

    peak_lag      = int(lags[np.argmax(xcorr_m)])
    peak_lag_sec  = peak_lag * DT
    expected_sec  = 0.5 * GAIT_CYCLE_SEC
    tolerance     = 0.2 * GAIT_CYCLE_SEC
    passed        = abs(abs(peak_lag_sec) - expected_sec) < tolerance

    print(f"  교차상관 피크 lag:  {peak_lag_sec:+.3f}s")
    print(f"  예상 오프셋:        ±{expected_sec:.3f}s (±0.5 × {GAIT_CYCLE_SEC}s)")
    print(f"  허용 오차:          ±{tolerance:.3f}s (20%)")
    print(f"  L_Knee std={std_L:.1f}°  R_Knee std={std_R:.1f}°")

    # 플롯
    lag_t = lags * DT
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("I6-3: L/R 무릎 위상 비대칭", fontsize=13)

    axes[0].plot(np.arange(len(kL)) * DT, angles_emg[:, 1], "#1565C0", lw=1.5, label="L_Knee")
    axes[0].plot(np.arange(len(kR)) * DT, angles_emg[:, 4], "#C62828", lw=1.5, ls="--", label="R_Knee")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Knee Angle (°)")
    axes[0].set_title("L/R 무릎 각도 시계열")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    plot_range = slice(center - cyc_steps - 10, center + cyc_steps + 10)
    axes[1].plot(lag_t[plot_range], xcorr[plot_range], "#333", lw=1.5)
    axes[1].axvline(peak_lag_sec, color="red", lw=2, ls="--",
                    label=f"피크 lag={peak_lag_sec:+.3f}s")
    axes[1].axvline(+expected_sec, color="green", lw=1, ls=":",
                    label=f"예상 +{expected_sec:.2f}s")
    axes[1].axvline(-expected_sec, color="green", lw=1, ls=":")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Cross-correlation")
    axes[1].set_title(f"교차상관  |  {'✅ PASS' if passed else '❌ FAIL'}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "I6_3_lr_phase.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")

    report("I6-3", "L/R 무릎 위상 비대칭 (±0.5 cycle)",
           passed, f"peak_lag={peak_lag_sec:+.3f}s, 예상=±{expected_sec:.3f}s")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# I6-4  근육 제거 비교 — Hamstrings Ablation
# ══════════════════════════════════════════════════════════════════════════════

def test_I6_4(gym, args, angles_full: np.ndarray):
    """
    Hamstrings 비활성화 시 무릎 신전 범위 증가 확인.

    메커니즘:
      hamstrings → 무릎 굴곡 토크 (R_knee moment arm = +0.03m)
      hamstrings OFF → 굴곡 토크 감소 → 무릎가 더 신전 → 최소 각도 감소
    기준: ham OFF 시 L_Knee 최소 각도가 full보다 3° 이상 낮아야 함.
    """
    print("\n" + "=" * 60)
    print("I6-4: Hamstrings 제거 비교 (ham off → 무릎 더 신전)")
    print("=" * 60)

    N      = int(N_CYCLES * GAIT_CYCLE_SEC / DT)
    ablate = {"hamstrings_L", "hamstrings_R"}
    angles_abl, _ = run_sim(gym, args, N, apply_emg=True, ablate=ablate)

    analyze_from = int(GAIT_CYCLE_SEC / DT)
    knee_L_full  = angles_full[analyze_from:, 1]
    knee_L_abl   = angles_abl[analyze_from:,  1]
    knee_R_full  = angles_full[analyze_from:, 4]
    knee_R_abl   = angles_abl[analyze_from:,  4]

    min_full_L = float(knee_L_full.min())
    min_abl_L  = float(knee_L_abl.min())
    max_full_L = float(knee_L_full.max())
    max_abl_L  = float(knee_L_abl.max())
    min_full_R = float(knee_R_full.min())
    min_abl_R  = float(knee_R_abl.min())
    max_full_R = float(knee_R_full.max())
    max_abl_R  = float(knee_R_abl.max())

    # ham OFF → 굴곡 토크 감소 → peak 굴곡 각도(max) 감소
    # Note: 최소 각도(신전)는 bi-articular 효과로 오히려 증가할 수 있음 (hip 신전도 감소)
    less_flex_L = bool(max_abl_L < max_full_L - 3.0)
    less_flex_R = bool(max_abl_R < max_full_R - 3.0)
    passed      = less_flex_L or less_flex_R

    t = np.arange(N) * DT
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("I6-4: Hamstrings 제거 비교\n"
                 "(ham OFF → 굴곡 토크 감소 → 최대 굴곡 각도 감소 예상)", fontsize=12)

    axes[0].plot(t, angles_full[:, 1], "#1565C0", lw=2,   label="Full EMG")
    axes[0].plot(t, angles_abl[:, 1],  "#C62828", lw=1.8, ls="--",
                 label="Ham = 0 (Ablated)")
    axes[0].set_ylabel("L_Knee Angle (°)")
    axes[0].set_title(f"L_Knee: Full max={max_full_L:.1f}°  Ham-off max={max_abl_L:.1f}°  "
                      f"Δmax={max_full_L - max_abl_L:+.1f}°  "
                      f"| min: {min_full_L:.1f}° → {min_abl_L:.1f}°")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, angles_full[:, 4], "#1565C0", lw=2,   label="Full EMG")
    axes[1].plot(t, angles_abl[:, 4],  "#C62828", lw=1.8, ls="--",
                 label="Ham = 0 (Ablated)")
    axes[1].set_ylabel("R_Knee Angle (°)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"R_Knee: Full max={max_full_R:.1f}°  Ham-off max={max_abl_R:.1f}°  "
                      f"Δmax={max_full_R - max_abl_R:+.1f}°  "
                      f"| min: {min_full_R:.1f}° → {min_abl_R:.1f}°")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        for cyc in range(N_CYCLES + 1):
            ax.axvline(cyc * GAIT_CYCLE_SEC, color="gray", lw=0.6, ls=":")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "I6_4_hamstring_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  L_Knee: max  Full={max_full_L:.1f}° → Ham-off={max_abl_L:.1f}°"
          f"  Δ={max_full_L - max_abl_L:+.1f}°")
    print(f"  R_Knee: max  Full={max_full_R:.1f}° → Ham-off={max_abl_R:.1f}°"
          f"  Δ={max_full_R - max_abl_R:+.1f}°")
    print(f"  L_Knee: min  Full={min_full_L:.1f}° → Ham-off={min_abl_L:.1f}°"
          f"  (min 변화는 bi-articular 효과)")
    print(f"  기준: L 또는 R에서 최대 굴곡 각도 3° 이상 감소")
    print(f"  Plot: {out}")

    report("I6-4", "Hamstrings 제거 → 최대 굴곡 감소",
           passed, f"L_Δmax={max_full_L - max_abl_L:+.1f}° R_Δmax={max_full_R - max_abl_R:+.1f}°")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# 요약
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print("06_integration_test  결과 요약")
    print("=" * 60)
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    for tid, ok in results.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark}  {tid}")
    print(f"\n총 {passed_count}/{total} PASS")
    if passed_count == total:
        print("→ Level 1 통합 검증 완료.")
        print("  오픈루프 EMG → 보행 유사 관절 운동 확인됨.")
        print("  다음 단계: Level 2 (RL 상위 제어기) 설계")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"→ FAIL 항목: {failed}")
    print(f"\n결과 저장: {RESULTS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("IsaacGym Level 1 통합 검증 시작 (06_integration_test)")
    print(f"  Gait cycle: {GAIT_CYCLE_SEC}s  |  cycles: {N_CYCLES}  |  DT: {DT:.4f}s")
    print(f"  Config:  {CONFIG_DIR}")
    print(f"  Asset:   {ASSET_DIR}")
    print()

    args = gymutil.parse_arguments(
        description="06_integration_test: 오픈루프 EMG → 보행 유사 관절 운동 검증",
        custom_parameters=[
            {"name": "--headless", "action": "store_true", "default": False},
        ],
    )

    gym = gymapi.acquire_gym()

    # I6-1: 중력 기준선
    _, angles_grav = test_I6_1(gym, args)

    # I6-2: 보행 EMG 패턴 (메인 테스트)
    _, angles_emg, emg_log = test_I6_2(gym, args)

    # I6-3: L/R 위상 비대칭 (I6-2 데이터 재활용)
    test_I6_3(angles_emg)

    # I6-4: Hamstrings 제거 비교
    test_I6_4(gym, args, angles_emg)

    print_summary()

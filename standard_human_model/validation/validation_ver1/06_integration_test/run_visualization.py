"""
06_integration_test / run_visualization.py
==========================================
IsaacGym 뷰어 + 실시간 matplotlib 모니터로 오픈루프 보행 EMG 검증을 시각화한다.

3개 시나리오를 나란히 비교:
  [파랑] Gait EMG  — 문헌 기반 보행 EMG 패턴
  [회색] No EMG   — 중력만 (기준선)
  [빨강] Ham-off  — Hamstrings 비활성화

실시간 모니터 (TkAgg, 롤링 윈도우):
  행 1: L_Knee + R_Knee  각도 (3개 시나리오)
  행 2: L_Hip  + L_Ankle 각도 (3개 시나리오)
  행 3: 핵심 EMG 활성화 (Gait EMG 시나리오: hip_flex, ham, quad, sol, tib_ant)

사용법:
  conda activate phc
  cd /home/gunhee/workspace/PHC

  # 뷰어 + 실시간 모니터 (기본)
  python standard_human_model/validation/06_integration_test/run_visualization.py

  # headless (플롯만 저장)
  python standard_human_model/validation/06_integration_test/run_visualization.py --headless

  # 모니터 없이 뷰어만
  python standard_human_model/validation/06_integration_test/run_visualization.py --no-monitor
"""

# CRITICAL: IsaacGym must be imported before torch
import sys
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import math
from collections import deque

import matplotlib
_is_headless = "--headless" in sys.argv

# sys.path 설정 (먼저 import 필요한 모듈들보다 앞에)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

VAL_DIR = os.path.dirname(__file__)
sys.path.insert(0, VAL_DIR)
from run_validation import make_emg_cmd, MUSCLE_NAMES, NUM_MUSCLES, GAIT_CYCLE_SEC

# run_validation.py가 matplotlib.use를 호출할 수 있으므로 그 이후에 재설정
matplotlib.use("Agg" if _is_headless else "TkAgg", force=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
except Exception:
    pass

from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import (
    NUM_DOFS, JOINT_DOF_RANGE, LOWER_LIMB_DOF_INDICES,
)

# ── 경로 ─────────────────────────────────────────────────────────────────────
ASSET_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../isaacgym_validation"))
CONFIG_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 상수 ─────────────────────────────────────────────────────────────────────
DT            = 1.0 / 60.0
DURATION      = 11.0         # 시뮬레이션 길이 (초, 10 gait cycles)
MONITOR_SECS  = 5.0          # 롤링 윈도우 (초) — DURATION보다 작아야 스크롤됨
MONITOR_EVERY = 6            # N step마다 모니터 갱신
TORQUE_CLIP   = 500.0

# ── DOF 인덱스 ───────────────────────────────────────────────────────────────
HIP_L_IDX   = JOINT_DOF_RANGE["L_Hip"][0]
KNEE_L_IDX  = JOINT_DOF_RANGE["L_Knee"][0]
ANKLE_L_IDX = JOINT_DOF_RANGE["L_Ankle"][0]
HIP_R_IDX   = JOINT_DOF_RANGE["R_Hip"][0]
KNEE_R_IDX  = JOINT_DOF_RANGE["R_Knee"][0]
ANKLE_R_IDX = JOINT_DOF_RANGE["R_Ankle"][0]

LOWER_LIMB_DOF_SET = set(LOWER_LIMB_DOF_INDICES)

# ── 시나리오 정의 ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name":       "Gait EMG",
        "color_rgb":  (0.20, 0.50, 1.00),
        "color_plot": "#1565C0",
        "apply_emg":  True,
        "ablate":     None,
    },
    {
        "name":       "No EMG",
        "color_rgb":  (0.60, 0.60, 0.60),
        "color_plot": "#757575",
        "apply_emg":  False,
        "ablate":     None,
    },
    {
        "name":       "Ham-off",
        "color_rgb":  (1.00, 0.30, 0.30),
        "color_plot": "#C62828",
        "apply_emg":  True,
        "ablate":     {"hamstrings_L", "hamstrings_R"},
    },
]
NUM_ENVS = len(SCENARIOS)

# 모니터에 표시할 핵심 EMG 근육 (Gait EMG 시나리오 기준)
EMG_SHOW = [
    ("hip_flexors_L",  "#AB47BC"),
    ("hamstrings_L",   "#E53935"),
    ("quadriceps_L",   "#1E88E5"),
    ("soleus_L",       "#43A047"),
    ("tibialis_ant_L", "#FB8C00"),
]


# ══════════════════════════════════════════════════════════════════════════════
# IsaacGym 환경 생성 (viewer 지원)
# ══════════════════════════════════════════════════════════════════════════════

def create_viz_sim(gym, args, graphics_device=None):
    """
    3개 env 생성 (각 시나리오).
    CPU pipeline 항상 사용 — viewer + headless 모두 동작 (02/03과 동일 방식).
    GPU pipeline은 tensor device 불일치 에러를 유발하므로 사용하지 않는다.
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

    # CPU pipeline: viewer 유무에 상관없이 항상 CPU (02/03과 동일)
    gfx = graphics_device if graphics_device is not None else -1
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(args.compute_device_id, gfx, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "sim 생성 실패"

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_opts = gymapi.AssetOptions()
    asset_opts.angular_damping        = 0.1
    asset_opts.linear_damping         = 0.0
    asset_opts.max_angular_velocity   = 100.0
    asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    asset_opts.fix_base_link          = True

    humanoid = gym.load_asset(sim, ASSET_DIR, "smpl_humanoid_fixed.xml", asset_opts)
    assert humanoid is not None, "에셋 로드 실패"
    num_dof = gym.get_asset_dof_count(humanoid)

    spacing = 3.5
    envs, handles = [], []
    for i, sc in enumerate(SCENARIOS):
        env = gym.create_env(
            sim,
            gymapi.Vec3(-spacing / 2, -spacing / 2, 0.0),
            gymapi.Vec3(spacing / 2,   spacing / 2,  spacing * 2),
            NUM_ENVS,
        )
        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(i * spacing, 0.0, 2.0)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi / 2)
        handle = gym.create_actor(env, humanoid, pose, sc["name"], i, 0, 0)

        # 시나리오별 색상 적용
        r, g, b = sc["color_rgb"]
        color   = gymapi.Vec3(r, g, b)
        n_bodies = gym.get_actor_rigid_body_count(env, handle)
        for body_idx in range(n_bodies):
            gym.set_rigid_body_color(env, handle, body_idx, gymapi.MESH_VISUAL, color)

        # DOF 모드 설정
        props = gym.get_actor_dof_properties(env, handle)
        for j in range(num_dof):
            if j in LOWER_LIMB_DOF_SET:
                props["driveMode"][j] = int(gymapi.DOF_MODE_EFFORT)
                props["stiffness"][j] = 0.0
                props["damping"][j]   = 2.0
            else:
                props["driveMode"][j] = int(gymapi.DOF_MODE_POS)
                props["stiffness"][j] = 200.0
                props["damping"][j]   = 5.0
        gym.set_actor_dof_properties(env, handle, props)

        envs.append(env)
        handles.append(handle)

    gym.prepare_sim(sim)

    _dst     = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dst)
    dof_pos  = dof_states[:, 0].view(NUM_ENVS, num_dof)
    dof_vel  = dof_states[:, 1].view(NUM_ENVS, num_dof)
    torques  = torch.zeros(NUM_ENVS * num_dof, dtype=torch.float32)  # CPU pipeline → cpu tensor

    return sim, envs, handles, dof_pos, dof_vel, torques, num_dof


def make_human_body():
    return HumanBody.from_config(
        muscle_def_path=os.path.join(CONFIG_DIR, "muscle_definitions.yaml"),
        param_path=os.path.join(CONFIG_DIR, "healthy_baseline.yaml"),
        num_envs=1,
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 실시간 모니터
# ══════════════════════════════════════════════════════════════════════════════

class RealtimeMonitor:
    """
    4행 레이아웃 (sagittal plane 집중):
      행 0: L_Hip  angle (°) | L_Hip  torque (Nm)
      행 1: L_Knee angle (°) | L_Knee torque (Nm)
      행 2: L_Ankle angle(°) | L_Ankle torque(Nm)
      행 3: 핵심 EMG 활성화 (Gait EMG 시나리오만, 전체 너비)
    """

    def __init__(self, dt):
        maxlen = int(MONITOR_SECS / dt) + 10
        self.dt = dt

        self.t_buf    = deque(maxlen=maxlen)
        # 각도 버퍼 (3 관절 × 3 시나리오)
        self.hip_L    = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        self.knee_L   = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        self.ankle_L  = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        # 토크 버퍼 (sagittal, Nm)
        self.tau_hL   = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        self.tau_kL   = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        self.tau_aL   = {sc["name"]: deque(maxlen=maxlen) for sc in SCENARIOS}
        self.emg_bufs = {name: deque(maxlen=maxlen) for name, _ in EMG_SHOW}

        plt.ion()
        self.fig = plt.figure(figsize=(12, 11))
        gs = gridspec.GridSpec(4, 2, figure=self.fig, hspace=0.55, wspace=0.35)
        self.fig.suptitle("실시간 모니터 — Sagittal plane (Angle / Bio-Torque)", fontsize=11, y=0.99)

        self.ax_hA  = self.fig.add_subplot(gs[0, 0])   # L_Hip angle
        self.ax_hT  = self.fig.add_subplot(gs[0, 1])   # L_Hip torque
        self.ax_kA  = self.fig.add_subplot(gs[1, 0])   # L_Knee angle
        self.ax_kT  = self.fig.add_subplot(gs[1, 1])   # L_Knee torque
        self.ax_aA  = self.fig.add_subplot(gs[2, 0])   # L_Ankle angle
        self.ax_aT  = self.fig.add_subplot(gs[2, 1])   # L_Ankle torque
        self.ax_emg = self.fig.add_subplot(gs[3, :])

        for ax, title in [
            (self.ax_hA, "L_Hip Angle (°)"),
            (self.ax_kA, "L_Knee Angle (°)"),
            (self.ax_aA, "L_Ankle Angle (°)"),
            (self.ax_hT, "L_Hip Torque (Nm)"),
            (self.ax_kT, "L_Knee Torque (Nm)"),
            (self.ax_aT, "L_Ankle Torque (Nm)"),
        ]:
            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        # 토크 축: 0선 참조
        for ax in [self.ax_hT, self.ax_kT, self.ax_aT]:
            ax.axhline(0, color="k", lw=0.5, ls=":")

        self.ax_emg.set_ylabel("EMG Activation (0–1)", fontsize=8)
        self.ax_emg.set_xlabel("Time (s)", fontsize=8)
        self.ax_emg.set_title("핵심 근육 활성화 (Gait EMG)", fontsize=9)
        self.ax_emg.grid(True, alpha=0.3)
        self.ax_emg.tick_params(labelsize=7)

        # 시나리오별 라인 생성
        self.lines_hA, self.lines_kA, self.lines_aA = {}, {}, {}
        self.lines_hT, self.lines_kT, self.lines_aT = {}, {}, {}
        for sc in SCENARIOS:
            nm = sc["name"]
            lw = 2.0 if sc["apply_emg"] else 1.2
            ls = "-" if nm == "Gait EMG" else ("--" if nm == "No EMG" else "-.")
            kw = dict(color=sc["color_plot"], lw=lw, ls=ls, label=nm[:10])
            self.lines_hA[nm], = self.ax_hA.plot([], [], **kw)
            self.lines_kA[nm], = self.ax_kA.plot([], [], **kw)
            self.lines_aA[nm], = self.ax_aA.plot([], [], **kw)
            self.lines_hT[nm], = self.ax_hT.plot([], [], **kw)
            self.lines_kT[nm], = self.ax_kT.plot([], [], **kw)
            self.lines_aT[nm], = self.ax_aT.plot([], [], **kw)

        for ax in [self.ax_hA, self.ax_kA, self.ax_aA,
                   self.ax_hT, self.ax_kT, self.ax_aT]:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.6)

        # EMG 라인 생성
        self.lines_emg = {}
        for name, color in EMG_SHOW:
            label = name.replace("_L", "").replace("_", " ")
            line, = self.ax_emg.plot([], [], color=color, lw=1.8, label=label)
            self.lines_emg[name] = line
        self.ax_emg.legend(fontsize=7, loc="upper right", framealpha=0.6, ncol=5)
        self.ax_emg.set_ylim(-0.05, 1.10)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.15)

    def push(self, t, angles_all, torques_all, emg_cmd):
        """
        angles_all:  dict[scenario_name] -> [hip_L, knee_L, ankle_L] (도)
        torques_all: dict[scenario_name] -> [tau_hip_L, tau_knee_L, tau_ankle_L] (Nm, sagittal)
        emg_cmd: (20,) 현재 EMG 명령
        """
        self.t_buf.append(t)
        for sc in SCENARIOS:
            nm   = sc["name"]
            ang  = angles_all[nm]
            tau  = torques_all[nm]
            self.hip_L[nm].append(ang[0])
            self.knee_L[nm].append(ang[1])
            self.ankle_L[nm].append(ang[2])
            self.tau_hL[nm].append(tau[0])
            self.tau_kL[nm].append(tau[1])
            self.tau_aL[nm].append(tau[2])
        for name, _ in EMG_SHOW:
            idx = MUSCLE_NAMES.index(name)
            self.emg_bufs[name].append(float(emg_cmd[idx]))

    def draw(self):
        if not self.t_buf:
            return
        ts    = list(self.t_buf)
        t_now = ts[-1]

        for sc in SCENARIOS:
            nm = sc["name"]
            self.lines_hA[nm].set_data(ts, list(self.hip_L[nm]))
            self.lines_kA[nm].set_data(ts, list(self.knee_L[nm]))
            self.lines_aA[nm].set_data(ts, list(self.ankle_L[nm]))
            self.lines_hT[nm].set_data(ts, list(self.tau_hL[nm]))
            self.lines_kT[nm].set_data(ts, list(self.tau_kL[nm]))
            self.lines_aT[nm].set_data(ts, list(self.tau_aL[nm]))

        for name, _ in EMG_SHOW:
            self.lines_emg[name].set_data(ts, list(self.emg_bufs[name]))

        # 롤링 윈도우
        if t_now < MONITOR_SECS:
            t_lo, t_hi = 0.0, MONITOR_SECS
        else:
            t_hi = t_now + 0.1
            t_lo = t_hi - MONITOR_SECS
        for ax in [self.ax_hA, self.ax_kA, self.ax_aA,
                   self.ax_hT, self.ax_kT, self.ax_aT,
                   self.ax_emg]:
            ax.relim()
            ax.autoscale_view(scalex=False)
            ax.set_xlim(t_lo, t_hi)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = gymutil.parse_arguments(
        description="06_integration_test 시각화: 오픈루프 보행 EMG → IsaacGym",
        custom_parameters=[
            {"name": "--headless",   "action": "store_true", "default": False,
             "help": "headless 모드 (플롯만 저장)"},
            {"name": "--no-monitor", "action": "store_true", "default": False,
             "help": "실시간 모니터 비활성화"},
            {"name": "--duration",   "type": float, "default": DURATION,
             "help": f"시뮬레이션 길이 (초, 기본={DURATION})"},
        ],
    )

    duration  = args.duration
    max_steps = int(duration / DT)
    gym       = gymapi.acquire_gym()

    # ── sim 생성 ──────────────────────────────────────────────────────────────
    gfx_device = args.graphics_device_id if not args.headless else None
    sim, envs, handles, dof_pos, dof_vel, torques, num_dof = \
        create_viz_sim(gym, args, graphics_device=gfx_device)

    # ── HumanBody 인스턴스 (시나리오별) ──────────────────────────────────────
    bodies = []
    for _ in SCENARIOS:
        b = make_human_body()
        b.activation_dyn.reset()
        b.reflex.reset()
        bodies.append(b)

    torques_2d = torques.view(NUM_ENVS, num_dof)
    ll_idx     = LOWER_LIMB_DOF_INDICES

    # ── 기록 배열 ─────────────────────────────────────────────────────────────
    t_arr        = np.arange(max_steps) * DT
    angle_hist   = np.zeros((NUM_ENVS, max_steps, 6))   # 6: kL, kR, hL, hR, aL, aR
    torque_hist  = np.zeros((NUM_ENVS, max_steps, 6))   # 동일 순서, sagittal torque (Nm)
    emg_hist     = np.zeros((max_steps, NUM_MUSCLES))

    KEY_DOFS = [KNEE_L_IDX, KNEE_R_IDX, HIP_L_IDX, HIP_R_IDX, ANKLE_L_IDX, ANKLE_R_IDX]

    # ── IsaacGym 뷰어 ─────────────────────────────────────────────────────────
    viewer = None
    if not args.headless:
        cam_props        = gymapi.CameraProperties()
        cam_props.width  = 1280
        cam_props.height = 720
        viewer = gym.create_viewer(sim, cam_props)
        if viewer is None:
            print("[경고] 뷰어 생성 실패. headless 모드로 계속.")
        else:
            # 3개 humanoid 전체가 보이도록 카메라 배치
            mid_x = (NUM_ENVS - 1) * 3.5 / 2
            gym.viewer_camera_look_at(
                viewer, None,
                gymapi.Vec3(mid_x,  -5.0, 2.5),   # 카메라 위치
                gymapi.Vec3(mid_x,   0.0, 1.5),    # 주시 방향
            )
            print(f"\n뷰어 켜짐. 3개 시나리오 나란히 표시.")
            for i, sc in enumerate(SCENARIOS):
                print(f"  [{sc['name']:10s}]  x={i * 3.5:.1f}m  색상: RGB{sc['color_rgb']}")
            print("Q 또는 창 닫기로 종료.\n", flush=True)

    # ── 실시간 모니터 ─────────────────────────────────────────────────────────
    monitor = None
    if not _is_headless and not args.no_monitor:
        try:
            monitor = RealtimeMonitor(dt=DT)
            print("실시간 모니터 활성화")
        except Exception as e:
            print(f"[모니터 생성 실패: {e}]")

    # ── 시뮬레이션 루프 ───────────────────────────────────────────────────────
    print("=" * 60)
    print("06_integration_test 시각화")
    print(f"  시나리오: {[sc['name'] for sc in SCENARIOS]}")
    print(f"  시간: {duration:.1f}s  ({duration/GAIT_CYCLE_SEC:.1f} gait cycles)")
    print("=" * 60)

    for step in range(max_steps):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        t = step * DT

        # ── 시나리오별 EMG 명령 + bio-토크 계산 ──────────────────────────────
        emg_gait = make_emg_cmd(t, ablate=None)   # Gait EMG 시나리오용 (모니터 표시)
        emg_hist[step] = emg_gait

        for i, sc in enumerate(SCENARIOS):
            if sc["apply_emg"]:
                cmd_np = make_emg_cmd(t, ablate=sc["ablate"])
            else:
                cmd_np = np.zeros(NUM_MUSCLES, dtype=np.float32)

            cmd   = torch.from_numpy(cmd_np).unsqueeze(0)
            pos_i = dof_pos[i].cpu().unsqueeze(0)
            vel_i = dof_vel[i].cpu().unsqueeze(0)
            bio_tau = bodies[i].compute_torques(pos_i, vel_i, cmd, dt=DT)

            tau_np = bio_tau[0].detach().numpy()
            torques_2d[i, ll_idx] = torch.tensor(
                np.clip(tau_np[ll_idx], -TORQUE_CLIP, TORQUE_CLIP),
                dtype=torch.float32,
            )

            # 관절 각도 + sagittal 토크 기록
            for k, dof_idx in enumerate(KEY_DOFS):
                angle_hist[i, step, k]  = np.degrees(dof_pos[i, dof_idx].item())
                torque_hist[i, step, k] = float(tau_np[dof_idx])

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)

        gym.fetch_results(sim, True)  # CPU pipeline 항상

        # ── 뷰어 갱신 ──────────────────────────────────────────────────────
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                max_steps = step + 1
                break
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        # ── 실시간 모니터 갱신 ─────────────────────────────────────────────
        if monitor is not None:
            angles_all = {
                sc["name"]: [
                    angle_hist[i, step, 2],   # hip_L
                    angle_hist[i, step, 0],   # knee_L
                    angle_hist[i, step, 4],   # ankle_L
                ]
                for i, sc in enumerate(SCENARIOS)
            }
            torques_all = {
                sc["name"]: [
                    torque_hist[i, step, 2],  # tau_hip_L  (sagittal)
                    torque_hist[i, step, 0],  # tau_knee_L (sagittal)
                    torque_hist[i, step, 4],  # tau_ankle_L(sagittal)
                ]
                for i, sc in enumerate(SCENARIOS)
            }
            monitor.push(t, angles_all, torques_all, emg_gait)
            if step % MONITOR_EVERY == 0:
                monitor.draw()

        # ── 콘솔 출력 (1초마다) ────────────────────────────────────────────
        if (step + 1) % 60 == 0:
            t_s = (step + 1) * DT
            parts = [f"{SCENARIOS[i]['name'][:8]:8s}: "
                     f"kL={angle_hist[i,step,0]:+5.1f}° "
                     f"kR={angle_hist[i,step,1]:+5.1f}°"
                     for i in range(NUM_ENVS)]
            print(f"  t={t_s:4.1f}s  |  {'  |  '.join(parts)}")

    # ── 뷰어 종료 대기 ────────────────────────────────────────────────────────
    if viewer is not None:
        print("\n시뮬레이션 완료. Q 또는 창 닫기로 종료.")
        while not gym.query_viewer_has_closed(viewer):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            # monitor.draw()는 이 루프에서 호출하지 않음 — plt.pause가 segfault 유발
        gym.destroy_viewer(viewer)

    if monitor is not None:
        monitor.close()

    gym.destroy_sim(sim)

    # ── 최종 플롯 저장 ─────────────────────────────────────────────────────────
    n = max_steps
    plt.ioff()

    # ── 플롯 1: 관절 각도 + EMG ──────────────────────────────────────────────
    fig1 = plt.figure(figsize=(18, 14))
    gs1  = gridspec.GridSpec(4, 2, figure=fig1, hspace=0.52, wspace=0.30)
    fig1.suptitle("06_integration_test — 관절 각도 (Sagittal plane)\n"
                  "(fix_base_link=True, 하지 EFFORT 모드)",
                  fontsize=12)

    joint_plots = [
        ("L_Knee (°)",  0, gs1[0, 0]),
        ("R_Knee (°)",  1, gs1[0, 1]),
        ("L_Hip (°)",   2, gs1[1, 0]),
        ("R_Hip (°)",   3, gs1[1, 1]),
        ("L_Ankle (°)", 4, gs1[2, 0]),
        ("R_Ankle (°)", 5, gs1[2, 1]),
    ]
    for title, k, subplot_spec in joint_plots:
        ax = fig1.add_subplot(subplot_spec)
        for i, sc in enumerate(SCENARIOS):
            lw = 2.0 if sc["apply_emg"] else 1.2
            ls = "-" if sc["name"] == "Gait EMG" else ("--" if sc["name"] == "No EMG" else "-.")
            ax.plot(t_arr[:n], angle_hist[i, :n, k],
                    color=sc["color_plot"], lw=lw, ls=ls, label=sc["name"])
        for cyc in range(int(duration / GAIT_CYCLE_SEC) + 1):
            ax.axvline(cyc * GAIT_CYCLE_SEC, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(title)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_emg1 = fig1.add_subplot(gs1[3, :])
    im = ax_emg1.imshow(emg_hist[:n, :10].T, aspect="auto",
                        extent=[0, t_arr[n - 1], -0.5, 9.5],
                        cmap="hot_r", vmin=0, vmax=1, origin="lower")
    ax_emg1.set_yticks(range(10))
    ax_emg1.set_yticklabels([nm.replace("_L", "") for nm in MUSCLE_NAMES[:10]], fontsize=7)
    ax_emg1.set_xlabel("Time (s)")
    ax_emg1.set_title("EMG 입력 (Gait EMG 시나리오, L side)")
    plt.colorbar(im, ax=ax_emg1, label="Activation (0–1)", shrink=0.8)
    for cyc in range(int(duration / GAIT_CYCLE_SEC) + 1):
        ax_emg1.axvline(cyc * GAIT_CYCLE_SEC, color="white", lw=0.8, ls=":")

    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "I6_viz_angles.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n각도 플롯 저장: {out1}")

    # ── 플롯 2: 관절 토크 (sagittal, Nm) ─────────────────────────────────────
    fig2 = plt.figure(figsize=(18, 14))
    gs2  = gridspec.GridSpec(4, 2, figure=fig2, hspace=0.52, wspace=0.30)
    fig2.suptitle("06_integration_test — Bio-Torque Sagittal (Nm)\n"
                  "(HumanBody 근육 모델 출력, 하지 EFFORT 모드)",
                  fontsize=12)

    torque_plots = [
        ("L_Knee Torque (Nm)",  0, gs2[0, 0]),
        ("R_Knee Torque (Nm)",  1, gs2[0, 1]),
        ("L_Hip Torque (Nm)",   2, gs2[1, 0]),
        ("R_Hip Torque (Nm)",   3, gs2[1, 1]),
        ("L_Ankle Torque (Nm)", 4, gs2[2, 0]),
        ("R_Ankle Torque (Nm)", 5, gs2[2, 1]),
    ]
    for title, k, subplot_spec in torque_plots:
        ax = fig2.add_subplot(subplot_spec)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        for i, sc in enumerate(SCENARIOS):
            lw = 2.0 if sc["apply_emg"] else 1.2
            ls = "-" if sc["name"] == "Gait EMG" else ("--" if sc["name"] == "No EMG" else "-.")
            ax.plot(t_arr[:n], torque_hist[i, :n, k],
                    color=sc["color_plot"], lw=lw, ls=ls, label=sc["name"])
        for cyc in range(int(duration / GAIT_CYCLE_SEC) + 1):
            ax.axvline(cyc * GAIT_CYCLE_SEC, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(title)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3행: L side angle+torque 오버레이 (hip/knee/ankle 각 1개씩)
    overlay_joints = [
        ("L_Hip",   2, gs2[3, 0]),
        ("L_Knee",  0, gs2[3, 1]),
    ]
    for jname, k, subplot_spec in overlay_joints:
        ax = fig2.add_subplot(subplot_spec)
        ax2 = ax.twinx()
        sc_gait = SCENARIOS[0]  # Gait EMG만
        i = 0
        ax.plot(t_arr[:n], angle_hist[i, :n, k],
                color="#1565C0", lw=2, label=f"{jname} Angle")
        ax2.plot(t_arr[:n], torque_hist[i, :n, k],
                 color="#E53935", lw=1.5, ls="--", label=f"{jname} Torque")
        ax.set_ylabel(f"{jname} Angle (°)", color="#1565C0", fontsize=8)
        ax2.set_ylabel(f"{jname} Torque (Nm)", color="#E53935", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{jname} — Angle vs Torque (Gait EMG)", fontsize=9)
        ax.grid(True, alpha=0.3)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")

    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "I6_viz_torques.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"토크 플롯 저장: {out2}")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    analyze_from = int(GAIT_CYCLE_SEC / DT)
    for i, sc in enumerate(SCENARIOS):
        kL_seg = angle_hist[i, analyze_from:n, 0]
        kR_seg = angle_hist[i, analyze_from:n, 1]
        if len(kL_seg) > 0:
            print(f"  [{sc['name']:10s}]  "
                  f"L_Knee ROM={kL_seg.max()-kL_seg.min():.1f}°  "
                  f"R_Knee ROM={kR_seg.max()-kR_seg.min():.1f}°")
    print("=" * 60)


if __name__ == "__main__":
    main()

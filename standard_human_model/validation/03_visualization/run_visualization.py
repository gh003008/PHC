"""
03_visualization / run_visualization.py
========================================
IsaacGym 뷰어에서 세 환자 프로파일(Healthy / Spastic / Flaccid)의
무릎 진자 운동을 실시간으로 시각화한다.

시연 내용:
    80° 무릎 굴곡에서 -5 rad/s 신전 kick 후,
    각 프로파일의 근골격 bio-torque 반응 차이를 물리 시뮬레이션으로 비교.
    Spastic > Healthy > Flaccid 순서로 무릎이 적게 신전된다.

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC

    # 뷰어 + 실시간 모니터 (기본)
    python standard_human_model/validation/03_visualization/run_visualization.py

    # 모니터 신호 선택 (angle/torque/vel 또는 all)
    python standard_human_model/validation/03_visualization/run_visualization.py --monitor angle,torque,vel

    # 모니터 없이 뷰어만
    python standard_human_model/validation/03_visualization/run_visualization.py --no-monitor

    # headless (플롯만 저장)
    python standard_human_model/validation/03_visualization/run_visualization.py --headless
"""

# CRITICAL: IsaacGym must be imported before torch
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import sys
from collections import deque

# matplotlib 백엔드: 뷰어 모드 → TkAgg(interactive), headless → Agg
import matplotlib
_is_headless = "--headless" in sys.argv
matplotlib.use("Agg" if _is_headless else "TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

VAL_DIR = os.path.join(os.path.dirname(__file__), "../02_isaacgym_integration")
sys.path.insert(0, VAL_DIR)
from run_validation import create_sim_and_envs, make_human_body

from standard_human_model.core.skeleton import JOINT_DOF_RANGE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DOF_IDX  = JOINT_DOF_RANGE["L_Knee"][0]
DT            = 1.0 / 60.0
INITIAL_ANGLE = 1.4    # 80°
INITIAL_VEL   = -5.0   # rad/s 신전 방향
DURATION      = 10.0   # seconds

PROFILES = [
    {"name": "Healthy",          "color_rgb": (0.25, 0.55, 1.0),  "mods": {}},
    {"name": "Spastic (Stroke)", "color_rgb": (1.0,  0.25, 0.25), "mods": {
        "reflex":   {"stretch_gain": 8.0,  "stretch_threshold": 0.02},
        "ligament": {"k_lig": 200.0, "damping": 25.0, "alpha": 15.0},
        "muscle":   {"damping_scale": 3.0},
    }},
    {"name": "Flaccid (SCI)",    "color_rgb": (0.2,  0.85, 0.35), "mods": {
        "reflex":   {"stretch_gain": 0.0,  "stretch_threshold": 999.0},
        "ligament": {"k_lig": 5.0,  "damping": 0.5,  "alpha": 5.0},
        "muscle":   {"f_max_scale": 0.05, "damping_scale": 0.1},
    }},
]
COLORS_PLOT = ["#2196F3", "#F44336", "#4CAF50"]


# ═══════════════════════════════════════════════════════════════════════════════
# 실시간 모니터
# ═══════════════════════════════════════════════════════════════════════════════

MONITOR_SIGNALS = {
    "angle":  ("L_Knee 각도",  "deg"),
    "torque": ("Bio-Torque",  "Nm"),
    "vel":    ("L_Knee 속도",  "rad/s"),
}
DEFAULT_MONITOR     = "angle,torque,vel"
MONITOR_WINDOW_SECS = 10.0   # 롤링 윈도우 길이 (초)
MONITOR_UPDATE      = 6      # N 스텝마다 갱신 (60fps → 10 updates/s)


class RealtimeMonitor:
    """IsaacGym 뷰어와 나란히 표시되는 실시간 파라미터 모니터."""

    def __init__(self, signal_keys, profile_names, profile_colors, dt):
        self.keys   = signal_keys
        self.names  = profile_names
        self.colors = profile_colors
        maxlen      = int(MONITOR_WINDOW_SECS / dt) + 10

        self.t_buf = deque(maxlen=maxlen)
        self.bufs  = {
            k: {n: deque(maxlen=maxlen) for n in profile_names}
            for k in signal_keys
        }

        plt.ion()
        n_sigs = len(signal_keys)
        self.fig, axes = plt.subplots(
            n_sigs, 1, figsize=(7, 2.6 * n_sigs), sharex=True
        )
        self.fig.suptitle("실시간 모니터 — 무릎 진자", fontsize=11, y=0.98)
        self.axes = [axes] if n_sigs == 1 else list(axes)

        self.lines = {}
        for i, key in enumerate(signal_keys):
            ax = self.axes[i]
            label, unit = MONITOR_SIGNALS[key]
            ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            self.lines[key] = {}
            for name, color in zip(profile_names, profile_colors):
                line, = ax.plot([], [], color=color, lw=1.5, label=name[:12])
                self.lines[key][name] = line
            ax.legend(loc="upper left", fontsize=7, framealpha=0.6)

        self.axes[-1].set_xlabel("Time (s)", fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 모니터 창을 IsaacGym 뷰어(1280px) 오른쪽에 배치
        try:
            mgr = self.fig.canvas.manager
            mgr.window.geometry("+1290+0")
        except Exception:
            pass

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def push(self, t, values):
        """values: {signal_key: {profile_name: float}}"""
        self.t_buf.append(t)
        for k in self.keys:
            for n in self.names:
                self.bufs[k][n].append(values[k][n])

    def draw(self):
        if not self.t_buf:
            return
        ts = list(self.t_buf)
        for i, key in enumerate(self.keys):
            ax = self.axes[i]
            for name in self.names:
                self.lines[key][name].set_data(ts, list(self.bufs[key][name]))
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = gymutil.parse_arguments(
        description="03_visualization: IsaacGym 뷰어 환자 프로파일 시연",
        custom_parameters=[
            {"name": "--headless",   "action": "store_true", "default": False,
             "help": "뷰어 없이 headless 실행 (플롯만 저장)"},
            {"name": "--monitor",    "type": str, "default": DEFAULT_MONITOR,
             "help": f"실시간 모니터 신호 (쉼표 구분 또는 'all'). 기본: {DEFAULT_MONITOR}"},
            {"name": "--no-monitor", "action": "store_true", "default": False,
             "help": "실시간 모니터 비활성화"},
        ],
    )

    num_envs  = len(PROFILES)
    max_steps = int(DURATION / DT)
    gym = gymapi.acquire_gym()

    # ── sim + env 생성 (I03와 동일 설정) ────────────────────────────────────
    gfx_device = args.graphics_device_id if not args.headless else None
    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=num_envs,
                            initial_knee_angles=[INITIAL_ANGLE] * num_envs,
                            graphics_device=gfx_device)

    # ── 프로파일별 색상 적용 ─────────────────────────────────────────────────
    try:
        for i, (env, handle) in enumerate(zip(envs, actor_handles)):
            r, g, b = PROFILES[i]["color_rgb"]
            color   = gymapi.Vec3(r, g, b)
            n_bodies = gym.get_actor_rigid_body_count(env, handle)
            for body_idx in range(n_bodies):
                gym.set_rigid_body_color(env, handle, body_idx, gymapi.MESH_VISUAL, color)
            print(f"  [{PROFILES[i]['name']}]  RGB({r:.2f},{g:.2f},{b:.2f})")
    except Exception as e:
        print(f"  [색상 적용 실패: {e}]")

    # ── 초기 kick velocity 설정 ──────────────────────────────────────────────
    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_vel_all[i, TEST_DOF_IDX] = INITIAL_VEL
    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states_kick   = gymtorch.wrap_tensor(_dof_state_tensor)
    env_ids = torch.arange(num_envs, dtype=torch.int32, device=dof_states_kick.device)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states_kick),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    # ── 바이오 모델 ──────────────────────────────────────────────────────────
    bodies = []
    for p in PROFILES:
        b = make_human_body(p["mods"] if p["mods"] else None)
        b.activation_dyn.reset()
        b.reflex.reset()
        bodies.append(b)

    torques_2d = torques.view(num_envs, num_dof)

    # ── IsaacGym 뷰어 ────────────────────────────────────────────────────────
    viewer = None
    if not args.headless:
        cam_props        = gymapi.CameraProperties()
        cam_props.width  = 1280
        cam_props.height = 720
        viewer = gym.create_viewer(sim, cam_props)
        if viewer is None:
            print("[경고] 뷰어 생성 실패. headless 모드로 계속.")
        else:
            gym.viewer_camera_look_at(
                viewer, None,
                gymapi.Vec3(0.0, -3.0, 2.0),
                gymapi.Vec3(0.0,  0.0, 2.0),
            )
            print("\n뷰어 켜짐. Q 또는 창 닫기로 종료.\n", flush=True)

    # ── 실시간 모니터 ────────────────────────────────────────────────────────
    monitor = None
    if not args.headless and not args.no_monitor:
        raw_keys = (list(MONITOR_SIGNALS.keys()) if args.monitor.lower() == "all"
                    else [k.strip() for k in args.monitor.split(",")
                          if k.strip() in MONITOR_SIGNALS])
        if raw_keys:
            try:
                monitor = RealtimeMonitor(
                    signal_keys    = raw_keys,
                    profile_names  = [p["name"] for p in PROFILES],
                    profile_colors = COLORS_PLOT,
                    dt             = DT,
                )
                print(f"모니터 신호: {raw_keys}")
            except Exception as e:
                print(f"[모니터 생성 실패: {e}]")

    # ── 기록 배열 ────────────────────────────────────────────────────────────
    time_arr       = np.arange(max_steps) * DT
    angle_history  = np.zeros((num_envs, max_steps))
    torque_history = np.zeros((num_envs, max_steps))

    print("=" * 60)
    print("03_visualization  (무릎 진자 — 환자 프로파일 비교)")
    print(f"  초기 각도: 80°  kick: {INITIAL_VEL} rad/s  시간: {DURATION}s")
    if monitor:
        print(f"  모니터 신호: {monitor.keys}")
    print("=" * 60)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────────
    for step in range(max_steps):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        for i in range(num_envs):
            pos_i   = dof_pos_all[i].cpu().unsqueeze(0)
            vel_i   = dof_vel_all[i].cpu().unsqueeze(0)
            cmd     = torch.zeros(1, bodies[i].num_muscles)
            bio_tau = bodies[i].compute_torques(pos_i, vel_i, cmd, dt=DT)
            tau_val = float(np.clip(bio_tau[0, TEST_DOF_IDX].item(), -500.0, 500.0))
            torques_2d[i, TEST_DOF_IDX] = tau_val
            angle_history[i, step]  = dof_pos_all[i, TEST_DOF_IDX].item()
            torque_history[i, step] = tau_val

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # ── IsaacGym 뷰어 갱신 ──────────────────────────────────────────────
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                max_steps = step + 1
                break
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        # ── 실시간 모니터 갱신 ──────────────────────────────────────────────
        if monitor is not None:
            t = step * DT
            vals = {
                "angle":  {PROFILES[i]["name"]: np.degrees(angle_history[i, step])
                           for i in range(num_envs)},
                "torque": {PROFILES[i]["name"]: torque_history[i, step]
                           for i in range(num_envs)},
                "vel":    {PROFILES[i]["name"]: dof_vel_all[i, TEST_DOF_IDX].item()
                           for i in range(num_envs)},
            }
            monitor.push(t, vals)
            if step % MONITOR_UPDATE == 0:
                monitor.draw()

        # ── 콘솔 출력 (1초마다) ─────────────────────────────────────────────
        if (step + 1) % 60 == 0:
            t = (step + 1) * DT
            parts = [f"{PROFILES[i]['name'][:7]}: {np.degrees(angle_history[i,step]):5.1f}° "
                     f"{torque_history[i,step]:+5.1f}Nm"
                     for i in range(num_envs)]
            print(f"  t={t:4.1f}s  |  {'  '.join(parts)}")

    # ── 시뮬레이션 종료 후 뷰어 유지 ────────────────────────────────────────
    if viewer is not None:
        print("\n시뮬레이션 완료. Q 또는 창 닫기로 종료.")
        while not gym.query_viewer_has_closed(viewer):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            if monitor is not None:
                monitor.draw()
        gym.destroy_viewer(viewer)

    if monitor is not None:
        monitor.close()

    gym.destroy_sim(sim)

    # ── 최종 플롯 저장 ───────────────────────────────────────────────────────
    n = max_steps
    plt.ioff()
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle("무릎 진자: 환자 프로파일 비교 (IsaacGym)\nSpastic > Healthy > Flaccid",
                  fontsize=13)

    for i, p in enumerate(PROFILES):
        axes2[0].plot(time_arr[:n], np.degrees(angle_history[i, :n]),
                      color=COLORS_PLOT[i], linewidth=2, label=p["name"])
        axes2[1].plot(time_arr[:n], torque_history[i, :n],
                      color=COLORS_PLOT[i], linewidth=1.5, alpha=0.85, label=p["name"])

    axes2[0].axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--", alpha=0.4)
    axes2[0].set_ylabel("L_Knee 굴곡 각도 (deg)")
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    axes2[1].axhline(0, color="gray", linewidth=0.5)
    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Bio-Torque (Nm)")
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)

    finals = [np.degrees(angle_history[i, n - 1]) for i in range(num_envs)]
    note = "  |  ".join(f"{PROFILES[i]['name'][:7]}: {finals[i]:.1f}°"
                        for i in range(num_envs))
    axes2[0].text(0.98, 0.02, note, transform=axes2[0].transAxes,
                  ha="right", va="bottom", fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "knee_pendulum_profiles.png")
    fig2.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"\n플롯 저장: {out_path}")

    # ── 요약 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    s_f = np.degrees(angle_history[1, n - 1])
    h_f = np.degrees(angle_history[0, n - 1])
    f_f = np.degrees(angle_history[2, n - 1])
    print(f"  Healthy   최종: {h_f:.1f}°")
    print(f"  Spastic   최종: {s_f:.1f}°")
    print(f"  Flaccid   최종: {f_f:.1f}°")
    ok = (s_f > h_f) and (h_f > f_f)
    print(f"  순서 Spastic > Healthy > Flaccid: {'✅' if ok else '❌'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

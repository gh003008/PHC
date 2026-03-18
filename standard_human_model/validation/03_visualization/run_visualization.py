"""
03_visualization / run_visualization.py
========================================
02_isaacgym_integration의 I03 검증 코드를 그대로 사용하여
IsaacGym 뷰어에서 세 환자 프로파일을 시각화한다.

- I03 코드 (동작 검증됨)를 직접 임포트하여 사용
- 환경 생성 후 프로파일별 색상 적용
- 뷰어 / 최종 플롯 저장

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC

    # 뷰어 켜기
    python standard_human_model/validation/03_visualization/run_visualization.py --pipeline cpu

    # headless (플롯만)
    python standard_human_model/validation/03_visualization/run_visualization.py --headless --pipeline cpu
"""

# CRITICAL: IsaacGym must be imported before torch
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# I03와 동일한 설정 (run_validation.py의 create_sim_and_envs + make_human_body 재사용)
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
DURATION      = 5.0

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


def main():
    args = gymutil.parse_arguments(
        description="03_visualization: IsaacGym 뷰어 환자 프로파일 시연",
        custom_parameters=[
            {"name": "--headless", "action": "store_true", "default": False},
        ],
    )

    num_envs  = len(PROFILES)
    max_steps = int(DURATION / DT)
    gym = gymapi.acquire_gym()

    # ── I03와 동일한 sim + env 생성 ─────────────────────────────────────────
    sim, envs, actor_handles, dof_pos_all, dof_vel_all, torques, num_dof = \
        create_sim_and_envs(gym, args, num_envs=num_envs,
                            initial_knee_angles=[INITIAL_ANGLE] * num_envs)

    num_bodies = gym.get_asset_rigid_body_count(
        gym.get_actor_asset(envs[0], actor_handles[0])
    ) if hasattr(gym, "get_actor_asset") else 24  # SMPL = 24 bodies

    # ── 색상 적용 (try: MESH_VISUAL, fallback: skip) ─────────────────────
    try:
        for i, (env, handle) in enumerate(zip(envs, actor_handles)):
            r, g, b = PROFILES[i]["color_rgb"]
            color = gymapi.Vec3(r, g, b)
            n_bodies = gym.get_actor_rigid_body_count(env, handle)
            for body_idx in range(n_bodies):
                gym.set_rigid_body_color(env, handle, body_idx, gymapi.MESH_VISUAL, color)
            print(f"  [{PROFILES[i]['name']}]  RGB({r:.2f},{g:.2f},{b:.2f})")
    except Exception as e:
        print(f"  [색상 적용 실패: {e}]")

    # ── 초기 kick velocity 설정 (I03와 동일: 별도 호출) ─────────────────
    gym.refresh_dof_state_tensor(sim)
    for i in range(num_envs):
        dof_vel_all[i, TEST_DOF_IDX] = INITIAL_VEL
    env_ids = torch.arange(num_envs, dtype=torch.int32)
    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states_kick = gymtorch.wrap_tensor(_dof_state_tensor)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_states_kick),
        gymtorch.unwrap_tensor(env_ids),
        num_envs,
    )

    # ── 바이오 모델 ──────────────────────────────────────────────────────
    bodies = []
    for p in PROFILES:
        b = make_human_body(p["mods"] if p["mods"] else None)
        b.activation_dyn.reset()
        b.reflex.reset()
        bodies.append(b)

    torques_2d = torques.view(num_envs, num_dof)

    # ── 뷰어 ─────────────────────────────────────────────────────────────
    viewer = None
    if not args.headless:
        cam_props = gymapi.CameraProperties()
        cam_props.width  = 1280
        cam_props.height = 720
        viewer = gym.create_viewer(sim, cam_props)
        gym.viewer_camera_look_at(
            viewer, None,
            gymapi.Vec3(12.0, -2.0, 4.0),
            gymapi.Vec3(0.0,   5.0, 1.5),
        )
        print("\n뷰어 켜짐. 'Q' 또는 창 닫기로 종료.\n")

    # ── 기록 배열 ────────────────────────────────────────────────────────
    time_arr       = np.arange(max_steps) * DT
    angle_history  = np.zeros((num_envs, max_steps))
    torque_history = np.zeros((num_envs, max_steps))

    print("=" * 60)
    print("03_visualization 시작  (I03 물리 설정 재사용)")
    print(f"  초기 각도: 80°  kick: {INITIAL_VEL} rad/s  시간: {DURATION}s")
    print("=" * 60)

    # ── 시뮬레이션 루프 (I03와 100% 동일 구조) ──────────────────────────
    for step in range(max_steps):
        gym.refresh_dof_state_tensor(sim)
        torques.zero_()

        for i in range(num_envs):
            pos_i = dof_pos_all[i].unsqueeze(0)
            vel_i = dof_vel_all[i].unsqueeze(0)
            cmd   = torch.zeros(1, bodies[i].num_muscles)
            bio_tau = bodies[i].compute_torques(pos_i, vel_i, cmd, dt=DT)
            tau_val = float(np.clip(bio_tau[0, TEST_DOF_IDX].item(), -500.0, 500.0))
            torques_2d[i, TEST_DOF_IDX] = tau_val
            angle_history[i, step]  = dof_pos_all[i, TEST_DOF_IDX].item()
            torque_history[i, step] = tau_val

        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                max_steps = step + 1
                break

        if (step + 1) % 60 == 0:
            t = (step + 1) * DT
            parts = [f"{PROFILES[i]['name'][:7]}: {np.degrees(angle_history[i,step]):5.1f}° "
                     f"{torque_history[i,step]:+5.1f}Nm"
                     for i in range(num_envs)]
            print(f"  t={t:4.1f}s  |  {'  '.join(parts)}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # ── 플롯 저장 ────────────────────────────────────────────────────────
    n = max_steps
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("무릎 진자: 환자 프로파일 비교 (IsaacGym)\nSpastic > Healthy > Flaccid", fontsize=13)

    for i, p in enumerate(PROFILES):
        axes[0].plot(time_arr[:n], np.degrees(angle_history[i, :n]),
                     color=COLORS_PLOT[i], linewidth=2, label=p["name"])
        axes[1].plot(time_arr[:n], torque_history[i, :n],
                     color=COLORS_PLOT[i], linewidth=1.5, alpha=0.85, label=p["name"])

    axes[0].axhline(np.degrees(INITIAL_ANGLE), color="gray", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("L_Knee 굴곡 각도 (deg)")
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Bio-Torque (Nm)")
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    finals = [np.degrees(angle_history[i, n-1]) for i in range(num_envs)]
    note = "  |  ".join(f"{PROFILES[i]['name'][:7]}: {finals[i]:.1f}°" for i in range(num_envs))
    axes[0].text(0.98, 0.02, note, transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "knee_pendulum_profiles.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n플롯 저장: {out_path}")

    # ── 요약 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    s_f = np.degrees(angle_history[1, n-1])
    h_f = np.degrees(angle_history[0, n-1])
    f_f = np.degrees(angle_history[2, n-1])
    print(f"  Healthy   최종: {h_f:.1f}°")
    print(f"  Spastic   최종: {s_f:.1f}°")
    print(f"  Flaccid   최종: {f_f:.1f}°")
    ok = (s_f > h_f) and (h_f > f_f)
    print(f"  순서 Spastic > Healthy > Flaccid: {'✅' if ok else '❌'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

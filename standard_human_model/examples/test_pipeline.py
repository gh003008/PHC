"""
전체 근골격 파이프라인 테스트.

IsaacGym 없이 CPU에서 동작 확인.

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python -m standard_human_model.examples.test_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from standard_human_model.core.human_body import HumanBody
from standard_human_model.core.skeleton import NUM_DOFS


def test_model_loading():
    """모델 로드 및 요약 출력."""
    print("=" * 70)
    print("1. 모델 로드 테스트")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=4,
        device="cpu",
    )

    print(body.summary())
    print(f"\nInfo: {body.info()}")
    print("모델 로드 성공")


def test_passive_dynamics():
    """수동 역학 테스트: descending_cmd=0 (능동 제어 없음)."""
    print("\n" + "=" * 70)
    print("2. 수동 역학 테스트 (능동 명령 0)")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1,
        device="cpu",
    )

    # 관절이 약간 벗어난 상태
    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_pos[0, 3] = 0.5   # L_Knee 굴곡 0.5 rad
    dof_vel = torch.zeros(1, NUM_DOFS)
    dof_vel[0, 3] = 1.0   # L_Knee 속도 +1 rad/s

    # 능동 명령 없음
    cmd = torch.zeros(1, body.num_muscles)

    torques = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/30)

    print(f"L_Knee 위치: {dof_pos[0, 3]:.2f} rad, 속도: {dof_vel[0, 3]:.2f} rad/s")
    print(f"L_Knee 토크 (x,y,z): {torques[0, 3:6].tolist()}")
    print("  → 능동 명령 0이어도 수동 토크 발생 (passive F-L + damping + ligament)")

    # 발목 배굴 위치에서의 수동 토크
    dof_pos2 = torch.zeros(1, NUM_DOFS)
    dof_pos2[0, 6] = 0.3  # L_Ankle 배굴
    dof_vel2 = torch.zeros(1, NUM_DOFS)

    torques2 = body.compute_torques(dof_pos2, dof_vel2, cmd, dt=1/30)
    print(f"\nL_Ankle 위치: {dof_pos2[0, 6]:.2f} rad (배굴)")
    print(f"L_Ankle 토크 (x,y,z): {torques2[0, 6:9].tolist()}")


def test_active_contraction():
    """능동 수축 테스트: 특정 근육 활성화."""
    print("\n" + "=" * 70)
    print("3. 능동 수축 테스트")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1,
        device="cpu",
    )

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)

    # 근육 이름 → 인덱스 매핑
    muscle_names = body.moment_arm.muscle_names
    print(f"근육 수: {len(muscle_names)}")
    print(f"근육 목록: {muscle_names}")

    # 햄스트링만 100% 활성화
    cmd = torch.zeros(1, body.num_muscles)
    ham_idx = muscle_names.index("hamstrings_L")
    cmd[0, ham_idx] = 1.0

    # 여러 step 실행 (activation dynamics 때문에 즉시 최대가 아님)
    for step in range(5):
        torques = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/30)

    activation = body.get_activation()
    print(f"\n햄스트링(L) 활성화 후 (5 steps):")
    print(f"  hamstrings_L activation: {activation[0, ham_idx]:.4f}")
    print(f"  L_Hip 토크 (x): {torques[0, 0]:.2f} Nm  (신전 방향)")
    print(f"  L_Knee 토크 (x): {torques[0, 3]:.2f} Nm  (굴곡 방향)")
    print("  → bi-articular: 하나의 근육이 두 관절에 동시 토크!")


def test_biarticular_coupling():
    """Bi-articular coupling 검증."""
    print("\n" + "=" * 70)
    print("4. Bi-articular Coupling 검증")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1,
        device="cpu",
    )

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    muscle_names = body.moment_arm.muscle_names

    # 이관절근 목록
    coupling = body.moment_arm.get_coupling_info()
    print("Bi-articular 근육:")
    for name, joints in coupling.items():
        if len(joints) > 1:
            print(f"  {name}: {' ↔ '.join(joints)}")

    # 비복근(gastrocnemius) 활성화 → 무릎 + 발목 동시 토크
    cmd = torch.zeros(1, body.num_muscles)
    gas_idx = muscle_names.index("gastrocnemius_L")
    cmd[0, gas_idx] = 1.0

    for _ in range(10):
        torques = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/30)

    print(f"\n비복근(L) 활성화 결과:")
    print(f"  L_Knee 토크 (x): {torques[0, 3]:.2f} Nm  (굴곡)")
    print(f"  L_Ankle 토크 (x): {torques[0, 6]:.2f} Nm  (족저굴)")
    print("  → 하나의 근육이 무릎 굴곡 + 발목 족저굴 동시 생성")


def test_stretch_reflex():
    """Stretch reflex 테스트: 빠른 신장 → 반사적 수축."""
    print("\n" + "=" * 70)
    print("5. Stretch Reflex 테스트")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=2,
        device="cpu",
    )

    # env 0: 느린 신장 (역치 이하)
    # env 1: 빠른 신장 (역치 이상, 반사 발동)
    dof_pos = torch.zeros(2, NUM_DOFS)
    dof_vel = torch.zeros(2, NUM_DOFS)
    dof_vel[0, 6] = 0.05    # L_Ankle 느린 배굴
    dof_vel[1, 6] = 2.0     # L_Ankle 빠른 배굴 → 족저굴근 빠르게 신장

    cmd = torch.zeros(2, body.num_muscles)

    for _ in range(5):
        torques = body.compute_torques(dof_pos, dof_vel, cmd, dt=1/30)

    print("느린 배굴 (0.05 rad/s):")
    print(f"  L_Ankle 토크: {torques[0, 6]:.4f} Nm")
    print("빠른 배굴 (2.0 rad/s):")
    print(f"  L_Ankle 토크: {torques[1, 6]:.4f} Nm")
    print("  → 빠른 신장에서 더 큰 저항 (stretch reflex)")


def test_activation_dynamics():
    """Activation dynamics 테스트: 시정수에 따른 지연."""
    print("\n" + "=" * 70)
    print("6. Activation Dynamics 테스트")
    print("=" * 70)

    body = HumanBody.from_config(
        muscle_def_path="muscle_definitions.yaml",
        param_path="healthy_baseline.yaml",
        num_envs=1,
        device="cpu",
    )

    dof_pos = torch.zeros(1, NUM_DOFS)
    dof_vel = torch.zeros(1, NUM_DOFS)
    cmd = torch.ones(1, body.num_muscles)  # 전체 근육 100% 명령

    muscle_names = body.moment_arm.muscle_names
    sol_idx = muscle_names.index("soleus_L")
    tib_idx = muscle_names.index("tibialis_ant_L")

    print(f"{'Step':>4}  {'Soleus act':>12}  {'TibAnt act':>12}")
    print("-" * 35)
    for step in range(15):
        body.compute_torques(dof_pos, dof_vel, cmd, dt=1/30)
        act = body.get_activation()
        if step % 2 == 0:
            print(f"{step:>4}  {act[0, sol_idx]:>12.4f}  {act[0, tib_idx]:>12.4f}")

    print("  → Soleus (tau_act=0.020s)가 TibAnt (tau_act=0.015s)보다 느리게 활성화")


if __name__ == "__main__":
    test_model_loading()
    test_passive_dynamics()
    test_active_contraction()
    test_biarticular_coupling()
    test_stretch_reflex()
    test_activation_dynamics()
    print("\n" + "=" * 70)
    print("모든 테스트 완료!")
    print("=" * 70)

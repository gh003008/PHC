"""
Standard Human Model 동작 확인 스크립트.

사용법:
    conda activate phc
    cd /home/gunhee/workspace/PHC
    python -m standard_human_model.examples.test_patient_dynamics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from standard_human_model.core.patient_profile import PatientProfile, JOINT_GROUPS
from standard_human_model.core.patient_dynamics import PatientDynamics, NUM_DOFS


def test_profile_loading():
    """모든 프로파일 로드 테스트."""
    profiles = [
        "healthy/healthy_adult",
        "sci/sci_t10_complete_flaccid",
        "sci/sci_incomplete_spastic",
        "stroke/stroke_r_hemiplegia",
        "parkinson/parkinson_moderate",
        "cp/cp_spastic_diplegia",
    ]

    print("=" * 60)
    print("프로파일 로드 테스트")
    print("=" * 60)

    for p_path in profiles:
        profile = PatientProfile.load(p_path)
        print(f"\n{profile}")
        print(f"  72-dim vector (처음 8개): {profile.get_param_vector()[:8]}")

    print("\n모든 프로파일 로드 성공")


def test_dynamics_computation():
    """토크 계산 테스트 (CPU)."""
    device = "cpu"
    num_envs = 4

    profiles_to_test = {
        "healthy": "healthy/healthy_adult",
        "SCI flaccid": "sci/sci_t10_complete_flaccid",
        "SCI spastic": "sci/sci_incomplete_spastic",
        "Stroke R hemi": "stroke/stroke_r_hemiplegia",
        "Parkinson": "parkinson/parkinson_moderate",
    }

    # 공통 입력
    dof_pos = torch.zeros(num_envs, NUM_DOFS, device=device)
    dof_vel = torch.ones(num_envs, NUM_DOFS, device=device) * 0.5  # 일정 속도
    pd_targets = torch.ones(num_envs, NUM_DOFS, device=device) * 0.1  # 약간의 목표
    kp = torch.ones(NUM_DOFS, device=device) * 200.0
    kd = torch.ones(NUM_DOFS, device=device) * 20.0
    sim_time = 1.0

    print("\n" + "=" * 60)
    print("토크 계산 비교 (dof_vel=0.5, pd_target=0.1)")
    print("=" * 60)

    # 관절 그룹별 DOF 인덱스 (대표 1개씩)
    from standard_human_model.core.patient_dynamics import GROUP_DOF_RANGES
    repr_dofs = {g: GROUP_DOF_RANGES[g][0] for g in JOINT_GROUPS}

    for name, path in profiles_to_test.items():
        profile = PatientProfile.load(path)
        dynamics = PatientDynamics(profile, num_envs, device)
        torque = dynamics.compute_torques(dof_pos, dof_vel, pd_targets, kp, kd, sim_time)

        print(f"\n[{name}]")
        info = dynamics.info()
        print(f"  tremor={info['has_tremor']}, spasticity={info['has_spasticity']}, "
              f"tau_active=[{info['min_tau_active']:.1f}, {info['max_tau_active']:.1f}]")

        for group in JOINT_GROUPS:
            dof_idx = repr_dofs[group]
            t = torque[0, dof_idx].item()
            print(f"  {group:<14}: torque = {t:>10.2f} Nm")


def test_profile_summary():
    """프로파일 요약 표시 테스트."""
    print("\n" + "=" * 60)
    print("프로파일 요약 (SCI Incomplete Spastic)")
    print("=" * 60)

    profile = PatientProfile.load("sci/sci_incomplete_spastic")
    print(profile.summary())

    print("\n" + "=" * 60)
    print("프로파일 요약 (Stroke R Hemiplegia)")
    print("=" * 60)

    profile = PatientProfile.load("stroke/stroke_r_hemiplegia")
    print(profile.summary())


def test_healthy_shortcut():
    """PatientProfile.healthy() 편의 메서드 테스트."""
    profile = PatientProfile.healthy()
    assert profile.name == "healthy_adult"
    assert profile.get_group_params("L_Hip")["tau_active"] == 1.0
    print("\nhealthy() shortcut 테스트 통과")


if __name__ == "__main__":
    test_profile_loading()
    test_dynamics_computation()
    test_profile_summary()
    test_healthy_shortcut()
    print("\n모든 테스트 통과!")

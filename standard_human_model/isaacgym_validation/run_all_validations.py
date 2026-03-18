"""
run_all_validations.py — 근골격계 파이프라인 IsaacGym 통합 전 정량 검증

IsaacGym 없이 CPU에서 실행. 각 관절별 근육 특성이 정확히 반영되는지 확인.

실험 목록:
  1. Per-joint passive torque profile (각도 sweep → passive force + ligament)
  2. Stretch reflex response (속도 sweep → reflex torque)
  3. Muscle activation → joint torque mapping (R matrix 검증)
  4. Co-contraction impedance (공수축 비율별 관절 강성)
  5. Patient profile comparison (healthy vs spastic vs flaccid)

사용법:
  conda activate phc
  cd /home/gunhee/workspace/PHC
  python standard_human_model/isaacgym_validation/run_all_validations.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from standard_human_model.isaacgym_validation.exp1_passive_torque_profile import run as run_exp1
from standard_human_model.isaacgym_validation.exp2_stretch_reflex import run as run_exp2
from standard_human_model.isaacgym_validation.exp3_muscle_torque_mapping import run as run_exp3
from standard_human_model.isaacgym_validation.exp4_cocontraction import run as run_exp4
from standard_human_model.isaacgym_validation.exp5_patient_comparison import run as run_exp5


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Musculoskeletal Pipeline Validation Suite")
    print("=" * 70)

    experiments = [
        ("Exp1: Passive Torque Profile", run_exp1),
        ("Exp2: Stretch Reflex Response", run_exp2),
        ("Exp3: Muscle→Torque Mapping", run_exp3),
        ("Exp4: Co-contraction Impedance", run_exp4),
        ("Exp5: Patient Profile Comparison", run_exp5),
    ]

    for name, func in experiments:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        func(output_dir)
        print(f"  [DONE] {name}")

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

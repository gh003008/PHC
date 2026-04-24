# KIT_425 PERSONAL v_cmd 3-way Ablation

상위 plan 문서: `01_research_docs/260424_kit425_personal_vcmd_3way_plan.md`

## 슬롯 요약

| Slot | retime | cmd_w | 의도 |
|---|---|---|---|
| **A** NO_RETIME | OFF | 0.3 | retime 없이 cmd_reward로만 v_cmd 학습 가능? |
| **B** RETIME_PURE | ON [0.7, 1.4] | 0.0 | retime만으로 충분 (main candidate) |
| **C** RETIME_CMDW | ON [0.7, 1.4] | 0.3 | retime + cmd_reward 안전망 |

## 데이터

- 모션: `sample_data/amass_isaac_walking_primitive.pkl` (KIT_425 3 clip 필터)
- 필터: `sample_data/amass_isaac_walking_primitive_kit425_only.json`
- 클립 속도: 0.371 / 0.652 / 0.845 m/s (단일 subject KIT_425)
- v_cmd 범위: U(0.40, 0.82) m/s

## 서버에서 실행

```bash
cd ~/PHC
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_A_gpu0.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_B_gpu1.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_C_gpu2.sh
squeue -u $USER
```

## 평가

```bash
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_A.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_B.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_C.sh
```

체크포인트: `output/KIT425_PERSONAL_{A,B,C}.pth` + milestone `*_NNNNNNNN.pth` (2500 단위).

## 코드 변경

**없음.** 기존 `HumanoidImVICCmdMultiClip` (KIT425에서 사용) 가 이미:
- Smart sampling (`v_diff.argmin`) 구현됨
- `multiclip_retime_enabled` 토글 처리
- `cmd_tracking_w` 처리

→ env yaml의 토글만 바꾸면 3가지 ablation 자동 분기.

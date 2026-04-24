# KIT_425 PERSONAL v_cmd 3-way Ablation 실행 plan (2026-04-24)

작성일: 2026-04-24
상위 맥락: `02_research_dev/260424_all_experiments_stacked_VIC04_to_KIT425.md`
실험 디렉토리: `exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/`

---

## 1. 목표

**최종 목표**: IsaacGym에서 사용자가 v_cmd를 입력하면, 단일 subject(KIT_425)의 walking style을 imitate하면서 그 v_cmd 속도로 걷는 personalized humanoid 정책.

**이번 실험 목표**: 위 정책을 학습 가능한지 검증하고, **retime / cmd_reward의 기여도를 분리 측정**.

---

## 2. 배경 (현재까지 학습 결과 요약)

전체 stack은 `02_research_dev/260424_all_experiments_stacked_VIC04_to_KIT425.md` 참조. 핵심:

- **AMASS_CMD_V2_B** (260419): single AMASS clip + v_cmd input, **86% success**. v_cmd 머신러리 검증됨.
- **KIT425_PHASEFIX MAIN** (260423): 3-clip multi-clip + per-clip phase fix, v_cmd **없음**. 로컬 test 280/88 (eps_len 64% of ceiling).
- **MULTICLIP_FWD** (260423): 22-clip + v_cmd, **0% success**. 원인: v_cmd와 reference clip 속도 mismatch 발산 + phase obs hardcoded(KIT425에서 fix).

**Gap**: v_cmd 입력은 single clip에서만 (CMD_V2_B), multi-clip은 v_cmd 없거나 (KIT425) 망함 (MULTICLIP_FWD). 이 두 갈래를 합쳐야 함.

---

## 3. 설계 — 3-way ablation

기존 `HumanoidImVICCmdMultiClip`이 이미 retime + smart sampling을 모두 지원하기 때문에 **코드 변경 0**, env yaml 토글만으로 3-way 가능.

| Slot | clips | smart sampling | retime | cmd_w | 측정하는 것 |
|---|---|---|---|---|---|
| **A** NO_RETIME | KIT_425 3 clip | ON (closest) | **OFF** | **0.3** | retime 없이 cmd_reward로만 v_cmd 학습 가능? |
| **B** RETIME_PURE | KIT_425 3 clip | ON | **[0.7, 1.4]** | **0.0** | retime만으로 충분? (cleanest, main 후보) |
| **C** RETIME_CMDW | KIT_425 3 clip | ON | **[0.7, 1.4]** | **0.3** | retime + cmd_reward 둘 다 (안전망) |

### 3.1 ablation 의미

- **A vs B**: retime의 기여 (smart sampling은 둘 다 ON)
- **B vs C**: cmd_reward의 기여 (retime은 둘 다 ON)
- A vs C: confounded — 직접 비교 X

### 3.2 retime이 무엇을 하는가

```
v_cmd ~ U(0.40, 0.82) sampling
  → smart sampling: argmin |v_clip - v_cmd| 으로 가까운 clip 선택
  → retime scale s = clamp(v_cmd / v_clip_natural, 0.7, 1.4)
  → reference motion이 시간 축 s 배 빨리/느리게 재생됨
  → 결과적으로 reference가 정확히 v_cmd 속도로 움직임
  → policy가 reference만 imitate해도 v_cmd 추종됨 (cmd_reward redundant)
```

### 3.3 retime을 끄면 (Slot A)

reference는 clip 원래 속도로 재생. v_cmd는 따로 obs로 들어감. **cmd_reward (cmd_w=0.3)** 가 v_cmd ↔ output speed 결합을 만듦. mismatch가 있어 학습이 더 어렵지만 가능 (single clip CMD_V2_B가 86% 성공한 메커니즘과 동일).

---

## 4. 데이터

- **모션 lib**: `sample_data/amass_isaac_walking_primitive.pkl` (전체 AMASS)
- **클립 필터**: `sample_data/amass_isaac_walking_primitive_kit425_only.json` (KIT_425 3 clip 선택)
- **사용 클립** (단일 subject KIT_425):
  - `0-KIT_425_walking_03_poses` (0.371 m/s, 6.5s)
  - `0-KIT_425_walking_medium08_poses` (0.652 m/s, 5.4s)
  - `0-KIT_425_walking_medium05_poses` (0.845 m/s, 4.2s)
- **v_cmd 범위**: U(0.40, 0.82) m/s — 데이터 범위 안. extrapolation 안 함.

향후 확장: `walking_fast05_poses` (0.91 m/s)을 추가하면 4 clip × 0.37–0.91 m/s 범위 가능. (이번 plan에서는 KIT425와 동일한 3 clip 유지하여 변수 단순화.)

---

## 5. 공통 설정 (3 슬롯 모두 동일)

| 영역 | 항목 | 값 |
|---|---|---|
| env | `cycle_motion` | False (KIT425 검증값) |
| env | `vic_phase_obs` | True (per-clip phase fix) |
| env | `vic_curriculum_stage` | 2 (CCF 학습 활성화) |
| env | `vic_ccf_num_groups` | 4 |
| env | `episode_length` | 300 |
| env | `power_coefficient` | 5e-7 |
| env | `reward_curriculum_switch_epoch` | 10000 |
| env | `multiclip_v_cmd_range` | [0.40, 0.82] |
| learning | `max_epochs` | 20000 |
| learning | `save_frequency` | 2500 (milestone ckpt 7개) |
| learning | `learning_rate` | 5e-5 |
| learning | mlp units | [1024, 1024, 512, 512] |
| learning | sigma_init `val` | -2.9 (PD action) |

slot간 차이: `multiclip_retime_enabled`, `retime_scale_range`, `cmd_tracking_w` 만.

---

## 6. 실행 절차 (서버 — exolab GPU server `server1`)

### 6.1 환경 준비

```bash
ssh server1                               # SSH 로그인
cd ~/PHC
git pull origin Jimin                     # 새 config 가져오기
ls exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/
# 출력: 3 env yaml + 1 learning yaml + 6 sbatch script + README
```

### 6.2 학습 (3 GPU 동시 제출)

```bash
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_A_gpu0.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_B_gpu1.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/train_personal_C_gpu2.sh
squeue -u $USER                           # 큐 확인
```

소요시간: **각 ~13–15시간** (KIT425와 동일 config), 병렬 → wall clock ~13–15h.

진행 모니터링:
```bash
tail -f logs/kit425_personal_A_*.out      # epoch별 reward / eps_len
tail -f logs/kit425_personal_B_*.out
tail -f logs/kit425_personal_C_*.out
```

체크포인트:
- canonical: `output/KIT425_PERSONAL_{A,B,C}.pth`
- milestone (2500 epoch 단위): `output/KIT425_PERSONAL_{A,B,C}_{NNNNNNNN}.pth`

### 6.3 평가 (학습 완료 후)

```bash
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_A.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_B.sh
sbatch exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/test_personal_C.sh
```

각 ~10분. 출력:
- `logs/test_kit425_personal_*.out` 의 `av reward: X av steps: Y` 라인
- (선택) v_cmd 고정 평가: 별도 스크립트로 v_cmd ∈ {0.45, 0.55, 0.70, 0.80} 각각 100 episode 후 v_err 계산 (다음 단계)

### 6.4 로컬 시각화 (각 ckpt를 scp 후)

```bash
# 로컬에서 (exolab-MS-7D56)
scp server1:~/PHC/output/KIT425_PERSONAL_B.pth ~/PHC/output/
cd ~/PHC
python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_B.yaml \
  --cfg_train exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
  --num_envs 1 --test --epoch -1 \
  --experiment KIT425_PERSONAL_B
```

(헤드 없이 빨리 보려면 `--no_virtual_display --headless` 추가)

---

## 7. 예상 결과 / 성공 기준

### 7.1 정성적 예측

| Slot | 예상 success rate | walk 가능성 | 근거 |
|---|---|---|---|
| **A** NO_RETIME | 65–80% | 🟡 중간 | MULTICLIP_FWD 0% 변형이지만 (1) phase fix (2) smart sampling 으로 보강. mismatch 0.15 학습되면 OK |
| **B** RETIME_PURE | 85–95% | 🟢 높음 | retime이 mismatch 0으로 만듦, cleanest |
| **C** RETIME_CMDW | 85–95% | 🟢 높음 | B와 동급, cmd_reward 안전망 |

→ **세 슬롯 중 적어도 하나가 90%+ 가능성 매우 높음**.

### 7.2 정량 성공 기준

- **walk OK**: test-greedy `av_steps ≥ 90` (episode_length 300의 30%. ceiling은 cycle=False라 clip duration × 30fps ≈ 130~190 frames이므로 90이면 의미 있는 step)
- **cmd 추종 OK**: v_cmd 고정 평가에서 v_err ≤ 0.1 m/s (다음 단계)
- **이상적**: av_steps ≥ 130 + v_err ≤ 0.05 → 사실상 100% 완주 + 정확 추종

### 7.3 결과 비교 표 (학습 후 채울 표 — 사전 정의)

```
| Slot | training rwd | training eps_len | test av_rwd | test av_steps | walk OK? |
|------|--------------|-------------------|-------------|----------------|----------|
| A NO_RETIME    | ?  | ?  | ?  | ?  | ?  |
| B RETIME_PURE  | ?  | ?  | ?  | ?  | ?  |
| C RETIME_CMDW  | ?  | ?  | ?  | ?  | ?  |
```

---

## 8. Failure mode 및 fallback

### 8.1 모든 슬롯 실패 (av_steps < 50)
- 원인 후보: `vic_curriculum_stage=2`가 너무 일찍 켜져 destabilize. → fallback: stage 1 (CCF=0)로 재실행.
- 또는 reward curriculum switch (Ep 10000) 이 destabilizer. → fallback: `reward_curriculum_switch_epoch: 999999` (사실상 비활성).

### 8.2 A만 실패, B/C 성공
- → "retime이 multi-clip + v_cmd에 필수" 결론. A는 폐기, B를 main으로.

### 8.3 B 실패, C 성공
- → "retime만으론 부족, cmd_reward 보강 필요". C를 main으로.

### 8.4 A 가장 좋음 (예상 X 시나리오)
- → "retime이 오히려 방해 (예: AMP disc와 충돌)". A를 main으로 채택, retime 코드 점검.

### 8.5 canonical < milestone
- KIT425에서도 있었던 케이스. milestone sweep (5k/10k/15k/20k)으로 best 찾기. `output/KIT425_PERSONAL_*_NNNNNNNN.pth` 사용.

---

## 9. 다음 단계 (이 실험 후)

1. **v_cmd grid 평가**: best slot ckpt로 v_cmd ∈ {0.40, 0.50, 0.60, 0.70, 0.80} 고정 평가, v_err 측정. 추종 정확도 정량화.
2. **결과 doc 작성**: `02_research_dev/YYMMDD_kit425_personal_vcmd_results.md` (CLAUDE.md 규칙 한글).
3. **만약 잘 되면**: 4 clip로 확장 (walking_fast05 추가) → v_cmd 범위 [0.4, 0.9].
4. **궁극 목표 (장기)**: KIT_424 다른 subject와 비교 → personalization의 의미 검증. VIC stage 2 본격 활용 (CCF 분석).

---

## 10. 변경 파일 요약 (이 plan으로 추가/생성된 것)

```
exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/
├── README.md
├── env_im_walk_vic_kit425_personal_A.yaml      # NO_RETIME, cmd_w=0.3
├── env_im_walk_vic_kit425_personal_B.yaml      # RETIME_PURE, cmd_w=0.0
├── env_im_walk_vic_kit425_personal_C.yaml      # RETIME_CMDW, cmd_w=0.3
├── im_walk_vic_kit425_personal.yaml            # shared learning config
├── train_personal_A_gpu0.sh                     # sbatch (idx0 GPU)
├── train_personal_B_gpu1.sh                     # sbatch (idx1 GPU)
├── train_personal_C_gpu2.sh                     # sbatch (idx2 GPU)
├── test_personal_A.sh                           # 평가 sbatch
├── test_personal_B.sh
└── test_personal_C.sh

01_research_docs/
└── 260424_kit425_personal_vcmd_3way_plan.md    # 이 문서
```

코드 변경: **없음** (기존 `phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` 가 모든 토글 지원).

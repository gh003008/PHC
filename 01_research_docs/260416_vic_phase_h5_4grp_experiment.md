# VIC_PHASE_H5_4grp 실험 설정 및 서버 실행 가이드

**Version**: 1.0 — 2026-04-16
**실험 ID**: `VIC_PHASE_H5_4grp`
**Related**:
- [260416_phase0_vic_feasibility_and_next_steps.md](./260416_phase0_vic_feasibility_and_next_steps.md)
- [260416_review_and_modified_plan_for_vic_impedance_action.md](./260416_review_and_modified_plan_for_vic_impedance_action.md)

---

## 1. 실험 목적

Phase 0 feasibility 분석 결과:
- **VIC 8grp (Cell C)**: 392 rwd / 130 steps — 성능 저하
- **VIC 4grp (Cell E)**: 330 rwd / 115 steps — 8grp 보다는 나으나 여전히 no-VIC 대비 열위
- **no-VIC (Cell D)**: 605 rwd / 192 steps — 현재 H5 최고

그러나 gunhee 의 VIC_PHASE (2026-03-12) 는 **932 rwd / 299 steps** (AMASS forward_single) 로 사실상 트립 없이 완주했음. 현재 H5 VIC 실험들과의 결정적 차이를 코드/설정 전 구간에서 비교한 결과:

> **유일한 config 차이는 `vic_phase_obs: True` 한 줄 누락. 그 외 AMP disc, PPO, MLP, CCF sigma, reward curriculum, task/disc 가중치는 전부 동일. 코드 drift 는 additive-only 로 default 비활성이므로 영향 없음.**

따라서 본 실험은 "**gunhee VIC_PHASE 설정 + 4 groups + H5 motion**" 으로 재구성하여 phase observation 과 4-group 축소의 결합 효과를 측정한다.

---

## 2. 구성 요약

### 베이스: gunhee VIC_PHASE
- `exp_config/forward_walking/260312_VIC_PHASE/env_im_walk_vic.yaml` 의 env 설정 전체 상속
- CCF sigma `-1.0`, reward curriculum switch 10k, power_coefficient 5e-7, cycle_motion True 등 그대로

### 변경점 (VIC_PHASE 대비)
| 항목 | VIC_PHASE (gunhee) | VIC_PHASE_H5_4grp (신규) |
|---|---|---|
| motion_file | `amass_isaac_walking_forward_single.pkl` | `h5_motion_library.pkl` |
| vic_ccf_num_groups | 8 | **4** (L_Hip+Knee, L_Ankle+Toe, R_Hip+Knee, R_Ankle+Toe; 상체 고정) |
| exp_name | VIC_PHASE | VIC_PHASE_H5_4grp |

### 유지되는 핵심 설정
| 항목 | 값 |
|---|---|
| vic_enabled | True |
| vic_curriculum_stage | 2 (learnable CCF) |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_ccf_min / max | -1.0 / 1.0 |
| **vic_phase_obs** | **True** (+2 obs dims, sin/cos gait phase) ← Cell C/E 에서 누락되었던 핵심 |
| power_coefficient | 5e-7 |
| cycle_motion | True |
| reward_curriculum_switch_epoch | 10000 |
| reward_w_stage1_task / disc | 0.7 / 0.3 |
| reward_w_stage2_task / disc | 0.3 / 0.7 |
| task_reward_w / disc_reward_w | 0.5 / 0.5 (learning yaml) |
| MLP units | [1024, 1024, 512, 512] |
| learning_rate | 5e-5 |
| max_epochs | 20000 |
| num_envs | 512 |

---

## 3. 수정된 파일 (git-tracked)

### 3.1 `phc/data/cfg/env/env_im_walk_vic_4grp.yaml`
```yaml
# VIC: CCF sigma override (log-std, -1.0 => std=0.37 vs PD default -2.9 => std=0.055)
vic_ccf_sigma_init: -1.0
# VIC: Phase observation (sin/cos encoding of gait cycle phase, +2 obs dims) — matches gunhee VIC_PHASE
vic_phase_obs: True   # ← 추가된 한 줄
# VIC: Reward Weight Curriculum ...
```

### 3.2 `phc/data/cfg/learning/im_walk_vic_4grp.yaml`
```yaml
config:
    name: VIC_PHASE_H5_4grp   # 기존: VIC_CCF_ON2_H5_4grp → 변경 (기존 Cell E 체크포인트 보존 목적)
```

### 3.3 `train_phc_vic_4grp.sh`
```bash
#SBATCH -J phc_vic_phase_4grp   # 기존: phc_vic_4grp → 변경 (로그 파일 구분)
```
나머지는 기존과 동일: partition `idx0`, `--gres=gpu:idx0:1`, `--mem=15G`, 24h wall, `no_log=True`.

---

## 4. 서버 실행 가이드

### 4.1 Prereq — 현재 실행 중인 Cell E 잡 종료
```bash
ssh server1-jiminyoun
squeue -u $USER
# phc_vic_4grp 잡 ID 확인 후
scancel <JOB_ID>
# 또는 이름 기반: scancel -n phc_vic_4grp
```

Cell E 체크포인트 (`~/PHC/output/VIC_CCF_ON2_H5_4grp.pth`) 는 보존됨 — 신규 실험은 `VIC_PHASE_H5_4grp` 이름으로 별도 저장.

### 4.2 최신 코드 가져오기
```bash
cd ~/PHC
git pull origin Jimin
```

확인 사항:
- `phc/data/cfg/env/env_im_walk_vic_4grp.yaml` 에 `vic_phase_obs: True` 있는지
- `phc/data/cfg/learning/im_walk_vic_4grp.yaml` 의 `name: VIC_PHASE_H5_4grp`
- `train_phc_vic_4grp.sh` job name `phc_vic_phase_4grp`

### 4.3 sbatch 제출
```bash
sbatch train_phc_vic_4grp.sh
# Submitted batch job <NEW_JOB_ID>
```

### 4.4 모니터링
```bash
squeue -u $USER
tail -f logs/phc_vic_phase_4grp_<JOB_ID>.out
# 기존 monitor_progress.sh 가 있으면 재활용 가능 (패턴은 동일)
```

주요 체크포인트:
- **Ep 1000**: `rwd > 150`, `eps_len > 50` 이면 phase_obs 효과 가시화 시작
- **Ep 5000**: 이전 Cell E (Ep 12,674 에서 330/115) 를 이미 상회해야 함
- **Ep 10000** (curriculum switch): 600 이상 기대
- **Ep 20000** (최종): 목표 **800+ rwd / 250+ steps**

---

## 5. 판단 기준 및 다음 단계

### 5.1 실험 성공 시 (Ep 20k: rwd ≥ 800, steps ≥ 250)
- `vic_phase_obs` 가 H5 VIC 학습의 핵심 missing piece 였음이 확정
- **이 체크포인트를 Phase 1 perturbation curriculum 의 baseline 으로 사용** (260416 review&modified_plan 문서 §6.2 Stage 2)
- 추가로 8grp 에도 phase_obs 추가해서 8grp vs 4grp 직접 비교 (ablation)

### 5.2 부분 성공 시 (rwd 500~800, steps 150~250)
- phase_obs 가 기여하긴 하나 H5 motion 의 본질적 난이도가 상한을 제한
- no-VIC Cell D (605/192) 와 유사 수준이면, "VIC 가 nominal H5 walking 에서 이득 없음" 결론 재확인
- 계획: residual VIC head 로 전환 (review&modified_plan 문서 §5)

### 5.3 실패 시 (rwd < 500, steps < 150)
- 코드 drift (비록 default-off 이지만) 혹은 H5 motion gait phase 사전계산 이슈 가능성
- `_precompute_gait_phase` 가 H5 의 긴 treadmill 시퀀스에서 제대로 작동하는지 로그 확인
- Fallback: gunhee 의 `humanoid_im_vic.py` 스냅샷 (`exp_config/forward_walking/260312_VIC_PHASE/humanoid_im_vic.py`) 으로 임시 교체해 코드 drift 가 원인인지 격리

---

## 6. 파일 위치 참고

### 로컬 및 서버 공통 (git-tracked, Jimin branch)
- `phc/data/cfg/env/env_im_walk_vic_4grp.yaml` ← 본 실험 env config
- `phc/data/cfg/learning/im_walk_vic_4grp.yaml` ← 본 실험 learning config
- `train_phc_vic_4grp.sh` ← sbatch 스크립트
- `exp_config/forward_walking/260312_VIC_PHASE/` ← 비교 기준 (gunhee 원본 스냅샷)

### 서버 전용 (학습 산출물)
- 체크포인트: `~/PHC/output/VIC_PHASE_H5_4grp.pth` (latest) + 100 epoch 간격 snapshot (`VIC_PHASE_H5_4grp_00000100.pth` 등)
- 로그: `~/PHC/logs/phc_vic_phase_4grp_<JOB_ID>.out` / `.err`

### 기존 비교 대상 체크포인트 (서버)
- `~/PHC/output/VIC_CCF_ON2_H5.pth` (Cell C, 8grp, 392/130)
- `~/PHC/output/VIC_CCF_ON2_H5_4grp.pth` (Cell E, 4grp no phase_obs, 330/115)
- `~/PHC/output/VIC_CCF_ON2_NoVIC_H5.pth` (Cell D, no VIC, 605/192)

---

## 7. 요약 한 줄

> **gunhee 가 잘 되던 VIC_PHASE 설정에서 impedance 그룹 수만 8 → 4 로 줄이고 motion 을 H5 로 교체. 기타 모든 하이퍼파라미터 및 `vic_phase_obs` 는 유지.** Phase observation 단독 기여도를 4-group 조건에서 정량 측정하고, 동시에 H5 정상 보행에서 VIC 의 실제 상한을 가늠한다.

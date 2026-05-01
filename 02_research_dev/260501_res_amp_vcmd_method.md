# RES_AMP_VCMD: PHC + Residual + AMP for Continuous v_cmd (260501)

작성일: 2026-05-01
브랜치: Jimin
관련 commit 범위: `7d1982c..f2eaa35`

---

## 1. 한 줄 요약

**Frozen phc_3 (PHC SMPL-PNN imitation 사전학습) + 256-256 residual MLP + AMP discriminator** 를 묶어, **연속 v_cmd ∈ [0.76, 1.23] m/s** 를 추적하는 보행 정책을 학습한다. 추론에서는 클립 전환이 사라지고 단일 정책이 v_cmd 한 값으로 자연스럽게 속도를 보간한다. **VIC4_VCMD S10-S19B 의 from-scratch 시도가 4060 Ti 에서 좌초된 후, frozen baseline 활용으로 재시작한 결과**.

---

## 2. 동기와 결정

### 왜 residual + AMP 인가
| 후보 | 핵심 | 채택 여부 |
|---|---|---|
| (A) phc_3 + 단일 클립 그대로 | v_cmd 변화 불가 | X |
| (B) phc_3 + 단일 클립 retime | 같은 패턴 stretch만 → 보폭/케이던스 미스매치 | X |
| (C) phc_3 + multi-clip + retime + a-1 hard switch (zero-training, v3 데모) | smooth 변속은 되지만 클립 사이 점프 노출 | X (인터랙티브 데모로만 유지) |
| **(D) phc_3 frozen + α head + AMP** | frozen baseline 활용 → 학습 비용 작음, 연속 변속 가능 | **채택** |
| (E) AMP-only from scratch | 4060 Ti 7.6 GB로는 사실상 불가 | X (VIC4_VCMD 시리즈에서 검증) |

### 7 가지 핵심 결정 (스펙 §2)
| ID | 결정 | 채택 |
|---|---|---|
| D1 | residual head on frozen phc_3 | a = π_phc_3(s) + Δa(s, v_cmd) |
| D2 | training-time clip switch 방식 | **a-1 (midpoints, hysteresis 0.02)** |
| D3 | AMP discriminator 학습 비중 | small (w_amp=0.3) |
| D4 | task reward | **R2 — r_track + r_im_anchor** |
| D5 | reward stage curriculum | **N1 — 처음부터 동시 활성** (no warm-up) |
| D6 | v_cmd ramp during episode | **S2 — episode-level + low-prob mid-ep ramp** |
| D7 | termination | **C2 — fall + pelvis < 0.4** |

세부 근거: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md` §2, §11

---

## 3. 아키텍처

```
                      env: HumanoidImResAMPVCmd
                      ├ self_obs (358) + task_obs (576, 24 SMPL track bodies)
                      └ + v_cmd (1)  → total obs 935-D
                                         │
                              ┌──────────┴────────────┐
                              ▼                       ▼
                       obs[:, :-1] = 934-D       full obs = 935-D
                              │                       │
                              ▼                       ▼
              ┌─────────────────────────┐   ┌──────────────────────────┐
              │  phc_3 PNN (FROZEN)     │   │  ResMLPHead (TRAINABLE)  │
              │  no_grad, eval mode     │   │  Linear→SiLU→Linear→     │
              │  hidden=[2048,1536,     │   │   SiLU→Linear            │
              │   1024,1024,512,512]    │   │  hidden=(256, 256)       │
              │  3 PNN primitives       │   │  σ_init=-1.0 (std=0.37)  │
              └──────────┬──────────────┘   └──────────┬───────────────┘
                         │ a_base (69-D)               │ Δa_mean, Δa_logstd
                         │                             │ (clamp ±0.2)
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                         a_total = a_base + Δa  (PD targets, 69-D)
                                    │
                                    ▼
                          isaac_pd torque + sim
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
             reward env        AMP disc           value (critic)
        r_track + r_im    on (s,a) sequences      935-D in
       (+ r_amp via       w_amp=0.3              [1024,512]→1
        disc reward)
```

**Reward**:
```
r_total = w_track · r_track + w_amp · r_amp + w_im · r_im + r_survive
        = 0.5 · exp(-5.0 · (v_act - v_cmd)²)
        + 0.3 · D(s, a)
        + 0.2 · exp(-2.0 · pose_dist)
        + 1.0 · alive
```

**Action clipping**: `Δa ← clamp(Δa_mean, ±0.2)`. residual 폭주 방지.

---

## 4. 신규 파일 (스펙 §4)

| 파일 | 역할 | LOC | commit |
|---|---|---|---|
| `phc/env/tasks/humanoid_im_res_amp_vcmd.py` | task class. v_cmd obs, multi-clip + a-1 switch, retime, S2 ramp, r_track + r_im, term | ~600 | Tasks 1, 4-9 |
| `phc/learning/res_amp_network.py` | ResMLPHead + ResAMPVCmdNetwork (AMPPNNBuilder.Network 상속) + ResAMPVCmdBuilder + load_frozen_phc_3 | ~280 | Tasks 10-13 + 108c0bc |
| `phc/learning/res_amp_agent.py` | ResAMPVCmdAgent(IMAmpAgent) + frozen-base no-grad assert | ~80 | Tasks 14-15 |
| `phc/data/cfg/env/env_im_res_amp_vcmd.yaml` | env config | ~135 | Task 2 + 108c0bc |
| `phc/data/cfg/learning/im_res_amp_vcmd.yaml` | smoke learning config | ~125 | Task 3 |
| `phc/data/cfg/learning/im_res_amp_vcmd_server.yaml` | server learning config (200K AMP buffers, 4096 amp batch) | ~125 | Task 19 (f57bd4c) |
| `scripts/launch_res_amp_smoke.sh` | local 4060 Ti smoke launch | – | Tasks 16, 108c0bc |
| `train_phc_res_amp_vcmd.sh` | server1 sbatch | – | Task 19 (f57bd4c) |
| `scripts/eval_res_amp_vcmd.py` | sweep + ramp eval, JSON output | ~270 | Task 21 (5848628) |
| `scripts/phc_walk_demo_residual.py` | trained-residual interactive demo | ~530 | Task 22 (5848628) |

**phc/utils/parse_task.py**: 1 줄 추가 (`HumanoidImResAMPVCmd` 등록).
**phc/run.py**: 3 줄 추가 (`amp_residual` model, `amp_pnn_residual` network, `im_amp_residual` agent + player).

---

## 5. 학습 결과

### 5.1 Stage 1 — local smoke (4060 Ti 7.6 GB)

`scripts/launch_res_amp_smoke.sh` (num_envs=32, max_epochs=500, ~7 min wall time)

| 메트릭 | 값 |
|---|---|
| 총 epoch | 501 (`MAX EPOCHS NUM!`) |
| rwd 시작/종료/평균 | 17.8 / 23.8 / 23.28 |
| eps_len 시작/종료/평균/최대 | 12.9 / 17.4 / 17.0 / 19.7 |
| OOM/NaN | 없음 |
| ckpt | `output/ResAMPVCmd{,_00000250,_00000500}.pth` |

**스펙 §8 Stage 1 통과**:
- `av_reward +30%` ✓ (+33.7%)
- `no NaN/OOM` ✓
- `av_eps_len ≥ 240` ❌ (17.4) — **smoke 한계, Stage 2 에서 회복 기대**
- 부팅 + 1 epoch + ckpt 저장 ✓

상세: `02_research_dev/260501_res_amp_vcmd_stage1_smoke_analysis.md`

### 5.2 Stage 2 — server1 본학습 (A5000 24 GB)

**계획됨, 사용자 액션 (Task 20 — sbatch 제출)**.
- 스크립트: `train_phc_res_amp_vcmd.sh`
- num_envs=512, episode_length=600, max_iterations=20000, minibatch_size=8192, AMP buffer 200K
- 예상 wall time: 36-48h
- 예상 결과: eps_len 240+ (=80% of 600), v_cmd sweep abs_err < 0.10 m/s

### 5.3 Eval — sweep + ramp

**계획됨 (Stage 2 ckpt 도착 후 자동 수행)**.
- 스크립트: `scripts/eval_res_amp_vcmd.py`
- Static sweep: 9 levels × 5s, 통과 기준 |v_act - v_cmd| < 0.10 m/s
- Dynamic ramp: 4 segment scripted (0.85→1.15→0.85→1.0), 통과 기준 settling time < 2.5s
- 출력: `02_research_dev/260501_res_amp_vcmd_eval_results.json`

### 5.4 Manual smoke (Task 23, 사용자 액션)

```bash
python scripts/phc_walk_demo_residual.py
# 30s 녹화:
python scripts/phc_walk_demo_residual.py --record_seconds 30 --out_name res_amp_vcmd_smoke
```

**500-epoch smoke ckpt 로 부팅 검증 완료**: `step=480 v_target=1.000 v_ramped=1.000 v_actual=0.521` — 부팅·렌더·키보드/패널 입력 정상. `v_actual` 이 낮은 이유는 학습 부족 (Stage 2 필요).

---

## 6. 비교: v3 데모 vs RES_AMP_VCMD (학습)

| 측면 | v3 데모 (zero-training) | RES_AMP_VCMD (Stage 2 후 예상) |
|---|---|---|
| 학습 요구 | 0 | 36-48h server |
| 클립 전환 가시성 | 사이클 경계에서 보임 (드물지만) | 없음 (정책 내부에서 흡수) |
| v_cmd 변속 부드러움 | retime ratio ±10% 범위, 클립 사이 점프 가능 | 부드럽게 보간 (예상) |
| 외삽 (V_CMD_MAX 초과) | 즉시 깨짐 | 학습 분포 외라 보장 못함 |
| 학습 곡선의 의존성 | 없음 (phc_3 그대로) | phc_3 frozen 가중치 의존 |
| 추론 환경 | full PHC pipeline | full PHC pipeline + residual head |
| 디버깅 난이도 | 낮음 (deterministic) | 중 (Δa 의 영향 추적 필요) |

v3 데모는 **인터랙티브 시연용으로 계속 유지**, RES_AMP_VCMD 는 **연구·논문용 정량 결과용**.

---

## 7. 핵심 함정 (다음 PHC residual 작업 체크리스트)

108c0bc 에서 한꺼번에 잡은 5 개. PHC frozen-base 위에 새 task class 만들 때 **첫 부팅에서 같은 패턴이 반복될 확률 높음**:

1. **trackBodies 필드** — 우리 yaml 에 적으면 `_track_bodies` 가 그 줄로 한정됨 → task_obs 줄어들고 frozen base 와 dim 안 맞음. **빈 채로 두면 24 SMPL bodies default**. (`02_research_dev/260501_phc_3_obs_dim_investigation.md`)

2. **AMPPNNBuilder.Network 의 `kwargs['input_shape']` mutation** — parent 가 자기 local kwargs 의 input_shape 를 `self_obs+task_obs` 로 덮어쓰지만, **`**kwargs` 가 함수마다 fresh dict 가 되므로 우리 outer scope 는 영향 없음**. 결과: residual_head 는 935 (정상), critic 만 934 (잘못). `super().__init__()` 후 critic_mlp 명시적 재빌드 필요.

3. **control_mode** — `pd` 면 SMPL p_gains 가 미초기화 (humanoid.py 가 h1/g1 만 채움). **phc_3 와 동일하게 `isaac_pd` 사용**.

4. **power_reward** — `True` 여야 reward_raw 가 5-col, AmpAgent logger 의 `reward_raw[4]` 가 OOB 안 남.

5. **OOM at backward (4060 Ti)** — frozen PNN forward activation 누적이 큼. **num_envs=32** 가 7.6 GB 상한, **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** 추가 권장.

6. **`--no_log` 가 wandb 만 끈다** — ckpt 저장은 살아있음. 폴더 구조: argparse 경로(`output/<exp>_<epoch>.pth`) vs hydra 경로(`output/<task>/<exp>/Humanoid_*.pth`). eval/demo 글롭 패턴은 양쪽 다 보는 게 안전.

7. **`fut_tracks=False` → `_num_traj_samples=1` 강제** — yaml 의 `numTrajSamples: 3` 는 분기 안에서만 의미. obs size 계산 시 주의.

---

## 8. 후속 / out of scope (스펙 §13)

이 작업에서 **명시적으로 미루는** 항목:

1. **2D / 3D 명령 확장**. (vy, ωz 추가). 현재 1D vx 만. Stage 2 가 잘 되면 자연스러운 follow-up.
2. **클립 풀 확장 (5-7 클립)**. 현재 3 자연속도 클립만. 더 넓은 v_cmd 범위 / 변속 다양성 시 필요.
3. **Distillation (γ)**. reference 없이 동작하는 student 정책. residual 까지 묶어서 한 정책으로 distill.
4. **Exoskeleton 연동**. 본 연구의 최종 목표인 인간+Exo 시스템 제어. 현재는 SMPL only.
5. **Isaac Lab (PhysX 5) 마이그레이션**. closed-chain (loop joint) 지원 필요할 때.

---

## 9. 명령 요약

```bash
conda activate phc

# Stage 1 smoke (~7 min, 로컬 4060 Ti)
bash scripts/launch_res_amp_smoke.sh

# 인터랙티브 데모 (Stage 1 또는 Stage 2 ckpt)
python scripts/phc_walk_demo_residual.py
python scripts/phc_walk_demo_residual.py --record_seconds 30 --out_name res_amp_vcmd_smoke

# 자동 평가 (Stage 2 ckpt 권장)
python scripts/eval_res_amp_vcmd.py
cat 02_research_dev/260501_res_amp_vcmd_eval_results.json

# Stage 2 server (사용자 액션)
# 1. server1 에 코드 sync:    rsync -av <local>/PHC/ jiminyoun@server1:/home/jiminyoun/PHC/
# 2. server1 에서:            cd ~/PHC && sbatch train_phc_res_amp_vcmd.sh
# 3. 모니터:                  squeue -u $USER  ;  tail -F logs/phc_res_amp_vcmd_*.out
# 4. 36-48h 후:               bash pull_ckpt.sh  (또는 rsync)
```

---

## 10. 관련 문서·자산 인덱스

**설계·계획**:
- 디자인 스펙: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md` (commit `79851c1`)
- 구현 플랜: `docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md` (commit `43f9d9c`)

**진행 중 분석 문서**:
- 방법론 정리 (구현 진행 중 스냅샷): `01_research_docs/260501_phc_residual_amp_vcmd_methodology.md`
- obs dim 미스매치 조사: `02_research_dev/260501_phc_3_obs_dim_investigation.md`
- Stage 1 smoke 결과 분석: `02_research_dev/260501_res_amp_vcmd_stage1_smoke_analysis.md`
- v3 zero-training 데모 방법: `02_research_dev/260501_phc_pretrained_walk_demo_method.md`

**자산**:
- frozen base: `output/HumanoidIm/phc_3/Humanoid.pth` + `.hydra/{overrides,config}.yaml`
- 모션 풀: `sample_data/amass_walking_3clips_seamless_60s_v9.pkl` (3 자연속도 클립)
- Stage 1 smoke 체크포인트: `output/ResAMPVCmd{,_00000250,_00000500}.pth`
- Stage 1 smoke 로그: `logs/res_amp_smoke_20260501_214452/train.out`
- exp_config 백업: `exp_config/forward_walking/260501_RES_AMP_VCMD/`

**커밋 마일스톤** (Jimin branch):
- `7d1982c` v3 데모 method doc
- `b525dbf` v3 데모 Bug 3 (CUDA assert) 분석
- `79851c1` design spec
- `43f9d9c` implementation plan
- Tasks 1-15 (atomic commits) — task class, network, agent
- `108c0bc` smoke boots end-to-end (5 fixes)
- `4d31561` 방법론 정리 (구현 진행 중 스냅샷)
- `2b6d1bf` obs dim 조사
- `f57bd4c` server sbatch + server learning yaml
- `5848628` eval + residual demo
- `f2eaa35` Stage 1 smoke 분석
- `<this commit>` 최종 method doc 통합

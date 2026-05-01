# PHC Residual + AMP for Continuous v_cmd — 구현 중 방법론 정리

작성일: 2026-05-01
상태: **구현 진행 중** — Task 1–15 완료, Task 16 (smoke launch) 에서 obs dim 불일치로 일시 중단 → 원인 파악 완료 (`02_research_dev/260501_phc_3_obs_dim_investigation.md`) → yaml 한 줄 수정 후 재시도 예정

관련 문서:
- 디자인 스펙: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md` (commit `79851c1`)
- 구현 플랜: `docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md` (commit `43f9d9c`)
- 선행 데모 방법: `02_research_dev/260501_phc_pretrained_walk_demo_method.md` (v3 무학습 데모, multi-clip + retime + a-1 hard switch)
- obs dim 조사: `02_research_dev/260501_phc_3_obs_dim_investigation.md`

---

## 1. 한 줄 요약

`phc_3` (PHC 의 SMPL-PNN imitation 사전학습 모델) 을 **frozen base** 로 두고, 그 위에 **256-256 residual MLP** 와 **AMP discriminator** 를 얹어 **연속 v_cmd ∈ [0.76, 1.23] m/s** 를 추적하는 정책을 학습한다. 추론 시점에는 클립 전환이 사라지고, 정책이 받은 v_cmd 한 값만으로 자연스럽게 속도가 보간되게 만드는 것이 목표.

핵심 명제: **v3 demo 의 multi-clip + retime + a-1 switch 는 학습용 환경(트레이닝)에만 사용하고, 추론(런타임) 에서는 단일 정책이 v_cmd 만 보고 결정한다.**

---

## 2. 왜 residual + AMP 인가 (브레인스토밍 결정)

브레인스토밍 단계에서 7 가지 핵심 결정 (`docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md` §2):

| ID | 결정 | 채택 |
|---|---|---|
| D1 | residual head on frozen phc_3 | a = π_phc_3(s) + Δa(s, v_cmd) |
| D2 | training-time clip switch 방식 | **a-1 (hard switch at midpoints, hysteresis 0.02)** |
| D3 | AMP discriminator 학습 비중 | small (w_amp=0.3) |
| D4 | task reward 형태 | **r_track + r_im** (ramped) |
| D5 | reward stage curriculum | **N1 (no stage curriculum)** — 처음부터 r_track + r_im + AMP 동시 활성 |
| D6 | v_cmd ramp during episode | **S2 ramp** — episode 중 v_cmd 가 ±0.5 m/s² 까지 천천히 변할 수 있게 |
| D7 | termination | C2 (fall detect + pelvis < 0.4m) |

선행 작업 (VIC4_VCMD S10–S19B 시리즈) 의 교훈:
- AMP-only from-scratch 는 4060 Ti 7.6 GB 로는 거의 불가능 (학습 불안정 + 시간)
- phc_3 가 이미 24-body imitation 을 정밀하게 학습한 상태이므로 **frozen baseline 활용** 이 결정적 advantage
- residual 만 학습하면 파라미터 수와 메모리, 학습 시간을 모두 크게 줄일 수 있음

---

## 3. 아키텍처

```
              ┌──────────────────────────────────────────────┐
              │  HumanoidImResAMPVCmd (env)                  │
              │  - obs = self_obs(358) + task_obs(576) + 1   │
              │  - 마지막 1 dim = v_cmd                       │
              │  - multi-clip pool + a-1 hard switch         │
              │  - per-step retime via _motion_start_times   │
              └─────────────┬────────────────────────────────┘
                            │ obs 935-D
                            ▼
        ┌──────────────────────────────────────────────────┐
        │  ResAMPVCmdNetwork (rl_games builder)             │
        │                                                   │
        │   v_cmd 떼고 934-D ──▶  phc_3 PNN (frozen, eval)   │
        │                            │                      │
        │                            ▼                      │
        │                       a_phc (69-D)                │
        │                            +                      │
        │   934-D + v_cmd ────▶  ResMLPHead (256-256)        │
        │       (937-D)              │                      │
        │                            ▼                      │
        │                       Δa (69-D)                   │
        │                                                   │
        │   final_action = clip(a_phc + Δa, ±0.2 from a_phc)│
        └──────────────────────────────────────────────────┘
```

- **Frozen base (phc_3)**: `output/HumanoidIm/phc_3/Humanoid.pth`, 25.5M parameters, PNN with 3 primitives, hidden `[2048, 1536, 1024, 1024, 512, 512]`. eval mode 고정 + grad 차단.
- **Residual head**: 256-256 MLP, sigma_init=-1.0 (std=0.37, 학습 가능). 입력 = `concat(934-D base obs, v_cmd 1-D)` = 937-D. 출력 = 69-D Δa.
- **Δa clipping**: `delta_action_clip=0.2` — phc_3 출력에서 ±0.2 이상 벗어나지 못하게 막아 baseline 의 안정성을 보존.
- **AMP discriminator**: rl_games 의 `AMPPNNBuilder.Network` 가 이미 가지고 있는 disc head 그대로 사용. `numAMPObsSteps=10`.

---

## 4. 새로 추가하는 파일 (스펙 §4)

| 파일 | 역할 | LOC | 상태 |
|---|---|---|---|
| `phc/env/tasks/humanoid_im_res_amp_vcmd.py` | 환경 task class. v_cmd obs, multi-clip switch, per-step retime, S2 v_cmd ramp, r_track + r_im, termination | ~600 | ✅ 완료 |
| `phc/learning/res_amp_network.py` | `ResMLPHead` + `ResAMPVCmdNetwork` (`AMPPNNBuilder.Network` 상속) + `ResAMPVCmdBuilder` + `load_frozen_phc_3()` | ~200 | ✅ 완료 |
| `phc/learning/res_amp_agent.py` | `ResAMPVCmdAgent(IMAmpAgent)`. frozen base no-grad assertion 만 추가 | ~80 | ✅ 완료 |
| `phc/data/cfg/env/env_im_res_amp_vcmd.yaml` | 환경 yaml. v_cmd 파라미터, reward weights, clip 정의 | ~130 | ⚠️ trackBodies 한 줄 제거 필요 |
| `phc/data/cfg/learning/im_res_amp_vcmd.yaml` | 학습 yaml. algo=im_amp_residual, network=amp_pnn_residual | ~150 | ✅ 완료 |
| `scripts/phc_walk_demo_residual.py` | 학습 후 inference demo (v3 demo 와 동일 UX, residual 정책 사용) | ~280 | ⏳ Task 22 |

플러스 `phc/utils/parse_task.py` 1 줄 등록 (`HumanoidImResAMPVCmd → eval(...)`) — 완료.

---

## 5. 학습 환경 핵심 동작

### 5.1 v_cmd observation (Task 4 완료, commit `b3...`)
- 환경 obs 끝에 `v_cmd` 1 dim 추가 → total dim = phc_3 입력 + 1.
- 네트워크는 입력을 두 갈래로 분기: `obs[:, :-1]` 는 phc_3, 전체 obs 는 residual head.

### 5.2 Multi-clip + a-1 hard switch (Task 5)
- 클립 풀: `sample_data/amass_walking_3clips_seamless_60s_v9.pkl` 안의 3 자연속도 클립 (~0.90 / 0.97 / 1.07 m/s).
- v3 demo 의 monkey-patch 로직을 task class 의 정식 메서드로 포팅.
- v_cmd 와 가장 가까운 자연속도 클립을 mid-point + hysteresis 0.02 로 결정.
- **사이클 경계에서만** 전환 (mid-cycle 전환 시 발생하는 IK pop / 모멘텀 단절 방지).

### 5.3 Per-step retime (Task 6)
- `_motion_start_times_offset` 누적값을 이용해 매 step 모션 진행 속도를 v_cmd / clip_speed 비율로 변형.
- 사이클 길이 보존 (전환 시 phase 일관성).

### 5.4 S2 v_cmd ramp (Task 7)
- 매 step 마다 확률 `v_cmd_ramp_prob=0.01` 로 v_cmd 가 +/- 방향으로 ramp 시작.
- 변화율 한계 `v_cmd_max_accel=0.5 m/s²` (= 1/120 m/s per 60Hz step).
- 에피소드 내 v_cmd 가 살아있어야 추론 시점의 키보드 ramp 와 분포가 일치.

### 5.5 Reward (스펙 §5.4, Task 8)
```
r_total = w_track · r_track + w_amp · r_amp + w_im · r_im

r_track = exp(-α_track · |v_actual - v_cmd|²)        # α_track = 5.0
r_im    = exp(-α_im · pose-tracking error)           # α_im    = 2.0  (from PHC's r_imitation)
r_amp   = AMP discriminator score                    # rl_games 기본

w_track = 0.5, w_amp = 0.3, w_im = 0.2  (D5 = N1 → 처음부터 동시 활성)
```

### 5.6 Termination (Task 9)
- C2: fall detect (PHC 표준) + pelvis height < 0.4 m.

### 5.7 Action clipping (스펙 §5.4)
- `delta_action_clip = 0.2` — Δa 가 baseline action 으로부터 ±0.2 이상 못 벗어남.

---

## 6. 학습 단계

| Stage | 목적 | 환경 | num_envs | episode_length | 예상 시간 | 상태 |
|---|---|---|---|---|---|---|
| 0 (개발 smoke) | 부팅·forward pass·loss·1 epoch 확인 | 4060 Ti 로컬 | 64 | 300 (5s) | ~10 min | ⚠️ 진행중 (obs mismatch 해결 후 재시도) |
| 1 (smoke run) | local 6h 단기 학습, ~500 epoch | 4060 Ti 로컬 | 64 | 300 | 6h | ⏳ Task 17 |
| 2 (full run) | server1 sbatch 본 학습, ~20K epoch | server1 | 512 | 600 | 36–48h | ⏳ Task 20 |
| 3 (eval) | v_cmd sweep + ramp + 정성 평가 | 4060 Ti | 1 (test) | – | 수십분 | ⏳ Task 21–24 |

Stage 1 통과 기준 (스펙 §8):
- av_steps ≥ 200 (bunny-hop 안 함)
- v_actual ↔ v_cmd 상관계수 ≥ 0.7 on a held-out v_cmd sweep
- AMP disc loss 가 폭주하지 않음
- phc_3 frozen 파라미터의 grad norm = 0 (Task 15 assertion)

---

## 7. 현재 상태 (2026-05-01 기준)

✅ Task 1–15 (코드 생성 부분) 모두 commit 완료.
- Task 1: skeleton + parse_task 등록 — `9f...`
- Task 2: env yaml — `4ebc0e1`
- Task 3: learning yaml — `3aed21a` + `3f7440b`
- Task 4: v_cmd obs — done
- Task 5: multi-clip a-1 switch — `952c2d6` + `edd7aa1`
- Task 6: per-step retime — `9299b06`
- Task 7: S2 ramp — `323d2af`
- Task 8: r_track + r_im — `08293da`
- Task 9: termination — `238d0f1`
- Task 10: ResMLPHead — `c1a2745`
- Task 11: ResAMPVCmdNetwork — `ab528e7`
- Task 12: phc_3 loader — `f792fb1`
- Task 13: builder registration — `a78185b`
- Task 14: ResAMPVCmdAgent — `68b6387`
- Task 15: frozen-base no-grad assertion — `a8c560d`

⚠️ Task 16 (smoke launch script + 부팅 확인) — 부팅 자체는 성공했으나 **forward pass 에서 obs dim 불일치 crash**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x454 and 934x2048)
```
원인 분석 완료 (`02_research_dev/260501_phc_3_obs_dim_investigation.md`).

**한 문장 요약**: `env_im_res_amp_vcmd.yaml` 의 `trackBodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]` 한 줄이 task_obs 를 24-body × 24 = 576 → 4-body × 24 = 96 으로 줄여 phc_3 가 학습된 934-D 와 어긋남. **이 한 줄을 삭제하면 자동으로 24 SMPL body default 로 fallback 되어 934 매칭됨.**

다음 단계:
1. Path A 적용 — `phc/data/cfg/env/env_im_res_amp_vcmd.yaml` 에서 `trackBodies` 줄 삭제
2. Task 16 smoke 재시도 — `bash scripts/launch_res_amp_smoke.sh`
3. 부팅 + 1 epoch 확인되면 Task 17 (사용자 6h smoke run) 으로 넘김

⏳ Task 17–24 — 사용자 액션 또는 후속 코드 (eval, demo, 분석 doc).

---

## 8. 위험 요소 (스펙 §10 발췌)

| 위험 | 완화책 |
|---|---|
| residual 이 phc_3 동작을 크게 깨고 자기 식대로 가버림 | `delta_action_clip=0.2` + AMP disc 가 OOD 펜널티 |
| AMP disc 폭주 (자기끼리 학습) | `w_amp=0.3` 으로 작게, `numAMPObsSteps=10` 그대로 |
| v_cmd 끝값 (0.76, 1.23) 외삽 시 깨짐 | 클립 mid-point 가 dataset 내부에 있어 외삽 영역 자체가 좁음 + S2 ramp 분포 학습 |
| frozen base 가 실수로 학습됨 | Task 15 의 `calc_gradients` override 에서 grad-norm assertion |
| 4060 Ti OOM | num_envs 64 + amp_buffer 20K (스펙 §7 기본) |
| obs dim 미스매치 (← 지금 우리가 만난 것) | trackBodies 필드 제거 + `.hydra/overrides.yaml` 와 yaml 정렬 점검 |

---

## 9. 학습된 교훈 (이 단계에서)

1. **`.hydra/overrides.yaml` 와 `.hydra/config.yaml` 은 frozen baseline 의 ground truth.** 호환 task class 를 만들 때 가장 먼저 봐야 할 파일.
2. **obs-affecting yaml 필드 (`obs_v`, `fut_tracks`, `numTrajSamples`, `trackBodies`, `numAMPObsSteps`, `self_obs_v`, `has_shape_obs`, `has_limb_weight_obs`) 는 baseline 과 일치시켜야 한다.** 한 줄만 어긋나도 입력 차원이 어긋남.
3. **`fut_tracks=False` → `_num_traj_samples=1` 강제** (yaml 의 `numTrajSamples` 무시). PHC 코드의 미묘한 분기.
4. **subagent-driven development 의 review loop 가 실제로 잡아냈다.** Task 4 의 wrong import, Task 5 의 dropped fallback, Task 13 의 missing yaml fields — 모두 spec/code reviewer 가 잡음. obs mismatch 는 코드만으로는 못 잡고 실제 부팅에서 노출됨 → 다음번엔 부팅 단계도 plan 의 explicit step 으로 넣는 게 좋음.

---

## 10. 참고

- 디자인 스펙: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md`
- 구현 플랜: `docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md`
- frozen base ground truth: `output/HumanoidIm/phc_3/.hydra/overrides.yaml` + `config.yaml`
- 무학습 선행 데모 (학습 환경의 인프라 출처): `scripts/phc_walk_demo.py` (v3) + `02_research_dev/260501_phc_pretrained_walk_demo_method.md`
- obs mismatch 조사: `02_research_dev/260501_phc_3_obs_dim_investigation.md`
- 선행 실패 시리즈 (AMP from scratch): `exp_config/forward_walking/2604*_VIC4_VCMD_*` 시리즈

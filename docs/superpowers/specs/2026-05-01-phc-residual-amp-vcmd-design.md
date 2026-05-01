# PHC + Residual + AMP for Continuous v_cmd — Design Spec

작성일: 2026-05-01
관련 자산: `output/HumanoidIm/phc_3/Humanoid.pth` (frozen base), `sample_data/amass_walking_3clips_seamless_60s_v9.pkl` (reference + AMP pool), `scripts/phc_walk_demo.py` v3 (multi-clip + retime 인프라)

## 1. 목표

PHC `phc_3` 사전학습 모델을 동결한 상태에서 **작은 residual head + AMP discriminator** 를 학습시켜, **연속 v_cmd ∈ [0.76, 1.23] m/s 입력에 대해 매 step 추적하는 정책**을 만든다. 디스크리트 클립 스위치 없이 사용자가 임의 v_cmd 를 키보드/패널로 변화시켜도 휴머노이드가 매끈하게 속도를 따라가는 게 최종 목표.

비목표 (v1 범위 밖):
- 2D / 3D 명령 (vy, ωz). 향후 확장.
- VIC (Variable Impedance Control) 결합. 별도 후속.
- 새 motion primitive (run, jump, turn). v_cmd-modulated forward walking 만.

## 2. 핵심 디자인 결정 (브레인스토밍 합의)

| # | 결정 | 선택 |
|---|---|---|
| 1 | 명령 차원 | **1D vx** ∈ [0.76, 1.23] m/s |
| 2 | Reference clip 전략 | **(a) Multi-clip + retime** (v3 데모와 동일) |
| 3 | 클립 전환 방식 | **(a-1) Hard switch at midpoint + hysteresis** (v3 데모 그대로) |
| 4 | Reward 구조 | **(R2) Tracking + AMP + small imitation anchor** |
| 5 | 네트워크 아키텍처 | **(N1) Output-residual** (a = phc_3(s,ref) + Δa(s,v_cmd)) |
| 6 | v_cmd sampling 전략 | **(S2) Episode-level + low-prob mid-ep ramp** |
| 7 | 학습 인프라 | **(C2) 로컬 4060 Ti smoke → server1 sbatch 본학습** |

각 결정의 근거는 §11 의 trade-off 표 참고.

## 3. 아키텍처

```
                 ┌─────────────────────────────────┐
                 │  HumanoidImResAMPVCmd (task)    │
                 │  - 기존 v3 데모의 multi-clip    │
                 │    + retime + a-1 hard switch   │
                 │    로직을 task class 안으로     │
                 │  - v_cmd 1-D obs 추가            │
                 │  - r_track + r_im_anchor 계산    │
                 │  - S2: 1%/step prob ramp         │
                 └──────────────┬──────────────────┘
                                │ obs (proprio + ref + v_cmd)
                                ▼
       ┌────────────────────────┼─────────────────────────────┐
       │           ResAMPVCmdNetwork                           │
       │                                                       │
       │   proprio, ref ─→ phc_3 PNN (FROZEN) ─→ a_base (69-D) │
       │                                                       │
       │   proprio, v_cmd ─→ residual MLP ─→ Δa (69-D)         │
       │                       (256-256, σ_init=-1.0)          │
       │                                                       │
       │   a_total = a_base + Δa,  exploration σ on Δa only    │
       └────────────────────────┬─────────────────────────────┘
                                │ a_total
                                ▼
                       physics step + reward
                                │
       ┌────────────────────────┴─────────────────────────────┐
       │  AMP discriminator on (s, a) — 3 walking clip 분포    │
       └───────────────────────────────────────────────────────┘

Total reward = w_track · exp(-|v_act-v_cmd|²)        (0.5)
             + w_amp   · log D(s,a)                  (0.3)
             + w_im    · exp(-pose_dist(s, ref))     (0.2)
             + r_survive                              (1.0 base)
```

## 4. 컴포넌트 / 파일 plan

**새로 만드는 파일**:

| 파일 | 역할 | LOC 예상 |
|---|---|---|
| `phc/env/tasks/humanoid_im_res_amp_vcmd.py` | Task class. v_cmd obs, multi-clip 스위칭, r_track/r_im 계산, S2 ramp logic | ~600 |
| `phc/learning/res_amp_network.py` | phc_3 freeze + residual head 의 nn.Module wrapper | ~150 |
| `phc/learning/res_amp_agent.py` | 학습 agent. AmpAgent 상속, residual params 만 grad | ~200 |
| `phc/data/cfg/env/env_im_res_amp_vcmd.yaml` | env config | ~70 |
| `phc/data/cfg/learning/im_res_amp_vcmd.yaml` | learning config (residual spec, AMP disc, hyperparams) | ~120 |
| `scripts/phc_walk_demo_residual.py` | 학습된 residual 로 inference 데모 (v3 데모 + 학습된 weight) | ~280 |

**기존 자산 재활용 (수정 없음)**:
- `phc/learning/amp_agent.py` — AMP discriminator 코드 그대로 사용
- `phc/utils/parse_task.py` — `HumanoidImResAMPVCmd` 한 줄 등록
- `output/HumanoidIm/phc_3/Humanoid.pth` — frozen base load
- `sample_data/amass_walking_3clips_seamless_60s_v9.pkl` — reference + AMP pool 둘 다
- v3 데모의 multi-clip + retime 알고리즘 — `_select_clip`, `_apply_pending_clip_switch`, retime 로직을 task class 메서드로 이식

**PHC 베이스 클래스는 일절 수정하지 않음.** 모든 변경은 새 파일 안에서.

## 5. 데이터 흐름

**Per-step (학습/추론 공통)**:

1. **Observation**:
   - `proprio` ~245-D (PHC 표준): root height/rot, joint pos/vel, body pos/vel, contact, key body, etc.
   - `ref` (현재 active clip 의 이번 frame): root pos/rot, joint pos/vel, body pos. v3 데모와 동일.
   - `v_cmd`: 1-D scalar (현재 v_cmd_ramped, normalized to [-1, 1] for residual head input).
   - 합쳐서 phc_3 와 residual head 에 dispatch.

2. **Action 합성**:
   - `a_base = phc_3.forward(proprio, ref)` — frozen, no grad.
   - `Δa_mean, Δa_log_std = residual_head(proprio, v_cmd)`.
   - 학습 시: `Δa = Δa_mean + N(0, exp(Δa_log_std))` (sample).
   - 추론 시: `Δa = Δa_mean` (deterministic).
   - `a = a_base + Δa` (PD targets, 69-D).

3. **Physics + state update** (PHC 표준).

4. **Reward** (per-env):
   - `r_track = exp(-α_track · |v_act - v_cmd|²)`. v_act = `||root_vel[:, :2]||`. α_track = 5.0.
   - `r_amp = log D(s, a)` (AMP discriminator). PHC AMP 코드 재사용.
   - `r_im = exp(-α_im · pose_dist(joint_pos, ref_joint_pos))`. α_im = 2.0. anchoring residual 가까이.
   - `r_survive = 1.0` (사망 시 0).
   - `r_total = 0.5 · r_track + 0.3 · r_amp + 0.2 · r_im + r_survive`.

5. **Termination** (PHC 표준 + 우리 추가):
   - Fall: pelvis height < 0.4 OR tracking error > 1.0 m.
   - max_episode_length: 학습 시 600 step (10s @ 60Hz physics). 추론 시 무제한.

## 6. v_cmd Sampling (S2)

**Episode 시작 시**:
- `v_cmd_target = uniform(0.76, 1.23)`
- `v_cmd_ramped = v_cmd_target` (즉시 도달, 시작 transient 회피)

**Episode 동안 매 step**:
- 1% 확률로 `v_cmd_target = uniform(0.76, 1.23)` 새로 sampling (ramp trigger).
- `v_cmd_ramped` 가 매 step `max_accel · dt` 만큼 `v_cmd_target` 으로 향함 (max_accel=0.5 m/s²).
- 600 step (10s) 에피소드에서 평균 6 회 ramp → 정책이 mid-walk transition 풍부하게 노출.

**클립 선택**:
- v_cmd_ramped 가 midpoint 통과 시 (hysteresis 0.02) `pending_clip` 큐잉 → 다음 pre_physics_step 에서 atomic switch (v3 a-1 그대로).

## 7. AMP Discriminator 클립 풀

**v1**: 기존 3 클립 (`amass_walking_3clips_seamless_60s_v9.pkl`) 그대로 사용. 자연속도 0.897 / 0.975 / 1.068 m/s.

근거:
- v_cmd 학습 분포가 [0.76, 1.23] 안에 있고, 클립 풀 [0.897, 1.068] 이 그 중심부 커버 → discriminator 가 가르치는 "natural walking" 분포가 task 와 일치.
- 풀 작아도 우리 task 가 narrow (forward walking 만) 이라 discriminator overfit 위험 적음.
- v_cmd extremes (0.76, 1.23) 는 retime + residual 로 보간되고 AMP disc 가 약간 외삽 패널티 → 자연스럽게 부드러운 정책.

**v2 옵션** (v1 학습 후 평가 보고 결정):
- 0.7-0.8 m/s 와 1.2-1.3 m/s 의 walking 클립 추가 → AMP 분포를 v_cmd 범위 양 끝까지 확장. 단, 데이터 수집/큐레이션 작업 필요 (KIT/AMASS 에서 검색).

## 8. 학습 곡선 / 커리큘럼

**Stage 1: Smoke (로컬, 4060 Ti)**:
- num_envs = 64, episode_length = 300 (5s)
- max_epochs = 500 (~6h wall-clock 예상)
- 통과 기준:
  - AMP disc accuracy 50-80% 사이 안정 (균형)
  - av_reward 가 epoch 0 대비 +30% 이상
  - av_episode_length ≥ 240 step (=80%)
  - NaN / OOM / crash 없음

**Stage 2: Full training (server1 sbatch)**:
- num_envs = 512, episode_length = 600 (10s)
- max_epochs = 20000 (~36-48h wall-clock 예상, A100 / 4090 기준)
- Checkpoint: 매 1000 epoch 저장, 매 5000 epoch 빼고 30 일 후 삭제.
- 통과 기준 (학습 종료):
  - av_episode_length ≥ 580 step (=97%)
  - v_cmd 추적 평균 오차 ≤ 0.10 m/s (sweep test)
  - AMP disc 의 fake/real loss 모두 안정 진동

**Curriculum 유무**: 없음. residual on phc_3 baseline 이므로 처음부터 안정적이라 가정. (S10-S19B 의 curriculum 은 from-scratch 라 필요했음.)

## 9. 평가 / 성공 기준

**평가 시나리오** (학습 완료 후 별도 eval script):

1. **Static v_cmd sweep** (각 5s × 9 단계):
   - v_cmd ∈ {0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18, 1.23}
   - 각 단계에서 평균 v_act 측정.
   - 합격: |v_act_avg - v_cmd| < 0.10 m/s 모든 9 단계, 넘어짐 0 회.

2. **Dynamic v_cmd ramp** (60s):
   - v_cmd: 1.0 → 0.78 → 1.20 → 0.85 → 1.10 → 1.0 (각 10s)
   - 각 transition 의 settling time 측정.
   - 합격: settling time < 2.5s, 넘어짐 0 회, 시각적으로 부드러운 보행.

3. **Sample qualitative video** (30s, 사용자 키보드 입력):
   - 사용자가 ↑↓ 키와 슬라이더로 자유 조작.
   - 합격 기준: "사람이 보기에 자연스럽다" (subjective).

**Eval script**: `scripts/eval_phc_residual_amp_vcmd.py` (Stage 3 산출물).

## 10. 학습 인프라 세부

**로컬 (Stage 1)**:
- 호스트: exolab-MS-7D56 (RTX 4060 Ti 7.6 GB)
- conda env: `phc`
- 명령: `python phc/run.py --task HumanoidImResAMPVCmd --cfg_env env_im_res_amp_vcmd.yaml --cfg_train im_res_amp_vcmd.yaml --headless --num_envs 64`
- 메모리 추정: phc_3 forward only ≈ 2 GB + residual training (64 envs) ≈ 3 GB → 5 GB / 7.6 GB. 안전.

**서버 (Stage 2)**:
- 호스트: server1 (Slurm)
- 파티션: idx0 또는 idx1 (사용 가능한 슬롯)
- sbatch 파일 위치: `slurm/train_res_amp_vcmd.sbatch` (Stage 2 산출물)
- num_envs=512, ~24-48h
- wandb 로깅 활성화

## 11. 결정의 trade-off 요약

| 결정 | 선택 vs 대안 | 트레이드오프 |
|---|---|---|
| 1D vx | vs 2D / 3D | 클립 풀 작게 유지, 학습 안정. 다차원은 후속. |
| Multi-clip + retime | vs single canonical | phc_3 baseline 이 v_cmd 근처 동작 → residual 부담 작음. |
| Hard switch (a-1) | vs phase-aligned blend (a-2) | 구현 부담 0, AMP disc 가 jitter 흡수. a-2 는 후속 옵션. |
| R2 (anchor 포함) | vs R1 (anchor 없음) | residual 이 baseline 근처 머무도록 anchor → 보행 quality 보존. |
| N1 output-residual | vs N2 / N3 (invasive) | phc_3 통째 freeze, 학습 메모리 적음, 디버깅 쉬움. |
| S2 mid-ep ramp | vs S1 episode-level fixed | 학습-추론 분포 일치, transient 자동 학습. |
| C2 local + server | vs server only | smoke 로 OOM/bug 미리 잡음, 서버 자원 낭비 방지. |

## 12. 위험 요소 / 완화

| 위험 | 가능성 | 완화책 |
|---|---|---|
| AMP disc 가 너무 강해서 residual 이 phc_3 와 똑같이 만 학습 (v_cmd 무시) | 중간 | w_amp 0.3 으로 작게 시작, w_track 0.5 우세하게. 학습 중 disc loss 비율 모니터. |
| Residual 이 발산해서 phc_3 동작 망가뜨림 | 중간 | sigma_init=-1.0 (std=0.37), Δa clipping ±0.2 rad, w_im=0.2 anchor. |
| Mid-ep ramp 시 클립 스위치 직후 transient 가 커서 termination | 낮음 | hysteresis=0.02 + max_accel=0.5 m/s² 가 이미 충분히 부드러움. v3 데모 검증 됨. |
| 4060 Ti smoke OOM | 낮음 | num_envs=64 로 시작, 안 되면 32. PHC base 가 num_envs=128 까지 검증 됨. |
| Server training 시 wandb 끊김 / Slurm queue 대기 | 중간 | local checkpoint 도 매 1000 epoch 저장, 끊기면 resume. |
| v3 데모 클립 풀 (3 개) 이 너무 작아 AMP disc 가 풀 외 motion 패널티 강함 | 중간 | v1 끝 evaluation 후 §7 v2 옵션 (5-7 클립) 으로 확장. |
| VIC4_VCMD S10-S19B 와 같이 episode_length 가 안 늘음 | **낮음** (그때는 from-scratch, 우리는 baseline 있음) | smoke 단계에서 즉시 발견. anchor reward 더 키워 baseline 가까이 잡음. |

## 13. Out of scope (이후 단계)

- 2D (vx, vy) / 3D (vx, vy, ωz) 명령 → spec v2.
- AMP 클립 풀 확장 (5-7 클립) → 본학습 후 v2 평가에 따라.
- Distillation (γ): residual 학습된 후 reference 없이 동작하는 student 정책 → spec v3.
- VIC (CCF) residual 결합 → 별도 spec.
- Phase-aligned blend (a-2) 업그레이드 → a-1 한계 발견 시 spec amendment.

## 14. 참고

- v3 데모: `scripts/phc_walk_demo.py`, `02_research_dev/260501_phc_pretrained_walk_demo_method.md`
- 이전 from-scratch 시도: `02_research_dev/260427_vic4_vcmd_runs.md`, `260430_S18_S19_S19B_setup_and_results.md`
- PHC AMP infra: `phc/learning/amp_agent.py`
- 사전학습 체크포인트: `output/HumanoidIm/phc_3/Humanoid.pth` (PNN, 6-layer hidden, 3 primitives)

# RES_AMP_VCMD Stage 2 학습 실패 분석

작성일: 2026-05-02
관련 ckpt: `output/ResAMPVCmd_*.pth`, `ResAMPVCmd_B_*.pth` (서버, 6300 epoch까지 학습 후 cancel)
관련 코드: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`, `phc/learning/res_amp_network.py`, `phc/learning/res_amp_agent.py`

---

## 1. 한 줄 요약

**13 시간 학습 후 metric은 eps_len 525, rwd 850 까지 올라갔지만, inference에서 휴머노이드가 15 step 안에 fall.** 학습 metric은 reward shaping 결함과 episode cap 메커니즘 차이로 부풀려진 misleading 지표였고, 정책은 실제로 walking을 학습하지 못했음.

User memory의 *"Numbers can be Goodharted"*, *"Video is the final truth"* 경고가 정확히 적중한 사례.

---

## 2. 증상

| 측정 | 값 | 해석 |
|------|------|------|
| 학습 metric (Slot A Ep 6300) | rwd 853, eps_len 561 / 600 (94%) | 좋아 보임 |
| 같은 ckpt + `--test` inference (서버) | eps_len 6-13 | 6 step 안에 fall |
| 같은 ckpt + 사용자 시각 검증 (local 비디오) | "한 step도 제대로 못 걸음" | 진짜 능력은 ~0 |
| `--test` 없이 training-mode rollout | Ep 5501: eps_len 15.9 → Ep 5506: 72.3 | 5 PPO update만에 4× — 즉, 정책이 수렴 근처도 아님 |

---

## 3. 원인 분석 (확정)

### 3.1 Reward shaping: r_survive 비중 너무 큼 (메인 원인)

`humanoid_im_res_amp_vcmd.py` line 252-254:

```python
self.rew_buf[:] = (W_TRACK * r_track   # 0.5
                 + W_IM * r_im          # 0.2
                 + r_survive)           # 1.0 fixed per step
```

**Reward의 60%가 단순 survival**. PPO는 누적 리턴을 maximize → "걷기"보다 "안 넘어지기" 우선순위 학습. 학습 후 분석:

- Ep 4800: per-step reward = 792.1 / 525.3 = **1.51**
- 1.0 (survive 만으로) + 0.51 (track + im 합쳐서) → survival이 reward의 **66%**

PPO가 진짜 walking을 학습할 incentive가 약하고, 어색한 자세로 "stand-balance + 약간 흔들기"로 episode_length cap 도달하는 전략을 학습.

### 3.2 Episode cap 학습/test 불일치 (metric inflation 메커니즘)

`phc/run.py` line 249-250:
```python
if args.test and not flags.small_terrain:
    cfg['env']['episodeLength'] = 99999999999999
```

- **학습**: episode_length=600 (CLI). 600 step 후 자동 reset → eps_len이 600 cap에 도달 가능.
- **`--test` inference**: cap이 99999... 강제. 진짜 fall만이 termination.

학습 시: 87% 에피소드가 cap (600)에 도달 (안 넘어짐), 13%가 진짜 fall (~15 step). Mean = 0.87 × 600 + 0.13 × 15 ≈ 524. **eps_len 525가 합리적으로 설명됨**.

Inference 시: cap이 사실상 무한 → fall만이 종료 → 진짜 fall time (~15) 노출.

### 3.3 phc_3 baseline이 우리 env에서 OOD

`phc_walk_demo.py` (phc_3 + 3-clip + retime, 학습 없음): 잘 걷음 (사용자 확인).
`HumanoidImResAMPVCmd` (residual=0, 매칭된 termination, retime/clip switch off): **steps 2-24, fall**.

차이는 task class:
- 잘 작동: `HumanoidIm` (phc_3 학습 환경 직속) + 외부 monkey-patch.
- 실패: `HumanoidImResAMPVCmd(HumanoidIm)` — 내부에 multi-clip, retime, v_cmd, custom reward 다 직접 구현.

phc_3 학습 시 사용된 PHC의 multi-clip + PNN 통합 (`HumanoidImMCP`)을 base로 안 쓰고 vanilla HumanoidIm 위에 직접 구현하면서 미묘하게 다른 obs/state distribution 발생 → phc_3 baseline 자체가 OOD가 됨.

### 3.4 Sanity check가 너무 늦었음

5500 epoch까지 학습 metric만 보고 정책 잘 되는 줄 알았음. 첫 inference smoke를 5500 epoch 후에 처음 함. **Goodharting 탐지가 12 시간 늦음**.

---

## 4. 4가지 Lesson (다음 시도에 반영)

| # | Lesson | 다음 시도 fix |
|---|--------|---------|
| 1 | Reward에 큰 fixed term (`r_survive=1.0`) 넣지 말 것 | r_survive 제거 또는 0.1 이하 |
| 2 | Episode cap이 학습/inference에서 다르면 metric 비교 불가 | 학습 / test 둘 다 episode_length=600 (또는 사용자 정의 동일값) |
| 3 | 검증된 base task (`HumanoidImMCP`)를 무시하고 직접 reimplement 하지 말 것 | 새 task class는 `HumanoidImMCP` 상속 (검증된 PNN 통합 그대로) |
| 4 | 학습 metric만 신뢰하지 말 것 | **1000 epoch마다 inference smoke** (v_actual 측정), Goodharting 즉시 감지 |

---

## 5. 부수 발견

### 5.1 `--test`가 활성화하는 코드 경로 (motion_times = 0 외 다수)

`flags.test = True` 시 `humanoid_im.py` 등에서 다양한 분기:
- L1016-1017: `motion_times[:] = 0` (모든 env가 t=0에서 시작)
- L83-84, L716-717, L830: obs/track noise 비활성
- L339-340 외: motion_lib `random_sample = (not flags.test)` (test 시 deterministic)

이전 실패 디버그 시 motion_times 패치만 시도해 봤지만 다른 분기 효과는 측정 안 함. 다음에 신경 써야 할 부분.

### 5.2 Cycle counter 60-step suppression

`humanoid_im.py` L1192-1194: motion clip이 끝나서 cycle 발생하면 60 step 동안 termination 억제. 학습 시 random motion_times로 일부 env가 clip 끝 근처에서 시작 → 짧은 시간 내 cycle → 60-step 면역. 이게 metric inflation에 약간 기여하지만 메인 원인은 아님 (3.1, 3.2가 메인).

### 5.3 결정타: `r_survive` 제거 시 학습 행동 예측

이전 학습은 r_survive로 "stand-balance"가 dominant strategy. 제거 시:
- 안 넘어지기만으로는 reward 0
- r_track 0.5 / r_im 0.2 / r_amp 0.3 합쳐서만 reward
- 진짜 walking이 아니면 reward 0 → PPO가 walking 학습 강제

다음 시도의 reward 설정:
```python
r_total = 0.5 * r_track + 0.3 * r_amp + 0.2 * r_im
        # r_survive 제거됨
```

### 5.4 추가 진단 (2026-05-02 저녁): phc_3 alone도 우리 env에서 fall

v2 시작 전 5분 진단으로 isolation 확인:
- 진단 hook (env-var 가드): `PHC_ZERO_RESIDUAL=1`, `PHC_DISABLE_DYNAMICS=1`, `PHC_DISABLE_PELVIS_TERM=1`, `PHC_FIX_VCMD_NEUTRAL=1`
- 코드: `phc/learning/res_amp_network.py:eval_actor`, `phc/env/tasks/humanoid_im_res_amp_vcmd.py:pre_physics_step` & `_compute_reset`
- 결과: residual=0 + multi-clip off + retime off + pelvis term off + v_cmd 중립 → **여전히 24-25 step 안에 fall, reward 34-35 cap**

추가로 env yaml의 `terminationHeight: 0.4 → 0.15`, `cycle_motion: True → False`, `has_pnn: False → True`, `num_prim: 1 → 3`, `zero_out_far: True → False` 모두 env_im_pnn (phc_3 학습) 매칭으로 변경 → **여전히 24-25 step**.

같은 phc_3 ckpt가 `phc_walk_demo.py` (HumanoidIm + Hydra path)에서 잘 걷는데, 우리 task class에서 fall.

→ **결정적: env yaml 파라미터만으로는 안 풀림. 가장 큰 의심은 RunningMeanStd shift** — `output/ResAMPVCmd.pth` 로드 시 trained running stats가 phc_3 학습 stats와 달라서 phc_3가 wrong-normalized obs 받음.

진단 hook 인프라는 v2 검증에도 그대로 재사용 (각 gate 통과 확인용).

### 5.5 v2 빌드 순서: 진단 gate 단계별 추가

v1 실패 핵심 이유 = 모든 변경을 한 번에 빌드 → 5500 epoch 후 fall 발견 → 어느 변경이 원인인지 모름.

v2는 7 단계 gate, 각 단계에 진단 (PHC_ZERO_RESIDUAL=1 + 모든 dynamics off → eps_len ≥ 500 walking) 통과 후에만 다음으로 commit:

| Gate | 추가 |
|------|------|
| 1 | `HumanoidImResAMPVCmdV2(HumanoidImMCP)` 스켈레톤 — single clip, no v_cmd, no dynamics, env yaml = env_im_pnn structure |
| 2 | phc_3 RunningMeanStd를 ckpt에서 로드 후 freeze |
| 3 | v_cmd obs 추가 (residual head input에만) |
| 4 | Multi-clip switch (motion_lib level, no obs override) |
| 5 | Retime in pre_physics_step |
| 6 | Residual head random init (ZERO_RESIDUAL flag disable) |
| 7 | Reward (r_track + r_im, no r_survive) — 학습 시작 |

학습 시작 후 1000 epoch마다 자동 진단 실행. 통과 못하면 즉시 alert.

### 5.7 Gate 2 통과 (2026-05-02 야간)

V2 task에 v_cmd 1-D obs 추가 + 새 learning yaml + 인프라 (phc_3 RMS 자동 적용 + ckpt 미존재 우아 처리).

**결과**: PHC_ZERO_RESIDUAL=1 + V2 architecture (frozen phc_3 + zero residual + v_cmd 935-D obs) → **eps_len 1799 step (full motion), avg reward 1666.97**. Gate 1 (1695.85) 대비 ~30 차이는 v_cmd obs가 critic에 미세하게 영향, **phc_3 동작은 유지** (last dim drop 검증).

**추가된 인프라:**
1. `res_amp_network.py`: phc_3 ckpt에서 running_mean_std 추출 후 `_frozen_rms_state`로 stash
2. `im_amp_players.py`: player init 시 `_frozen_rms_state` 발견 → 935-D으로 확장 후 model.running_mean_std에 적용 (`PHC_LOAD_FROZEN_RMS` 자동)
3. `amp_players.py`: restore("Base"/missing file) 우아 처리 — V2 첫 학습 시 ckpt 없이도 부팅 가능

이로써 **각 gate 검증이 stub ckpt 없이 가능**. v2 학습 시작은 phc_3 RMS로 시작하고 PPO 진행하면서 자연스럽게 적응.

### 5.6 Gate 1 통과 (2026-05-02 늦은 저녁)

**결과**: V2 task class (empty subclass HumanoidIm) + 새 env yaml (env_im_pnn 파라미터 미러 + full robot config) → phc_3 alone이 **eps_len 1799 step (= 60s 모션 clip 전체)** 까지 walking, avg reward 1695.85.

**v1 vs V2 차이 (확정):**

| 변경 | v1 | V2 | 영향 |
|------|----|----|------|
| env yaml 구조 | `env:` block + missing robot params | `env:` block + full robot config | obs dim 36개 차이 (AMP disc) |
| `has_dof_subset` | 누락 (default False) | True | **AMP discriminator 차원 결정적 영향** |
| `has_pnn` | False (잘못됨) | True (phc_3에 맞춤) | 영향 미미 (HumanoidIm path) |
| `cycle_motion` | True | False (env_im_pnn 매칭) | episode 길이 |
| `terminationHeight` | 0.4 | 0.15 | 영향 미미 (단독으로는 안 풀림) |
| Task class | HumanoidIm-derived 다양한 override | Empty subclass | obs override 제거 |

**가장 중요한 단일 fix**: `has_dof_subset: True` (robot config에). 이게 없으면 AMP obs가 step당 36 features 더 커지고, phc_3 ckpt와 disc shape mismatch.

**DIAG infra (V2 gate 검증용)**:
- `PHC_DIAG_DONE_PRINT=1` env var → im_amp_players.py가 매 done 이벤트마다 eps_len 출력 (print_stats 무시)
- `--small_terrain` 플래그 → run.py:249 episode_length=99999 강제 우회
- 진단 명령 (Gate 1 검증):
```bash
PHC_DIAG_DONE_PRINT=1 python phc/run.py \
  --task HumanoidImResAMPVCmdV2 \
  --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml \
  --cfg_train phc/data/cfg/learning/im_pnn_big.yaml \
  --motion_file sample_data/amass_walking_3clips_seamless_60s_v9.pkl \
  --network_path output/HumanoidIm/phc_3 \
  --num_envs 4 --test --epoch -1 --no_virtual_display \
  --headless --small_terrain --episode_length 100
```

---

## 6. 다음 시도 (v2) 개요

자세한 spec: `docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md`

핵심 변경:
- Base task: `HumanoidImMCP` (vanilla `HumanoidIm` 안 씀)
- Reward: `r_survive` 제거, weights는 [0.5, 0.3, 0.2]
- Episode cap: 학습/test 동일
- Sanity check: 1000 epoch마다 inference smoke
- Network: 256-256 (15GB / 7.6GB 한도 유지)
- delta_clip: ±0.1 (보수적)
- 3 trial 병렬:
  - (A) Single residual MLP (단순)
  - (B) Soft-gated 3-residual MoE (속도 specialized)
  - (C) phc_3 column 0/1/2 ablation (학습 없음, local)

---

## 7. 사용한 컴퓨트

| 슬롯 | 학습 시간 | 결과 |
|------|---------|------|
| Slot A (4061, idx0, seed=0) | ~13 시간 | metric 올라감, 실재 학습 안 됨 |
| Slot B (4094, idx2, seed=1) | ~12 시간 | 동일 |
| 디버깅 inference jobs | ~30 분 (10+ jobs) | 진단 |

총 컴퓨트: ~25 GPU-hour, 결과: 0 정책. 다만 4가지 lesson + setup 결함 진단 → 다음 시도 ROI ↑.

---

## 8. 보존된 자산

- `output/HumanoidIm/phc_3/Humanoid.pth` — frozen base (그대로 사용 가능)
- `output/ResAMPVCmd_smoke_ep500.pth.bak` — Stage 1 smoke ckpt (참조용)
- `videos/res_amp_stage2_slotA_ep5500*.mp4` — 실패한 정책의 시각 증거 (다음 시도와 비교 baseline)
- 이 문서 — lesson 보존

# 최근 6개 실험 종합 정리 (AMASS 기반 VIC 학습)

**Date**: 2026-04-20
**Scope**: Velocity command 도입 전후 6개 실험을 하나의 문서로 정리
- **Phase 0 검증** (no cmd): VIC4, VIC8
- **Round 1** (cmd, no personalization): CMD_A, CMD_B
- **Round 2** (cmd + personalization): V2_A, V2_B

**데이터 출처**:
- `02_research_dev/260418_amass_pipeline_verify_result_analysis.md`
- `02_research_dev/260419_amass_cmd_result_analysis.md`
- `02_research_dev/260419_amass_cmd_rounds_combined_analysis.md`
- `exp_config/forward_walking/260417_AMASS_VERIFY/`, `260418_AMASS_CMD/`, `260419_AMASS_CMD_V2/`
- **신규 (260420)**: 로컬 test-greedy 로그 `output/amass_cmd_test_logs/{cmd_a,cmd_b,v2_a,v2_b}_test.log`

---

## 0. 한 눈에 보기

| # | 실험 | Slurm | cmd_w | v_cmd | Shape var. | 그룹 | 완주? | Training rwd/eps_len | Test-greedy rwd/eps_len | Success rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **AMASS_VERIFY_VIC4** | 3733 | — | — | ✗ | 4 | ✅ 20k | ~552 / ~180 | **951.19 / 299** | 100% (15/15) |
| 2 | **AMASS_VERIFY_VIC8** | 3734 | — | — | ✗ | 8 | ✅ 20k | ~588 / ~194 | **940.71 / 299** | 100% (7/7) |
| 3 | **AMASS_CMD_A** | 3756 | 0.3 | [0.8, 1.3] | ✗ | 4 | ✅ 20k | 388.3 / 184.6 | **526.95 / 236.4** | **65.5%** (513/783) |
| 4 | **AMASS_CMD_B** | 3757 | 0.5 | [0.8, 1.3] | ✗ | 4 | ✅ 20k | 278.8 / 185.0 | **440.39 / 278.7** | **86.1%** (371/431) |
| 5 | **AMASS_CMD_V2_A** | 3773 | 0.5 | [0.8, 1.3] | ✓ | 4 | ❌ Ep 8500 (42%) | 203.0 / 138.4 | ⚠ 로컬 segfault (§4.2) |
| 6 | **AMASS_CMD_V2_B** | 3774 | 0.3 | [0.6, 1.4] | ✓ | 4 | ❌ Ep 8238 (41%) | 266.5 / 131.2 | ⚠ 로컬 segfault (§4.2) |

**참조 기준 (gunhee)**: VIC_PHASE test-greedy **932.44 / 299** on 동일 motion.

---

## 1. Phase 0 — AMASS 파이프라인 검증 (VIC4, VIC8)

### 1.1 목적
Phase 0 H5 4-cell 매트릭스(C/D/E/F)에서 최대 605/192(≈50%)로 정체 → gunhee VIC_PHASE(932/299, ≈100%)와 큰 격차. 원인을 두 가설로 분해:
1. **H5 데이터 변환 품질 문제** (상체 7관절 zero-mapping)
2. **프레임워크 드리프트** (amp_agent, humanoid_im_vic 등)

H5 수정 전에 **프레임워크가 여전히 gunhee 수준으로 작동하는지** 검증하기 위해, gunhee가 쓰던 `amass_isaac_walking_forward_single.pkl`로 재현.

### 1.2 공통 설정
- Motion: `sample_data/amass_isaac_walking_forward_single.pkl` (5.43s, 1 clip)
- Task: `HumanoidImVIC`
- max_epochs=20000, num_envs=512, PPO+AMP
- VIC Stage 2, sigma_init=-1.0 (CCF std=0.37), phase_obs=True
- Reward curriculum switch @ Ep 10000 (stage1 task 0.7 / stage2 task 0.3)
- MLP [1024, 1024, 512, 512], lr=5e-5
- 2 variant 차이: **VIC ccf 그룹 수만 다름** (4 vs 8)

### 1.3 결과 (Test-Greedy, deterministic)

| Metric | VIC4 | VIC8 | gunhee 참고 |
|---|---|---|---|
| 에피소드 수 (집계) | 15 | 7 | 1+ |
| eps_len | **299 / 299 (100%)** | **299 / 299 (100%)** | 299 |
| reward mean | **951.19** | 940.71 | 932.44 |
| reward std | **0.35** | 1.55 | — |
| reward min/max | 950.65 / 951.67 | 938.07 / 942.15 | — |

### 1.4 해석
- **두 variant 모두 gunhee 수준 재현/초과** → 프레임워크 회귀 없음. H5 50% 정체는 **데이터 품질 문제로 거의 확정**.
- **VIC4 > VIC8**: mean +10.48, **std 4.4× 작음**, 에피소드 처리 2× 빠름.
- 8-group의 상체 2 dims(Upper_L/R)는 전진 보행 reward와 연결되지 않음 → 노이즈만 추가.
- **260416 mainline 권장(4-group) 데이터로 뒷받침**.

### 1.5 주요 교훈 — Training vs Test 메트릭 괴리
- Training log(552, 588)는 Stage 2 CCF sigma=0.37의 exploration noise로 **실제 성능을 ~35% 과소평가**.
- **모든 VIC 실험의 합/불 판단은 반드시 test-greedy로**.

---

## 2. Round 1 — Velocity Command 도입 (CMD_A, CMD_B)

### 2.1 목적
"CoM velocity command 추종"을 추가한 보행 정책 학습. 단일 axis(cmd_tracking_w)만 변화시켜 비교.

### 2.2 구현 변경
**New task**: `HumanoidImVICCmd` (phc/env/tasks/humanoid_im_vic_cmd.py)
- `HumanoidImVIC` 상속
- task obs += `[v_cmd_x, v_cmd_y, w_cmd_yaw]` (+3 dims)
- reward: `rew = (1 - w) * base + w * exp(-2·|v_err|² - 1·|w_err|²)`
- 에피소드 reset 시 command 재샘플

### 2.3 공통 설정 (Round 1)
- Motion, curriculum switch, ccf groups, max_epochs 등 **Phase 0 VIC4와 동일**
- v_cmd 분포: `U(0.8, 1.3)` m/s (forward only)
- ω_cmd: 0 고정 (단일 forward clip이라 turning 불가)
- Warm-start 없음 (obs dim +3으로 state_dict 불일치)

### 2.4 두 variant
| 항목 | A | B |
|---|---|---|
| cmd_tracking_w | 0.3 (imitation 우선) | 0.5 (command 우선) |
| exp_name | AMASS_CMD_A | AMASS_CMD_B |

### 2.5 결과 (Training log, final 100-ep 평균)

| Variant | Final rwd | Final eps_len | 학습 시간 |
|---|---|---|---|
| A (w=0.3) | **388.3** | 184.6 | 15h 39m |
| B (w=0.5) | 278.8 | **185.0** | 15h 07m |
| (참조) VIC4 no-cmd | 551.0 | 180 | 11h 35m |

**학습 궤적 샘플** (1000-ep 샘플링):

| Ep | A rwd/eps_len | B rwd/eps_len |
|---|---|---|
| 1 | 4 / 2 | 3 / 2 |
| 1001 | 76 / 104 | 129 / 141 |
| 5001 | 320 / 157 | 217 / 153 |
| 10001 (switch) | 344 / 166 | 265 / 177 |
| 15001 | 337 / 158 | 266 / 174 |
| 20000 | 387 / 184 | **312 / 202** |

### 2.6 해석

**Reward 수치는 설계대로**:
- A 예상: `0.7·551 + 0.3·1.0 = 386` ≈ 실측 388 ✓
- B 예상: `0.5·551 + 0.5·1.0 = 275` ≈ 실측 279 ✓
- 두 variant 모두 **cmd_reward ≈ 1.0 달성**, imitation은 VIC4 baseline(551) 수준 유지.

**eps_len 비교**:
- A=184.6, B=185.0 → 무승부. 높은 tracking weight이 보행을 파괴하지 않음.
- **Ep 20000에서 B는 eps_len 202로 여전히 성장 중**, A는 184로 plateau.

### 2.7 Winner 판정 — **B (w=0.5)**
1. Objective가 "velocity command 추종" → B의 w=0.5가 더 직접 반영
2. eps_len 동등 또는 B 우위
3. B는 후반 성장 중이므로 25k~30k까지 연장 시 추가 개선 여지
4. A는 이미 수렴, 추가 학습 효용 낮음

### 2.8 Test-Greedy 평가 (260420 로컬 실행, `--num_envs 1 --test --epoch -1`)

| 지표 | CMD_A (w=0.3) | CMD_B (w=0.5) |
|---|---|---|
| av_reward | **526.95** | **440.39** |
| av_steps | 236.4 | **278.7** |
| 집계 에피소드 수 | 783 | 431 |
| 성공 에피소드 (steps ≥ 290) | 513 | 371 |
| **Success rate** | **65.5%** | **86.1%** |
| Best episode (reward / steps) | 688.7 / 299 | 507.2 / 299 |

### 2.8.1 배율 비교 (training → test-greedy)

| 실험 | Training rwd | Test-greedy rwd | 배율 | 예상 (VIC4 1.73×) | 실제 vs 예상 |
|---|---|---|---|---|---|
| CMD_A | 388 | 526.95 | 1.36× | ~672 | -145 |
| CMD_B | 279 | 440.39 | 1.58× | ~483 | -43 |
| (참조) VIC4 | 551 | 951 | 1.73× | — | — |

**관찰**: VIC4의 1.73× 배율은 CMD 조건에서 낮아짐 (1.36×~1.58×). Cmd reward는 이미 1.0 근처라서 stochastic → greedy 전환으로 얻는 "숨겨진 성능"이 base(imitation)에만 적용됨. 이론 상한 분석은 §2.8.2.

### 2.8.2 Best-episode 역산 — policy가 성공할 때의 base 성능

식: `rew = (1-w)·base + w·cmd_reward`. Best episode에서 cmd_reward≈1.0 가정 시:

| 실험 | Best reward | Implied base | vs VIC4 951 |
|---|---|---|---|
| CMD_A | 688.7 | (688.7 - 0.3)/0.7 ≈ **983.4** | +32 (초과!) |
| CMD_B | 507.2 | (507.2 - 0.5)/0.5 ≈ **1013.4** | +62 (초과!) |

**결론**: 성공 에피소드에서는 **imitation 수준이 VIC4 baseline과 동등 또는 초과**. 평균 rwd가 낮은 건 **실패 에피소드의 비율**이 원인. CMD_A는 실패가 35%, CMD_B는 14%.

### 2.9 Winner 재판정 — **B가 더 robust** (Test-greedy 확인)

Round 1 training metric에서 A가 rwd 높고 B는 후반 성장 중이었으나, **test-greedy에서는 B가 명확한 승자**:

1. **Success rate**: B 86.1% vs A 65.5% (**21%p 차이**)
2. **av_steps**: B 278.7 vs A 236.4 (+42 steps = 1.4초 더 걸음)
3. **Best-episode imitation**: B 1013 > A 983 (둘 다 VIC4 초과)
4. **Av_reward는 공식에 의한 희석 때문에 B가 낮게 보임** — 실질 성능은 B 우세

**해석**: 더 높은 cmd_tracking_w(0.5)가 정책을 "command 궤도"로 더 강하게 유도 → 발산 덜 발생 → robust 보행. Training metric은 stochastic rollout이라 이 차이가 드러나지 않음.

### 2.10 남은 불확실성
- **v_err / ω_err 수치**: reward_cmd≈1.0 시사하지만, 직접 측정 (env yaml의 `cmd_v_range: [1.0, 1.0]` 고정 후 실속도 측정) 미실행
- **B의 Ep 20000 후 추가 학습 효과**: training에서 eps_len 202까지 성장 중이었으므로, 25k~30k 학습 시 success rate 추가 개선 여지

---

## 3. Round 2 — Personalization 추가 (V2_A, V2_B)

### 3.1 목적
사용자 원본 목표 "CoM velocity input **including personalization**"의 personalization 축 추가.

### 3.2 Personalization 방식
PHC 기존 인프라 활용 (코드 수정 없음):
- `robot.has_shape_variation: True` — per-env 서로 다른 SMPL betas(10 params) → 다른 body
- `robot.has_shape_obs: True` — policy 입력에 betas 10 dims 추가 (+10 obs)
- `robot.has_shape_obs_disc: False` — AMP discriminator는 shape 안 봄 (demo는 단일 subject, 혼란 방지)

기각된 대안: subject 임베딩(단일 subject data), motion style 다양화(단일 clip), joint range(shape에 포함됨).

### 3.3 두 variant (Round 1 결과 반영 설계)

| 항목 | V2_A | V2_B |
|---|---|---|
| cmd_tracking_w | **0.5** (Round 1 winner 유지) | **0.3** (Round 1 안정 학습) |
| cmd_v_range | [0.8, 1.3] | **[0.6, 1.4]** (확장) |
| has_shape_variation | True | True |
| has_shape_obs | True | True |
| 나머지 | Round 1과 동일 | Round 1과 동일 |

**두 variant의 축 의미**:
- V2_A = winner + **personalization 추가만**
- V2_B = 안정 w + **personalization + generalization 동시** 추가

### 3.4 결과 — ⚠ **디스크 Full로 중단**

**중단 직전 (final 100-ep 평균)**:

| Variant | 도달 Ep | Final rwd | Final eps_len |
|---|---|---|---|
| V2_A (w=0.5 + pers) | 8500 / 20000 (**42%**) | 203.0 | 138.4 |
| V2_B (w=0.3 + pers + wider) | 8238 / 20000 (**41%**) | 266.5 | 131.2 |

### 3.5 Round 1 동등 epoch 비교 (Ep 8001 기준)

| Config | Round 1 | Round 2 | Δ |
|---|---|---|---|
| w=0.3 | A: 338 rwd / 163 eps_len | V2_B: 276 / 134 | **-62 rwd, -29 eps_len** |
| w=0.5 | B: 194 / 177 | V2_A: 204 / 139 | **+10 rwd, -38 eps_len** |

### 3.6 해석 (잠정)

- **eps_len 전면 감소** (~180 → ~135): 다양한 body가 중간 실패 유발. **예상됨**.
- **Reward는 w에 따라 상반**:
  - w=0.3: personalization이 -62 rwd → imitation 집중 정책이 shape variation으로 어려워짐
  - w=0.5: +10 rwd → command tracking 목표가 shape에 덜 민감
- **V2_B > V2_A** (rwd 266 vs 203): Round 1과 동일 패턴 (낮은 w 우세). eps_len은 V2_A 약간 높음.

### 3.7 **중요 주의사항**
Round 2는 **~40%만 학습**된 상태. 나머지 60%에서:
1. Curriculum switch (Ep 10000) 효과 미확인 (Round 1에선 중요)
2. 수렴 후 personalization 정상화 가능성
3. eps_len 회복 가능성

**→ 현 데이터만으로 "personalization이 나쁘다"고 단정 불가**.

### 3.8 V2 Test-greedy 시도 (260420 로컬) — ⚠ Segfault

로컬 RTX 4060 Ti 8GB에서 V2 partial ckpt 평가 시도:
```bash
python phc/run.py --task HumanoidImVICCmd \
  --cfg_env .../env_im_walk_vic_cmd_V2_A.yaml \
  --cfg_train .../im_walk_vic_cmd_V2_A.yaml \
  --num_envs 1 --test --epoch 8400 --no_virtual_display
```

**결과**: IsaacGym asset 로딩 단계 (SMPL XML 생성, 26/26 진행) 직후 **Segmentation fault (exit 139)**.
- V2_A_00008400.pth (91MB, 정상 크기) → segfault
- V2_B_00008200.pth (91MB) → segfault
- seed 변경(42) 시도 → 동일 segfault

**원인 추정**: `has_shape_variation=True` 코드 경로가 로컬 환경에서 불안정. 서버에서 512 envs 학습은 정상 실행되었으므로 **로컬 특정 이슈** (GPU 메모리, IsaacGym+SMPL 조합, 또는 tmp XML 생성 동시성).

**대안**: 서버에서 test-greedy 평가 필요. 또는 `has_shape_variation: False + has_shape_obs: False`로 env 수정 후 재학습 체크포인트 필요 (현 ckpt는 shape obs 읽어서 state_dict shape 불일치 발생).

**결정**: V2 test-greedy 결과는 **본 라운드에서 취득 불가**. 재학습 또는 서버 평가로 보류.

### 3.9 디스크 위기 사고 요약
- 누적 ckpt 1976 × 90MB ≈ 174G + V2 매 5분 저장 → 09:14 KST 디스크 100%
- V2_A: `PytorchStreamWriter failed writing file data/12` → 즉시 사망
- V2_B: `OSError: [Errno 28] No space left on device` → 8h CPU idle loop (Slurm은 RUNNING 표시하지만 실제 학습 0 진행)
- 복구: Tier1 72.3G + Tier2 ~82G + pip cache 5.8G = **~160G 회수** → 현재 288G/468G(65%), 157G 여유

**재발 방지 교훈**:
1. 실험 시작 전 `df -h` 필수 (150G+ 여유)
2. `save_frequency 100 → 500` 권고 (디스크 5-10× 절약)
3. `scancel` 시 `ps -ef | grep python`으로 실제 프로세스 확인
4. Disk full 경고 모니터 (cron)

---

## 4. 6개 실험 통합 분석

### 4.1 "Command 도입 vs 도입 전" — 설계된 reward 희석 + Test-greedy 검증

| 실험 | 성공 시 base | cmd_reward | training rwd | test-greedy av rwd | test-greedy success |
|---|---|---|---|---|---|
| VIC4 (no cmd) | ~951 | — | 551 | **951** | 100% |
| VIC8 (no cmd) | ~940 | — | 588 | **941** | 100% |
| CMD_A (w=0.3) | 983 (>951) | ≈1.0 | 388 ≈ 0.7·551+0.3·1 | **527** | **65.5%** |
| CMD_B (w=0.5) | 1013 (>951) | ≈1.0 | 279 ≈ 0.5·551+0.5·1 | **440** | **86.1%** |

**핵심 발견**:
1. **Training reward의 희석**: 공식에 의한 구조적 감소. CMD_A 388 ≈ 0.7·551 + 0.3·1 = 386, CMD_B 279 ≈ 0.5·551 + 0.5·1 = 275.
2. **성공 시 imitation은 VIC4 baseline 초과**: CMD_A best 983, CMD_B best 1013. Cmd tracking이 imitation에 추가 보너스처럼 작용.
3. **Test-greedy success rate가 진짜 metric**: B 86% > A 65% → **B가 robust 승자**.
4. **실패 에피소드가 av_reward를 깎음**: CMD_A 35% 실패, CMD_B 14% 실패.

### 4.2 "Personalization 도입" 잠정 비용 (Round 2, 40% 데이터)

| 지표 | Round 1 B @Ep 8k | Round 2 V2_A @Ep 8k | 비용 |
|---|---|---|---|
| rwd (w=0.5) | 194 | 204 | +10 (노이즈 수준) |
| eps_len (w=0.5) | 177 | 139 | **-38** |
| rwd (w=0.3, V2_B vs A) | 338 | 276 | **-62** |
| eps_len (w=0.3) | 163 | 134 | **-29** |

**잠정 비용**: training rwd 15-35%, eps_len 30 정도. 단 **수렴 시 회복 여부 미확인**.

### 4.3 연구 축 한 줄 정리
1. **Phase 0 (VIC4/8)**: 파이프라인 건강 확인, 4-group mainline 확정 (test-greedy 951/299)
2. **Round 1 (CMD_A/B)**: Velocity command 메커니즘 성공 구현, **test-greedy로 B가 robust 승자 확정** (success 86% vs A 65%)
3. **Round 2 (V2_A/B)**: Personalization 추가 학습 난이도 상승 확인 — 완주 데이터 필요 + 로컬 test-greedy 실행 불가 (segfault)

---

## 5. 권고 다음 단계

### 5.1 완료 (260420)
- ✅ **Round 1 A/B test-greedy 평가**: CMD_A 527/236 (65.5%), CMD_B 440/279 (**86.1%**)
- ✅ Winner 재확인: B (w=0.5) — success rate 대폭 우위

### 5.2 즉시 가능 (로컬)
1. **v_cmd 고정 평가**: env yaml의 `cmd_v_range`를 `[0.8, 0.8]`, `[1.0, 1.0]`, `[1.2, 1.2]`, `[1.3, 1.3]` 등으로 고정해서 각각 실행 → 실제 policy 속도 측정 → v_err 계산 (command tracking 정확도 직접 측정)
2. **Ep 5k/10k/15k/20k 체크포인트 비교**: 언제 "명령 추종"이 생겼는지 학습 곡선 확인
3. **CMD_A/B 시각화**: `--num_envs 1 --test --epoch -1` (headless 빼고) 실행 시 보행 품질 눈으로 확인

### 5.3 서버에서 필요
1. **V2_A/V2_B test-greedy 평가**: 로컬은 shape variation 경로에서 segfault. 서버에서 `--num_envs 1 --test --epoch 8400 (또는 8200)` 실행 필요.

### 5.4 서버 재실행 우선순위
**A. Round 2 재시작** (가장 설득력) — V2는 40%만 돌았음. `save_frequency: 100 → 500`으로 디스크 리스크 완화 후 from-scratch 재시작.

**B. Round 1 B (승자) 연장 학습** — Ep 20k 시점에 여전히 성장 중 (eps_len 185→202). Ep 30k까지 연장 시 success rate 86% → 95%+ 기대. 새 ckpt 필요없이 resume으로 저비용.

**C. Multi-clip motion library 빌드** — AMASS primitive.pkl에서 walking subset 추출 (원래 플랜 §2.2). Direction 축 (ω_cmd) 진입의 전제.

### 5.5 구조적 개선 후보
- **Residual impedance head** (`260416_review_and_modified_plan...`): impedance를 gated residual로 강등
- **학습 재개 메커니즘**: rl-games epoch_num 보존 (현재는 resume 시 0부터 시작 → ckpt overwriting 리스크)

---

## 6. 체크포인트 & 산출물

### 6.1 완주 체크포인트
| 실험 | 최종 ckpt | 중간 마일스톤 |
|---|---|---|
| AMASS_VERIFY_VIC4 | `output/AMASS_VERIFY_VIC4.pth` | 5k/10k/15k/20k |
| AMASS_VERIFY_VIC8 | `output/AMASS_VERIFY_VIC8.pth` | 5k/10k/15k/20k |
| AMASS_CMD_A | `output/AMASS_CMD_A.pth` | 5k/10k/15k/20k |
| AMASS_CMD_B | `output/AMASS_CMD_B.pth` | 5k/10k/15k/20k |

### 6.2 부분 체크포인트
| 실험 | 마지막 ckpt |
|---|---|
| AMASS_CMD_V2_A | `output/AMASS_CMD_V2_A_00008500.pth` |
| AMASS_CMD_V2_B | `output/AMASS_CMD_V2_B_00008200.pth` |

### 6.3 Test-greedy 로그 (260420 신규)
| 파일 | 내용 |
|---|---|
| `output/amass_cmd_test_logs/cmd_a_test.log` | CMD_A av_reward 526.95, av_steps 236.4 |
| `output/amass_cmd_test_logs/cmd_b_test.log` | CMD_B av_reward 440.39, av_steps 278.7 |
| `output/amass_cmd_test_logs/v2_a_test.log` | V2_A segfault (shape variation 경로) |
| `output/amass_cmd_test_logs/v2_b_test.log` | V2_B segfault |
| `output/amass_verify_test_logs/vic4_test.log` | VIC4 951.19 / 299 (기존) |
| `output/amass_verify_test_logs/vic8_test.log` | VIC8 940.71 / 299 (기존) |

### 6.4 관련 문서
| 파일 | 내용 |
|---|---|
| `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` | Phase 0 VIC4/8 상세 (test-greedy 951/940) |
| `02_research_dev/260419_amass_cmd_result_analysis.md` | Round 1 A/B 상세 |
| `02_research_dev/260419_amass_cmd_rounds_combined_analysis.md` | Round 1 + Round 2(부분) 종합 + 디스크 사고 postmortem |
| `02_research_dev/260420_recent_6_experiments_summary.md` | **본 문서** |
| `exp_config/forward_walking/260417_AMASS_VERIFY/README_FOR_SERVER.md` | Phase 0 실행 문서 |
| `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md` | Round 1 실행 문서 |
| `exp_config/forward_walking/260419_AMASS_CMD_V2/RUNNING_ON_SERVER_V2.md` | Round 2 실행 문서 |

### 6.5 구현 코드
- `phc/env/tasks/humanoid_im_vic_cmd.py` — Round 1에서 신규 작성, V2에서 재사용
- `phc/utils/parse_task.py` — `HumanoidImVICCmd` 등록

---

*2026-04-20 작성 — 사용자 요청으로 최근 6개 실험을 한 문서에 통합.*
*2026-04-20 업데이트 — 로컬 test-greedy 결과 추가 (CMD_A/B 완료, V2 segfault).*

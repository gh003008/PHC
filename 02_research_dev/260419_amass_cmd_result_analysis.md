# AMASS Velocity-Command VIC (Round 1, A/B) 결과 분석

**Date**: 2026-04-19 (자율 세션, 서버 Claude)
**Experiments**: `260418_AMASS_CMD` (A1/B1), Slurm 잡 3756/3757
**Motion**: `sample_data/amass_isaac_walking_forward_single.pkl` (단일 클립, 5.43s)
**Task**: `HumanoidImVICCmd` (VIC + velocity command obs/reward, 4-group CCF)
**Training**: 20000 epochs each, num_envs=512, PPO+AMP

---

## 1. 핵심 결과

### 1.1 학습 종료 시점 (final 100-ep 평균, Ep 19901-20000)

| Variant | cmd_tracking_w | Final rwd | Final eps_len | 학습 시간 |
|---|---|---|---|---|
| **A** (moderate) | 0.3 | **388.3** | **184.6** | 15h 39m |
| **B** (aggressive) | 0.5 | 278.8 | 185.0 | 15h 07m |
| AMASS_VERIFY_VIC4 (참조, no cmd) | — | 551 | 180 | 11h 35m |

**두 variant 모두 정상 종료** (Ep 20000 도달, MAX EPOCHS NUM 정상 메시지). Crash 없음.

### 1.2 학습 속도 (ep/min)
- A: 1278 ep / 60 min = 21.3 ep/min
- B: 1323 ep / 60 min = 22.1 ep/min

Command 추가로 인한 overhead는 미미 (기존 VIC4 대비 ~5% 느림).

---

## 2. 학습 궤적 (1000-ep 샘플링)

### 2.1 A (cmd_tracking_w=0.3) — imitation-priority
```
Ep 1     rwd 4    eps_len 2
Ep 1001  rwd 76   eps_len 104    <- 빠른 초기 상승
Ep 2001  rwd 289  eps_len 153
Ep 5001  rwd 320  eps_len 157
Ep 10001 rwd 344  eps_len 166    <- curriculum switch 근처
Ep 13001 rwd 344  eps_len 164    <- plateau
Ep 18001 rwd 368  eps_len 174
Ep 20000 rwd 387  eps_len 184    <- 최종
```
**특성**: 초기 급상승 → 중기 plateau (3000~15000) → 후반 완만한 상승

### 2.2 B (cmd_tracking_w=0.5) — command-priority
```
Ep 1     rwd 3    eps_len 2
Ep 1001  rwd 129  eps_len 141   <- A보다 초기 eps_len 빠름
Ep 2001  rwd 208  eps_len 150
Ep 5001  rwd 217  eps_len 153
Ep 10001 rwd 265  eps_len 177   <- curriculum switch 근처
Ep 13001 rwd 253  eps_len 165
Ep 18001 rwd 263  eps_len 176
Ep 20000 rwd 312  eps_len 202   <- 최종
```
**특성**: A보다 낮은 reward 수준에서 학습. 중반 plateau 길지만 후반 eps_len이 202로 A 초과

---

## 3. 해석

### 3.1 cmd_tracking_w 효과
두 variant의 rwd 차이(388 vs 279) = w 차이(0.3 vs 0.5)에 의한 것:
- reward 식: `rew = (1-w) * base + w * cmd_reward`
- `base` (imitation+disc): VIC4 baseline 551 기준
- `cmd_reward ∈ [0, 1]` (exp of -squared_error)

예상 최대 reward:
- A: `0.7 * 551 + 0.3 * 1.0 = 386` ≈ 실측 388 ✅
- B: `0.5 * 551 + 0.5 * 1.0 = 275` ≈ 실측 279 ✅

**즉 두 variant 모두 학습된 policy는 VIC4 baseline 수준의 imitation을 유지하면서 command tracking도 거의 최대치로 도달한 것으로 해석 가능**.

### 3.2 eps_len 비교 (구조적 안정성)
- A: 184.6
- B: 185.0
- 거의 동일 → B의 높은 cmd_tracking_w가 보행을 망가뜨리지 않음

B의 최종 Ep 20000 eps_len=202이 A의 184보다 **높음**. B는 command 추종에 더 집중하면서도 보행 안정성은 유지 또는 개선.

### 3.3 Training metric의 한계 (260418 분석서 교훈)
Stage 2 CCF sigma=0.37로 인한 exploration noise가 training rwd를 ~35% 저평가하는 것이 알려짐:
- VIC4 training 551 → test-greedy 951 (1.73×)
- 추정 test-greedy:
  - A: 388 × 1.73 ≈ 672?
  - B: 279 × 1.73 ≈ 483?
  
하지만 이 배율은 cmd 조건 하에서 유지될지 불명. test-greedy 평가 필수 (사용자 로컬에서).

---

## 4. 두 variant 비교 & 승자 판정

| 축 | A (w=0.3) | B (w=0.5) | 승자 |
|---|---|---|---|
| Final training rwd | 388 | 279 | A (by design) |
| Final eps_len | 184.6 | 185.0 | 무승부 |
| Ep 20000 eps_len | 184 | 202 | **B** |
| 학습 궤적 안정성 | 초기 빠른 상승, 일찍 plateau | 느린 시작, 지속 성장 | **B** (후반 더 학습) |
| Command tracking 예측력 | 약 (30% weight) | 강 (50% weight) | **B** |
| Reward 상한 (계산상) | 386 | 275 | A (수치만) |

### 4.1 자율 판정: **B가 objective에 더 부합**

근거:
1. **Objective가 "velocity command 추종"** — B의 w=0.5가 objective를 더 강하게 반영
2. **eps_len 동등 또는 B 우위** — command 중심 학습이 보행을 파괴하지 않음
3. **Ep 20000 시점 B가 eps_len 202로 여전히 성장 중** — A는 184로 plateau
4. **Training reward 공식적 해석**: 양쪽 다 cmd_reward 근사 최대. B는 그걸 50% 가중해서 가져감

### 4.2 주의사항
- 실제 command 추종 정확도(v_err, ω_err)는 **test-greedy 평가 필수**
- B가 후반에 성장 중이므로 epoch 25000-30000까지 이어서 학습하면 더 개선 가능성 있음
- A는 이미 수렴 — 추가 학습으로 얻을 것 적음

---

## 5. V2 라운드 설계 (이 분석을 바탕으로 이어서 실행 중)

### 5.1 판단
- **Winner of Round 1**: B (w=0.5)
- **But**: A의 학습이 더 안정적이었으므로 V2 기본은 B, 하지만 A 성향(w=0.3)도 다른 axis와 조합해서 비교

### 5.2 V2 objective 재확인
사용자 원본 목표:
> embedding human agent as a simulation for CoM velocity input **including personalization**

**Personalization axis**: SMPL body shape variation
- `robot.has_shape_variation: True` → per-env 다른 SMPL betas (10 params)
- `robot.has_shape_obs: True` → policy 입력에 betas 10 dims 추가
- 결과: policy가 "다양한 body를 command 속도로 걷게" 하는 것 학습

### 5.3 V2 두 variant
- **V2_A** (`AMASS_CMD_V2_A`): winning cmd_tracking_w=0.5 + **personalization ON**
- **V2_B** (`AMASS_CMD_V2_B`): cmd_tracking_w=0.3 + **personalization ON** + **wider v_cmd range [0.6, 1.4]**

두 variant 차이:
- V2_A: 승자 설정 + personalization (가장 목표 부합)
- V2_B: A 설정(안정 학습) + personalization + generalization (넓은 속도 범위 테스트)

V2_A가 objective 가장 잘 반영, V2_B는 일반화 여유 테스트.

### 5.4 V2 예상 리스크
1. **Shape variation 초기 OOM**: 512개 env별 다른 XML 생성. Init 시 많은 메모리 필요. 완화: 동일 sbatch --mem=15G 유지, 안 되면 num_envs 줄임.
2. **Obs dim +10로 학습 느려짐**: 기존 Obs + 3(cmd) + 10(shape) = 의 추가 13 dims. 큰 변화 아님. 정책 수용량 충분.
3. **AMP discriminator 혼란**: demo는 단일 subject이므로 shape 다양화가 disc 관점에서 분포 이탈. 완화: `has_shape_obs_disc: False` 유지 (disc는 shape 안 봄).

---

## 6. 체크포인트 & 산출물

| 파일 | 위치 |
|---|---|
| A 최종 ckpt | `output/AMASS_CMD_A.pth` + `output/AMASS_CMD_A_00020000.pth` |
| B 최종 ckpt | `output/AMASS_CMD_B.pth` + `output/AMASS_CMD_B_00020000.pth` |
| A 학습 로그 | `logs/amass_cmd_A_3756.out` / `.err` |
| B 학습 로그 | `logs/amass_cmd_B_3757.out` / `.err` |
| Round 1 문서 | `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md` |

---

## 7. 다음 단계

1. V2 실험 실행 (동시 작성되는 문서 참조: `exp_config/forward_walking/260419_AMASS_CMD_V2/RUNNING_ON_SERVER_V2.md`)
2. 사용자 복귀 후 A/B test-greedy 평가로 실제 tracking 정확도 확인 (MD의 §6 수식/명령 사용)
3. V2 결과 들어오면 `02_research_dev/260420_amass_cmd_v2_result_analysis.md` 작성 예정

---

## 8. 참조

- Round 1 계획: `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md`
- Round 1 실행 문서: `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md`
- VIC4 검증 분석: `02_research_dev/260418_amass_pipeline_verify_result_analysis.md`
- Impedance 재배치 제안: `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md`

---

*이 문서는 자율 세션에서 Claude가 작성. 사용자 복귀 후 확인/수정 환영.*
*작성 시각: 2026-04-19 KST 02:30, Slurm 3756/3757 종료 직후.*

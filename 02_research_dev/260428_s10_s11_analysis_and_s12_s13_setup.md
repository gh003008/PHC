# 260428 — S10/S11 학습 결과 분석 + S12/S13 셋업 차이 정리

S10·S11 종료 시점(사용자 요청에 따른 scancel) 결과 분석과 현재 학습 중인 S12/S13의 의도·차이를 정리한다. 최종 목적은 "왜 S10가 두 발로 걷지만 관절 각도가 레퍼런스와 다르게 보이는가"의 원인 가설 분리와, 다음 두 슬롯에서 그 가설을 검증할 수 있도록 설계 근거를 남기는 것이다.

## 1. 요약 표 (4슬롯 비교)

| 항목 | S10 | S11 | **S12** (학습 중) | **S13** (학습 중) |
|---|---|---|---|---|
| 학습 데이터 | v8 (2-clip) | v8 (2-clip) | **v9 (3-clip + KIT_11)** | **v9 (3-clip + KIT_11)** |
| 자연 v_x 범위 | [0.975, 1.068] | [0.975, 1.068] | **[0.897, 0.975, 1.068]** | **[0.897, 0.975, 1.068]** |
| multiclip_v_cmd_range | [0.82, 1.23] | [0.82, 1.23] | **[0.76, 1.23]** | **[0.76, 1.23]** |
| foot_pos_reward (w, k) | 0.3, 30 | 0.3, 30 | 0.3, 30 | 0.3, 30 |
| foot_clearance_reward | OFF | **w=1.0, k=30, thresh=0.05** | OFF | OFF |
| reward_w_stage2_task | 0.3 | 0.3 | 0.3 | **0.5** |
| reward_w_stage2_disc | 0.7 | 0.7 | 0.7 | **0.5** |
| num_envs | 512 | 512 | 384 | 384 |
| 학습 종료 epoch (or 진행) | ~16221 (cancel) | ~16315 (cancel) | running | running |

## 2. S10 결과 분석

### 2.1 학습 지표 (마지막 관측)
- Train rwd: ~330 / Train eps_len: ~150 (300의 50%)
- Eval (ep14500, 4 envs × 20s, termDist=0.5 복원):
  - 7개 에피소드 중 4개가 ceiling(299) 도달, 3개가 조기 종료(2/95/186)
  - 최대 보상: 1022
- 영상 기록: `videos/VIC4_VCMD_S10_ep14500.mp4`

### 2.2 정성 평가
- ✓ 두 발로 번갈아 스텝 (S5/S6/S7/S8에서 보였던 sliding 문제 해결)
- ✓ 넘어지지 않음 (terminationDistance=0.5 안정)
- ✓ 전진 운동 발생
- ✗ **관절 각도가 레퍼런스와 시각적으로 많이 다름**
- ✗ eps_len 50% (절반의 episode가 끝까지 못 감)

### 2.3 핵심 관찰
foot_pos_reward를 w=0.1→0.3로 강화하니 발이 적극적으로 ref를 추종하여 stepping이 살아났다. 그러나 stage 2 reward curriculum이 task=0.3 / disc=0.7로 disc가 우세하기 때문에, "그럴듯한 보행처럼 보이면" disc 보상은 충족되지만 정확한 관절 각도까지는 추종되지 않는다. 이것이 "걷긴 하는데 폼이 이상하다"의 정량 원인 가설.

## 3. S11 결과 분석

### 3.1 학습 지표
- Train rwd: ~450 / Train eps_len: ~150 (300의 50%)
- Eval (ep14600, 4 envs × 20s):
  - 5개 에피소드 중 3개가 ceiling(299), 2개가 즉시 fall(2)
  - 최대 보상: 1197
- 영상 기록: `videos/VIC4_VCMD_S11_ep14600.mp4`

### 3.2 정성 평가
- ✓ 궤적(root) 추종은 S10보다 명확히 우수 (수치상 train rwd가 더 높음)
- ✗ **앞발이 한 번 step한 뒤 뒤발이 따라오지 않고 정지** ("back foot stuck")
- ✗ 즉, foot_clearance_reward(swing-only z height)가 averaging trap에 걸려 한쪽 발만 일을 해도 보상이 충족됨

### 3.3 핵심 관찰
foot_clearance_reward는 4개 발 중 swing 마스크가 켜진 발들에 대해 z height shortfall을 평균한다. 따라서 한 발이 잘 들어 올려지면 다른 발은 정지해 있어도 reward가 양호한 값을 갖는 local optimum이 존재한다. 이는 reward shaping의 설계상 결함으로, 다음 회차에서는 per-foot multiplicative 형태로 변경하거나 forward-displacement reward를 추가하는 방향이 필요하다.

## 4. S10 vs S11 정리

| 차원 | S10 우위 | S11 우위 |
|---|---|---|
| 두 발 stepping | ✓ | ✗ (back foot stuck) |
| 궤적 추종 | △ | ✓ |
| 폼(관절 각도) | ✗ | ✗ (둘 다 어색) |
| 학습 reward 절댓값 | 330 | 450 |
| eps_len ceiling 도달률 | 50% | 50% |

핵심: S10는 stepping은 살리되 폼이 어색하고, S11은 reward 수치는 높으나 시각적으로는 한 발만 일하는 quasi-gait가 됨. 둘 다 episode_length 50% 천장으로 막혀 있어 "eps_len is truth" 기준으로는 모두 sub-optimal.

## 5. S12·S13으로 가는 가설 분리

S10/S11이 공통적으로 가진 문제 — 관절 각도가 ref와 다름 — 의 원인 가설은 두 가지로 나뉜다.

### 가설 A. **데이터 다양성 부족**
v8(2 clip)은 모두 KIT_425 동일 피험자, 자연 속도 0.97–1.07 m/s 거의 동일. AMP disc가 이 좁은 분포만 학습하여 정책에 변화 압력이 약함 → 정책이 ref와 다르게 가도 disc는 "그럴듯한 보행"으로 판정.

### 가설 B. **AMP disc 우세 (stage 2 보상 가중)**
stage 2에서 task 0.3 / disc 0.7로 disc가 우세 → "보행처럼 보이면" 보상이 70% 충족되므로 task(관절 위치/회전 추종)는 30%만 끌어주어도 정책이 안정점에 정착. 이로 인해 정책이 ref 관절 각도에서 떨어진 자세도 받아들임.

### 분리 설계
- **S12 = v9 데이터 + S10의 보상 셋업** → A만 변경, B는 유지. 데이터 다양성만으로 폼 문제가 해결되는지 측정.
- **S13 = v9 데이터 + S10의 보상 셋업 + stage 2 task=0.5/disc=0.5** → A·B 동시 변경. 둘 다 적용한 상한을 측정.

## 6. v9 데이터 (3-clip) 빌드 결과

새 클립 후보 검사 후 **KIT_11_WalkingStraightForwards05** 추가:
- root z mean = 0.8987 (v8 medium08 baseline 0.8964 대비 **+0.2 cm**) → 운동학적 미스매치 거의 없음
- β = [0,0,0] neutral (canonical SMPL)
- 직진 보행 (v_y << v_x)
- raw 중간창 v_x = -0.674 m/s

seamless 60s loop 빌드 후 SNR picker가 가장 깨끗한 steady cycle (HS[1]→HS[3], stddev=0.110)을 선택, 결과적으로 seamless v_x = **-0.897 m/s**가 되었다. 원시 클립 후반 -0.30 / -0.09 m/s 구간은 피험자가 정지하기 위해 감속하는 부분이라 picker가 정확히 reject. 그래서 속도 범위 확장은 modest:

- v8 union: [0.829, 1.228]
- v9 union: [0.762, 1.228] (저측 +0.07, 7%)

속도 범위 자체는 크게 늘지 않았지만, **3번째 피험자가 들어가면서 보행 스타일(arm swing, trunk lean, knee lift) 분산이 늘어남**이 본 슬롯에서 노린 효과. AMP disc가 더 다양한 분포를 학습해야 정책에 더 강한 압력이 걸리고, 그 압력이 있어야 task가 작은 가중치로도 ref 관절 각도까지 끌어올 수 있다는 가설.

## 7. S12·S13 결과 해석 매트릭스

| S12 | S13 | 해석 |
|---|---|---|
| ✓ ceiling, ref-like | ✓ | 데이터 다양성만으로 충분. S12 레시피 채택. |
| ✗ 폼 여전히 이상 | ✓ | reward rebalance가 필수. S13 레시피 채택. |
| ✓ | ✗ degraded | (예외적) disc curriculum이 도움이 되었다는 뜻. 0.3/0.7 유지. |
| ✗ 둘 다 | — | 보상 디자인 / track obs / 모델 capacity 등 다른 병목. 재설계 필요. |

## 8. 실행상의 작은 이슈 정리

1. S9(데이터 변경만, S8 보상 그대로)는 처음 두 차례 모두 OOM-killed (15G 한도). v9 적용 직전이므로 이번에는 S9 baseline을 따로 만들지 않음. S12가 사실상 "데이터 변경만"의 역할을 겸함.
2. PHC 측 CLI `--num_envs` 옵션은 `cfg.env.numEnvs` (camelCase)만 덮어쓰고, env 생성 코드는 `cfg.env.num_envs` (snake_case)를 직접 읽음. 이로 인해 buffer-size mismatch가 발생. 해결: yaml에서 양쪽 필드를 모두 384로 직접 명시.
3. partition `MaxMemPerNode = 15500 MB`. 24G 요청 시 PD(pending) 상태로 들어감. 15G는 partition 한도 그대로.

## 9. 다음 의사결정 분기 (S12/S13 종료 후)

1. **둘 다 ceiling + ref-like 폼** → 정책 정착. S14에서 v_cmd controllability 검증(속도 슬라이드, 회전 등)로 이행.
2. **S13만 성공** → AMP disc 우세 가설 확인. stage 2 0.5/0.5를 default로 채택, foot_clearance를 다시 도입할 때 per-foot 곱셈 형태로 재설계.
3. **둘 다 실패** → 보상이 아닌 obs/모델 병목 가능성. trackBodies 확장(현재 4개 → 모든 lower-body), numTrajSamples 증가 등 재검토.

---

학습 종료 예상 시각: 2026-04-29 08–10 KST (16k epoch 기준). 종료 후 ckpt rsync → 시각화 → 위 매트릭스에 기입 예정.

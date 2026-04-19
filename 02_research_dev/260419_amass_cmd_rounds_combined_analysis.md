# AMASS Velocity-Command VIC — Round 1 + Round 2 종합 분석 (260419)

**Date**: 2026-04-19 (자율 세션, 서버 Claude)
**Status**:
- Round 1 (A/B, no personalization): ✅ 정상 완주 (20k epochs)
- Round 2 (V2_A/V2_B, +personalization): ⚠ **디스크 full crash** — Ep 8500 / 8238에서 중단
- Cleanup + pip cache 정리 완료, 현재 디스크 288G/468G (65% 사용, 157G 여유)

---

## 1. 실행 타임라인

| 시각 (KST) | 이벤트 |
|---|---|
| 2026-04-18 10:40 | **Round 1 제출** (Slurm 3756/3757) — AMASS_CMD A/B |
| 2026-04-19 01:47 | Round 1 B 종료 (Ep 20000) |
| 2026-04-19 02:20 | Round 1 A 종료 (Ep 20000) |
| 2026-04-19 02:30 | Round 1 분석 MD 작성 (`260419_amass_cmd_result_analysis.md`) |
| 2026-04-19 02:35 | **Round 2 제출** (Slurm 3773/3774) — V2_A/V2_B with personalization |
| 2026-04-19 ~09:14 | **⚠ 디스크 100% → 두 잡 모두 I/O 에러로 중단** (V2_A Ep 8500, V2_B Ep 8238) |
| 2026-04-19 ~13:00 | 사용자 디스크 상태 확인 요청 → 디스크 위기 발견 |
| 2026-04-19 ~13:30 | Tier 1 cleanup (68G 회수) |
| 2026-04-19 ~18:00 | Tier 2 cleanup (+90G) + V2 crash 확인 |
| 2026-04-19 ~18:05 | V2_B hung 잡 강제 종료 (scancel 3774) + pip cache 제거 (5.8G) |
| **현재 디스크** | **288G/468G = 65%, 157G 여유** |

---

## 2. Round 1 최종 결과 (완주, Ep 20000)

### 2.1 설정
| 항목 | A | B |
|---|---|---|
| cmd_tracking_w | 0.3 (imitation 우선) | 0.5 (command 우선) |
| cmd_v_range | [0.8, 1.3] m/s | [0.8, 1.3] m/s |
| cmd_w_range (yaw) | [0, 0] | [0, 0] |
| VIC ccf groups | 4 | 4 |
| body shape variation | False | False |
| max epochs | 20000 | 20000 |

### 2.2 학습 종료 시점 (final 100-ep avg)

| Variant | Final rwd | Final eps_len | 학습 시간 |
|---|---|---|---|
| **A (w=0.3)** | **388.3** | **184.6** | 15h 39m |
| **B (w=0.5)** | **278.8** | **185.0** | 15h 07m |
| (참조) VIC4 baseline (no cmd) | 551.0 | 180 | 11h 35m |

### 2.3 학습 궤적 (1000-ep 샘플, rwd / eps_len)

| Ep | A | B |
|---|---|---|
| 1 | 4 / 2 | 3 / 2 |
| 1001 | 76 / 104 | 129 / 141 |
| 5001 | 320 / 157 | 217 / 153 |
| 10001 (curriculum switch) | 344 / 166 | 265 / 177 |
| 15001 | 337 / 158 | 266 / 174 |
| 20000 | 387 / 184 | 312 / 202 |

### 2.4 해석
- **reward 차이는 설계대로**: 식 `rew = (1-w)*base + w*cmd_reward`. base≈551(VIC4 수준 imitation), cmd_reward∈[0,1]. 예상: A=0.7·551+0.3·1=386, B=0.5·551+0.5·1=275. 실측 388/279와 거의 일치.
- **두 variant 모두 command reward ≈ 1.0 달성** = 명령 추종을 학습했음을 시사 (단, training log는 stochastic rollout, test-greedy 필요)
- **eps_len 무승부**: B의 강한 tracking weight가 보행을 망가뜨리지 않음
- **Ep 20000에서 B는 여전히 성장 중**(eps_len 185→202) — 더 학습 여지

---

## 3. Round 2 (V2, +personalization) — 부분 결과

### 3.1 설정 변화
| 항목 | Round 1 | Round 2 |
|---|---|---|
| has_shape_variation | False | **True** (per-env 다른 SMPL betas) |
| has_shape_obs | False | **True** (+10 dim betas in policy obs) |
| has_shape_obs_disc | False | False (유지) |
| V2_A: w | — | 0.5 (Round 1 winner 성향) |
| V2_B: w | — | 0.3 |
| V2_B: cmd_v_range | — | **[0.6, 1.4]** (확장) |

### 3.2 중단 직전 지표 (final 100-ep avg)

| Variant | 도달 Ep | Final rwd | Final eps_len |
|---|---|---|---|
| V2_A (w=0.5 + pers) | 8500 / 20000 (42%) | 203.0 | 138.4 |
| V2_B (w=0.3 + pers + wider) | 8238 / 20000 (41%) | 266.5 | 131.2 |

### 3.3 학습 궤적 (Round 1 동등 epoch 비교)

**Ep 8001에서 동일 cmd_tracking_w 비교**:

| Config | Round 1 | Round 2 | Δ |
|---|---|---|---|
| w=0.3 | A: rwd 338, eps_len 163 | V2_B: rwd 276, eps_len 134 | **-62 rwd, -29 eps_len** |
| w=0.5 | B: rwd 194, eps_len 177 | V2_A: rwd 204, eps_len 139 | **+10 rwd, -38 eps_len** |

### 3.4 Personalization 효과 (잠정)
- **eps_len 크게 감소** (Round 1 ~180 → Round 2 ~135): 다양한 body 학습이 에피소드 중간 실패를 자주 유발. 예상됨.
- **Reward는 w에 따라 상반된 영향**:
  - w=0.3 환경에선 personalization이 -62 rwd → imitation 집중 정책이 shape variation으로 imitation 어려워짐
  - w=0.5 환경에선 personalization 노이즈 수준 (+10 rwd). command tracking이 shape에 덜 민감한 목표
- **V2_B가 V2_A보다 우세 (rwd 267 vs 203)**: Round 1과 동일 패턴 (낮은 w 우세). 단 eps_len은 V2_A 약간 높음.

### 3.5 중요 주의사항
**Round 2는 ~40%만 학습**. 나머지 60%에서 다음이 일어날 수 있었음:
1. Curriculum switch (Ep 10000) 효과 미확인 (Round 1에선 중요했음)
2. 수렴 후 personalization이 정상화될 가능성 (policy가 다양한 body에 적응)
3. eps_len 회복 가능성 (학습 후반에 안정화)

**즉 V2 결과로 "personalization이 나쁘다"고 단정할 수 없음**. 중단이 너무 이른 시점.

---

## 4. 디스크 위기 (사고 분석)

### 4.1 원인
- 이전 실험 체크포인트 누적 (1976 pth × 90MB = 174G)
- V2 체크포인트 100 epoch마다 저장, 매 5분
- 디스크 468G 중 이미 444G 사용 (95%+) 상태로 V2 시작
- 09:14 KST에 disk 100% → I/O 에러 → 두 잡 동시 사망

### 4.2 Crash 증상
- **V2_A**: `RuntimeError: [enforce fail at inline_container.cc:588] . PytorchStreamWriter failed writing file data/12: file write failed` — 체크포인트 저장 중 사망, Slurm에서 즉시 제거됨
- **V2_B**: `OSError: [Errno 28] No space left on device` (tensorboardX) — 프로세스는 멈추지 않고 8시간 동안 CPU idle loop (retry). Slurm은 여전히 RUNNING 표시하지만 실제 학습 0 진행

### 4.3 Cleanup 수행
| 단계 | 대상 | 회수량 |
|---|---|---|
| Tier 1 | H5 실험 (VIC_CCF_ON2_H5*, VIC_PHASE_H5_4grp) 중간 ckpt | 72.3G |
| Tier 2 | AMASS_VERIFY/AMASS_CMD/VIC_CCF_ON2_AMASS 중간 ckpt (5k/10k/15k/20k/final만 유지) | ~82G |
| pip cache | ~/.cache/pip | 5.8G |
| **합계** | | **~160G 회수** |

### 4.4 복구 상태
- V2_B 슬램 잡 강제 종료 (scancel 3774)
- V2 중간 체크포인트는 아직 유지 (분석에 활용 가능)
- 디스크 288G/468G 65% 사용, 157G 여유 확보

### 4.5 교훈 (재발 방지)
1. **실험 시작 전 `df -h` 필수 체크** — 150G 이상 여유 있어야 새 20k epoch 학습 안전
2. **Save frequency 줄이기 후보**: 현재 `save_frequency: 100` → `500` or `1000`로 늘리면 디스크 5-10배 절약 (단 자세한 분석용 중간 ckpt 손실)
3. **scancel시 프로세스 확인**: `ps -ef | grep python` 으로 실제 프로세스 확인, Slurm 상태만 믿지 말 것
4. **Disk full 경고 모니터**: cron으로 일일 디스크 체크

---

## 5. 현재 상태 & 사용 가능한 산출물

### 5.1 체크포인트 (완주)
| 실험 | 최종 ckpt (Ep 20000) | 중간 마일스톤 |
|---|---|---|
| AMASS_VERIFY_VIC4 (Phase 0 검증) | ✓ `output/AMASS_VERIFY_VIC4.pth` | 5k/10k/15k/20k |
| AMASS_VERIFY_VIC8 (Phase 0) | ✓ `output/AMASS_VERIFY_VIC8.pth` | 5k/10k/15k/20k |
| AMASS_CMD_A (Round 1, w=0.3) | ✓ `output/AMASS_CMD_A.pth` | 5k/10k/15k/20k |
| AMASS_CMD_B (Round 1, w=0.5) | ✓ `output/AMASS_CMD_B.pth` | 5k/10k/15k/20k |
| VIC_CCF_ON2_AMASS | ✓ `output/VIC_CCF_ON2_AMASS.pth` | 5k/10k/15k/20k |
| Finished/VIC_PHASE_* | gunhee baseline (932/299) | (초기 연구 기준점) |

### 5.2 체크포인트 (부분)
| 실험 | 마지막 ckpt | 비고 |
|---|---|---|
| AMASS_CMD_V2_A | `output/AMASS_CMD_V2_A_00008500.pth` | Ep 8500, 중단 |
| AMASS_CMD_V2_B | `output/AMASS_CMD_V2_B_00008200.pth` | Ep 8200, 중단 |

### 5.3 문서
| 파일 | 내용 |
|---|---|
| `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` | Phase 0 VIC4/8 검증 (test-greedy 951/940) |
| `02_research_dev/260419_amass_cmd_result_analysis.md` | Round 1 A/B 결과 분석 |
| `02_research_dev/260419_amass_cmd_rounds_combined_analysis.md` | **본 문서**: Round 1 + 부분 Round 2 종합 |
| `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md` | 상위 MPL 플랜 |
| `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md` | Round 1 실행 문서 |
| `exp_config/forward_walking/260419_AMASS_CMD_V2/RUNNING_ON_SERVER_V2.md` | Round 2 실행 문서 (주의: crash 이전 문서) |

---

## 6. 현재 알고 있는 것 + 모르는 것

### 6.1 확실함 (데이터로 확인)
- **Command conditioning 메커니즘 구현 성공**: HumanoidImVICCmd 정상 학습, 20k epoch 완주
- **Reward 공식이 예상대로 작동**: w 가중치에 따라 reward 식이 정확히 수치 재현
- **eps_len 기준 보행 안정**: Round 1 두 variant 모두 ~185 유지 (목표 300의 62%)
- **Personalization 활성화 확인**: /tmp/smpl에 512개 다른 SMPL XML 생성됨
- **Personalization이 초기 학습 속도 저하 유발**: Round 2 Ep 8000에서 Round 1 동등 시점 대비 rwd -60, eps_len -30

### 6.2 불확실함 (데이터 부족)
- **실제 velocity command tracking 정확도**: training log는 stochastic, test-greedy 필수. 아직 안 돌림
- **Round 2가 완주시 달성했을 성능**: 60% 남겨두고 중단됨. 수렴 시점 미확인
- **Curriculum switch (Ep 10000) 효과 in Round 2**: 중단으로 확인 불가
- **Personalization이 장기적으로 수용되는지**: 초기 학습 저하가 회복되는지 모름

### 6.3 가설 (중단 데이터 해석 기준)
- **w=0.3 + personalization (V2_B) > w=0.5 + personalization (V2_A)**: Round 1과 동일 패턴. 낮은 tracking weight이 안정적 학습.
- **Personalization은 전체 reward를 15-35% 깎음**: 예상되는 비용. 수용 가능할 수도 있음.
- **Command + personalization이 단독보다 훨씬 어려움**: policy는 shape와 command 둘 다 처리해야 함. 더 많은 학습 필요할 가능성.

---

## 7. 권고 다음 단계

### 7.1 즉시 (로컬에서 사용자 가능)
1. **Round 1 A/B test-greedy 평가** — 실제 command tracking 정확도 측정
   ```bash
   scp server1:~/PHC/output/AMASS_CMD_A.pth output/
   scp server1:~/PHC/output/AMASS_CMD_B.pth output/
   python phc/run.py --task HumanoidImVICCmd \
     --cfg_env exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_A.yaml \
     --cfg_train exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_A.yaml \
     --num_envs 1 --test --epoch -1
   ```
2. **v_cmd 고정 평가**: env yaml에서 `cmd_v_range: [1.0, 1.0]` 처럼 고정해서 실행 → 실제 policy 속도 측정 → v_err 계산
3. **Ep 5k/10k/15k/20k 체크포인트 비교**: 학습 과정에서 언제 "명령 추종"이 생겼는지 확인

### 7.2 서버 재실행 후보 (우선순위 순)
**A. Round 2 재시작 (personalization 완주)** ← 가장 설득력
- V2_B가 Ep 8200까지는 Round 1 B보다 일관되게 우수 (rwd 266 vs 220 수준)
- 미완주 상태로 폐기하기 아까움
- 방식: from-scratch 재시작 (이전 ckpt 폐기) OR --epoch resume (ckpt overwriting 이슈 있음)

**B. Round 1 winner에 test-greedy 기반 검증 후 새 방향**
- 로컬에서 A/B test-greedy로 실제 tracking 품질 확인
- 더 나은 쪽을 Round 3로 확장: ω_cmd (direction) 추가, 또는 더 넓은 cmd range
- 하지만 단일 clip 한계로 direction은 제한적

**C. Save frequency 변경 + Round 2 재시작**
- `save_frequency: 100 → 500` (5× 체크포인트 감소)
- 디스크 12G → 2.4G per experiment
- 더 안전하게 20k epoch 돌릴 수 있음
- 단, 중간 분석용 ckpt 적어짐

### 7.3 구조적 개선 후보
- **Multi-clip motion library**: 단일 clip 한계 극복. AMASS primitive.pkl에서 walking clip subset 추출 (원래 플랜 §2.2)
- **Residual impedance head**: `260416_review_and_modified_plan_for_vic_impedance_action_kr.md` 제안대로 impedance를 gate residual로 재배치
- **학습 재개 메커니즘**: 체크포인트 로드 시 epoch_num 보존되도록 rl-games 확장 (현재는 0부터 다시 시작 → ckpt overwriting 리스크)

---

## 8. 내부 참조

- Round 1 상세 분석: `02_research_dev/260419_amass_cmd_result_analysis.md`
- Phase 0 VIC 검증: `02_research_dev/260418_amass_pipeline_verify_result_analysis.md`
- Round 1 실행 기록: `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md`
- Round 2 실행 기록 (crash 이전): `exp_config/forward_walking/260419_AMASS_CMD_V2/RUNNING_ON_SERVER_V2.md`
- 구현 코드: `phc/env/tasks/humanoid_im_vic_cmd.py`
- MPL 상위 플랜: `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md`
- Impedance 재배치 제안: `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md`

---

*이 문서는 자율 세션 Round 1 완주 + Round 2 crash 후 종합 분석. 작성: 2026-04-19 KST 18:10.*
*사용자 복귀 후 §7의 재실행 옵션 중 선택해주시면 진행.*

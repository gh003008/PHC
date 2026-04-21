# Multi-clip Retrieval + Retime Hybrid — 학습 방향 재정립

**Date**: 2026-04-21
**Context**: Local Claude × User 디자인 토론 결과. 단일-clip 위에서의 v_cmd 설계 한계를 구조적으로 인식하고, Motion Plan Layer(G module)의 학습 방향을 multi-clip retrieval + retime hybrid로 전환.

**이 문서의 독자**: server-side Claude (gh003008 서버, EXOLAB GPU). 본 문서를 기준으로 현재 진행 중인 Baseline A/B를 완료시킨 뒤 multi-clip 파이프라인 구축으로 넘어가기 위한 기준 문서.

---

## 1. 한 줄 요약

> **"임의 v_cmd → 적절한 학습 동작"이라는 Motion Plan Layer의 최종 목표는 단일-clip에서 원리적으로 불가능. Multi-clip library + retrieval + retime hybrid 구조만이 해결책이며, 현재 진행 중인 Baseline A/B(260421_AMASS_CMD_B_EXTEND, 260421_AMASS_RETIME)는 "다음 단계의 recipe를 정하는 진단 실험"으로서만 의미를 가진다.**

---

## 2. 왜 단일-clip으로는 근본적으로 불가능한가

### 2.1 용어 정리 (다시)

- **v_motion**: reference motion clip의 natural pelvis forward 속도. Clip 내용으로 고정.
- **v_cmd**: task obs에 주입되는 명령 속도. 에피소드마다 달라짐.
- **base imitation rwd**: 시뮬 상태 vs reference 상태 일치도. 간접적으로 v_motion 추종.
- **cmd tracking rwd**: exp(−k·|v_pelvis − v_cmd|²). 직접적으로 v_cmd 추종.

### 2.2 Round 1 (v_cmd obs + reward blend)의 모순

단일 clip + 랜덤 v_cmd = 학습 정책이 내야 할 동작이 정의 불능.

- Clip은 하나의 v_motion만 가짐.
- v_cmd는 [0.8, 1.3]에서 독립 샘플.
- v_cmd ≠ v_motion 인 에피소드에서 base imitation과 cmd tracking이 서로 다른 방향을 가리킴.
- cmd_tracking_w는 두 reward의 가중치만 바꿀 뿐 충돌 자체는 유지.
- 실험 결과: CMD_A 실패율 35%, CMD_B 실패율 14%. 성공 시 품질은 VIC4 baseline(951) 초과 (best-ep 역산). 비용은 "실패 에피소드 비율"로 나타남.

### 2.3 Retime (Round 3 / Baseline B)의 한계

Teacher motion을 s·t로 retime해서 reward 충돌을 구조적으로 제거한 것은 맞음. 그러나:

- s ∈ [0.9, 1.1] 범위만 커버. v_motion의 ±10%.
- 그 밖의 속도는 생체역학적으로 무의미. 사람이 빨리 걸을 때는 time-stretch가 아니라 stride length·cadence·팔 진자까지 바뀜.
- Turning(ω_cmd) 불가. 단일 forward clip이라 회전 데이터 자체가 없음.
- Walk-stop-walk, 가속/감속 전이 등 상태 변화 불가.

**즉 Retime은 "임의 입력 처리 능력"을 주는 방법이 아니라, 단일 clip에서 reward 구조의 토대를 검증하는 중간 단계다.** 이 점을 명확히 하고 시작해야 함.

---

## 3. 올바른 방향 — Multi-clip Retrieval + Retime Hybrid

### 3.1 핵심 구조

```
Episode reset
  ├─ v_cmd ~ Sample(training distribution)
  ├─ retrieve clip* = argmin over library  |v_motion(clip) − v_cmd|
  ├─ compute scale s = v_cmd / v_motion(clip*)
  └─ teacher = retime(clip*, scale=s)       # v_ref_retimed = s · v_ref, time_retimed = start + s·(t − start)

Training
  └─ base imitation reward (vs retimed teacher)        # cmd_tracking_w = 0
      → teacher 자체가 v_cmd를 담음 → 구조적 충돌 없음

Testing
  └─ fixed v_cmd bins × {success rate, v_err, imitation rwd, failure-mode 분포}
```

### 3.2 왜 이 조합인가 — 대안 비교

| 접근 | 원리 | 장 | 단 |
|---|---|---|---|
| (A) Nearest clip only | v_cmd에 가장 가까운 clip을 teacher로, retime 없음 | 단순 | v_err = |v_cmd − v_clip*|가 강제로 남음 → reward 충돌 miniature 재발 |
| (B) **Retrieval + Retime (추천)** | nearest clip 선택 → s = v_cmd / v_clip로 retime | 각 clip 근처 ±k% 연속 커버. Clip 간격 ≤ 2k%면 전역 연속 | Retime bandwidth k 튜닝 필요 |
| (C) Latent 보간 | VAE/diffusion으로 clip 사이 샘플링 | 완전 연속 커버 | 현 단계 과잉, 학습 불안정 |
| (D) 고밀도 clip | 0.1 m/s 간격 고밀도 library | retime 불필요 | 데이터 수집·저장·indexing 비용 ↑ |

**(B)가 실질적 답**. 이유:

1. Retrieval이 커버하는 거친 격자 + retime이 커버하는 미세 보간의 **조합**이 자연스럽게 "finite clip으로 연속 속도 커버"를 달성함.
2. 각 clip 근처 ±k% 밴드가 retime의 안전 구간. k ≈ 10~15% 정도로 가정하면 clip 간격 20~30%로 설정 → 6~8개 clip이면 [0.6, 1.6] m/s 연속 커버.
3. MPL framework 문서의 "retrieval된 future reference window + style discriminator" 구조와 정확히 부합.
4. Baseline B(retime)의 결과가 직접 **k 파라미터 튜닝 근거**로 재활용됨.

### 3.3 왜 base imitation만으로 충분한가

"clip 속도 자체가 reward가 되어야"라는 직관이 정확히 이 구조에서 성립:

- Retrieved clip의 v_motion이 v_cmd와 완전 일치 (retime 후)
- Base imitation reward는 retimed teacher와 시뮬 상태의 일치도 측정
- 따라서 "imitation 잘 함 ⇔ v_cmd 잘 추종"이 구조적으로 동치가 됨
- 별도 cmd_tracking_w·cmd reward 필요 없음
- AMP discriminator도 retimed demo 분포로 맞춰야 함 (Baseline B와 동일 논리)

---

## 4. Acceptance Criterion — Testing Protocol

### 4.1 핵심 질문

> **훈련 속도 분포 내의 임의 v_cmd가 주어졌을 때, 정책이 (a) 그 속도를 잘 따라가면서 (b) 넘어지지 않는가?**

### 4.2 평가 배터리

각 fixed v_cmd bin마다 N=50~100 에피소드:

| 지표 | 정의 | 합격선(초안) |
|---|---|---|
| Success rate | steps ≥ 290 / total | ≥ 95% |
| v_err_mean | |mean v_pelvis − v_cmd| | ≤ 0.05 m/s |
| v_err_rms | RMS(v_pelvis − v_cmd) over episode | ≤ 0.10 m/s |
| Imitation rwd | 성공 에피소드 평균 | ≥ 900 (VIC4 baseline 951 대비) |

### 4.3 Bin 설계

- **훈련 범위 내 sweep**: 훈련 분포 범위를 N=5~10 bin으로 균등 분할. 예: 훈련 [0.6, 1.6] → bin {0.7, 0.9, 1.1, 1.3, 1.5}.
- **Clip 경계 근처 집중 샘플**: 두 clip 사이 중간점(retime 최대 구간)에서 성능 저하 여부 확인.
- **훈련 범위 바깥**: degradation curve 측정. v_cmd ∈ {0.4, 0.5, 1.7, 1.8} 등. 합격 요구는 없고 extrapolation 경계 확인만.

### 4.4 추가 테스트

- **Ramp test**: 에피소드 중간에 v_cmd가 smooth하게 변화. 전이 능력 확인.
- **Step test**: v_cmd가 중간에 급변. 응답 지연·overshoot 측정.
- **Failure-mode 분포**: 실패 에피소드들의 v_cmd가 clip 경계 근처인지, 훈련 bin edge인지 분석.

---

## 5. Training Reward 구조 — 권장

```yaml
# reward 구성 (Round 4 기준안)
base_imitation:   1.0   # retimed retrieved teacher와의 일치도 (PHC default)
survival:         (유지)
cmd_tracking:     0.0   # 제거. teacher 자체가 cmd 신호.
amp_style:        (유지, 단 demo pool도 retime 일치)

# curriculum
stage1: warm-up with CCF=0 (VIC Stage 1 그대로)
stage2: CCF 학습 on (Stage 2)
reward_curriculum_switch_epoch: 10000  # 유지
```

핵심 원칙: **구조적 충돌 가능성이 있는 reward 항은 아예 넣지 않는다**. Round 1의 실패는 충돌 reward를 "weight로 달래려 한" 데서 왔음.

---

## 6. Clip 간격 설계 — Baseline B 결과로 결정

### 6.1 현재 Baseline B의 역할 재정의

Baseline B(260421_AMASS_RETIME)의 s ∈ [0.9, 1.1] 학습 결과는 **"단일 clip을 ±10% retime했을 때 정책이 robust한가"를 측정**하는 실험.

- **성공 (≥95% success at v_cmd = 0.9~1.1 · v_nat)** → ±10% retime bandwidth가 안전. Multi-clip에서 clip 간격 **≤ 20%** (예: 0.6, 0.72, 0.86, 1.03, 1.24, 1.48 — 등비수열)
- **부분 성공 (<95%)** → retime bandwidth를 좁혀야 함. Clip 간격 축소 or retime 범위 축소로 대응.
- **실패 (실질적 개선 없음)** → retime 접근 자체 재검토. 이 경우 대안(C) 또는 (D)로 전환.

### 6.2 Clip 간격 초안

- **등비수열**: 0.60, 0.72, 0.86, 1.03, 1.24, 1.48 m/s (ratio 1.20)
- 각 clip이 ±10% retime 커버 → 전체 [0.54, 1.63] m/s 연속 커버
- 6개 clip이면 충분. 필요 시 촘촘하게 조정.

### 6.3 속도 이외 축

- **Cadence**: v와 상관관계 있지만 독립적 변이 있음. 2차 retrieval 축으로 고려.
- **Contact phase**: 초기 stage에선 무시, 나중에 contact-conditioned retrieval로 확장.
- **Turning (ω_cmd)**: forward walking 단계에선 ω_cmd = 0 고정. Turning clip 추가 시점은 별도 단계.

---

## 7. 구현 체크리스트

### 7.1 데이터 파이프라인

- [ ] AMASS walking subset 필터링
  - Label 기반 (KIT/CMU metadata: "walk"/"walking" 서브스트링)
  - 또는 rule 기반 (pelvis height 일정, 주기적 발 접촉, |v_pelvis| > 0.3 m/s, 평균 v가 특정 범위)
- [ ] Per-clip annotation
  - `v_motion_mean = mean(pelvis forward velocity)` (주기 내 평균)
  - `v_motion_cyclic = mean per gait cycle` (cycle당 평균의 평균)
  - `cadence = steps / duration`
  - `stride_length = v_motion_mean / (cadence / 2)`
  - `direction = yaw rate 평균` (forward-only subset에서는 ≈ 0)
- [ ] Clip 저장 포맷 (통합 pkl 또는 개별 pkl + 인덱스)
- [ ] Retrieval 인덱스 (v_motion → clip_id 매핑. KDTree 1D 충분)

### 7.2 코드 변경

- [ ] `phc/utils/motion_lib_smpl.py` (또는 `motion_lib_base.py`): multi-clip retrieval 경로
  - `_sample_motion(v_cmd)` → nearest clip 반환
  - 기존 `cycle_motion=True` 버그 재확인 (이미 알려진 2nd cycle velocity 불연속)
- [ ] `HumanoidImVICCmdRetime` 확장 or 신규 task class (`HumanoidImVICCmdMulti`)
  - `_get_state_from_motionlib_cache`가 per-env clip_id도 함께 처리하도록
  - retime scale = v_cmd / v_motion(clip_id) per env
- [ ] AMP demo pool: 여러 clip에서 샘플링, per-demo scale도 동일 범위에서 샘플링
- [ ] Env yaml: `motion_file` → `motion_library_dir` + `clip_index_file`

### 7.3 AMP demo pool

- 기존 Baseline B에선 단일 clip의 demo를 per-demo s'로 scale.
- Multi-clip에선: demo 샘플링 시 clip 먼저 선택 (uniform 또는 train 분포 proportional) → 그 clip 내에서 demo window 뽑기 → per-demo s'로 scale.
- Discriminator positive 분포 ≈ rollout 분포 유지.

### 7.4 평가 코드

- [ ] Fixed v_cmd 배터리 스크립트 (sbatch or local): 각 bin N 에피소드 × 지표 집계
- [ ] Ramp/step test 모드 (env yaml에 `cmd_schedule: [(t0, v0), (t1, v1), ...]` 형태)
- [ ] Failure-mode 로깅: 실패 에피소드의 v_cmd, 실패 시점의 clip_id, phase 기록

---

## 8. Baseline A / B 결과의 재활용 방식

두 트랙은 이미 돌고 있으므로 중단 없이 완료. 결과는 multi-clip 설계의 **recipe 확정**에 사용:

| 시나리오 | Baseline A (CMD_B→30k) | Baseline B (Retime R1) | Multi-clip 설계 함의 |
|---|---|---|---|
| A↑ B↑ | 수렴 후 CMD_B도 개선 | Retime도 성공 | Retime-only reward (w=0) 구조를 multi-clip에 이식. Retime bandwidth ±10% 확정. |
| A↑ B↓ | 연장만으로 충분 | Retime 문제 | Multi-clip도 reward blend로. 단 clip retrieval로 충돌 최소화. |
| A↓ B↑ | 구조적 한계 재확인 | Retime이 옳은 길 | 곧바로 multi-clip retime으로 이동. bandwidth 결과 그대로 활용. |
| A↓ B↓ | 양쪽 다 부족 | 근본 원인 재검토 | Retime 구현 버그? AMP demo 분포? reward 항 자체? — 디버그 우선 |

어떤 시나리오든 multi-clip으로 가야 한다는 결론은 유지. A/B 결과는 "어떻게"를 결정함.

---

## 9. 주의 엣지

### 9.1 Clip boundary 연속성

- 에피소드 길이 > clip 길이일 때 cycle 처리 문제 (기존 `cycle_motion=True`의 2nd cycle velocity 불연속).
- 대책: 긴 에피소드는 clip을 pad 또는 cycle 없이 "도중에 terminate" 시점 전까지만.

### 9.2 Retime 후 clip 길이 변화

- s < 1이면 clip이 길어짐(느려짐). s > 1이면 짧아짐.
- 에피소드 길이(max_episode_steps)는 고정이므로 s에 따라 "clip 몇 번 반복"이 달라짐. AMP demo sampling에도 동일 영향.
- 대책: episode_steps를 s-independent하게 두되, s=1 기준의 frame 수가 아니라 "retimed duration"으로 계산.

### 9.3 AMP demo 분포

- Baseline B에서 per-demo s' 샘플해 velocity field만 스케일. Multi-clip에선 (clip_id, s')가 모두 샘플링 대상.
- Discriminator positive 분포가 rollout 분포와 밀리면 learning signal 붕괴.

### 9.4 Retrieval noise

- v_cmd가 두 clip 사이 정확히 중간일 때 → 어느 쪽 retrieve해도 s가 극단(±bandwidth edge). 학습 시 이 영역이 많으면 training instability.
- 대책: 훈련 v_cmd 분포를 clip center에 약간 치우치게 설계 or soft retrieval (top-2 clip 간 weighted).

### 9.5 데이터 편향

- AMASS walking subset은 healthy young adult에 편향. Personalization(B-1)이 다시 의미를 갖는 시점은 이 편향이 모델링 병목이 될 때.

---

## 10. 제안 타임라인

| 주 | 항목 |
|---|---|
| W0 (이번 주) | Baseline A/B 완료 대기 (16h × 2). 결과 나오면 fixed-v 평가 배터리 실행. |
| W0 병렬 | AMASS walking subset 파이프라인 초안: filter → annotate → indexing. 실제 학습에 안 쓰더라도 "6~10 clip subset"부터. |
| W1 | `HumanoidImVICCmdMulti` 구현, motion_lib multi-clip retrieval 경로, AMP demo pool 변경. |
| W1 후반 | Multi-clip 첫 학습 실행 (20k epochs × 2 GPU: A = narrow range (0.7~1.3), B = wider (0.5~1.5)). |
| W2 | Fixed-v 배터리 평가 + failure-mode 분석. Clip 간격/bandwidth 튜닝. |
| W3 | ω_cmd(turning) clip 추가 검토. 또는 cadence 축 추가. |

---

## 11. Server-side Claude 액션 아이템

**(1) 즉시 — 진행 중 작업 보호**
- Baseline A (AMASS_CMD_B_EXTEND, idx0), Baseline B (AMASS_RETIME, idx1) 모니터링. 중단 없이 완료.
- 완료 시 `output/AMASS_CMD_B_EXTEND.pth`, `output/AMASS_RETIME_R1.pth` ckpt 확인.

**(2) 평가 배터리 준비 (A/B 완료 전 선작업)**
- 평가 스크립트 `scripts/eval_fixed_v.py` 작성: `cmd_v_range: [v, v]` 고정 + N 에피소드 × 5~10 bin.
- 출력: CSV with columns `[v_cmd, ep_idx, steps, mean_v_pelvis, v_err_rms, imit_rwd, success]`.
- 분석 노트북: bin별 success rate, v_err curve, failure-mode 플롯.

**(3) AMASS walking subset 파이프라인 착수 (A/B 완료 전부터 병렬)**
- `scripts/data/extract_amass_walking.py`:
  - AMASS primitive.pkl 로드
  - Label/rule 기반 walking 필터
  - Per-clip annotation (v_mean, v_cyclic, cadence, stride, duration)
  - 출력: `sample_data/amass_walking_subset_v0.pkl` + `sample_data/amass_walking_index_v0.json`
- Sanity check: 상위 20개 clip의 v 분포 플롯. 너무 편향되어 있으면 additional 필터.

**(4) Multi-clip 학습 인프라 (W1)**
- `phc/env/tasks/humanoid_im_vic_cmd_multi.py`:
  - `HumanoidImVICCmdRetime` 상속
  - `_resample_clip_and_scale(env_ids)`: v_cmd 샘플 → 가장 가까운 clip retrieve → s 계산 → per-env 저장
  - `_get_state_from_motionlib_cache` 오버라이드 (per-env clip_id 분기)
- `phc/utils/motion_lib_smpl.py` 또는 `motion_lib_base.py`: multi-clip 경로 검증. 기존 `cycle_motion` 버그 회피.
- `parse_task.py`: `HumanoidImVICCmdMulti` 등록.

**(5) 첫 multi-clip 실험 (W1 후반)**
- Narrow range A: v_cmd ∈ [0.7, 1.3], 6 clip, ±10% retime bandwidth
- Wider range B: v_cmd ∈ [0.5, 1.5], 8 clip, ±10% retime bandwidth
- max_epochs 20000, save_frequency 2500
- 평가는 고정 v bin 배터리로 (3) 스크립트 재활용.

---

## 12. 참고 문서

### 내부
- `02_research_dev/260415_MPL_Framework_draft_v11.docx` — MPL 전체 프레임워크 원 설계
- `02_research_dev/260420_recent_6_experiments_summary.md` — Round 1/2/0 6개 실험 종합
- `02_research_dev/260420_velocity_command_design_analysis.md` — v_cmd 충돌 진단
- `02_research_dev/260420_velocity_command_recommendation.md` — Retime 권고안
- `02_research_dev/260421_velocity_retime_execution_plan.md` — Baseline A/B 실행 계획
- `02_research_dev/260421_motion_plan_layer_method_progress.pptx` — 회의용 현재 요약

### 구현 코드
- `phc/env/tasks/humanoid_im_vic.py` — VIC 기반 (Phase 0)
- `phc/env/tasks/humanoid_im_vic_cmd.py` — Round 1 v_cmd obs (충돌 확인된 설계)
- `phc/env/tasks/humanoid_im_vic_cmd_retime.py` — Round 3 Retime (현재 Baseline B, 단일 clip)
- `phc/utils/motion_lib_smpl.py`, `phc/utils/motion_lib_base.py` — multi-clip 확장 대상

---

*Authored 2026-04-21 local session. Server-side Claude에게 전달 목적 — 다음 학습 방향 확정.*

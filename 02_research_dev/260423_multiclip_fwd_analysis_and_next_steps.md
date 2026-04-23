# Multi-Clip FWD 결과 분석 및 다음 단계 (2026-04-23)

본 문서는 `260423_multiclip_fwd_results.md` (서버 세션 작성) 의 결과를
코드 레벨에서 검증하고, 제안된 next step 들을 우선순위별로 실행 가능한
형태로 정리한 것이다.

---

## 1. 결과 요약 (pull 한 내용 기준)

### 테스트-그리디 (test-greedy) 최종 수치

| Variant | av_reward | av_steps | 성공률 (≥290 steps) |
|---|---|---|---|
| FWD_NR (no retime, 22 clips) | 215.37 | 65.51 | ~0 % |
| FWD_RT (retime, 22 clips)    | 211.78 | 64.29 | ~0 % |
| **참고: VIC4 single-clip**       | **951.2** | **299** | **100 %** |
| **참고: CMD_B single-clip**      | **440.4** | **279** | **86.1 %** |

### 한 줄 해석
- NR ≈ RT. retime 레이어는 dense 22-clip 라이브러리에서는 사실상 no-op
  (scale s 가 거의 1.0 근처에 머물러 NR 과 동일한 분포로 수렴).
- 10 s 에피소드 중 평균 **2.1 s 만에 쓰러짐**. Single-clip 86~100 % 대비
  multi-clip 0 % 는 "단지 더 어려워서"로 설명 불가 → 구조적 결함.

---

## 2. 결과 md 의 가설 중 코드 레벨로 검증된 것

### (A) Phase-obs 테이블이 multi-clip 을 따라가지 못함 → **확인됨**

`phc/env/tasks/humanoid_im_vic.py:995-1073` `_precompute_gait_phase()`
의 핵심 라인:

```python
motion_id = 0
motion_length = self._motion_lib._motion_lengths[motion_id].item()
# ... 아래 전부 motion_id=0 에 대한 샘플링 ...
self._gait_phase_table = gait_phase      # [1000]
self._gait_motion_length = motion_length # scalar (clip 0 의 길이)
```

그리고 obs 생성에서 (`humanoid_im_vic.py:973-989`):

```python
frame_idx = (time_in_clip / self._gait_motion_length * self._gait_phase_num_samples).long()
gait_phase = self._gait_phase_table[frame_idx]
```

→ **22 개 clip 이 올라가도 phase 테이블은 index 0 clip (최저속 0.21 m/s 포즈) 하나로
계산됨.** index 0 의 stride period 로 모든 clip 의 time_in_clip 을
0~1 정규화하므로, 빠른 clip (0.99 m/s) 은 실제 보행 주기와
완전히 어긋난 sin/cos 값이 policy 에 들어간다.

`vic_phase_obs: True` 가 두 multiclip config 모두에서 켜져 있음
(`env_im_walk_vic_multiclip_fwd_NR.yaml:54`, `..._RT.yaml:54`).

**결론**: 이 버그는 관찰 가능하고, 22 clip 의 속도 range 가 넓을수록
policy 가 받는 observation noise 는 커진다. 결과 md 의 "관찰 잡음이
clip 다양성에 비례해서 커진다" 는 기술과 부합.

### (B) cycle_motion 불연속 → **설정 확인됨**

두 config 모두 `cycle_motion: True` (`env_im_walk_vic_multiclip_fwd_NR.yaml:31`,
`..._RT.yaml:31`). 에피소드 300 steps × dt 0.033 = **10 s**,
클립 길이는 대체로 3~5 s → 에피소드당 1~2 회 clip-wrap 이 발생.

코드: `phc/env/tasks/humanoid_im.py:1122-1137` 에서 cycle_motion 시
참조 motion 이 clip 시작으로 순간 점프. 22 clip × 서로 다른 wrap point
× 서로 다른 start-pose → clip 간 다양성이 클수록 치명적.

### (C) Phase-CCF 분석기의 4-vs-8 그룹 broadcasting 에러 → **확인됨**

Config 에 `vic_ccf_num_groups: 4` (`env_im_walk_vic_multiclip_fwd_NR.yaml:52`).
기존 `analyze_phase_ccf.py` 및 `im_amp_players.py` 의 phase_ccf_log 저장부는
8-group 기준으로 작성되어 있어 shape (4,) → (8,) broadcasting 에러 발생.
이건 post-hoc 분석 파이프라인 문제이지 학습 성능 결함과는 별개.

---

## 3. 재검토한 가설 우선순위

결과 md 는 next step 으로 4 개를 제안했는데, 위 코드 검증을 반영하면
**(3) phase-obs 수정이 (1) 소규모 재학습보다 먼저 시도할 값어치**가
있다. 이유:
- phase-obs 버그는 확정된 사실이고 local 에서 수정 가능한 수 십 줄 코드.
- (1) "4 clip 만으로 재학습" 은 20k epoch × 2 GPU 시간을 다시 태우는
  것 → cheap 해도 하루 일감.
- phase-obs 를 먼저 고쳐 놓으면 (1) 재학습도 고친 파이프라인으로 돌릴 수
  있어, 변수 하나만 (clip 다양성) 바꾼 clean A/B 가 됨.

---

## 4. 다음 단계 (실행 순)

### Step 1 — phase-obs 를 clip-per-env 로 수정 (local, 코드만)

목표: `_gait_phase_table` 을 "로드된 모든 clip 에 대해 precompute" →
런타임에 env 별 현재 motion_id 로 lookup.

개략 변경:

```python
# _precompute_gait_phase
num_clips = self._motion_lib._motion_lengths.shape[0]
self._gait_phase_table = torch.zeros(num_clips, num_samples, device=self.device)
self._gait_motion_lengths = self._motion_lib._motion_lengths.clone()  # [num_clips]
for cid in range(num_clips):
    # 기존 heel-strike 로직을 motion_id=cid 로 반복
    self._gait_phase_table[cid] = ...
```

```python
# _compute_gait_phase_obs (또는 해당 obs 섹션)
motion_ids = self._sampled_motion_ids           # [num_envs]
mlen       = self._gait_motion_lengths[motion_ids]  # [num_envs]
fidx       = (time_in_clip / mlen * self._gait_phase_num_samples).long().clamp(0, N-1)
gait_phase = self._gait_phase_table[motion_ids, fidx]
```

복잡도: 본질적으로 loop-over-clips 로 번역. Heel strike 가 검출 안 되는
clip 은 기존 fallback (clip-level linear phase) 유지.

검증: 2 clip 이상 로드 후 startup 로그에서 per-clip stride period 가
다르게 출력되면 OK. num_envs=1 no-train dry-run 으로 phase 관찰
(obs dump 한 번) 하여 clip 스위치 시 phase 가 실제로 바뀌는지 확인.

### Step 2 — 4-clip 소규모 재학습 (서버, NR only)

목표: "phase-obs 수정 × clip 다양성 축소" 조합이 single-clip 성공률을
복원하는지 확인.

- 소스: `amass_isaac_walking_primitive_fwd_only.json` 에서
  `WalkingStraightForwards*` 4 clip (|v_x| ≈ 0.23~0.53).
- variant: NR 만 (RT 는 현 결과로 NR ≈ RT 확정).
- epochs 20 000, save_frequency 2 500, num_envs 512.
- 성공 기준: test-greedy av_steps ≥ 200 (success rate ≥ 60 %) 이면
  "clip-diversity × phase-obs" 가 주원인으로 확정.

config 배치: `exp_config/forward_walking/260424_AMASS_MULTICLIP_FWD4_PHASEFIX/`
에 4-clip 전용 json + yaml + sbatch 생성.

### Step 3 — 기존 canonical ckpt 의 per-clip 실패모드 감사 (서버, cheap)

목표: cycle_motion wrap 이 실패의 트리거인지 확정.

- 현재 `output/AMASS_MULTICLIP_FWD_NR.pth` 를 재사용.
- test-greedy 를 v_cmd 를 22 clip 각각의 |v_x| 로 **고정**하고 각 20
  회, 종료 step 을 clip duration 과 비교.
- wrap 직후 (time_in_clip ≈ clip_len) 근방에서 terminate 가 집중되면
  cycle_motion 이 주범. 그렇지 않으면 phase-obs 가 더 지배적.
- 구현: `phc/run.py --test` 래퍼 쉘 스크립트에 v_cmd 강제 세팅.
  (기존 `_sample_ref_state` 의 v_cmd sampling 만 우회.)

Step 1 과 병렬 가능. Step 1 이 local, Step 3 이 서버 GPU 이므로 동시 진행해도 블로킹 없음.

### Step 4 — cycle_motion 제거 또는 long-clip concat 재학습

Step 3 가 wrap-dominant 라고 말하면:
- 옵션 A: `cycle_motion: False` + 더 짧은 episode_length (예: 180 steps
  = 6 s). 짧은 clip 을 끝까지 쓰고 재시작. 단순하지만 eps_len 절반으로 감.
- 옵션 B: `motion_lib_smpl` 레벨에서 같은 dir_class 의 clip 을
  랜덤 concat 해 10 s 이상 합성 시퀀스 만들기. 구현량 많음.

어느 쪽이든 Step 2 가 성공 (≥ 60 %) 한 뒤에만 시도. 실패면 step 2 가
"phase-fix + small clipset 조차 부족" 을 의미하므로 원인 재탐색 우선.

### Step 5 — AMP discriminator 분포 점검 (low priority)

결과 md §6 tertiary suspect. Reward 세부 항목별 로그 분리 (task vs
disc vs power) 를 wandb panel 에 추가하고 NR_4clip 학습 중 disc
accuracy 를 시계열로 관찰. Step 1+2 로 문제가 회복되면 건드릴 필요 없음.

---

## 5. 구체적 결정 사항 및 변수 정리

- **다음 커밋 단위**: phase-obs multi-clip 화 (Step 1) 를 별도 브랜치 또는
  현 `Jimin` 브랜치의 독립 커밋으로. 기존 single-clip 경로는 하위 호환
  유지 (num_clips == 1 이면 기존 경로와 동치).
- **서버 GPU 예산**: Step 2 는 20 k × 512 env 1 GPU, 2 variant 아닌 1
  variant 이므로 이전 multiclip 런의 1/2 (약 8~10 h).
- **결과 문서화**: Step 2 완료 후 `02_research_dev/YYMMDD_multiclip_fwd4_phasefix_results.md` 작성 (CLAUDE.md 규칙에 따라 한글).

## 6. 짧은 결론

- multi-clip 실패의 1차 원인은 **phase-obs 테이블이 clip 0 에 하드코딩되어 있는 버그** 일 가능성이 매우 높다 (코드 검증됨).
- 2차는 cycle_motion × clip-wrap 이 짧은 clip 여러 개에서 증폭된 효과.
- 먼저 (Step 1) phase-obs 수정 → (Step 2) 4-clip 재학습 → (Step 3) wrap 감사 순으로 진행하면, 2 일 내에 "성공 vs 계속 실패" 의 갈림길이 명확해진다.

---

*Authored 2026-04-23. 결과 md 는 서버 세션이 작성, 본 문서는 local 에서
코드 검증 추가 후 실행 plan 으로 재정리.*

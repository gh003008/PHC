# Multi-clip 방향 혼입 문제 — 분석과 즉시 조치

**Date**: 2026-04-21 (local session, server Claude 앞으로 전달)
**TL;DR**: `amass_isaac_walking_primitive.pkl` (50 clips)은 "속도별 50개"가 아니라 직선 전진·뒷걸음·원 궤적·다양한 속도 라벨이 섞여있다. 현재 `HumanoidImVICCmdMultiClip`의 `argmin |v_nat − v_cmd|` retrieval은 방향을 전혀 고려하지 않고 `v_mean_mid`도 magnitude만 저장돼 있어, forward v_cmd에 대해 뒷걸음 clip이나 turning clip을 teacher로 집어올 수 있다. 지금 돌고 있는 NR/RT 학습 결과의 해석을 오염시킬 가능성이 있으니, 속도 검증은 forward-straight 서브셋 위에서 먼저 한 다음 방향·속도를 분리해서 확장하는 쪽이 맞다.

---

## 1. 무엇을 확인했나

### 1.1 50 clip의 동작 다양성 (filename 기반)

| 분류 | 개수 | 대표 패턴 |
|---|---|---|
| Straight forward | 4 | `WalkingStraightForwards0x` |
| Straight backward | 9 | `WalkingStraightBackwards0x` |
| Clockwise circle (우회전) | 10 | `WalkInClockwiseCircle0x` |
| Counter-CW circle (좌회전) | 4 | `WalkInCounterClockwiseCircle0x` |
| 속도 라벨 (slow/medium/fast) | 6+8+4 | `walking_slow/medium/fast0x`, 방향 미지 |
| "run" (빠른 걸음) | 7 | `walking_run0x` |
| Handrail | 1 | `walk_with_handrail_right0x` |

→ 속도 축만이 아니라 **방향 축 (전·후·좌회전·우회전)** 이 교차돼 있음.

### 1.2 기존 메타(`amass_isaac_walking_primitive_v_nat.json`)의 한계

- `v_mean_mid`은 부호 없는 **magnitude**. 뒷걸음 clip도 양수.
    - 예: `KIT_7_WalkingStraightBackwards04` → `v_mean_mid = 0.403`
- `is_backward_walking` 플래그는 **자동 휴리스틱으로 보이며 신뢰성 낮음**:
    - `KIT_11_WalkingStraightForwards05` → `is_backward_walking = True` (오태깅)
    - `KIT_7_WalkingStraightBackwards04` → `is_backward_walking = False` (오태깅)
- 따라서 현재 파일만으로는 forward vs backward를 구분할 수 없다.

### 1.3 현재 학습 구조의 실제 행동

`phc/env/tasks/humanoid_im_vic_cmd_multiclip.py` (line 51–55):
```python
self._clip_v_nat = torch.tensor(
    [v_nat_dict[k]["v_mean_mid"] for k in motion_keys],
    ...
)
# retrieval: chosen = argmin(|v_nat - v_cmd|)  across ALL 50 clips
```

- v_cmd = +0.55 (forward 의도)를 샘플해도 뒷걸음 clip(v_mean_mid = 0.53)을 teacher로 집어올 수 있다.
- 원 궤적 clip도 v_mean_mid는 양수이고 측면 속도·yaw가 있는데 retrieval은 speed magnitude만 비교.
- obs의 `v_cmd_x = s * v_nat(clip)`은 양수(forward)로 기록되는데, teacher는 뒷걸음 또는 회전을 함.
- Imitation reward는 "teacher 따라가기"니까 **실제 정책은 뒷걸음/회전을 학습**하면서도 obs 상에선 forward v_cmd를 받은 것처럼 보임.

### 1.4 현재 진행 중인 학습에 주는 영향

- NR (idx0, job 3853) / RT (idx1, job 3854) 두 variant 모두 영향 받음.
- 성공률/v_err이 낮게 나올 때 원인이 "retrieval/방향 혼입"인지 "retime bandwidth 부족"인지 구분 불가.
- AMP positive 분포에도 뒷걸음·회전이 섞여들어감 → discriminator가 "walking"의 의미를 "방향 혼합 걷기"로 배움.

---

## 2. 사용자의 정리

> 우선은 속도로만 해봐야 할 것 같은데, 동작이 달라지면 그건 또 다른 이야기가 될 것 같아.

→ **Multi-clip 가설 (속도별 clip + retrieval로 임의 v_cmd 커버)의 1차 검증은 forward-straight 서브셋 위에서만.** 방향·회전 축은 분리해서 이후 단계로 미룸.

---

## 3. 즉시 조치

### 3.1 분석 스크립트 (신규 커밋, 서버 실행)

`scripts/data/compute_walking_direction_metadata.py` 추가. 기능:

1. `amass_isaac_walking_primitive.pkl`을 로드.
2. Clip마다 pelvis 위치의 signed `(v_x_mean_mid, v_y_mean_mid)`와 root yaw의 `yaw_rate_mean_mid`를 중간 구간에서 계산. 중간 구간 = 앞뒤 0.5초 제외.
3. 분류:
    - `forward_straight`: `v_x > +0.15 m/s`, `|v_y| < 0.20`, `|yaw_rate| < 0.30 rad/s`
    - `backward_straight`: `v_x < -0.15`, `|v_y| < 0.20`, `|yaw_rate| < 0.30`
    - `turning`: `|yaw_rate| ≥ 0.30 rad/s`
    - `other`: 그 외 (대각선 이동 등)
4. 두 개 JSON 저장:
    - `amass_isaac_walking_primitive_dirmeta.json` — 전체 50 clip의 부호 있는 메타
    - `amass_isaac_walking_primitive_fwd_only.json` — forward_straight 서브셋만 (오름차순 v_x 정렬)

임계값(V_FWD_MIN, V_LATERAL_MAX, YAW_RATE_MAX)은 스크립트 상단에 상수로 두고, 첫 실행 후 분포 보고 조정 가능하게 함.

### 3.2 서버 실행 절차

```bash
cd /home/server1/PHC   # 또는 서버 경로
conda activate phc
python scripts/data/compute_walking_direction_metadata.py
```

출력 JSON 두 개가 `sample_data/` 아래에 생성됨. 각 clip의 v_x / v_y / yaw_rate / dir_class가 print로도 표시됨.

### 3.3 학습 config 전환

`phc/env/tasks/humanoid_im_vic_cmd_multiclip.py`와 `env_im_walk_vic_multiclip_*.yaml`을 수정:

- `multiclip_v_nat_path`를 `sample_data/amass_isaac_walking_primitive_fwd_only.json`으로 교체.
- 해당 json의 entry 키만 motion_lib에서 로드하도록 `motion_file` 경로 로직 조정.
   - 가장 깨끗한 방법: `amass_isaac_walking_primitive_fwd_only.pkl`도 별도로 저장하고 `motion_file`을 그쪽으로 지정.
   - 빠른 방법: 기존 pkl 유지 + task class의 `_clip_v_nat` 로드 시 fwd_only json 키만 필터링하고, retrieval이 그 subset에서만 argmin.
- `multiclip_v_cmd_range`는 subset의 v_x 분포에 맞게 조정 (예: subset의 10~90 퍼센타일).
- 코드 내 `v_mean_mid` → `v_x_mean_mid` 필드 참조로 변경 (부호 있는 forward speed).

### 3.4 현재 돌고 있는 NR/RT는?

두 옵션:

**(a) 중단 + 재시작 (권장)**: fwd_only subset으로 NR/RT 재실행. Checkpoint는 버림. Fresh 20k × 2 × 16h. 결과 해석이 깨끗함.

**(b) 완주시키고 참고용**: 지금 것은 "방향 혼합 multi-clip의 가장 나이브한 설정"의 예시로 기록. 이후 fwd_only 결과와 비교하면 "방향 혼입이 성공률을 얼마나 깎는가"를 정량화 가능.

**결정은 서버 Claude가 디스크·시간 여유 보고 선택.** 문서·분석 맥락에선 (a)가 깨끗하지만, 이미 돌고 있는 것을 버리는 비용이 아깝다면 (b)도 의미 있음.

---

## 4. 이후 단계 (분리 연구 축)

forward-straight-only에서 multi-clip 구조가 정상 작동하는 걸 확인한 뒤 별도 단계로:

1. **Direction 축**: `v_cmd_x` + `v_cmd_y` 또는 yaw rate `ω_cmd`를 obs에 추가, retrieval을 2~3차원 거리로 확장.
    - 즉시 사용할 수 있는 후보: `backward_straight` clip 9개 → forward + backward 쌍방향 걷기
    - 그 다음: `turning` clip 14개 → turning 포함
2. **"다양한 동작 = 다른 이야기"의 경계 식별**: 뒷걸음과 회전은 같은 "Walking" 카테고리지만 의미 상 추가 입력(방향)이 필요한 축. 학습 난이도도 다름 — 각각 분리 실험으로 다루는 게 맞다.
3. MPL framework 문서 §4 retrieval 설계와 정합 — `u ≡ (pelvis 선속도 + yaw rate + next step placement)` 좁은 벡터 정의로 확장하는 흐름이 자연스럽다.

---

## 5. 첨부

- `scripts/data/compute_walking_direction_metadata.py` — 본 커밋에 포함.
- 임계값은 서버 실행 후 출력 분포 보고 조정할 수 있음. 대안 임계값:
    - 매우 엄격: `V_FWD_MIN = 0.30, YAW_RATE_MAX = 0.15` (= 거의 순수 forward straight 만 유지)
    - 느슨: `V_FWD_MIN = 0.10, YAW_RATE_MAX = 0.50` (subset 크기 최대화)

---

*Authored 2026-04-21 local session. 서버 Claude는 (1) 스크립트 실행 → 서브셋 결정, (2) 현재 NR/RT를 (a)/(b) 중 선택, (3) `humanoid_im_vic_cmd_multiclip.py` 필드명·서브셋 참조 업데이트 후 재실행을 권함.*

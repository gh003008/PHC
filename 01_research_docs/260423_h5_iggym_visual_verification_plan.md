# H5 보행 데이터 IsaacGym 시각 검증 계획 (Step A)

작성일: 2026-04-23
작성자: Jimin
환경: Linux (server1 또는 로컬 Isaac Gym 환경)

---

## 1. 배경

현재 VIC 학습은 AMASS `amass_isaac_walking_primitive.pkl`로 진행 중이나, 궁극적으로는 자체 취득한 H5 mocap 데이터(Vicon Plug-in Gait 기반)로 학습을 돌리는 것이 목표다. 그 전에 H5 데이터가 "실제 사람 보행"처럼 동작하는지 검증이 필요하다.

지난 점검에서 H5→SMPL 변환 후 모션이 어색해 보였으나, 구체적으로 뭐가 틀렸는지 기록이 남지 않은 상태. 이번에는 **체크리스트 기반 시각 검증**으로 재현 가능한 판정을 남긴다.

## 2. 전체 검증 로드맵 (H5 데이터 사용 승인 조건)

- **Step A (본 문서)**: IsaacGym에서 H5-유래 SMPL 모션을 눈으로 재생 → "사람이 걷는다"로 보이는지 질적 판정
- **Step B (후속)**: AMASS 보행 클립 대비 관절각/속도/주기 정량 비교
- **Step C (최종)**: H5 pkl로 VIC 학습 돌려서 reward/steps 수렴이 AMASS 기준과 동등한지 확인

Step A에서 명백한 관절 매핑 이슈가 발견되면 `h5_to_motion_lib.py`의 변환 로직(특히 `VICON_TO_SMPL_MAP`과 sign/axis convention)을 수정하고 Step A를 재실행.

## 3. Step A 목표

H5에서 변환된 SMPL pkl을 IsaacGym에서 단독 재생(reference motion only, 학습 policy 없음)하고, 정의된 체크리스트를 채워 각 항목을 OK / 이상 / 모호로 판정한다.

## 4. 접근 방식

**선택안**: 기존 `phc/run.py --task HumanoidIm --test`의 reference motion 재생 경로 재사용.

- 이유: PHC에 이미 motion library 로딩 + IsaacGym 렌더 경로가 구현되어 있음. 새 스크립트 작성 비용 없음.
- `motion_file`을 H5 변환 pkl로 지정하고 `num_envs=1`, `--test` 플래그로 렌더.
- 학습된 policy가 없어도 reference motion 자체는 재생되도록 PHC 내부 경로 확인 필요 (필요 시 기존 AMASS motion replay 커맨드를 기반으로 미세 조정).

**대안 (fallback)**: 기존 `scripts/vis/vis_h5_motion_mj.py`의 MuJoCo 대응판이 있으므로, IsaacGym 경로에 막히면 MuJoCo로 대체. 단 최종 실험 환경이 IsaacGym이므로 IsaacGym에서의 검증을 우선한다.

## 5. 실행 순서 (Linux 환경)

### 5.1 원본 H5 sanity check

```bash
conda activate phc
python scripts/data/h5_explorer.py \
  --h5 data/combined_data_from_csv.h5 \
  --summary
```

확인 항목:
- trial 수와 각 trial의 duration(s)
- NaN/Inf 없음
- 관절각 단위가 degree이고 범위가 생체역학적으로 합리적 (hip flexion ~-20~+30°, knee flexion 0~+60°, ankle ~-20~+20° 등)
- GRF vertical이 체중 수준(수백~1000N)에서 heel strike 피크 패턴
- CoM 궤적 단위(mm)와 높이 변화 주기
- Treadmill speed가 의도한 속도와 일치

### 5.2 대표 trial 1개 pkl 변환

```bash
python scripts/data/h5_to_motion_lib.py \
  --h5 data/combined_data_from_csv.h5 \
  --output sample_data/h5_walk_S001_lv0_trial01.pkl \
  --subjects S001 \
  --tasks level_100mps \
  --assist_level lv0 \
  --fps_out 30
```

(subject/task/trial 선택은 환경에서 실제 경로 확인 후 조정)

### 5.3 IsaacGym 재생

기본 커맨드 템플릿:
```bash
python phc/run.py \
  --task HumanoidIm \
  --cfg_env phc/data/cfg/env/env_im_walk.yaml \
  --cfg_train phc/data/cfg/learning/im_walk.yaml \
  --motion_file sample_data/h5_walk_S001_lv0_trial01.pkl \
  --num_envs 1 --test --epoch -1
```

주의:
- 학습 checkpoint 없이 pure reference motion replay가 되는 경로 확인. 필요 시 `flags.test=True` 또는 motion visualization용 별도 플래그 조사.
- 시각화 실패 시 MuJoCo fallback: `python scripts/vis/vis_h5_motion_mj.py --pkl sample_data/h5_walk_S001_lv0_trial01.pkl`.

### 5.4 체크리스트 판정

아래 7개 항목을 각각 OK / 이상 / 모호로 기록하고, "이상"이면 구체적 증상과 의심 지점(예: "R_Knee axis sign 뒤집힘")을 남긴다.

| # | 항목 | 정상 기준 |
|---|---|---|
| 1 | 발 접촉 패턴 | 좌우 heel strike → toe off 순서, stance/swing 교대, 이중 지지 구간 존재 |
| 2 | 무릎 flexion 방향 | 앞쪽으로 굽힘, hyperextension 없음, 좌우 대칭 |
| 3 | 골반/코어 | 축 뒤집힘 없음, 보행 주기가 한눈에 보임 |
| 4 | 팔 스윙 | 반대쪽 다리와 counter-phase, 자연스러운 range |
| 5 | 상체 자세 | 대체로 수직, 과도한 기울기·흔들림 없음 |
| 6 | 발-지면 관계 | 지면 관통/공중부양 없음 (fix_height 유효) |
| 7 | 축/좌표계 | 휴머노이드가 뒤집히거나 옆으로 눕지 않음, facing +X, Z-up |

### 5.5 결과 기록

`02_research_dev/YYMMDD_h5_iggym_replay_A_result_analysis.md`에 다음 내용 작성:
- 사용한 trial과 변환 커맨드
- 체크리스트 결과 표
- 스크린샷 또는 녹화 2~3장 (`output/h5_visual_check/` 아래)
- 발견된 이슈 요약과 의심 원인 (매핑/축/sign)
- 다음 단계 결정: (a) Step B 진행 / (b) `h5_to_motion_lib.py` 수정 후 Step A 재실행 / (c) 접근 (b)(AMASS와 나란히 재생)로 escalation

## 6. 성공 기준

- 체크리스트 7개 항목 중 최소 5개 이상 "OK" → Step B 진행 승인
- 1~2개 "이상" → 해당 항목만 수정 후 재판정
- 3개 이상 "이상" 또는 전반적 기괴함 → 매핑/축 convention 전면 재검토 및 접근 (b) escalation

## 7. 주의 및 리스크

- **환경 전환**: 본 계획은 Windows 워크스페이스에서 작성되었고 실행은 Linux에서 수행. 경로 구분자, `scripts/` 상대 경로 등에 주의.
- **H5 파일 위치**: 실행 환경에서 `data/combined_data_from_csv.h5`의 실제 경로를 먼저 확인하고 커맨드 조정 필요.
- **Policy 없는 재생 경로**: PHC의 `--test`가 policy checkpoint를 요구할 수 있음. 이 경우 기존 dummy checkpoint 사용 또는 motion replay 전용 경로 사용.
- **주관성**: "사람처럼 보이는지"는 질적 판단이나, 체크리스트 항목을 명시해 재현성을 확보.

## 8. 다음 단계 (참고용)

Step A 통과 후:
- **Step B**: H5 pkl과 AMASS pkl의 관절각 분포, 보행 주기(푸리에), CoM 궤적을 정량 비교하는 스크립트 작성
- **Step C**: H5 pkl로 `--task HumanoidImVIC` 학습 돌리고 reward/steps 수렴 곡선을 AMASS 기반 VIC_CCF_ON2와 비교

# 260408 PHC VIC 실행 셋업 및 수정 사항

본 문서는 이 머신에서 PHC VIC 코드를 처음 실행하기 위해 수행한 환경 셋업 과정과, 원본 PHC 대비 적용된 수정 사항을 정리한다.

---

## 1. 실행 환경 현황

- Conda env: `phc` (Python 3.8.20) — 이미 구성되어 있음
- IsaacGym 1.0rc4 — `/home/exolab/Downloads/isaacgym/python` 에서 editable 설치
- GPU: NVIDIA GeForce RTX 4060 Ti (8 GB VRAM)
- Repo: `/home/exolab/Documents/GitHub/PHC` (branch: `Jimin`)

---

## 2. 셋업 과정에서 수행한 수정

### 2.1 Motion 파일 경로 변경
원본 설정은 존재하지 않는 모션 파일을 참조하고 있었다.

- 파일: `phc/data/cfg/env/env_im_walk_vic.yaml` (line 7)
- 변경 전: `motion_file: "sample_data/amass_isaac_walking_forward_single.pkl"` (파일 없음)
- 변경 후: `motion_file: "sample_data/amass_isaac_walking_primitive.pkl"` (로컬에 존재)

주의: 기존 VIC_PHASE 실험들은 `walking_forward_single.pkl` 기준이므로, 본 설정으로 평가한 수치는 CLAUDE.md 성능 히스토리 표와 직접 비교할 수 없다.

### 2.2 SMPL body model 파일 배치
`phc/env/tasks/humanoid_amp.py` 가 `data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` 를 요구한다. 사용자가 SMPL v1.1.0 원본 파일을 `data/smpl/` 에 넣었으며, 파일명이 달라 코드가 기대하는 이름으로 복사했다.

```
data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl → SMPL_NEUTRAL.pkl
data/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl       → SMPL_MALE.pkl
data/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl       → SMPL_FEMALE.pkl
```

원본 파일은 그대로 보존되어 있다 (`cp` 사용).

### 2.3 체크포인트 경로 심볼릭 링크
`phc/utils/config.py` 의 `--epoch -1` 로직은 `output/<exp_name>.pth` 를 탐색하지만, 기존 체크포인트는 `output/Finished/` 하위에 있었다.

```
output/VIC_PHASE_04.pth → Finished/VIC_PHASE_04.pth  (symlink)
```

실험 이름은 `phc/data/cfg/learning/im_walk_vic.yaml` 의 `name: VIC_PHASE_04` 에서 결정된다.

### 2.4 원본 PHC 대비 연구 수정 (pre-existing, 본 세션 이전)
본 세션에서 추가한 것은 아니지만, 원본 PHC 대비 저장소 차원에서 이미 적용되어 있는 VIC 관련 수정 사항은 다음과 같다.

- `phc/env/tasks/humanoid_im_vic.py` — VIC 환경 (CCF 그루핑, torque 계산, phase obs, phase-CCF 로깅)
- `phc/learning/amp_agent.py` — CCF sigma override, reward curriculum, wandb 로깅
- `phc/learning/im_amp_players.py` — 평가 시 `phase_ccf_log.npy` 저장
- `phc/data/cfg/env/env_im_walk_vic.yaml`, `phc/data/cfg/learning/im_walk_vic.yaml` — VIC 전용 설정
- `analyze_phase_ccf.py` — Phase별 임피던스 분석/플롯 스크립트

자세한 설계는 `CLAUDE.md` 의 Research Overview 섹션 참고.

---

## 3. 실행 방법

모든 명령 실행 전 conda 활성화 필수.

```bash
conda activate phc
cd /home/exolab/Documents/GitHub/PHC
```

### 3.1 평가 (headless, 렌더 없음)
기존 체크포인트 `VIC_PHASE_04.pth` 로 one-shot 평가 수행.

```bash
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display
```

### 3.2 평가 (IsaacGym viewer 표시)
데스크탑 세션이 있어야 한다. SSH 사용 시 `ssh -Y` 또는 로컬 터미널에서 실행.

```bash
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch -1
```

### 3.3 학습
```bash
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --headless --num_envs 512
```

8 GB VRAM 환경에서는 `--num_envs 512` 가 OOM 날 수 있다. 그럴 경우 `--num_envs 256` 또는 `128` 로 낮춘다.

---

## 4. 최초 실행 결과 (VIC_PHASE_04 체크포인트)

- av reward: 174.96
- av steps: 52.70
- 출력물:
  - `output/VIC_PHASE_04/phase_ccf_log.npy`
  - `output/VIC_PHASE_04/phase_ccf_per_group.png`
  - `output/VIC_PHASE_04/phase_ccf_LR_comparison.png`

Group별 impedance_scale (phase 구간별 평균):

```
L_Hip       1.422 / 1.386 / 1.417  (전체 1.427)
L_Knee      0.953 / 0.954 / 1.044  (전체 0.989)
L_Ankle+Toe 1.663 / 1.620 / 1.660  (전체 1.638)
R_Hip       1.108 / 1.041 / 1.199  (전체 1.128)
R_Knee      0.941 / 0.959 / 1.025  (전체 0.978)
R_Ankle+Toe 1.728 / 1.767 / 1.666  (전체 1.734)
Upper-L     0.636 / 0.656 / 0.644  (전체 0.646)
Upper-R     0.694 / 0.669 / 0.735  (전체 0.696)
```

발목 > 힙 > 무릎 > 상체 순의 임피던스 패턴은 생체역학 문헌의 보행 관절 강성 분포와 질적으로 일치한다. 다만 모션 파일이 원본 실험과 달라 보상 수치(174.96)는 CLAUDE.md 히스토리 표(VIC_PHASE 계열 ~900+)와 직접 비교 불가.

---

## 5. 알려진 이슈

- Motion loading hang: `phc/utils/motion_lib_base.py` ~line 235 의 `mp.set_sharing_strategy('file_system')` 주석 해제 필요할 수 있음.
- `eval_success_rate` vs `success_rate`: wandb에서는 `eval_success_rate` 사용.
- IsaacGym viewer 는 X display 없으면 뜨지 않는다.

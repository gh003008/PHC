# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Rules (from `.agent/rules/code-command.md`)

**Always read `.agent/rules/code-command.md` before starting work.**

1. **모든 명령어 실행 전**: `conda activate phc` 필수
2. **문서 작성**: 한글로 작성, 파일명 `YYMMDD_topic_name.extension` 형식
   - `01_research_docs/`: 실험 세팅, 구현 방식, 학습 전략
   - `02_research_dev/`: 학습 결과 및 상세 분석
   - `03_QnA/`: 질문-답변 정리
   - `.txt` 파일 작성시: 마크다운(`###`, `**`) 사용 금지, 문장 중심으로 작성
3. **VIC 구현**: 기존 PHC 스크립트 직접 수정 금지, `_vic` 접미사 복사본 사용 (예: `humanoid_im_vic.py`). Low-level 유틸 수정 시 `# VIC` 주석 명시.
4. **실험 설정 백업**: 학습 실행 전 `exp_config/forward_walking/YYMMDD_실험이름/`에 yaml·py 스냅샷 백업.
5. **평가 후 결과 분석**: 자동으로 `02_research_dev/YYMMDD_forward_walk_vicXX_result_analysis.md` 작성.

---

## Research Overview

### 한 줄 요약
> PHC 기반 SMPL 휴머노이드에 Variable Impedance Control(VIC)을 구현하여, 보행 모방 중 관절별 임피던스를 자율 학습시키고 있다. 최종 목표는 Exoskeleton 제어기 학습.

### 핵심 개념: VIC (Variable Impedance Control)
- 기존 PD 제어: `torque = kp * (q_target - q) - kd * dq`
- VIC 추가: `torque = kp * 2^ccf * (q_target - q) - kd * 2^ccf * dq`
- **CCF (Compliance Control Factor)**: 관절별 임피던스 배율. `ccf=0` → 기존 동일, `ccf>0` → rigid, `ccf<0` → compliant.
- CCF는 policy가 action으로 출력. 8 그룹 (L_Hip, L_Knee, L_Ankle+Toe, R_Hip, R_Knee, R_Ankle+Toe, Upper_L, Upper_R).

### 커리큘럼 구조
- **Stage 1**: CCF=0 고정 (기존 PD와 동일). Warm-up으로 보행 자체를 먼저 학습.
- **Stage 2**: CCF 학습 활성화. `vic_curriculum_stage: 2`로 수동 전환.
- **Reward Curriculum**: epoch < switch_epoch → task=0.7/disc=0.3, 이후 역전(task=0.3/disc=0.7).

### 최종 목표
인간+Exoskeleton 시뮬레이션에서 Exo 제어기 RL 학습. VIC는 "사람이 임피던스를 어떻게 조절하는지" 모델링하는 사전 단계.
- IsaacGym(PhysX 4)은 closed kinematic chain 미지원 → Isaac Lab(PhysX 5) 마이그레이션 필요.

---

## Current Research Status (2026-03-11)

### 최신 실험: VIC_CCF_ON2
- CCF 8 dims에 별도 sigma=-1.0 (std=0.37) 적용 → **CCF가 실제로 학습됨!**
- 결과: av_reward=939.51, av_steps=297.39 (V4 baseline 대비 2배+)
- 발목 1.28~1.49x (높음), 무릎 0.91~0.94x (낮음), 상체 0.80~0.86x (낮음) → 생체역학 문헌과 질적 일치

### 실험 성능 히스토리

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON | 945.30 | 297.98 | Stage2, CCF sigma=-2.9 공유 → 미학습 |
| **VIC_CCF_ON2** | **939.51** | **297.39** | **CCF sigma=-1.0 분리 → CCF 학습 확인** |

### 다음 실험 방향
1. **VIC_PHASE** (우선순위 1): Phase obs (sin/cos) +2 dims 추가 → Stance/Swing 구분 강화
2. **Contact-aware CCF 분석** (우선순위 2): contact force로 Stance/Swing별 임피던스 비교
3. **주기적 보행 모션 교체** (우선순위 3): walk-stop → cyclic walking loop

---

## Architecture Overview

PHC는 **IsaacGym** + **rl-games** 기반 물리 휴머노이드 제어 프레임워크. 핵심 알고리즘: AMP (Adversarial Motion Prior) + 모션 모방.

### Entry Points
- `phc/run.py` — argparse 기반 (VIC 실험에서 주로 사용)
- `phc/run_hydra.py` — Hydra config 기반

### Task Class Hierarchy (`phc/env/tasks/`)
```
VecTask (base_task.py)
  └→ HumanoidAMPTask (humanoid_amp_task.py)
       └→ HumanoidAMP (humanoid_amp.py)
            └→ HumanoidIm (humanoid_im.py, 93KB)
                 ├→ HumanoidImVIC (humanoid_im_vic.py, 102KB) ← 현재 연구
                 ├→ HumanoidImMCP (humanoid_im_mcp.py)
                 └→ HumanoidImMCPVIC (humanoid_im_mcp_vic.py)
```
Task 등록: `phc/utils/parse_task.py`에서 `eval(args.task)(...)` 으로 매핑.

### Learning Classes (`phc/learning/`)
- `im_amp.py` (IMAmpAgent) — 메인 학습 에이전트 (imitation + AMP discriminator)
- `amp_agent.py` (AmpAgent) — PPO + AMP reward, CCF sigma override (`pre_epoch()`), reward curriculum, wandb 로깅
- `im_amp_players.py` (IMAmpPlayer) — 평가 시 사용, phase_ccf_log.npy 저장
- `network_builder.py`, `amp_network_*.py` — MLP policy/value/disc 네트워크

### Config System (`phc/data/cfg/`)
Hydra 계층 구조. Root: `config.yaml`.
- `env/` — 환경·태스크 파라미터 (13개 yaml)
- `learning/` — 네트워크, PPO, AMP 설정 (9개 yaml)
- `robot/` — 로봇 모델: SMPL, SMPLX, Unitree H1/G1 (7개 yaml)
- `sim/` — PhysX 물리 설정
- `control/`, `domain_rand/` — 제어 모드, 도메인 랜덤화

### Utilities (`phc/utils/`)
- `parse_task.py` — 태스크 이름 → 클래스 매핑
- `flags.py` — 글로벌 런타임 플래그 (`test`, `im_eval`, `debug`)
- `motion_lib_smpl.py` / `motion_lib_real.py` — 모션 데이터 로딩
- `torch_utils.py`, `rotation_conversions.py` — 회전 수학, 포즈 변환

### Output
- 체크포인트: `output/<experiment_name>.pth` 또는 `output/HumanoidIm/<exp_name>/`
- Phase 분석: `output/phase_ccf_log.npy`, `output/phase_ccf_log_plot.png`

---

## Key Files (VIC 관련)

| 파일 | 역할 |
|---|---|
| `phc/env/tasks/humanoid_im_vic.py` | VIC 환경. CCF 그루핑, torque 계산, wandb/phase-CCF 로깅 |
| `phc/learning/amp_agent.py` | CCF sigma override, reward curriculum, wandb 로깅 |
| `phc/learning/im_amp_players.py` | 평가 후 phase_ccf_log.npy 저장 |
| `phc/data/cfg/env/env_im_walk_vic.yaml` | 환경 설정 (VIC params, reward specs) |
| `phc/data/cfg/learning/im_walk_vic.yaml` | 학습 설정 (네트워크, PPO, AMP) |
| `analyze_phase_ccf.py` | Phase별 임피던스 분석·플롯 스크립트 |
| `exp_config/forward_walking/` | 실험별 설정 스냅샷 아카이브 |

### 현재 VIC 설정값 (VIC_CCF_ON2 기준)
```yaml
# env yaml
vic_enabled: True
vic_curriculum_stage: 2          # 1=CCF고정, 2=CCF학습
vic_ccf_num_groups: 8
vic_ccf_sigma_init: -1.0         # CCF sigma 분리 (std=0.37)
vic_ccf_min: -1.0 / vic_ccf_max: 1.0
reward_curriculum_switch_epoch: 10000
reward_w_stage1_task: 0.7 / disc: 0.3
reward_w_stage2_task: 0.3 / disc: 0.7

# learning yaml
sigma_init val: -2.9             # PD action sigma (std=0.055)
max_epochs: 20000
mlp units: [1024, 1024, 512, 512]
learning_rate: 5e-5
```

---

## Common Commands

```bash
conda activate phc

# === VIC 학습 (run.py) ===
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --headless --num_envs 512

# === VIC 평가 (headless) ===
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display

# === VIC 평가 (시각화) ===
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch -1

# === 특정 epoch 평가 ===
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --num_envs 1 --test --epoch 15000 --no_virtual_display

# === 이어서 학습 ===
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
  --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
  --headless --num_envs 512 --epoch 10400

# === Phase-CCF 분석 ===
python analyze_phase_ccf.py --npy output/phase_ccf_log.npy

# === Hydra 기반 (non-VIC) ===
python phc/run_hydra.py learning=im_walk env=env_im_walk robot=smpl_humanoid \
  exp_name=<name> env.num_envs=512 headless=True

# === 유틸리티 ===
python check_ckpt.py                    # 체크포인트 정보 확인
python scripts/vis/vis_motion_mj.py     # SMPL 모션 시각화
```

---

## Directory Structure

```
PHC/
├── phc/                          # 메인 소스코드
│   ├── run.py                    # argparse 기반 학습 엔트리
│   ├── run_hydra.py              # Hydra 기반 학습 엔트리
│   ├── env/tasks/                # 태스크 구현 (19개)
│   │   ├── humanoid_im_vic.py    # VIC 환경 (현재 연구)
│   │   ├── humanoid_im.py        # 베이스라인 모방
│   │   └── ...
│   ├── learning/                 # 학습 알고리즘 (20개)
│   │   ├── im_amp.py             # IMAmp 학습 에이전트
│   │   ├── amp_agent.py          # PPO+AMP (CCF sigma override 포함)
│   │   ├── im_amp_players.py     # 평가 플레이어
│   │   └── ...
│   ├── utils/                    # 유틸리티 (21개)
│   │   ├── parse_task.py         # 태스크 등록
│   │   ├── motion_lib_smpl.py    # 모션 데이터 로딩
│   │   └── ...
│   └── data/
│       ├── cfg/                  # Hydra 설정 (47개 yaml)
│       │   ├── env/              # 환경 설정
│       │   ├── learning/         # 학습 설정
│       │   ├── robot/            # 로봇 설정
│       │   └── sim/, control/, domain_rand/
│       └── assets/               # 물리 에셋 (mesh, mjcf, urdf)
├── 00_basic/                     # 기본 참고자료 (commands 등)
├── 01_research_docs/             # 실험 세팅·구현 문서 (28개)
├── 02_research_dev/              # 학습 결과·분석 문서 (20개)
├── 03_QnA/                       # 질문-답변 (6개)
├── 04_Context_Bridge/            # 컨텍스트 브릿지
├── exp_config/                   # 실험별 설정 스냅샷 아카이브
│   └── forward_walking/          # V1~V4, VIC01~VIC11, CCF_ON, CCF_ON2
├── output/                       # 체크포인트·학습 결과물
├── scripts/                      # 시각화·데이터 처리 스크립트
├── sample_data/                  # 모션 데이터 (AMASS pkl)
├── poselib/                      # Pose 라이브러리
└── analyze_phase_ccf.py          # Phase-CCF 분석 스크립트
```

---

## Known Issues

- **Motion loading hang**: `phc/utils/motion_lib_base.py` ~line 235에서 `mp.set_sharing_strategy('file_system')` 주석 해제 필요
- **Success rate metric**: wandb에서 `success_rate`가 아닌 `eval_success_rate` 사용
- **Bunny hop**: cycle_motion=True 시 2번째 사이클부터 velocity 불연속 → Phase obs 추가로 완화 예정
- **Remote push**: SSH 키 미설정으로 pending 상태

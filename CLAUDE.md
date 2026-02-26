# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Rules (from `.agent/rules/code-command.md`)

**Always read `.agent/rules/code-command.md` before starting work.**

1. **모든 명령어 실행 전**: `conda activate phc` 필수
2. **문서 작성**: 한글로 작성, 파일명 `YYMMDD_topic_name.extension` 형식
   - `01_research_docs/`: 실험 세팅, 구현 방식, 학습 전략
   - `02_research_dev/`: 학습 결과 및 상세 분석
   - `03_궁금한 것들 정리/`: 질문-답변 정리
   - `.txt` 파일: 마크다운(`###`, `**`) 사용 금지, 문장 중심으로 작성
3. **VIC 구현**: 기존 PHC 스크립트 직접 수정 금지, `_vic` 접미사 복사본 사용 (예: `humanoid_im_vic.py`). Low-level 유틸 수정 시 `# VIC` 주석 명시.

## Common Commands

```bash
# Training
conda activate phc
python phc/run_hydra.py learning=<learning_cfg> env=<env_cfg> robot=smpl_humanoid \
    exp_name=<name> env.num_envs=512 headless=True

# VIC01 current experiment
python phc/run_hydra.py learning=im_walk_vic env=env_im_walk_vic robot=smpl_humanoid \
    exp_name=VIC01_forward_walk_v4 env.num_envs=512 headless=True

# Evaluation
python phc/run_hydra.py learning=im_walk_vic env=env_im_walk_vic robot=smpl_humanoid \
    exp_name=VIC01_forward_walk_v4 env.num_envs=1 test=True epoch=-1 headless=False im_eval=True

# Inspect config (dry-run)
python phc/run_hydra.py env=env_im_walk_vic learning=im_walk_vic --cfg job

# Check checkpoint info
python check_ckpt.py

# Verify SMPL setup
python scripts/vis/vis_motion_mj.py
```

## Architecture Overview

PHC is a physics-based humanoid control framework built on **IsaacGym** + **rl-games**. The core algorithm is AMP (Adversarial Motion Prior) combined with motion imitation.

### Entry Point
`phc/run_hydra.py` — uses Hydra for config composition. All configs live in `phc/data/cfg/`.

### Config System (Hydra, `phc/data/cfg/`)
Hierarchical YAML composition. Root: `config.yaml`. Compose via command-line:
```
python phc/run_hydra.py env=<env_cfg> learning=<learning_cfg> robot=<robot_cfg>
```
- `env/` — environment and task parameters
- `learning/` — network architecture, PPO hyperparameters, AMP settings
- `robot/` — humanoid model (SMPL, SMPLX, Unitree H1/G1)
- `sim/` — IsaacGym PhysX physics settings
- `control/`, `domain_rand/` — control mode and randomization

### Task Classes (`phc/env/tasks/`)
Registered by name string in `phc/utils/parse_task.py` via `eval(args.task)(...)`.

| Class | File | Purpose |
|---|---|---|
| `HumanoidImVIC` | `humanoid_im_vic.py` | **Current**: VIC imitation task |
| `HumanoidIm` | `humanoid_im.py` | Baseline imitation |
| `HumanoidImMCPVIC` | `humanoid_im_mcp_vic.py` | MCP + VIC |
| `HumanoidImMCP` | `humanoid_im_mcp.py` | Mixture-of-primitives |
| `HumanoidAMPTask` | `humanoid_amp_task.py` | AMP base class |
| `VecTask` | `base_task.py` | IsaacGym vectorized task base |

Inheritance: `VecTask → HumanoidAMPTask → HumanoidIm → HumanoidImVIC`

### Learning (`phc/learning/`)
- `im_amp.py` — main training agent (imitation + AMP discriminator)
- `amp_agent.py` — PPO agent with AMP reward
- `network_builder.py`, `amp_network_*.py` — MLP policy/value/discriminator networks

### Utilities (`phc/utils/`)
- `parse_task.py` — maps task name strings → class constructors (add new tasks here)
- `flags.py` — global runtime flags (`test`, `im_eval`, `debug`, etc.)
- `motion_lib_smpl.py` / `motion_lib_real.py` — motion data loading
- `torch_utils.py` — rotation math, pose transforms

### Output
Checkpoints saved to `output/HumanoidIm/<exp_name>/`.

## Current Research: VIC01

- Branch: `single_primitives`
- Files: `phc/env/tasks/humanoid_im_vic.py`, `phc/data/cfg/env/env_im_walk_vic.yaml`, `phc/data/cfg/learning/im_walk_vic.yaml`
- Stage 1 (Warm-up): `vic_curriculum_stage: 1`, β=0 고정
- Stage 2 전환: `vic_curriculum_stage: 2`로 수동 변경 후 재시작
- VIC 관련 핵심 파라미터: `vic_enabled`, `vic_ccf_min`, `vic_ccf_max`, `vic_metabolic_reward_w`

## Known Issues

- **Motion loading hang**: `phc/utils/motion_lib_base.py` ~line 235에서 `mp.set_sharing_strategy('file_system')` 주석 해제
- **Success rate metric**: wandb에서 `success_rate`가 아닌 `eval_success_rate` 사용

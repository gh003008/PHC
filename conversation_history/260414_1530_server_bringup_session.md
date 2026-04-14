# 260414 15:30 세션 — EXOLAB 서버 환경 구축 + 첫 학습 제출

## 논의 주제
1. 서버 conda env 구축 (phc)
2. 연쇄 실패한 sbatch 5번의 원인별 수정
3. `env_im_walk.yaml` 구조 버그 발견·flatten
4. 1000-epoch 단위 모니터 스크립트 구축

## 실행 순서 요약

### Conda env 구축
- `conda tos accept` 필요 (Anaconda 채널, 서버 최초 생성이라)
- `~/phc_env_clean.yml` 의 PyPI-incompatible 엔트리 제거 후 설치:
  - `smpl-sim==0.0.1` 제거 → github source 로 별도 설치 (`pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master`) *(SERVER_GUIDE.md 업데이트도 사용자가 선반영)*
  - `chumpy==0.71` → `0.70` 로 일단 통과, 이후 `requirement.txt` 가 다시 0.71 로 upgrade (resolve됨)
  - `torch==2.4.1+cu118`, `torchvision==0.19.1+cu118`, `triton==3.0.0`, `nvidia-*-cu11` 6줄 제거 → torch cu121 로 별도 설치
- `pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121` → CUDA 검증 성공 (4 GPUs)
- `~/isaacgym/python` 및 `~/PHC` editable 설치 (PHC 는 `setup.py` 없어서 `pip install -r requirement.txt` 로 대체)

### 서버 실행 이슈 및 수정
| Job | 실패 원인 | 수정 |
|---|---|---|
| 3589 | PartitionConfig (mem 32G > per-partition 한도 15.5G) | `--mem=15G` |
| 3590 | `num_envs=512` Hydra override 거부 | `env.num_envs=512` 시도 |
| 3591 | `env.num_envs` 도 struct key 없음 | override 제거 |
| 3592 | wandb `No API key configured` | `no_log=True` 추가 |
| 3595 | (사용자 요청) idx3 → idx2 전환 + `cfg.env.num_envs` AttributeError | `env_im_walk.yaml` flatten |
| 3597 | `IndexError: motion data empty` | motion_file 교체 |
| **3598** | **RUNNING** | — |

### `env_im_walk.yaml` flatten (git-tracked 파일 수정)
원본은 Hydra 구조상 깨져있었음:
- 외부 `env:` wrapper 로 인해 `cfg.env.env.num_envs` 가 됨 (run_hydra.py line 315 는 `cfg.env.num_envs` 기대)
- 내부에 `robot:`, `control:`, `sim:`, `domain_rand:` sub-block 이 섞여있었음 — 이들은 별도 config 파일로 Hydra defaults 에서 로드되는 것이 정상
- 추측: **이 config 는 로컬에서도 실제로 돌아간 적 없음**. 로컬은 `env_im_walk_mpl.yaml` (축소본, flat 구조) 로만 실험해왔음

**수정 내용**:
- 외부 `env:` 제거 (하위 키들을 최상위로 올림)
- `task: HumanoidIm` 최상위 추가
- `robot:`/`control:`/`sim:`/`domain_rand:` 블록 삭제
- `motion_file`: `amass_isaac_walking_forward.pkl` → `amass_isaac_walking_primitive.pkl` (서버에 존재하는 파일)

**커밋 여부**: 아직 안 함. 1000 epoch 결과 판정 후 결정.

### Monitor
- `~/PHC/monitor_progress.sh` (신규) — 60초 간격 polling, 1000 epoch 경계마다 `progress_report.md` 테이블에 추가
- 로그 패턴: `PHC_Server_Baseline_v1-Ep: <N>\trwd: <R>\t...\teps_len: <L>`
- Background PID 1785910

## 현재 상태 (세션 종료 시점)
- **Job 3598 RUNNING** (15:35 KST 시작, partition idx2)
- Ep 1: rwd 6.7, eps_len 2.1 → Ep 41: rwd 23.1, eps_len 8.7 (로컬 shrunk 대비 초반부터 건강한 상승세)
- 속도 ~5.7초/epoch → 1000 epoch 도달 예상 **16:52 KST**
- 모니터 백그라운드 실행 중
- Wakeup 16:38 예약 (첫 milestone 근처 체크)

## 결정사항
- **Slurm GRES 규칙**: `#SBATCH -p idx{N}` + `#SBATCH --gres=gpu:idx{N}:1` + `--mem=15G` (per-partition 한도). {N} 은 비어있는 GPU 번호, 현재 기본 idx2.
- **바이섹션 순서**: AMASS + no-VIC (Step 1, 현재) → AMASS + VIC → H5 + no-VIC → H5 + VIC (Phase 0, 최종 목표)
- **wandb 는 서버 미설정 상태로 진행**. 필요 시 사용자가 interactive `wandb login` 후 `no_log=True` 제거.

## 수정 인벤토리 (복원 가이드)
| # | 파일 | 수정 유형 | 복원 방법 |
|---|---|---|---|
| 1 | `~/phc_env_clean.yml` (server home) | 수정 | 로컬에서 rsync 재업로드 |
| 2 | `~/PHC/train_phc.sh` | 신규 | 삭제 |
| 3 | `~/PHC/monitor_progress.sh` | 신규 | 삭제 |
| 4 | `~/PHC/phc/data/cfg/env/env_im_walk.yaml` | **git-tracked 수정** | `git checkout -- phc/data/cfg/env/env_im_walk.yaml` |

## 미해결 이슈

### 즉시
- [ ] 1000 epoch 시점 `av_reward` / `av_steps` 확인 (예상 16:52 KST)
- [ ] 결과 양호하면 env_im_walk.yaml 커밋 + push

### Plateau 판정 시 Fallback 계획
1. **가설 A — stripped sub-block 중 필요한 것 있었음**
   - `git checkout -- phc/data/cfg/env/env_im_walk.yaml` 로 원복
   - 대안 실행: `env=env_im` (기본, 잘 동작) + `+env.motion_file=sample_data/amass_isaac_walking_primitive.pkl` override
2. **가설 B — motion 데이터셋 차이**
   - 로컬에 `amass_isaac_walking_forward.pkl` 존재 여부 사용자 확인 → rsync
3. **가설 C — 그 외**
   - `env_im_walk_mpl.yaml` (로컬에서 실제 성공했던 버전) vs 현재 flat 버전 diff

### 문서 갱신 필요 (추후)
- `SERVER_GUIDE.md §5`: `num_envs=512` 오류, `--gres=gpu:1` → `--gres=gpu:idx{N}:1`, `--mem=32G` → `--mem=15G` 반영
- `CLAUDE.md` Common Commands: `env.num_envs=512` 잘못된 override 예시 제거

## 다음 단계
1. 16:38 wakeup → 현재 epoch/reward 보고
2. 16:52 전후 1000 epoch 도달 → `progress_report.md` 확인
3. 결과 따라:
   - 성공 → env_im_walk.yaml 커밋, 이 세션 파일 + SERVER_GUIDE 갱신 커밋, push. 이후 Step 2 (VIC) 준비
   - 실패 → 위 Fallback 계획대로 분기 진단

## 로컬 재현 가이드 (shrunk num_envs)

로컬 GPU 7.6GB 에서는 `num_envs=512` 로 OOM. yaml 을 수정하지 말고 cmdline 에서 override:

```bash
conda activate phc

# 로컬용: num_envs 64 or 128 로 축소
python phc/run_hydra.py \
    learning=im_walk \
    env=env_im_walk \
    exp_name=Local_FlatConfig_Reproduce \
    env.num_envs=64 \
    env.numEnvs=64 \
    headless=True \
    no_log=True
```

**주의**:
- `env.num_envs=64` 는 flatten 된 현재 yaml 에서 정상 override 됨 (이전엔 `env.env.num_envs` 라 실패했음)
- `numEnvs` 도 같이 낮춰야 함 (일부 PHC 코드가 `numEnvs` 읽음)
- OOM 시 32 까지 낮추거나 `+env.episode_length=150` 도 override
- 서버에서 학습한 체크포인트는 `output/HumanoidIm/humanoid_smpl/PHC_Server_Baseline_v1_*/` 에 저장됨 (서버 local, git 에 없음). 평가하려면 rsync 로 로컬 가져오기.

## 1000 epoch 결과 (2026-04-14 16:43 기록)

| 지표 | Ep 1000 | 로컬 V4 (20k 최종) | 로컬 VIC_CCF_ON2 (20k 최종) |
|---|---|---|---|
| reward | **128.7** | 461 (av) | 940 (av) |
| eps_len | **42.2** | 143 (av) | 297 (av) |

로컬 shrunk plateau (eps_len ≈12) 를 **3.5× 돌파**. "config 축소가 학습 실패의 원인" 가설 **강하게 확증**.

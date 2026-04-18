# AMASS Velocity-Command VIC V2 (Personalization) — 서버 실행 현황

**Date**: 2026-04-19 (자율 세션 Round 2, 서버 Claude)
**Parent analysis**: `02_research_dev/260419_amass_cmd_result_analysis.md`
**Parent plan**: `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md`
**Round 1 결과 요약**: A(w=0.3) rwd 388/eps_len 184, B(w=0.5) rwd 279/eps_len 185 — 둘 다 20k epoch 정상 완주. B가 objective(command 추종)에 더 부합하다고 판정.

---

## 1. 사용자 원본 목표 (복기)
> embedding human agent as a simulation for CoM velocity input **including personalization**

Round 1에서는 command 축만 건드렸고, Round 2에서 **personalization 축**을 추가.

---

## 2. 자율 판단 요약

### 2.1 Personalization 구현 방식 선택
PHC에 이미 존재하는 인프라 활용:
- `robot.has_shape_variation: True` → per-env 다른 SMPL betas (10 params) 샘플링해서 다른 body 생성
- `robot.has_shape_obs: True` → policy 입력에 betas 10 dims 추가 (자기 body 정보 앎)
- `robot.has_shape_obs_disc: False` (기본값 유지) → AMP discriminator는 shape 안 봄 (demo는 단일 subject이므로 혼란 방지)

다른 personalization 옵션들 고려 후 기각:
- Subject 임베딩 (ID별 latent vector): 단일 subject data라 불가
- Motion style 다양화: 단일 clip이라 불가
- Joint efforts/range 다양화: 필요하긴 한데 shape에 포함됨

**결론**: SMPL body shape variation이 가장 즉각 구현 가능하고 의미 있는 personalization.

### 2.2 두 variant 설계

**V2_A — Winner + personalization** (objective 최우선)
- cmd_tracking_w=0.5 (Round 1 winner B config)
- cmd_v_range [0.8, 1.3] (Round 1과 동일)
- has_shape_variation=True, has_shape_obs=True

**V2_B — Stable + personalization + wider range** (generalization 탐색)
- cmd_tracking_w=0.3 (Round 1 안정적 학습 A config)
- cmd_v_range [0.6, 1.4] (확장)
- has_shape_variation=True, has_shape_obs=True

두 variant의 축 의미:
- V2_A: personalization **추가**만, 나머지는 winner 유지
- V2_B: personalization + generalization **동시** 추가 (더 어려운 과제)

만약 V2_A 실패하고 V2_B 성공하면: w=0.3이 personalization 하에서도 더 적합
만약 V2_A 성공하고 V2_B 실패하면: wider range가 shape variation과 충돌
둘 다 성공하면: personalization은 수용되므로 다음 단계는 방향(ω_cmd) 추가

---

## 3. 실행 구성

### 3.1 새 파일
| 파일 | 내용 |
|---|---|
| `env_im_walk_vic_cmd_V2_A.yaml` | V2_A 환경 config (winner + shape variation) |
| `env_im_walk_vic_cmd_V2_B.yaml` | V2_B 환경 config (stable + shape variation + wider range) |
| `im_walk_vic_cmd_V2_A.yaml` | V2_A 학습 config (name: AMASS_CMD_V2_A) |
| `im_walk_vic_cmd_V2_B.yaml` | V2_B 학습 config (name: AMASS_CMD_V2_B) |
| `train_cmd_V2_A_gpu2.sh` | Sbatch idx2 |
| `train_cmd_V2_B_gpu3.sh` | Sbatch idx3 |

### 3.2 코드 변경 없음
기존 `HumanoidImVICCmd` (Round 1에서 작성)가 `HumanoidImVIC` → `Humanoid` 체인으로 상속되어 있어서 `has_shape_variation/obs`는 기본 Humanoid 클래스에서 처리됨. 추가 코드 수정 **불필요**.

### 3.3 핵심 설정 비교

| 필드 | Round 1 A | Round 1 B | **V2_A** | **V2_B** |
|---|---|---|---|---|
| cmd_tracking_w | 0.3 | 0.5 | **0.5** | **0.3** |
| cmd_v_range | [0.8,1.3] | [0.8,1.3] | [0.8,1.3] | **[0.6,1.4]** |
| cmd_w_range | [0,0] | [0,0] | [0,0] | [0,0] |
| has_shape_variation | False | False | **True** | **True** |
| has_shape_obs | False | False | **True** | **True** |
| vic_ccf_num_groups | 4 | 4 | 4 | 4 |
| reward curriculum switch | 10000 | 10000 | 10000 | 10000 |
| max_epochs | 20000 | 20000 | 20000 | 20000 |
| parent ckpt | — | — | — | — |

Parent ckpt 쓰지 않은 이유: obs dim 변경 (+10 shape obs, task_obs도 같은 +3 cmd 유지), state_dict 모양 달라짐. Warm-start는 network partial load 구현 필요하며 검증 시간 부족. From scratch로 20k epoch 도는 게 더 안전.

---

## 4. 제출한 Slurm 잡

| Job ID | Name | Partition | GPU | Variant | 기대 시간 |
|---|---|---|---|---|---|
| 3773 | amass_cmd_V2_A | idx2 | idx2 | A (winner+pers) | ~18h 예상 |
| 3774 | amass_cmd_V2_B | idx3 | idx3 | B (stable+pers+wider) | ~18h 예상 |

제출 시각: 2026-04-19 02:35 KST.
**예상 완료**: 2026-04-19 ~20:30 KST (shape variation은 init에서 +~1min, training rate는 Round 1과 비슷할 것으로 예상)

---

## 5. 성공 기준 (자율 판정 사전 설정)

### 5.1 학습 종료 시점 (training log)
shape variation 추가는 학습 난이도를 **올림**. Round 1보다 낮은 reward도 정상일 수 있음.

| 지표 | 합격 | 부분 성공 | 실패 |
|---|---|---|---|
| Final av_reward | ≥ Round 1의 80% | 60~80% | <60% |
| Final av_eps_len | ≥ 150 | 100~150 | <100 |
| Traceback/crash | 없음 | — | 즉시 분석 |
| Shape range 실행 | 다양 | — | 모두 같은 shape (버그) |

V2_A 기대: rwd ~220+ (Round 1 B 279 × 0.8), eps_len ~150+
V2_B 기대: rwd ~310+ (Round 1 A 388 × 0.8), eps_len ~150+
  단, V2_B는 wider cmd range로 V2_A보다 낮을 수도 있음

### 5.2 test-greedy 평가 (사용자 복귀 후)
Round 1의 경험: training rwd × ~1.7 = test-greedy rwd (VIC4 551→951)
- V2_A 예상 test-greedy: ~350-500
- V2_B 예상 test-greedy: ~450-600

### 5.3 Personalization 검증 방법
로컬에서:
```bash
# 1. 같은 seed + 같은 cmd 고정해서 여러번 돌려보기
#    다른 body shape에서도 안정적으로 걷는지 시각 확인
python phc/run.py \
  --task HumanoidImVICCmd \
  --cfg_env exp_config/forward_walking/260419_AMASS_CMD_V2/env_im_walk_vic_cmd_V2_A.yaml \
  --cfg_train exp_config/forward_walking/260419_AMASS_CMD_V2/im_walk_vic_cmd_V2_A.yaml \
  --num_envs 1 --test --epoch -1

# 2. 여러 env 동시 시각화 (num_envs 큼)
python phc/run.py \
  --task HumanoidImVICCmd \
  --cfg_env exp_config/forward_walking/260419_AMASS_CMD_V2/env_im_walk_vic_cmd_V2_A.yaml \
  --cfg_train exp_config/forward_walking/260419_AMASS_CMD_V2/im_walk_vic_cmd_V2_A.yaml \
  --num_envs 16 --test --epoch -1
```

16개 env가 서로 다른 body로 동시에 걷는 것을 확인 — 이게 "personalization 성공" 시각 증거.

---

## 6. 리스크와 완화

| 리스크 | 발생 가능성 | 완화 현황 |
|---|---|---|
| Shape variation init 시 메모리 폭발 | 중 | cpus-per-task=8이면 multiprocessing=1로 fallback. 느리지만 동작. `--mem=15G` 충분 |
| 512개 XML 파일 디스크 공간 부족 | 낮 | /tmp에 생성됨, 크지 않음 |
| Shape 범위가 너무 극단적 | 중 | SMPL betas 기본 sampling 범위가 reasonable. 극단 체형 나와도 policy 학습 가능 |
| AMP discriminator가 shape 혼란 일으킴 | 낮 | has_shape_obs_disc=False로 disc는 shape 안 봄 |
| From scratch 20k epoch 부족 | 중 | Round 1에서 수렴 확인됨. shape variation으로 +10~20% 더 걸릴 수도. 그래도 학습 경향은 보임 |
| Wider v_cmd [0.6,1.4] + shape variation 동시 로딩 (V2_B) | 중 | V2_A가 먼저 수렴 확인 후 V2_B 평가. 만약 V2_B 학습 부진하면 cmd_v_range 축소 재시도 권고 |

---

## 7. 사용자 복귀 후 해야 할 것

### 7.1 학습 진행 확인
```bash
ssh server1
cd ~/PHC
squeue -u $USER
tail logs/amass_cmd_V2_A_3773.out
tail logs/amass_cmd_V2_B_3774.out
grep "^env_im_walk_vic_cmd_V2" logs/amass_cmd_V2_A_3773.out | tail -5
grep "^env_im_walk_vic_cmd_V2" logs/amass_cmd_V2_B_3774.out | tail -5
```

### 7.2 학습 완료 후 체크포인트 로컬로
```bash
scp server1:~/PHC/output/AMASS_CMD_V2_A_00020000.pth output/
scp server1:~/PHC/output/AMASS_CMD_V2_B_00020000.pth output/

# 또한 Round 1 결과 비교용
scp server1:~/PHC/output/AMASS_CMD_A_00020000.pth output/
scp server1:~/PHC/output/AMASS_CMD_B_00020000.pth output/
```

### 7.3 로컬 평가 배터리
1. **Test-greedy (정해진 policy, 단일 env)**: eps_len 300 도달 여부 확인
2. **Multi-env 시각화 (num_envs=16)**: 16개 서로 다른 body가 동시에 걷는지 시각 확인
3. **Command sweep**: v_cmd를 환경 yaml에서 고정해서 [0.6, 0.9, 1.2, 1.4] 각각 측정

### 7.4 다음 라운드 기획 후보
- V2 성공 → **Round 3 (direction)**: ω_cmd ∈ [-0.3, 0.3] 추가. 하지만 단일 forward clip 한계. 멀티 clip 라이브러리 빌드 필요 (원래 plan §2)
- V2_A만 성공 → cmd_tracking_w=0.5 final, wider range 포기 → Round 3는 cmd_v_range 유지하면서 다른 personalization 축 (AMP discriminator 보강?)
- 둘 다 실패 → personalization 완화 (has_shape_variation만, has_shape_obs 끄기) 또는 shape 분포 축소

---

## 8. 실행 타임라인

| 시각 (KST) | 이벤트 |
|---|---|
| 2026-04-18 10:40 | Round 1 A/B 제출 (Slurm 3756, 3757) |
| 2026-04-19 01:47 | Round 1 B 종료 (rwd 279, eps_len 185) |
| 2026-04-19 02:20 | Round 1 A 종료 (rwd 388, eps_len 184) |
| 2026-04-19 02:30 | 분석 MD 작성 완료 |
| **2026-04-19 02:35** | **V2 A/B 제출 (Slurm 3773, 3774)** |
| 2026-04-19 ~20:30 | V2 A/B 종료 예상 |
| 2026-04-19 이후 | 사용자 복귀, 평가 진행 |

---

## 9. 참조 파일

### 내부
- `02_research_dev/260419_amass_cmd_result_analysis.md` — Round 1 분석 (이 라운드의 근거)
- `exp_config/forward_walking/260418_AMASS_CMD/RUNNING_ON_SERVER.md` — Round 1 실행 문서
- `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` — VIC4 검증 (training vs test-greedy 배율 근거)
- `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md` — 상위 MPL 플랜
- `phc/env/tasks/humanoid_im_vic_cmd.py` — Command-conditioned task (Round 1에 작성)

### 외부
1. Peng et al., **AMP** (TOG 2021)
2. Luo et al., **PHC: Perpetual Humanoid Control** (ICCV 2023) — body shape randomization 포함
3. Won et al., **Physics-based Character Controllers Using Conditional VAEs** (TOG 2022)
4. Loper et al., **SMPL: A Skinned Multi-Person Linear Model** (TOG 2015) — 10-param body shape space

---

*이 문서는 자율 세션에서 Claude가 작성. 사용자 복귀 후 확인/수정 환영.*
*작성 시각: 2026-04-19 KST 02:40, Slurm 3773/3774 제출 직후.*

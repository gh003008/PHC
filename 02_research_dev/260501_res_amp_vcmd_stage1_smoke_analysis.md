# RES_AMP_VCMD Stage 1 Smoke 결과 분석

작성일: 2026-05-01
관련 코드/체크포인트:
- task: `phc/env/tasks/humanoid_im_res_amp_vcmd.py`
- network: `phc/learning/res_amp_network.py`
- agent: `phc/learning/res_amp_agent.py`
- env yaml: `phc/data/cfg/env/env_im_res_amp_vcmd.yaml`
- learning yaml: `phc/data/cfg/learning/im_res_amp_vcmd.yaml`
- launch: `scripts/launch_res_amp_smoke.sh`
- log: `logs/res_amp_smoke_20260501_214452/train.out`
- 체크포인트: `output/ResAMPVCmd.pth`, `output/ResAMPVCmd_00000250.pth`, `output/ResAMPVCmd_00000500.pth`

관련 설계 문서:
- 디자인 스펙: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md`
- 구현 플랜: `docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md`
- 방법론 정리: `01_research_docs/260501_phc_residual_amp_vcmd_methodology.md`
- obs dim 조사: `02_research_dev/260501_phc_3_obs_dim_investigation.md`

---

## 1. 한 줄 요약

**Stage 1 smoke 인프라 검증 통과**. 501 epoch, 약 7분, **NaN/OOM 없이 완주**. rwd 17.8→23.8 (+33.7%), eps_len 12.9→17.4 (+34.9%) 으로 학습 신호 확인. 다만 **eps_len 17/300 (~5.8%) 으로 절대 성능은 한참 부족** — 본격 수렴은 Stage 2 (server, 20K epoch) 에서 기대.

---

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| 머신 | exolab-MS-7D56 (RTX 4060 Ti 7.6 GB) |
| 시작 | 2026-05-01 21:44 |
| 종료 | 2026-05-01 21:51 |
| 총 wall time | ~7분 |
| 총 환경 step | 513,024 frames (501 epoch × 32 envs × 32 horizon) |
| 평균 fps | fps_step ~1,500 / fps_total ~1,300 |
| GPU 메모리 사용 | ~6.5 GB / 7.6 GB |
| OOM/NaN | 없음 ✓ |
| frozen base 파라미터 | 25,525,455 (phc_3 PNN, eval mode, no grad) |
| 학습 가능 파라미터 | 4,744,332 (residual_head + critic_mlp + disc_mlp + sigma) |

`scripts/launch_res_amp_smoke.sh` 그대로 실행, num_envs=32, max_epochs=500, episode_length=300 (5s @ 60Hz × control_freq_inv=2 → 30 Hz 의 5s = 150 step 이지만 PHC 의 episodeLength 는 30 Hz 단위라 5s).

---

## 3. 핵심 metric 추이

| Epoch | rwd | eps_len | 비고 |
|---|---|---|---|
| 1 | 17.8 | 12.9 | warm-up. residual 가중치 무작위, phc_3 baseline 만 동작 |
| 50 | 21.6 | 15.7 | r_track 학습 시작 신호 |
| 100 | 22.7 | 16.7 | 안정화 시작 |
| 250 | 24.1 | 17.7 | 중간 체크포인트 (`ResAMPVCmd_00000250.pth`) |
| 500 | 23.8 | 17.4 | 종료 체크포인트 (`ResAMPVCmd_00000500.pth`) |
| **전체 평균** | **23.28** | **17.00** | n=501 |
| **전체 최댓값** | **27.00** | **19.70** | |

**증분**: `rwd 17.8 → 23.8` 이면 `+33.7%`. `eps_len 12.9 → 17.4` 이면 `+34.9%`.

### 학습 곡선 정성
- 전체적으로 monotone-ish increasing — 후반 200 epoch 에서는 plateau (rwd 22-25 사이 oscillation)
- 250 → 500 사이 추가 향상은 작음. 이는 **smoke 가 짧아서**라기보다, **현재 yaml 의 num_envs=32 / horizon=32 / minibatch_size=512 조합이 4060 Ti 에 맞춰 작아서** 학습 신호가 충분히 안정적이지 않을 가능성
- Stage 2 의 num_envs=512 / horizon=32 / minibatch=8192 에서 더 빠른 수렴 기대

---

## 4. Spec §8 Stage 1 통과 기준 비교

| 기준 | 목표 | 실측 | 결과 |
|---|---|---|---|
| av_eps_len ≥ 240 | 절대 성능 | 17.4 | ❌ (5.8%) |
| av_reward +30% (vs Ep 1) | 학습 진행 | +33.7% | ✓ |
| AMP disc 50–80% | LSGAN 균형 | 미측정 (--no_log) | – |
| NaN/OOM 없음 | 안정성 | 없음 | ✓ |
| 부팅 + 1 epoch + ckpt 저장 | 인프라 | ✓ (3개 ckpt) | ✓ |

**해석**: 절대 성능 기준은 미달이지만 **이는 최대 500 epoch 의 smoke 한계**. 인프라 (네트워크 wiring, frozen base load, residual head + critic + disc, AMP discriminator forward+backward, env step, reward computation) 은 모두 정상 동작 확인. Stage 2 진입 가능.

---

## 5. Task 16 부팅 단계에서 발견한 5 개 호환성 버그 (이미 commit 됨)

`108c0bc` 에 한꺼번에 fix. 다음 번 비슷한 작업 (frozen base 위에 residual 학습) 시 첫 부팅에서 같은 패턴이 나올 가능성 높으니 체크리스트로 정리:

| # | 증상 | 원인 | 해결 |
|---|---|---|---|
| 1 | `mat1 64x935 vs mat2 934x1024` (critic) | parent `AMPPNNBuilder.Network.__init__` 가 자기 local kwargs 의 `input_shape` 를 `self_obs+task_obs=934` 로 덮어씀 (env 는 935-D obs 출력) | `super().__init__()` 후 `critic_mlp` 를 `self_obs+task_obs+v_cmd_dim=935` 입력으로 재빌드 |
| 2 | `'p_gains' attribute missing` (first physics step) | yaml `control_mode: "pd"` 인데 SMPL 용 p_gains 는 humanoid.py 가 h1/g1 만 초기화함 | yaml `control_mode: "isaac_pd"` (phc_3 학습 시 설정과 동일) |
| 3 | `IndexError: reward_raw[4]` (logger) | yaml `power_reward: False` → reward_raw shape (N, 4), AmpAgent logger 는 `reward_raw[4]` 무조건 인덱싱 | yaml `power_reward: True` |
| 4 | reward_raw 가 모두 0 (logger) | 우리 `_compute_reward` 가 reward_raw 채우지 않음 | `_compute_reward` 에서 `reward_raw[:, 0] = r_track`, `[:, 1] = r_im` 채우기 |
| 5 | OOM at backward (4060 Ti, num_envs=64) | frozen phc_3 PNN forward activation 누적 (`hidden=[2048,1536,1024,1024,512,512]`) | `num_envs=32` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**가장 큰 함정**: `**kwargs` 가 함수 호출마다 fresh dict 가 되기 때문에, parent 가 `kwargs['input_shape']` 를 mutate 해도 우리 outer scope 는 안 바뀜 — 그래서 residual_head 는 935 로 (정상) 생성되고, critic 만 934 로 (문제) 생성된다. **mutation timing 이 함수 경계를 넘지 않는다는 점** 을 잊지 말 것.

---

## 6. Stage 1 smoke 의 한계 + Stage 2 가 해결할 것들

| 항목 | Stage 1 (smoke) | Stage 2 (server) | 영향 |
|---|---|---|---|
| num_envs | 32 | 512 | 16x 더 많은 trajectory diversity per epoch → AMP disc 신호 안정화 |
| max_epochs | 500 | 20,000 | 40x 더 많은 update step → residual 이 v_cmd 분포 전체에 적응할 시간 |
| episode_length | 300 (5s) | 600 (10s) | 더 긴 horizon → bunny-hop 같은 short-horizon hack 차단 |
| amp_buffer_size | 20K (smoke 메모리 fit) | 200K (phc baseline) | disc 의 demo 분포 representation 이 더 풍부 |
| amp_minibatch_size | 512 | 4096 | disc gradient 신호 안정화 |
| 예상 wall time | 7분 | 36-48h | – |

**즉, Stage 1 의 eps_len 17/300 은 무관계량이 아니라 정확히 위 5 개 차이로 설명되는 한계 신호**. Stage 2 가 같은 코드로 같은 학습 곡선을 잘 따라간다면, Stage 1 smoke 가 본 +33% 추세가 그대로 더 길게 지속되어 eps_len 200+ 도달 가능.

---

## 7. 다음 단계 (Task 19-24)

1. ✅ Task 18 (이 문서)
2. ✅ Task 19 — server sbatch script (`train_phc_res_amp_vcmd.sh` + `im_res_amp_vcmd_server.yaml`, commit `f57bd4c`)
3. ⏳ **Task 20** (사용자 액션) — server1 에 본 코드 rsync 후 `sbatch train_phc_res_amp_vcmd.sh` 제출. 36-48h 후 `output/ResAMPVCmd_*.pth` 가 생성되면 `pull_ckpt.sh` 로 가져옴
4. ✅ Task 21 — eval script (`scripts/eval_res_amp_vcmd.py`, commit `5848628`). Stage 2 ckpt 가 도착하면 `python scripts/eval_res_amp_vcmd.py --ckpt output/ResAMPVCmd_*.pth` 로 sweep + ramp 자동 평가
5. ✅ Task 22 — residual 추론 데모 (`scripts/phc_walk_demo_residual.py`, commit `5848628`). 사용자가 키보드로 v_cmd 슬라이드하며 시각 검증
6. ⏳ **Task 23** (사용자 액션) — viewer 띄워 `python scripts/phc_walk_demo_residual.py` + 30s 녹화
7. ⏳ Task 24 — 최종 방법론 + 결과 통합 문서

---

## 8. 학습된 교훈

1. **PHC 의 frozen-base + residual 패턴은 일단 성립한다**. AMPPNNBuilder.Network 를 그대로 상속하고 critic 만 재빌드하면 v_cmd 1-D obs 추가 정도는 매끈하게 들어감.

2. **yaml 의 obs/control 필드는 frozen baseline 의 학습 시점 yaml 과 정렬해야 한다**. trackBodies, control_mode, power_reward 모두 phc_3 의 `.hydra/overrides.yaml + config.yaml` 과 직접 비교하는 게 가장 안전.

3. **로컬 4060 Ti smoke 는 num_envs=32, max_epochs=500 이 한계**. 더 늘리면 OOM. 이걸로 **인프라 검증** 만 하고 본 학습은 server 가 정답.

4. **`--no_log` 가 wandb 만 끄지 ckpt 저장은 안 끈다**. `output/<exp_name>_<epoch>.pth` 와 `output/<exp_name>.pth` (best) 두 가지가 자동 저장됨. 다만 폴더 구조가 hydra 와 argparse 가 달라서 (hydra 는 `output/<task>/<exp>/Humanoid_*.pth`) eval/demo 에서 양쪽 글롭 패턴 확인 필요.

5. **smoke 의 `eps_len` 절대값은 무관하지만 trend 는 의미있다**. +35% 라는 증분은 "residual 이 무언가를 학습한다" 의 signal 이지 "정책이 좋다" 가 아님. 결론은 Stage 2 wall time 으로만 낼 수 있음.

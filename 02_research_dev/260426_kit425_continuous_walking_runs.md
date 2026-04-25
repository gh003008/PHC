# KIT_425 Continuous Walking — 3-Slot Job Submission

Submitted: 2026-04-26
Spec: `docs/superpowers/specs/2026-04-26-kit425-continuous-walking-design.md`
Plan: `docs/superpowers/plans/2026-04-26-kit425-continuous-walking-plan.md`

## 제출된 학습

| Slot | Job ID | GPU | Started (KST) | Expected finish | Sbatch |
|---|---|---|---|---|---|
| S1 TERM_OFF | 3954 | idx0 | 2026-04-26 ~01:20 | +24-36h | `train_S1_termoff_gpu0.sh` |
| S2 CYCLE_BASE | 3955 | idx1 | 2026-04-26 ~01:20 | +24-36h | `train_S2_cyclebase_gpu1.sh` |
| S3 CYCLE_VCMD_HOP | 3956 | idx2 | 2026-04-26 ~01:20 | +24-36h | `train_S3_vcmdhop_gpu2.sh` |

**참고**: 첫 제출 (3951/3952/3953) 은 motion library multiprocessing 의 `received 0 items of ancdata` 에러로 startup 직후 실패.
Fix commit `b95f87a` (`mp.set_sharing_strategy('file_descriptor')` → `'file_system'`) 적용 후 재제출 (3954/3955/3956) 에서 정상 학습 진행.

## 코드 변경사항 (이번 batch 에서 origin/Jimin 에 push)

| Commit | 내용 |
|---|---|
| `f778b3f` | docs: design spec |
| `fd0633b` | docs: implementation plan |
| `c6edfbb` | exp: 3-slot env configs |
| `ba3b18d` | feat: cycle-boundary v_cmd resampling (multiclip) |
| `cdbe8ce` | exp: sbatch scripts |
| `b95f87a` | fix: motion_lib sharing strategy `file_system` |

## 학습 setup 요약 (3 슬롯 공통)

- num_envs: 512
- max_epochs: 20000
- v_cmd: continuous [0.40, 0.82] m/s
- KIT_425 motion clip 만 (`amass_isaac_walking_primitive.pkl` + `_kit425_only.json` 필터)
- VIC stage 2 (CCF 학습 ON, sigma_init=-1.0, 4 groups)
- Reward curriculum: switch @ Ep 10000 (task 0.7→0.3, disc 0.3→0.7)

## Slot 별 차이 (B canonical 대비)

| 항목 | B canonical (이전) | S1 | S2 | S3 |
|---|---|---|---|---|
| terminationDistance | 0.25 | **100.0** | **100.0** | **100.0** |
| fallInitProb | 0.3 | **0.0** | **0.0** | **0.0** |
| hybridInitProb | 0.5 | **0.0** | **0.0** | **0.0** |
| cycle_motion | False | False | **True** | **True** |
| episode_length | 300 | 300 | **3000** | **3000** |
| multiclip_resample_on_cycle | n/a | False | False | **True** |

## 학습 후 평가 계획

1. `rsync` 체크포인트: `KIT425_CONT_{S1,S2,S3}.pth` (canonical + 의미있는 milestone)
2. `scripts/test_termination_stats.py --slot S1/S2/S3` 로 termination breakdown
3. B canonical baseline (98.1% drift / 1.9% success) 와 비교
4. S2/S3 가 ceiling 도달하면 다음 milestone: keyboard demo wrapper

## 관찰 시작 (모니터링 메모)

- t = 3 분 시점 S1 진행: Ep 30~37, eps_len 88-111 (ceiling 137 의 64-81%) — 이미 B canonical canonical greedy 평균 (80) 와 비슷한 수준의 stochastic eps_len. 학습 초반에 이 정도면 좋은 신호.
- S2/S3 는 cycle_motion=True 로 episode 가 길어서 첫 Ep 도달이 느림. 정상.

추가 관찰은 학습 진행하면서 `tail -F ~/PHC/logs/kit425_cont_*.out` 로.

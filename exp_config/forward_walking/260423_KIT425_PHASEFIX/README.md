# KIT_425 Phase-Fix Experiment (2026-04-23)

KIT_425 피험자 3-clip 으로 multi-clip policy 학습. 2×2 factorial 의 3 cell 실행.

## 슬롯 구성

| Slot | experiment_name | yaml                                  | 특이사항 |
|------|-----------------|---------------------------------------|----------|
| #1   | KIT425_MAIN     | env_im_walk_vic_kit425_main.yaml      | cycle=False, phase_obs=True  (메인) |
| #2   | KIT425_NOPHASE  | env_im_walk_vic_kit425_nophase.yaml   | cycle=False, phase_obs=False (phase obs ablation) |
| #3   | KIT425_LEGACY   | env_im_walk_vic_kit425_legacy.yaml    | cycle=True,  phase_obs=True  (legacy wrap 비교) |

공통: KIT_425 3 clip (`|v_x|` ∈ {0.371, 0.652, 0.845} m/s), per-clip phase fix 코드 (`humanoid_im_vic._precompute_gait_phase` per-clip 리팩터), NR variant, v_cmd [0.40, 0.82], 20k epoch, 512 envs, `cmd_tracking_w: 0.0`.

Wandb project: `PHC_Walk_KIT425_PHASEFIX` (공유 — 다만 `--no_log` 이 기본이므로 wandb 실제 활성화는 sbatch 수정 필요).

## 실행

```bash
# Training (병렬, ~각 12-16h)
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_main_gpu0.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_nophase_gpu1.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_legacy_gpu2.sh

# Test-greedy (각 ~2-3분)
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_main.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_nophase.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_legacy.sh
```

## 평가 Metric (단일 정량값 안 만듦, 병렬 보고)

- `av_steps`          (raw, 평균 에피소드 길이, frames)
- `av_time_s`         (`av_steps × dt`, dt≈0.0333)
- fall-free rate      (fall 아닌 이유로 종료된 에피소드 비율)
- clip_end rate       (cycle=False 에서 자연 clip 종료 비율)
- `max_possible_avg`  (cycle=False 시 clip 평균 길이, 참고값: ≈ 5.38s × fps ≈ 161 frames)

Termination reasons (`humanoid_im._last_reset_reason` + `im_amp_players` 집계 출력):
0=no reset, 1=fall, 2=clip_end, 3=max_episode.

## 주의

- `motion_file` 은 50-clip `primitive.pkl` 그대로. 필터링은 task 코드의 `_clip_eligible` 마스크가 수행 (KIT_425 3 clip 만 retrievable).
- Slot 2 는 `vic_phase_obs=False` 로 obs dim 이 2 작음 → 학습 ckpt 는 Slot 1/3 와 호환 안 됨 (fresh train 전용).
- Slot 3 는 cycle=True 라 episode 가 clip_end 에서 종료되지 않고 wrap. Fall 과 wrap 가 섞이므로 fall-free rate 해석에 주의.

## 참고 문서

- Plan: `02_research_dev/260423_kit425_phase_fix_plan.md`
- Results (TBD): `02_research_dev/YYMMDD_kit425_phasefix_results.md`
- Upstream: `02_research_dev/260423_multiclip_fwd_analysis_and_next_steps.md`

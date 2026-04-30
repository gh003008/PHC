# 260430 — S18 / S19 / S19B 셋업 및 중간 결과

## 배경

S16 (= S14 base + knee_angle_reward, k=20, w=0.3, mean over 6 DOFs) 학습 결과를 영상으로 확인했더니 무릎이 여전히 안 굽혀지고 다리가 펴진 채 미끄러지듯 걷는 문제가 그대로였다.

원인 분석:
- knee_angle_reward의 per-DOF 유효 가중치 ≈ 0.05 (w=0.3 / 6 DOFs)
- 일반적인 stretched-knee err 0.3 rad → reward 기여 ~0.05 → 전체 reward (~30+) 대비 1% 미만
- foot_pos_reward (w=0.3, k=30)가 발 위치만 맞추면 만점 → 무릎 굽힐 인센티브 없음
- 결과: knee 신호가 reward landscape에서 noise 수준으로 묻힘

## 실험 설계 (S18/S19/S19B)

| 항목 | S16 (base) | S18 (A) | 옛 S19 (A+C, 망) | **S19B (Aggressive A)** |
|---|---|---|---|---|
| knee_k | 20 | **60** | 60 | **120** |
| knee_w | 0.3 | 0.3 | **0.5** | **0.5** |
| foot_pos_w | 0.3 | 0.3 | **0.1** ← 망 원인 | **0.3** ← 복원 |
| reward_curriculum_switch_epoch | 10000 | **100000** | 100000 | 100000 |

- 모두 num_envs=256, max_epochs=20000, multiclip_resample_on_cycle=False
- curriculum_switch=100000 → max_epochs 안에 stage 2 안 들어감 → 항상 task 0.7 / disc 0.3 (tighter imitation)
- save_frequency 100→500 (서버 디스크 절감)

## 옛 S19 실패 원인

S19는 Fix A (knee_k 60) + Fix C (foot_pos_w 0.3→0.1) 동시 적용했는데, foot_pos를 떨어뜨리니 발 추적 자체가 약해져서 보행 자체가 망가짐. ep 9500 영상 확인 → 무릎은 여전히 안 굽혀지고 거기다 발도 잘못 찍는 더 안 좋은 결과.

→ S19 (job 4039) 스캔셀, S19B로 대체.

## S19B 설계 의도

옛 S19의 실패 = "발 추적 약화의 부작용". 그래서 S19B는:
- foot_pos는 그대로 유지 (0.3) — 발 위치 안정성 보존
- knee 신호만 더 세게: k 60→120 (4x harder vs S16), w 0.3→0.5
- knee err 0.3 rad → reward ~1e-5 (= 거의 0) → 펴진 무릎 페널티 가혹
- knee err 0.1 rad → reward 0.150 (= 비슷한 발 페널티 기여 0.22와 균형)

## 자동 스케줄 셋업 (06:00 4/30)

서버에 `at` 명령이 없어서 sbatch 자체로 스케줄링:
- S18 sbatch (idx0): `--begin=2026-04-30T06:00:00`
- S19 sbatch (idx2): `--begin=2026-04-30T06:00:00`
- 스캔셀 sbatch (idx1): `--begin=2026-04-30T06:00:00` 으로 S16/S17 이름으로 죽이는 5초짜리 잡

06:00에 세 잡 동시 트리거 → S16(4031)/S17(4032) 종료 → idx0/idx2 free → S18(4038)/S19(4039) 잡힘.

S19 영상 확인 후 S19B(4041) 로 11:53경 수동 교체.

## 중간 결과 (4/30 12:00 시점)

| Slot | Job | Ep | rwd | eps_len | 비고 |
|---|---|---|---|---|---|
| S18 | 4038 | 14340 | 349 | 164 | idx0, 6h+ |
| S19B | 4041 | 3665 | 235 | 118 | idx2, 막 시작 (~10분) |

S18은 ep 14000+ 에서 안정 학습, eps_len 164 (max 300) — 절반 이상 생존.
S19B는 epoch 0부터 새로 시작 (S19 체크포인트 안 이어받음), 초기 학습 단계라 비교 무의미.

## 영상 (4 envs, 15s @ 30fps)

- `videos/VIC4_VCMD_S18_ep9000.mp4` — S18 ep 9000 (3.1 MB)
- `videos/VIC4_VCMD_S19_ep9500.mp4` — 옛 S19 ep 9500 (망인 거 확인용, 3.1 MB)

S19B 영상은 ep 9000 부근 도달 시점에 추가 예정.

## 다음 마일스톤

- ep 20000 도달 또는 walltime 36h 컷 (대략 5/1 18:00)
- 영상 비교 우선순위: S18 (현재 진행 중) ↔ S19B (현재 학습 시작) 같은 epoch 매칭 후 무릎 굽힘 양상 비교
- S18에서 무릎이 어느 정도 굽혀지는지가 1차 평가 기준 (k=60도 충분한지)
- S19B와 S18 차이 = "k 60 vs 120 + w 0.3 vs 0.5"의 ablation

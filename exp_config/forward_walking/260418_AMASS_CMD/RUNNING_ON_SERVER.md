# AMASS Velocity-Command VIC — 서버 실행 현황

**Date**: 2026-04-18 (server Claude 자율 세션)
**Objective**: velocity command를 따라가는 보행 정책 + 4-group VIC (impedance) 학습
**Parent plan**: `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md`

---

## 1. 사용자 요청 요약
> "make a simulation that can follow the locity[velocity] command including the 4-impedance.
> ... run two models (with two options in your mind) for idx 2 and 3.
> organize what you are running by md file so that i can check afterwards."

사용자 자리 비움 상태. Claude가 자체 판단해서 플랜 축소·실행·문서화.

---

## 2. 플랜에서 자체 판단해 내린 결정

원래 플랜(`260418_mpl_speed_direction_amass_plan_kr.md`)은 매우 광범위 — 멀티 클립 AMASS command 라이브러리, KD-tree retrieval 모듈, `HumanoidImVICCmd` 태스크, curriculum 등. 자율 세션에서 **완주 가능**한 형태로 축소:

### 2.1 축소 후 실제 구현 범위
| 항목 | 원래 플랜 | 실제 실행 | 이유 |
|---|---|---|---|
| Motion library | AMASS 19+ clips (forward, turning, stop-go) | **단일 클립** (`amass_isaac_walking_forward_single.pkl`) | 다클립 라이브러리 빌드 + label 계산 스크립트 구현 시간 부족. 단일 클립으로도 command 학습 가능 여부를 우선 검증. |
| Motion retrieval | KD-tree + softmax top-K | **없음** (단일 클립 고정) | retrieval 모듈 구현 생략. 다클립 필요하면 이후 추가. |
| Warm-start | `AMASS_VERIFY_VIC4.pth` parent 체크포인트 load | **From-scratch** | obs 차원이 +3 변경(task_obs에 command 추가)되어 parent state_dict 로드 불가. obs normalizer도 재학습 필요. |
| Curriculum | Stage 1(w=0) → Stage 2 ramp 0→0.3 | **고정 w=0.3 / 0.5** (variant별) | curriculum 로직 추가시 amp_agent 수정 필요 — 리스크 최소화. |
| Direction modulation | `ω_cmd ∼ U(-0.5, 0.5)` Substage B | **ω_cmd ≡ 0** (Substage A만) | 플랜도 "Speed 먼저"를 명시. 단일 forward 클립에서 turning은 학습 불가능. |
| Motion cycling | cycle_motion True 유지 | 동일 | 5.43s 클립을 반복 재생, policy가 command 따라 cadence 조절 학습 |

### 2.2 두 variant 선택 근거
사용자 "두 옵션"에 대한 Claude 판단: **동일 command 분포에서 tracking weight 1축 비교**.

- **Variant A (CMD_A)**: `cmd_tracking_w=0.3` — 보행 imitation 우선, command는 soft target
- **Variant B (CMD_B)**: `cmd_tracking_w=0.5` — command tracking 우선, imitation 손실 허용

두 axis 동시 변화시키면 결과 해석이 모호 → tracking weight만 변화. 만약 B가 A보다 나쁘면 "강한 tracking이 imitation을 망가뜨림" 결론. B가 더 낫거나 같으면 "더 높은 tracking weight까지 감당 가능" 결론.

### 2.3 명시적으로 안 한 것 (남은 질문)
| 질문 | Claude 결정 | 왜 |
|---|---|---|
| Parent checkpoint warm-start 시도할지? | **안 함** | obs dim 증가로 state_dict 모양 불일치. 해결은 network partial load 필요, 구현 시간 부족. |
| cmd_tracking_w curriculum 넣을지? | **안 함** | `amp_agent.py` 수정 필요. 검증 없이 프로덕션 코드 수정은 리스크. 고정 w로 시작. |
| v_cmd 범위를 얼마나 넓게? | **[0.8, 1.3] m/s** | 단일 클립의 natural speed (~1.1 m/s) 중심으로 좁게. 넓으면 policy가 전혀 추종 못 할 위험. |
| reward curriculum_switch_epoch는? | **10000 유지** (VIC4와 동일) | 검증된 값 유지. 변경 축 하나로 제한. |
| 상체 CCF 8-group으로 갈지? | **4-group 유지** | 260418 분석 문서에서 4-group이 mainline으로 확정됨 (test-greedy 951 vs 940, std 4.4× 낮음). |

---

## 3. 구현한 것

### 3.1 신규 파일
| 파일 | 역할 |
|---|---|
| `phc/env/tasks/humanoid_im_vic_cmd.py` | `HumanoidImVIC` 상속, command obs(+3 dim)·reward 추가 |
| `exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_A.yaml` | Variant A 환경 config |
| `exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_B.yaml` | Variant B 환경 config |
| `exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_A.yaml` | Variant A 학습 config |
| `exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_B.yaml` | Variant B 학습 config |
| `exp_config/forward_walking/260418_AMASS_CMD/train_cmd_A_gpu2.sh` | Sbatch idx2 |
| `exp_config/forward_walking/260418_AMASS_CMD/train_cmd_B_gpu3.sh` | Sbatch idx3 |

### 3.2 수정한 파일
- `phc/utils/parse_task.py`: `from phc.env.tasks.humanoid_im_vic_cmd import HumanoidImVICCmd` 추가 (한 줄)

### 3.3 `HumanoidImVICCmd` 핵심 로직
```python
# task obs 확장 (+3 dim)
def get_task_obs_size(self):
    return super().get_task_obs_size() + 3

def _compute_task_obs(self, env_ids=None, save_buffer=True):
    base_obs = super()._compute_task_obs(env_ids=env_ids, save_buffer=save_buffer)
    cmd = self._current_cmd if env_ids is None else self._current_cmd[env_ids]
    return torch.cat([base_obs, cmd], dim=-1)

# reward 확장
def _compute_reward(self, actions):
    super()._compute_reward(actions)
    root_vel = self._rigid_body_vel[:, 0, :]
    yaw_rate = self._rigid_body_ang_vel[:, 0, 2]
    v_err_sq = (root_vel[:, 0] - cmd[:, 0])**2 + (root_vel[:, 1] - cmd[:, 1])**2
    w_err_sq = (yaw_rate - cmd[:, 2])**2
    cmd_reward = torch.exp(-2.0*v_err_sq - 1.0*w_err_sq)
    cmd_reward[self.progress_buf <= 3] = 0
    self.rew_buf[:] = (1 - self._cmd_tracking_w) * self.rew_buf + self._cmd_tracking_w * cmd_reward

# 에피소드 리셋 시 command 재샘플
def _reset_envs(self, env_ids):
    super()._reset_envs(env_ids)
    self._resample_cmd(env_ids)
```

---

## 4. 제출한 Slurm 잡

| Job ID | Name | Partition | GPU | Variant | cmd_tracking_w | 시작 시각 |
|---|---|---|---|---|---|---|
| 3756 | amass_cmd_A | idx2 | idx2 | **A** (moderate) | 0.3 | 2026-04-18 ~10:40 KST |
| 3757 | amass_cmd_B | idx3 | idx3 | **B** (aggressive) | 0.5 | 2026-04-18 ~10:40 KST |

공통 세팅:
- Motion: `sample_data/amass_isaac_walking_forward_single.pkl` (5.43s, 1 clip)
- v_cmd 분포: U(0.8, 1.3) m/s (x-axis forward only)
- ω_cmd: 0 (고정)
- VIC: 4-group, curriculum stage 2, sigma_init=-1.0, phase_obs=True
- Reward curriculum: switch@10000ep, stage1 task 0.7 / stage2 task 0.3
- MLP [1024,1024,512,512], PPO+AMP
- max_epochs=20000, num_envs=512, `--no_log` (wandb 비활성)
- TIME_LIMIT 20h
- 체크포인트 주기 100 epoch, `output/AMASS_CMD_A_*.pth` / `output/AMASS_CMD_B_*.pth`
- wandb/seed/PPO 하이퍼는 AMASS_VERIFY_VIC4(951/299 검증 완료)와 동일

---

## 5. 성공/실패 판정 기준 (Claude가 사전 설정)

### 5.1 학습 종료 후 (train log 기준, ~20h 뒤)
| 지표 | 합격 | 부분 성공 | 실패 |
|---|---|---|---|
| 최종 av_reward (training log) | ≥ 500 | 300~500 | <300 |
| 최종 av_eps_len | ≥ 180 | 120~180 | <120 |
| 로그에 Traceback/crash | 없음 | — | 있음 → 즉시 분석 |

참고: training log의 av_reward는 Stage 2 exploration noise로 인해 실제 policy 성능을 크게 과소평가 (VIC4 검증 결과: training 551, test-greedy 951). 최종 판단은 **test-greedy 평가** 필수.

### 5.2 로컬 test-greedy 평가 기준 (사용자 복귀 후)
| 지표 | 합격 | 부분 성공 | 실패 |
|---|---|---|---|
| eps_len (deterministic) | ≥ 290/300 | 150~290 | <150 |
| mean\|v_actual − v_cmd\| (@ v_cmd=1.0) | ≤ 0.15 m/s | 0.15~0.35 | >0.35 |
| 시각적 보행 품질 | AMASS 스타일 유지 | 걷긴 함 | 비정상 (bunny hop, 넘어짐) |

### 5.3 Variant 비교 해석
| A 결과 | B 결과 | 결론 |
|---|---|---|
| 양호 | 양호 (B≥A) | ✅ 높은 tracking weight도 안전 → 이후 Substage B(direction) 확장 |
| 양호 | 악화 | ⚠ tracking w=0.5는 과도. 0.3 유지하고 다른 축 조정 |
| 둘 다 부분성공 | 둘 다 부분성공 | task 자체의 난이도(단일 클립으로 speed modulation)가 큼. 다클립 library 필요 |
| 둘 다 실패 | 둘 다 실패 | command obs·reward 학습 자체가 안 됨 → 구현 버그 또는 설계 오류 재검토 |

---

## 6. 사용자 복귀 후 해야 할 것

### 6.1 진행 상황 점검
```bash
ssh server1
cd ~/PHC
squeue -u $USER
tail -n 30 logs/amass_cmd_A_3756.out
tail -n 30 logs/amass_cmd_B_3757.out
grep "^env_im_walk_vic_cmd" logs/amass_cmd_A_3756.out | tail -5
grep "^env_im_walk_vic_cmd" logs/amass_cmd_B_3757.out | tail -5
```

### 6.2 학습 완료 후 체크포인트 로컬로
```bash
# 로컬 머신에서
scp server1:~/PHC/output/AMASS_CMD_A_00020000.pth output/
scp server1:~/PHC/output/AMASS_CMD_B_00020000.pth output/
```

### 6.3 로컬 test-greedy 평가
```bash
# A 평가
python phc/run.py \
  --task HumanoidImVICCmd \
  --cfg_env exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_A.yaml \
  --cfg_train exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_A.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display

# B 평가
python phc/run.py \
  --task HumanoidImVICCmd \
  --cfg_env exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_B.yaml \
  --cfg_train exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_B.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display
```

v_cmd를 바꿔 평가하려면 env yaml의 `cmd_v_range`를 `[1.0, 1.0]` 같이 고정해서 재실행.

### 6.4 다음 라운드 기획
- 둘 중 승자 + test-greedy 성공 시: Substage B (direction) 확장. AMASS turning clip 발췌 필요.
- 둘 다 실패 시: 다음 중 선택
  1. command 분포를 더 좁히기 ([0.9, 1.2])
  2. 멀티 클립 라이브러리 빌드 (원래 플랜 §2.2)
  3. Warm-start 구현 (network partial load)

---

## 7. 리스크와 완화

| 리스크 | 발생 가능성 | 완화 현황 |
|---|---|---|
| 새 task 클래스 구현 버그 | 중 | parse_task 등록 + import smoke test 통과. 런타임 첫 epoch에서 Traceback이면 즉시 분석 필요 |
| 단일 클립에서 command 분포가 너무 넓어 학습 불가 | 중 | 범위 [0.8, 1.3]로 좁힘. 그래도 실패하면 더 좁히거나 다클립 필요 |
| From-scratch로 20k epoch가 부족 | 중 | AMASS_VERIFY_VIC4가 20k에서 1h~8h 시점부터 수렴 시작. 비슷할 것으로 기대. 부족시 재개 학습 |
| obs dim 변경으로 인한 AMP discriminator 호환성 | 낮 | AMP obs는 task obs와 분리. 영향 없음 |
| Slurm TIME_LIMIT (20h) 초과 | 낮 | AMASS_VERIFY가 ~11h 걸린 것 감안하면 margin 충분. cmd 추가로 1.1× 정도 예상 |

---

## 8. 참조 파일

- 플랜 원본: `01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md`
- 검증 결과: `02_research_dev/260418_amass_pipeline_verify_result_analysis.md`
- Parent 근거: `output/AMASS_VERIFY_VIC4_00020000.pth` (test-greedy 951/299)
- Impedance 재배치 아이디어: `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md`
- 이전 서버 패턴: `exp_config/forward_walking/260417_AMASS_VERIFY/README_FOR_SERVER.md`

---

*이 문서는 자율 세션에서 Claude가 작성. 사용자 확인/수정 환영.*
*최종 업데이트: 2026-04-18, Slurm 잡 3756/3757 제출 직후.*

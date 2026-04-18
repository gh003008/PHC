# MPL Speed/Direction Modulation on AMASS — 실행 계획
**Date**: 2026-04-18
**Author session**: Local Claude (exolab-MS-7D56) → 서버 Claude (server1) 전달용
**범위**: Speed + direction 명령 조건화 + AMP 기반 개인 특성 보존, **AMASS 데이터만 사용 (H5 유보)**

---

## 0. 배경 & 핵심 근거 문서

이 계획은 다음 두 문서의 정식 제안을 **AMASS 데이터 범위로 축소**한 실행 플랜이다.

- `01_research_docs/260413_MPL_methodology_refined.md` — MPL 통합 방법론 (α, 모션 라이브러리, AMP, 개인화)
- `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md` — Impedance를 residual/gated 모듈로 재배치
- `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` — AMASS 파이프라인 검증 결과 (VIC4=951/299 확정)

H5 데이터 수정은 **본 계획에서 다루지 않는다**. AMASS forward_single + (필요 시) AMASS multi-speed/multi-direction 클립 서브셋으로 진행한다.

---

## 1. 문제 정의 & 목표

### 1.1 구체 목표 (사용자 요청 재구성)
1. **Speed modulation**: 임의 CoM 전진 속도 명령 `v_cmd ∈ [0.5, 1.5] m/s`를 입력받아 그 속도로 걷는 정책
2. **Direction modulation**: 임의 heading yaw rate `ω_cmd ∈ [-0.5, 0.5] rad/s`를 입력받아 해당 방향으로 회전/주행하는 정책
3. **Personal characteristics 보존**: 학습된 정책의 모션이 **참조 데이터셋의 스타일**을 유지 (어깨 흔들림, 보폭 타이밍, 상체 기울기 등). AMP discriminator + 모션 라이브러리 검색으로 달성

### 1.2 실행 순서 (Speed-First)
사용자 결정(260418): **Speed 먼저, Direction 이후**. 같은 코드베이스에서 command 분포만 바꾸어 단계별 확장 (§5 참조):
1. **Substage A**: speed-only (`ω_cmd≡0`) — 우선
2. **Substage B**: +direction (`ω_cmd ∼ U(-0.5, 0.5)`) — A 성공 후
3. **Substage C**: stop-and-go / accel — B 성공 후

### 1.3 범위 밖 (이번 플랜에서 하지 않음)
- 외란 회복 (Phase 2, 260413 §5 perturbation curriculum) — 나중 단계
- α stability margin 계산 (260413 §3) — 나중 단계
- H5 데이터 재처리 — 유보
- 외골격 상호작용 (260413 §7) — 나중 단계

### 1.3 성공 기준 (수용/기각 임계값)
| 메트릭 | 기준 |
|---|---|
| Command-tracking: `mean │v_actual − v_cmd│` | ≤ 0.15 m/s |
| Command-tracking: `mean │ω_actual − ω_cmd│` | ≤ 0.1 rad/s |
| 정상 보행 eps_len (test-greedy) | ≥ 290 / 300 (97%) |
| AMP reward (평균) | ≥ baseline VIC4의 80% (즉 disc_reward 정상 학습 확인) |
| 시각적 검증 | Isaac Gym test 모드에서 걸음걸이가 "AMASS 같은" 리듬 유지 |

---

## 2. 데이터 준비

### 2.1 필요한 AMASS 클립 집합

현재 보유: `sample_data/amass_isaac_walking_forward_single.pkl` (1 clip, KIT 11 WalkingStraightForwards05, 5.43s) — forward only.

**추가 필요한 것**: 다양한 speed + direction 조합의 AMASS walking 클립 서브셋. 최소 요구 사항:

| 카테고리 | 권장 클립 수 | 속도 범위 | 방향 |
|---|---|---|---|
| Forward straight (slow 0.8 m/s) | ≥ 3 | 0.6–0.9 m/s | 0 rad |
| Forward straight (normal 1.1 m/s) | ≥ 5 | 0.9–1.3 m/s | 0 rad |
| Forward straight (fast 1.4 m/s) | ≥ 3 | 1.3–1.6 m/s | 0 rad |
| Turning left | ≥ 3 | 0.8–1.2 m/s | +0.2–0.5 rad/s |
| Turning right | ≥ 3 | 0.8–1.2 m/s | -0.5–-0.2 rad/s |
| Stop-and-go | ≥ 2 | 0–1.0 m/s transitions | 0 rad |
| **합계** | **≥ 19 clips** | | |

AMASS 원본 데이터셋에서 `CMU`, `KIT`, `BMLrub` 서브셋에 이 조건을 만족하는 클립이 다수 존재.

### 2.2 AMASS 클립 선정 & 조립 절차

**단계 1**: AMASS 메타데이터에서 walking 클립 필터링
- 기존 `sample_data/amass_isaac_walking_primitive.pkl`(17.8 MB, multiple clips) 내용 검사 — 이미 walking 다클립 세트일 가능성 있음
- 부족 시 `data/amass/` 원본에서 추가 발췌

**단계 2**: 각 클립에 command label 계산
- `v_cmd_i = mean(|| pelvis_xy_velocity ||)`
- `ω_cmd_i = mean(pelvis_yaw_rate)`
- `type_i ∈ {walk_straight, walk_turn_L, walk_turn_R, walk_stop_go}` (yaw rate 임계값 기반 자동 분류)

**단계 3**: `sample_data/amass_walking_commands.pkl` 저장 (AMASS 표준 dict 포맷 유지 + `metadata` 키 추가)

**산출 스크립트 (신규)**: `scripts/data/build_amass_command_library.py`
- 입력: AMASS raw pkl 또는 기존 primitive pkl
- 출력: `sample_data/amass_walking_commands.pkl` + `.meta.pkl`

---

## 3. 아키텍처 & 구현 설계

### 3.1 현재 → 목표 아키텍처 비교

**현재 (AMASS_VERIFY_VIC4)**:
```
obs = [proprio, phase, ref_motion_window]
action = [q_ref_12dim, ccf_4dim]
reward = 0.5 * task + 0.5 * disc + (phase의 ccf reward)
```

**목표 (AMASS_CMD_VIC4)**:
```
obs = [proprio, phase, ref_motion_window, command(v_cmd, ω_cmd)]  ← +3 dims
action = [q_ref_12dim, ccf_4dim]                                  ← 동일
reward = 0.5 * task_cmd + 0.5 * disc + ccf_reward
  where task_cmd = imitation_error + command_tracking_error
```

핵심 변경 2가지:
1. **Observation에 command vector 추가**: `(v_cmd_x, v_cmd_y, ω_cmd_yaw)` = 3 dim
2. **Command-conditioned motion retrieval + reward**:
   - 학습 시: 각 에피소드 시작 시 `(v_cmd, ω_cmd)` 샘플 → 해당 command에 가장 가까운 클립 top-K (K=3) 선택 → softmax 가중 평균을 reference motion으로 사용
   - Reward에 `r_track = exp(-k_v · |v_pelvis - v_cmd|² - k_ω · |ω_pelvis - ω_cmd|²)` 추가

### 3.2 코드 변경 범위 (최소 diff 원칙)

**신규 파일**:
| 경로 | 역할 |
|---|---|
| `scripts/data/build_amass_command_library.py` | AMASS 클립에 command label 붙이기 |
| `phc/utils/motion_retrieval_cmd.py` | Command 기반 top-K 검색 + softmax 보간 |
| `phc/env/tasks/humanoid_im_vic_cmd.py` | `HumanoidImVIC` 확장: command obs + command reward |
| `exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd.yaml` | 환경 config |
| `exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd.yaml` | 학습 config |
| `exp_config/forward_walking/260418_AMASS_CMD/train_cmd_gpu2.sh` | Sbatch 스크립트 |
| `exp_config/forward_walking/260418_AMASS_CMD/README_FOR_SERVER.md` | 서버 실행 가이드 |

**수정 필요 파일** (기존 PHC 스크립트 직접 수정 금지 규칙이 있으므로 `_cmd` 접미사로 복사):
| 원본 | 복사본 | 변경 포인트 |
|---|---|---|
| `phc/utils/parse_task.py` | — (수정) | `HumanoidImVICCmd` task 등록 |

### 3.3 주요 구현 세부사항

**① `phc/env/tasks/humanoid_im_vic_cmd.py`** (신규, `HumanoidImVIC` 상속)
```python
class HumanoidImVICCmd(HumanoidImVIC):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        self._command_dim = 3                       # vx, vy, ω
        self._obs_dim += self._command_dim
        self._cmd_v_range = cfg["env"].get("cmd_v_range", [0.5, 1.5])
        self._cmd_w_range = cfg["env"].get("cmd_w_range", [-0.5, 0.5])
        self._cmd_tracking_w = cfg["env"].get("cmd_tracking_w", 0.3)
        self._current_cmd = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids):
        # 에피소드 시작 시 command 샘플
        v = uniform(*self._cmd_v_range, (len(env_ids),))
        w = uniform(*self._cmd_w_range, (len(env_ids),))
        self._current_cmd[env_ids] = torch.stack([v, 0*v, w], dim=1)
        # Command로 top-K clip 검색 → motion_lib 동적 변경
        self._retrieve_by_command(env_ids)
        super().reset(env_ids)

    def _compute_obs(self):
        base_obs = super()._compute_obs()
        return torch.cat([base_obs, self._current_cmd], dim=-1)

    def _compute_reward(self, actions):
        base_reward = super()._compute_reward(actions)
        v_pelvis = self._humanoid_root_states[:, 7:10]
        w_pelvis_yaw = self._humanoid_root_states[:, 12]
        v_err = torch.norm(v_pelvis[:, :2] - self._current_cmd[:, :2], dim=-1)
        w_err = (w_pelvis_yaw - self._current_cmd[:, 2]).abs()
        cmd_reward = torch.exp(-2.0 * v_err**2 - 1.0 * w_err**2)
        return (1 - self._cmd_tracking_w) * base_reward + self._cmd_tracking_w * cmd_reward
```

**② `phc/utils/motion_retrieval_cmd.py`** (신규)
- 학습 시작 시 `amass_walking_commands.meta.pkl` 로드
- Command vector의 KD-tree 인덱스 구축
- `retrieve(cmd, k=3) → (clip_ids, weights)` 반환 (weights는 softmax(-dist/σ))
- Motion library는 여전히 `motion_lib_smpl.py` 사용하되, `sample_motions()`를 command-weighted로 override

**③ YAML config 핵심 변경**
```yaml
# env_im_walk_vic_cmd.yaml (신규 키만 발췌)
env:
  motion_file: sample_data/amass_walking_commands.pkl
  cmd_v_range: [0.5, 1.5]
  cmd_w_range: [-0.5, 0.5]
  cmd_tracking_w: 0.3
  cmd_retrieval_k: 3
  cmd_retrieval_sigma_v: 0.3
  cmd_retrieval_sigma_w: 0.2
  vic_ccf_num_groups: 4       # 260418 검증 결과 반영: 4-group mainline
  vic_curriculum_stage: 2
  vic_phase_obs: True
```

---

## 4. 학습 커리큘럼

### 4.1 Stage 1: Command 관찰만 추가 (warm-start, epoch 0–5000)
- Parent checkpoint: `AMASS_VERIFY_VIC4.pth` (951/299 baseline) load
- Command vector는 obs에 추가되지만 **reward는 기존 task=0.5/disc=0.5 유지** (cmd_tracking_w=0)
- Purpose: policy가 새 obs 차원에 적응하는 시간
- 기대: eps_len 유지 (≥ 290), reward ~900 유지

### 4.2 Stage 2: Command tracking 활성 (epoch 5000–15000)
- `cmd_tracking_w: 0.3` 점진 증가 (5k에서 0.1, 10k에서 0.2, 15k에서 0.3)
- Reward = 0.7 · task + 0.3 · cmd_track (+ disc 유지)
- 기대: v_err / ω_err 감소, eps_len 유지, reward 감소 허용 (tracking 비용)

### 4.3 Stage 3: AMP 스타일 강화 (epoch 15000–25000)
- `disc_reward_w` 점진 증가 0.5 → 0.7 (260413 §5 stage 3 원칙)
- Purpose: "personal characteristics" 보존 강화. policy가 command tracking을 위해 스타일을 망가뜨리지 않도록 밀어넣음
- 기대: disc_reward 상승, AMP discriminator 판별 어려워짐 (그만큼 real-like)

### 4.4 Total: epoch 25000 (기존 20000에서 +5k)

---

## 5. 실험 매트릭스 (Speed-First 스테이징)

### 5.0 기본 원칙: Speed 먼저 → Direction 확장
사용자 결정(260418): **Substage A (speed-only)**로 시작해서 파이프라인 검증 후 **Substage B (+direction)**로 확장. 구현은 처음부터 2D command를 지원하도록 하되, 초기 학습에서는 `ω_cmd` 분포를 0에 고정하여 speed 축만 학습.

### 5.1 Substage A — Speed-only (우선 실행)
Command 분포: `v_cmd ∼ U(0.5, 1.5) m/s`, `ω_cmd ≡ 0`
필요 AMASS: §2.1 표의 forward straight 3~5개 카테고리 (≥ 11 clips)

| Run name | parent | cmd_tracking_w | disc_w max | 목적 |
|---|---|---|---|---|
| **AMASS_CMD_A1** | AMASS_VERIFY_VIC4.pth | 0.3 | 0.5 | speed baseline |
| **AMASS_CMD_A2** | AMASS_VERIFY_VIC4.pth | 0.3 | 0.7 | speed + AMP 강화 |

**성공 판정**: `|v_actual − v_cmd| ≤ 0.15` m/s, eps_len ≥ 290 → Substage B 착수

### 5.2 Substage B — Speed + Direction (A 성공 후)
Command 분포: `v_cmd ∼ U(0.5, 1.5)`, `ω_cmd ∼ U(-0.5, 0.5)`
필요 AMASS: 추가로 turning L/R clips (≥ 6 clips)

| Run name | parent | 변경점 |
|---|---|---|
| **AMASS_CMD_B1** | A1 또는 A2 winner | `ω_cmd` 활성화 |
| **AMASS_CMD_B2** | A winner | + AMP discriminator 재학습 (turning 포함) |

### 5.3 Substage C — Others (B 성공 후, 이번 플랜 밖으로 넘길 수 있음)
- Stop-and-go transitions (`v_cmd` 급변)
- 다양한 가속도 프로필
- (멀리 나중) 외란, α, 개인화 — 별도 Phase

---

## 6. 서버 실행 절차

### 6.1 사전 준비 (로컬)
```bash
# 1. 본 계획 문서 및 신규 파일들을 git commit
git add 01_research_docs/260418_mpl_speed_direction_amass_plan_kr.md \
        scripts/data/build_amass_command_library.py \
        phc/utils/motion_retrieval_cmd.py \
        phc/env/tasks/humanoid_im_vic_cmd.py \
        phc/utils/parse_task.py \
        exp_config/forward_walking/260418_AMASS_CMD/
git commit -m "feat: AMASS command-conditioned MPL plan + skeleton"
git push origin Jimin

# 2. Parent checkpoint 확인
ls output/AMASS_VERIFY_VIC4.pth   # 필수

# 3. AMASS 클립 라이브러리 빌드 (아직 없으면)
python scripts/data/build_amass_command_library.py \
    --src sample_data/amass_isaac_walking_primitive.pkl \
    --out sample_data/amass_walking_commands.pkl

# 4. 서버로 전송 (gitignore된 pkl은 scp)
scp output/AMASS_VERIFY_VIC4.pth server1-jiminyoun:~/PHC/output/
scp sample_data/amass_walking_commands*.pkl server1-jiminyoun:~/PHC/sample_data/
```

### 6.2 서버에서 실행
```bash
cd ~/PHC
git pull origin Jimin

# A, B 병렬 제출 (GPU idx 2/3 희망)
sbatch exp_config/forward_walking/260418_AMASS_CMD/train_cmd_A_gpu2.sh
sbatch exp_config/forward_walking/260418_AMASS_CMD/train_cmd_B_gpu3.sh

squeue -u $USER
tail -f logs/amass_cmd_A_*.out
```

### 6.3 학습 시간 예상
- 25k epoch × 4~5s/epoch = ~30–35h per run (A, B 병렬이면 wall-clock ~35h)

---

## 7. 평가 & 분석

### 7.1 학습 종료 후 (로컬 pull)
```bash
scp server1-jiminyoun:~/PHC/output/AMASS_CMD_A.pth output/
scp server1-jiminyoun:~/PHC/output/AMASS_CMD_B.pth output/
```

### 7.2 평가 배터리 (로컬)
각 체크포인트에 대해:

**Test 1 — 고정 command 그리드**
```python
# 9점 그리드: (v_cmd, ω_cmd) ∈ {0.7, 1.0, 1.3} × {-0.3, 0, 0.3}
# 각 점에서 20 에피소드 × eps_len / cmd_tracking_error 기록
```

**Test 2 — Command sweep** (연속 변화)
- 에피소드 내에서 v_cmd를 0.6→1.4→0.6로 5초 주기 sweep
- 정책이 부드럽게 추종하는지 확인

**Test 3 — 시각 검증**
- Isaac Gym GUI로 각 command point 1회씩 재생
- "AMASS 스타일" 보존 여부 정성 평가

### 7.3 분석 문서
학습 완료 후: `02_research_dev/260420_amass_cmd_vic_result_analysis.md` (KR) 자동 작성

---

## 8. 결정 트리 (예상 결과별 후속)

| AMASS_CMD_A 결과 | AMASS_CMD_B 결과 | 다음 단계 |
|---|---|---|
| v_err ≤ 0.15, eps_len ≥ 290 | B ≥ A 스타일 | ✅ 성공. MPL Phase 2 (α + 외란)로 진입 |
| v_err > 0.3 또는 eps_len < 200 | — | ❌ command obs 학습 실패. cmd_tracking_w 스케줄 재설계 또는 reward scale 재조정 |
| A 양호 / B 불량 (disc_w=0.7이 망가뜨림) | — | ⚠ disc weight 0.7 과도. 0.6으로 회귀 |
| A=B 동일 | — | AMP 효과 미미. discriminator 아키텍처 재검토 |

---

## 9. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| AMASS에 충분한 turning 클립이 없음 | primitive.pkl 내용 부족 시 CMU AMASS subset 추가 발췌, 또는 straight-only부터 시작하여 turning을 Stage 2로 유예 |
| Command obs 추가로 parent policy 급격한 성능 붕괴 | Stage 1 warm-start (cmd_tracking_w=0)으로 완충. 필요 시 첫 1k epoch lr 1e-5로 더 낮춤 |
| Motion retrieval top-K softmax 보간이 quaternion 공간에서 불연속 | slerp 사용, K=3으로 낮게 유지, σ를 크게 (0.3 m/s) |
| AMP disc가 command에 무관하게 "느린 걸음"으로만 수렴 | amp_obs_demo_buffer에 command 분포 고루 샘플링 강제 |

---

## 10. 향후 확장 지점 (이번 플랜 밖)

- **Phase 2 (외란 회복)**: α 계산 + perturbation curriculum 추가 (260413 §5)
- **Phase 3 (Residual VIC)**: impedance head 분리 + gate (260416 §5.2)
- **Phase 4 (개인화)**: subject-specific AMASS 서브셋으로 fine-tune
- **Phase 5 (H5 복귀)**: H5 데이터 품질 수정 후 library 교체

---

## 11. 체크리스트 (착수 순서)

- [ ] `amass_isaac_walking_primitive.pkl` 내용 분석 — clip 수, 속도/방향 분포 파악
- [ ] (필요 시) AMASS 추가 클립 발췌 → `amass_walking_commands.pkl` 빌드
- [ ] `build_amass_command_library.py` 구현
- [ ] `motion_retrieval_cmd.py` 구현 (KD-tree + softmax)
- [ ] `humanoid_im_vic_cmd.py` 구현 (obs 확장 + reward 확장)
- [ ] `parse_task.py` 업데이트 (`HumanoidImVICCmd` 등록)
- [ ] env/learning yaml 작성
- [ ] Local에서 smoke test (num_envs=2, epoch=50)
- [ ] Sbatch 스크립트 & 서버 README 작성
- [ ] Git commit + push
- [ ] Checkpoint + pkl scp 서버
- [ ] 서버에서 A, B 병렬 제출
- [ ] 학습 진행 모니터링
- [ ] 완료 후 평가 + 분석 문서

---

## 12. 참고 문헌 & 내부 링크

### 내부 문서
- `01_research_docs/260413_MPL_methodology_refined.md` — 풀 MPL 설계
- `01_research_docs/260410_mpl_development_progress.md` — MPL Phase 1 (H5) 진행 현황
- `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md` — Impedance residual 재배치
- `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` — AMASS 검증 결과 (parent 근거)
- `exp_config/forward_walking/260417_AMASS_VERIFY/README_FOR_SERVER.md` — 서버 실행 패턴 참고

### 외부 근거
1. Peng et al., **AMP: Adversarial Motion Priors** (TOG 2021) — 스타일 보존 메커니즘
2. Tessler et al., **CALM: Conditional Adversarial Latent Models** (TOG 2023) — command-conditioned directable characters
3. Luo et al., **PHC: Perpetual Humanoid Control** (ICCV 2023) — 본 코드베이스 기반
4. Peng et al., **DeepMimic** (TOG 2018) — reference motion tracking
5. Won et al., **Physics-based Character Controllers Using Conditional VAEs** (TOG 2022) — command-conditioned motion latent

---

*최종 업데이트: 2026-04-18. 서버 전달 시 이 파일과 §11 체크리스트 내 구현 파일들을 함께 푸시할 것.*

# VIC 연구 현황 인수인계 (260311)

---

## 0. 한 줄 요약

> **PHC 기반 SMPL 휴머노이드에 Variable Impedance Control(VIC)을 구현하여, 정상 보행 모방 중 관절별 임피던스를 자율 학습시키고 있다. 최종 목표는 Exoskeleton 제어기 학습이다.**

---

## 1. 연구 배경 및 최종 목표

### 지금 하고 있는 것
PHC (Physics-based Humanoid Control) 프레임워크에 VIC를 추가하여, 보행 중 각 관절의 임피던스(stiffness, damping)를 policy가 자율적으로 조절하도록 학습시키고 있다.

기존 PHC의 PD 제어: `torque = kp * (q_target - q) - kd * dq`
VIC 추가 후: `torque = kp * 2^ccf * (q_target - q) - kd * 2^ccf * dq`

여기서 CCF (Compliance Control Factor)가 관절별 임피던스 배율이다. `ccf=0`이면 기존과 동일, `ccf>0`이면 더 rigid, `ccf<0`이면 더 compliant.

### 최종 목표
인간이 Exoskeleton을 착용한 시뮬레이션 환경에서 Exo 제어기를 RL로 학습시키는 것. VIC는 그 사전 단계로, "사람 자체가 임피던스를 어떻게 조절하는지" 모델링하기 위한 연구다.

### 시뮬레이터 한계 관련 최근 논의
Exo 시뮬레이션은 인간 body와 Exo body를 물리적으로 연결(weld constraint)해야 하는데, 현재 쓰는 **IsaacGym(PhysX 4)은 closed kinematic chain을 지원하지 않는다.** 이를 지원하는 옵션:
- **Isaac Lab(PhysX 5)**: GPU 병렬 유지 + closed-loop 지원. 마이그레이션 필요. 가장 현실적.
- **MuJoCo**: equality constraint로 검증됨. CPU only로 RL 스케일 제한.
- 현재 IsaacGym 유지 시: contact force 근사만 가능 → Exo attachment 표현 부적합.

---

## 2. 실험 히스토리

### 2-1. VIC 이전 (PHC V4)
- PHC 오리지널 baseline. forward walking 단일 모션.
- av_reward: ~461, av_steps: ~143 (4.8초)

### 2-2. VIC01~VIC10 시리즈 (단계적 개선)
핵심 개선 누적:
- Stage 1(CCF=0 고정) warm-up → Stage 2(CCF 학습 활성화) 전환 커리큘럼 설계
- CCF action grouping: per-DOF(69 dims) → 8 그룹으로 축소 (action 138→77)
- Reward weight curriculum 도입: epoch<threshold: task=0.7/disc=0.3, 이후 역전

### 2-3. VIC11 (260309, 최고점 도달)
**설정:** 8그룹 CCF, Stage 1 (CCF=0 고정), reward curriculum(7500 epoch 전환), 25000 epochs
**결과:** av_reward=947.11, av_steps=300.1 (10초, 에피소드 최대 도달)
V4 대비 2배 이상 성능 향상. 단, CCF를 아직 활성화하지 않은 상태.
**한계:** cycle_motion=True인 단발 walk-stop 모션 반복 시 2번째 사이클부터 velocity 불연속으로 bunny hop 발생.

### 2-4. VIC_CCF_ON (260310, CCF 최초 활성화)
**설정:** vic_curriculum_stage=2, sigma=-2.9 전체 공유, 20000 epochs
**결과:** av_reward=945.30, av_steps=297.98
**핵심 발견:** CCF가 실질적으로 학습되지 않았다. sigma=-2.9(std=0.055) 공유 시 CCF 탐색 범위 ±0.11 → impedance_scale [0.93, 1.08]. 이 범위에서 reward gradient 차이가 noise에 묻힌다.
체크포인트 직접 분석 결과: 모든 CCF bias ≈ 0, impedance_scale ≈ 1.0x.

### 2-5. VIC_CCF_ON2 (260310~260311, 현재 완료)
**핵심 변경:** CCF 8 dims에만 별도 sigma=-1.0 (std=0.37) 적용. PD dims sigma=-2.9 유지.
구현 위치: `amp_agent.py`의 `pre_epoch()` 최초 1회 실행으로 sigma override.
**결과:** av_reward=939.51, av_steps=297.39 (성능 유지)
**핵심 발견: CCF가 실제로 학습됐다!**

| 그룹 | CCF_ON (이전) | CCF_ON2 (이번) |
|---|---|---|
| L_Hip | 1.003x | **1.180x** |
| L/R_Knee | 0.97~0.98x | **0.91~0.94x** (낮아짐) |
| L/R_Ankle+Toe | 1.02~1.04x | **1.28~1.49x** (높아짐) |
| Upper-L/R | 0.97~1.01x | **0.80~0.86x** (낮아짐) |

---

## 3. 최근 주요 작업 (260310~260311)

### 3-1. CCF Wandb 모니터링 추가
`humanoid_im_vic.py`의 `_compute_torques()`에 Stage 2 활성 시 CCF stats 저장:
```python
self._last_ccf_mean = ccf_raw.mean().item()
self._last_ccf_std  = ccf_raw.std().item()
self._last_ccf_group_mean = ccf_raw.mean(dim=0).detach()  # [n_groups]
```
`amp_agent.py`의 `_assemble_train_info()`에서 wandb 로깅:
- `vic/ccf_mean`, `vic/ccf_std`, `vic/ccf_g0~g7`

### 3-2. Phase-resolved CCF 분석 구현
**방법:** test 모드 시 `_compute_torques()`에서 매 스텝 `(phase, ccf_raw[env=0])` 로깅.
`im_amp_players.py`의 `run()` 종료 시 `output/phase_ccf_log.npy`로 자동 저장.
분석 스크립트: `analyze_phase_ccf.py` (phase 20 구간 binning → 그룹별 impedance_scale 플롯).

**주요 수치 (598 스텝, 30 에피소드 평균):**

| Group | 초반(0~20%) | 중반(40~60%) | 후반(80~100%) | 전체평균 |
|---|---|---|---|---|
| L_Ankle+Toe | 1.018 | 1.241 | 1.312 | **1.278** |
| R_Ankle+Toe | **1.349** | **1.517** | **1.554** | **1.494** |
| L/R_Knee | 0.58~0.62 | 0.70~0.71 | 0.65~0.71 | ~**0.685** |
| Upper-L/R | 0.58~0.62 | 0.65~0.70 | 0.62~0.66 | ~**0.650** |

그래프: `output/phase_ccf_log_plot.png`

### 3-3. 선행 연구 문헌 비교 (Sartori 2015, Lee 2016, Vlutters 2022)
전체 패턴 `발목 > 엉덩이 > 무릎 ≈ 상체`가 생체역학 문헌과 **질적으로 일치**.
- 발목: 문헌에서 stance push-off 시 최대 stiffness, swing에서 급락 → 본 실험에서 가장 높음 ✅
- 무릎: 문헌에서 swing≈0 Nm/rad, mid-stance 수동 잠금으로 낮음 → 본 실험에서 0.68x ✅
- 엉덩이: 문헌에서 보행 전반 중간 수준 → 0.95~1.19x ✅
- 상체: near-passive → 0.63~0.67x ✅

**한계**: 현재 phase=0→1이 walk-stop 시퀀스 전체이므로 Stance/Swing별 분화는 직접 확인 불가. Contact force 로깅 추가 필요.

### 3-4. SMPL 모델 무릎 잠금 분석
- `L/R_Knee_x range [0, 145]도`: Hard stop으로 과신전 방지는 있음
- `stiffness=0`: 범위 내 점진적 stiffness 증가(screw-home mechanism) 없음
- 결론: 생물학적 knee locking은 미구현. Hard stop + PD control로 근사.

---

## 4. 현재 코드 상태

### 핵심 파일들
| 파일 | 주요 내용 |
|---|---|
| `phc/env/tasks/humanoid_im_vic.py` | VIC 환경. CCF 그루핑, sigma init, wandb 로깅, phase-CCF 로깅 |
| `phc/learning/amp_agent.py` | CCF sigma override (pre_epoch), reward curriculum, wandb 로깅 |
| `phc/learning/im_amp_players.py` | 평가 후 phase_ccf_log.npy 저장 |
| `phc/data/cfg/env/env_im_walk_vic.yaml` | 환경 설정 (현재: stage=2, sigma_init=-1.0) |
| `phc/data/cfg/learning/im_walk_vic.yaml` | 학습 설정 (현재: name=VIC_CCF_ON2, max_epochs=20000) |
| `analyze_phase_ccf.py` | phase별 임피던스 분석/플롯 스크립트 |

### 현재 yaml 설정 (VIC_CCF_ON2 기준)
```yaml
vic_enabled: True
vic_curriculum_stage: 2
vic_ccf_num_groups: 8
vic_ccf_sigma_init: -1.0      # CCF sigma 분리 핵심
reward_curriculum_switch_epoch: 10000
reward_w_stage1_task: 0.7 / disc: 0.3
reward_w_stage2_task: 0.3 / disc: 0.7
max_epochs: 20000
```

### 브랜치 현황
- 작업 브랜치: `single_primitives`
- 최근 커밋들 master에 병합 완료 (로컬)
- Remote push: SSH 키 미설정으로 pending 상태 (이전 세션에서 진행 중이었음)

---

## 5. 다음 실험 방향

### 우선순위 1: VIC_PHASE
Phase obs (sin/cos(2π×phase)) +2 dims를 obs에 추가.
구현 계획 문서: `01_research_docs/260309_phase_state_plan.md`

예상 효과:
- 에이전트가 cycle 경계를 사전 인지 → bunny hop 감소
- Phase-dependent CCF 조절 명확해질 것 (에이전트가 "지금이 stance인지 swing인지" 더 잘 파악)
- Stance/Swing별 임피던스 분화 가시화 기대

구현 방법 (humanoid_im_vic.py만 수정):
```python
# get_task_obs_size(): +2
# _compute_task_obs(): phase sin/cos concat
# __init__(): self._vic_phase_obs = cfg["env"].get("vic_phase_obs", False)
```
주의: obs 차원이 바뀌므로 기존 체크포인트 재사용 불가, 처음부터 학습 필요.

### 우선순위 2: Contact-aware CCF 분석
Stance leg vs. Swing leg 임피던스 비교를 위해:
- `_compute_torques()`에서 `net_contact_forces`도 함께 로깅
- contact 여부로 Stance/Swing 분류 후 CCF 비교

### 우선순위 3: 주기적 보행 모션으로 교체
현재 walk-stop 단발 모션 → cyclic walking loop 모션 교체 시:
- 사이클 경계 velocity 불연속 근본 해결
- Phase가 진짜 gait cycle에 대응 → 분석 의미 강화

---

## 6. 학습/평가 명령어

```bash
# 학습 (현재 설정: VIC_CCF_ON2 기준)
conda activate phc
python phc/run.py --task HumanoidImVIC \
    --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
    --num_envs 512 --headless

# 정량 평가 (headless)
python phc/run.py --task HumanoidImVIC \
    --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
    --num_envs 1 --test --epoch -1 --headless

# 시각화 평가 (창 열림)
python phc/run.py --task HumanoidImVIC \
    --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
    --num_envs 1 --test --epoch -1

# Phase-CCF 분석 (평가 후 자동 저장된 npy 사용)
python analyze_phase_ccf.py --npy output/phase_ccf_log.npy
```

---

## 7. 참고 문서 목록

| 경로 | 내용 |
|---|---|
| `01_research_docs/260309_phase_state_plan.md` | Phase obs 구현 계획 |
| `01_research_docs/260310_VIC_CCF_ON2_implementation.md` | VIC_CCF_ON2 구현 상세 |
| `02_research_dev/260309_forward_walk_vic11_result_analysis.md` | VIC11 결과 분석 |
| `02_research_dev/260310_forward_walk_vic_ccf_on_result_analysis.md` | VIC_CCF_ON 결과 (CCF 미학습 원인 분석) |
| `02_research_dev/260311_forward_walk_vic_ccf_on2_result_analysis.md` | VIC_CCF_ON2 결과 + 문헌 비교 |

---

## 8. 실험 성능 히스토리 요약

| 실험 | Avg Reward | Avg Steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | - |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF, Stage1 warm-up |
| VIC_CCF_ON | 945.30 | 297.98 | Stage2 활성화, CCF sigma=-2.9 공유 → 미학습 |
| **VIC_CCF_ON2** | **939.51** | **297.39** | **CCF sigma=-1.0 분리 → CCF 학습 확인** |

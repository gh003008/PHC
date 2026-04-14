# 260408 PHC 개요 및 VIC 수정 사항 정리

본 문서는 이 저장소의 기반이 되는 **PHC** 프레임워크가 무엇인지, 그리고 원본 PHC 대비 본 연구(VIC, Variable Impedance Control)에서 추가·수정한 내용을 정리한다.

---

## 1. PHC 란 무엇인가

**PHC (Perpetual Humanoid Control)** — Luo et al., 2023 (arXiv:2305.06456)

한 줄 요약: **IsaacGym 기반 물리 시뮬레이션에서 SMPL 휴머노이드에게 모션 캡처 데이터를 모방시키는 RL 제어기 프레임워크**.

핵심 특징
- **AMP (Adversarial Motion Prior)**: Discriminator 가 "이 동작이 사람 같은가" 를 판별하여 자연스러움 보상을 생성. 기존 모방 학습 + AMP reward 혼합으로 고품질 모방.
- **PMCP (Progressive Multiplicative Control Policy)**: Catastrophic forgetting 없이 10K+ 모션 클립을 순차 학습 가능하게 하는 정책 구조.
- **외력 없음**: 외부 힘 투입 없이도 noisy pose 입력·낙상 복구·다양한 스킬 모방 가능.
- **제어 방식**: 관절별 PD 제어. 정책이 `q_target` (목표 관절 각도)을 출력하면 저수준 컨트롤러가 `torque = kp * (q_target - q) - kd * dq` 로 변환.

프레임워크 구성
- Task 계층: `VecTask → HumanoidAMPTask → HumanoidAMP → HumanoidIm` (모션 모방 메인)
- 학습: `rl-games` PPO + AMP discriminator (`phc/learning/amp_agent.py`)
- 모션 데이터: AMASS 로부터 전처리된 `.pkl` (`sample_data/`)
- 로봇 모델: SMPL (`data/smpl/SMPL_*.pkl`)

---

## 2. VIC (Variable Impedance Control) 연구 개요

**목적**: 사람이 걸을 때 관절별 강성(stiffness)·감쇠(damping)를 어떻게 동적으로 조절하는지를 정책이 스스로 학습하게 만드는 것. 최종 목표는 **Exoskeleton 제어기 학습**을 위한 사전 단계로, "사람의 임피던스 조절 전략" 을 모델링하는 것.

### 2.1 핵심 아이디어: CCF

기존 PHC:

```
torque = kp * (q_target - q) - kd * dq
```

VIC 추가:

```
torque = kp * 2^ccf * (q_target - q) - kd * 2^ccf * dq
```

- **CCF (Compliance Control Factor)**: 관절 그룹별 임피던스 배율 (로그 스케일).
  - `ccf = 0` → 원본 PHC 와 동일
  - `ccf > 0` → rigid (강성 증가)
  - `ccf < 0` → compliant (유연)
- **CCF 는 정책의 action 으로 출력**된다. 즉 매 스텝 정책이 "지금 이 관절을 얼마나 뻣뻣하게 할지" 도 함께 결정.

### 2.2 8-group CCF
69개 DoF 전체에 CCF 를 부여하면 차원이 커지므로, 8개 기능 그룹으로 묶어 CCF 를 공유한다.

```
L_Hip, L_Knee, L_Ankle+Toe,
R_Hip, R_Knee, R_Ankle+Toe,
Upper-Left, Upper-Right
```

### 2.3 커리큘럼
- **Stage 1 (warm-up)**: `ccf=0` 으로 고정. 보행 자체를 먼저 학습.
- **Stage 2**: CCF 학습 활성화 (`vic_curriculum_stage: 2` 로 수동 전환).
- **Reward curriculum**: `switch_epoch` 이전에는 task 보상 70% / disc 보상 30%, 이후에는 역전 (task 30% / disc 70%).

### 2.4 Phase observation
보행 주기 내 위치(stance/swing 구분)를 정책이 인지할 수 있도록 `sin(phase), cos(phase)` 2차원 추가 (`vic_phase_obs: True`).

---

## 3. 원본 PHC 대비 수정 사항 (파일 단위)

본 저장소는 기존 PHC 스크립트를 직접 수정하지 않고 `_vic` 접미사 사본을 만들어 나란히 두는 원칙을 따른다 (CLAUDE.md 규칙). 저수준 유틸에 불가피한 수정이 필요한 경우 `# VIC` 주석을 명시한다.

### 3.1 신규 Task 클래스

| 파일 | 역할 |
|---|---|
| `phc/env/tasks/humanoid_im_vic.py` | VIC 메인 환경. CCF 그루핑 (`_build_ccf_group_dof_map`), torque 스케일링, phase observation, biomechanical reward, wandb·phase-CCF 로깅. 원본 `humanoid_im.py` 의 사본 + VIC 기능 추가. |
| `phc/env/tasks/humanoid_im_mcp_vic.py` | Mixture-of-Control-Primitives (MCP) 변형 + VIC. |

### 3.2 신규 Config

| 파일 | 내용 |
|---|---|
| `phc/data/cfg/env/env_im_walk_vic.yaml` | VIC 환경 파라미터: `vic_enabled`, `vic_curriculum_stage`, `vic_ccf_num_groups`, `vic_ccf_sigma_init`, `vic_ccf_min/max`, `vic_phase_obs`, `vic_bio_ccf_reward_w`, reward curriculum weights. |
| `phc/data/cfg/learning/im_walk_vic.yaml` | 네트워크·PPO·AMP 설정. `name: VIC_PHASE_04` 등 실험명도 여기에서 지정. |

### 3.3 학습 측 수정 (`phc/learning/`)

| 파일 | 변경 내용 |
|---|---|
| `phc/learning/amp_agent.py` | `pre_epoch()` 내부에 (a) CCF action dim 에 대한 별도 `sigma` override, (b) reward curriculum stage switching (task vs disc weight), (c) wandb 로깅 (CCF 통계 포함) 추가. |
| `phc/learning/im_amp_players.py` | 평가 플레이어. 매 스텝 `(phase, ccf)` 쌍을 수집하여 `output/<exp_name>/phase_ccf_log.npy` 로 저장. |

### 3.4 신규 분석 스크립트

| 파일 | 역할 |
|---|---|
| `analyze_phase_ccf.py` (repo root) | `phase_ccf_log.npy` 를 읽어 phase 구간별 그룹별 impedance_scale 평균 테이블, gait-cycle stance/swing 음영, per-group / L-R 비교 플롯 생성. |

### 3.5 실험 설정 아카이브

| 경로 | 내용 |
|---|---|
| `exp_config/forward_walking/` | 실험별 (V1~V4 baseline, VIC01~VIC11, CCF_ON, CCF_ON2, VIC_PHASE 시리즈) env/learning yaml 및 `humanoid_im_vic.py` 스냅샷. 학습 실행 전 자동 백업. |

### 3.6 연구 문서
- `01_research_docs/` — 실험 세팅·구현 문서
- `02_research_dev/` — 학습 결과·분석 문서
- `03_QnA/` — 질문-답변 정리

---

## 4. 현재 기본 VIC 설정값 (VIC_PHASE_04 기준)

```yaml
# phc/data/cfg/env/env_im_walk_vic.yaml
vic_enabled: True
vic_curriculum_stage: 2                 # 1=CCF 고정(warm-up), 2=CCF 학습
vic_ccf_num_groups: 8
vic_ccf_sigma_init: -1.0                # CCF log-std (std≈0.37) — 탐색 보장
vic_ccf_min / vic_ccf_max: -1.0 / 1.0   # CCF 범위 → 강성 0.5x ~ 2.0x
vic_phase_obs: True                     # +2 dims (sin/cos phase)
reward_curriculum_switch_epoch: 10000
reward_w_stage1_task / disc: 0.7 / 0.3
reward_w_stage2_task / disc: 0.3 / 0.7

# phc/data/cfg/learning/im_walk_vic.yaml
sigma_init: -2.9                        # PD action sigma (std≈0.055)
learning_rate: 5e-5
max_epochs: 20000
mlp.units: [1024, 1024, 512, 512]
```

---

## 5. 실험 성능 히스토리 (CLAUDE.md 에서 발췌)

| 실험 | av_reward | av_steps | 핵심 변경 |
|---|---|---|---|
| V4 (PHC baseline) | ~461 | ~143 | 원본 PHC (walk) |
| VIC11 | 947.11 | 300.1 | 8그룹 CCF + Stage1 warm-up |
| VIC_CCF_ON | 945.30 | 297.98 | Stage2 활성화, CCF sigma 공유(-2.9) → CCF 미학습 |
| **VIC_CCF_ON2** | **939.51** | **297.39** | CCF sigma 분리(-1.0) → CCF 실제 학습 확인 |

VIC 가 원본 PHC baseline(V4) 대비 av_reward 를 2배 이상, av_steps 를 2배 이상으로 끌어올렸다는 점이 핵심 결과. CCF_ON2 단계에서 발목·무릎·상체 그룹이 생체역학 문헌과 질적으로 일치하는 임피던스 분포를 보이기 시작했다.

---

## 6. 최종 목표

인간 + Exoskeleton 공동 시뮬레이션에서 **Exo 제어기 RL 학습**. VIC 는 "사람이 임피던스를 어떻게 조절하는지" 를 먼저 모델링해 두는 사전 단계다.

기술적 이슈: IsaacGym (PhysX 4) 은 closed kinematic chain 미지원 → 최종적으로 **Isaac Lab (PhysX 5) 마이그레이션 필요**.

# Standard Human Model + IsaacGym 통합 구현 정리

작성일: 2026-03-18
브랜치: `human_model_v1`

---

## 1. 개요

PHC 기반 VIC 연구에서 한 단계 더 나아가, **근골격계(musculoskeletal) 모델**을 구현하고 IsaacGym에 통합하는 작업을 진행했다. 최종 목표는 VIC 제어기에 근육 생체역학(Hill 근육 모델, 척수 반사, 인대)을 더해 보다 생리학적으로 사실적인 인간 모델을 만드는 것이다.

---

## 2. Standard Human Model 구현 (`standard_human_model/`)

### 2.1 전체 파이프라인

```
상위 명령 (u) → Reflex Layer → Activation Dynamics → Muscle Kinematics
                                                      → Hill Force Generation
                                                      → Torque Mapping (τ = F @ R)
                                                      → Ligament Torque
                                                      → 최종 관절 토크
```

IsaacGym의 `pre_physics_step`에서 매 스텝 `HumanBody.compute_torques()`를 호출하는 구조.

### 2.2 핵심 모듈 (`standard_human_model/core/`)

| 파일 | 역할 |
|---|---|
| `skeleton.py` | SMPL 관절 정의, DOF 범위, kp/kd 텐서 |
| `muscle_model.py` | Hill-type 근육 모델 (F-L, F-V, Passive) |
| `moment_arm.py` | R 행렬 (20근육 × 69DOF), 근육 길이/속도 계산 |
| `activation_dynamics.py` | 1차 활성화 ODE (tau_act, tau_deact) |
| `reflex_controller.py` | 척수 반사 (신장 반사, GTO, 상반 억제) |
| `ligament_model.py` | 관절낭 soft-limit 토크 (지수 모델) |
| `patient_profile.py` | 8 그룹 × 9 파라미터 환자 프로파일 로더 |
| `human_body.py` | 전체 파이프라인 통합 클래스 |

### 2.3 핵심 수식

**Hill 근육 모델:**
```
F_total = a * F_max * f_FL(l) * f_FV(v) * cos(pennation)   # active
        + F_max * f_PE(l)                                   # passive
        + damping * F_max * v                              # damping
```

**관절 토크 매핑 (moment arm matrix):**
```
τ = R^T @ F         # (69,) = (20×69)^T @ (20,)
l = l_slack - R @ q # 근육 길이
v = -(dof_vel @ R^T) # 근육 속도
```

**인대 soft-limit 토크:**
```
τ_lig = -k_lig * exp(alpha * max(0, q - q_soft_upper))
       + k_lig * exp(alpha * max(0, q_soft_lower - q))
```

### 2.4 근육 구성

- **총 20개 근육군** (좌우 각 10개)
- 좌/우: hip_flexors, gluteus_max, hip_abductors, hip_adductors, quadriceps, rectus_femoris, hamstrings, gastrocnemius, soleus, tibialis_ant
- **이관절근(bi-articular)**: rectus_femoris (고관절+무릎), hamstrings (고관절+무릎), gastrocnemius (무릎+발목)
- R 행렬은 YAML(`muscle_definitions.yaml`)에서 수동 정의 → DOF ordering은 skeleton.py = IsaacGym MJCF와 완전 일치

### 2.5 환자 프로파일

기준 파일: `config/healthy_baseline.yaml` (근육별 f_max, l_opt, 반사 파라미터, 인대 파라미터)

환자 프로파일 (`profiles/`):
- `healthy/healthy_adult.yaml` — 정상 성인 (모든 기본값)
- `stroke/stroke_r_hemiplegia.yaml` — 뇌졸중 우측 편마비 (우측 tau_active 0.15~0.3, spasticity 5~9)
- `sci/sci_t10_complete_flaccid.yaml` — T10 완전 척수손상 (하체 tau_active=0, k_passive 최소)
- `parkinson/` — (추가 예정)

---

## 3. IsaacGym 통합 (Strategy C — Hybrid)

### 3.1 통합 전략

3가지 전략 검토 후 **Strategy C (2단계 하이브리드)** 선택:

| 단계 | 방식 | 목적 |
|---|---|---|
| **Phase 1** | `τ = τ_PD + α * τ_bio` | PD 제어에 bio-torque 혼합 (α=0이면 기존 VIC와 동일) |
| **Phase 2** | Action space → 20 근육 명령 | 완전 근육 기반 제어 (장기 목표) |

Phase 1에서 `blend_alpha=0.0` (기본값)이면 기존 VIC_PHASE_04와 완전 동일 → **회귀 안전 보장**.

### 3.2 구현 파일

**`phc/env/tasks/humanoid_im_vic_msk.py`** (신규)
```python
class HumanoidImVICMSK(HumanoidImVIC):
    def _init_human_body(self):
        self._human_body = HumanBody.from_config(...)
        self._msk_blend_alpha = cfg.get("blend_alpha", 0.0)

    def _compute_torques(self, actions):
        τ_pd = super()._compute_torques(actions)
        if self._msk_blend_alpha == 0.0:
            return τ_pd  # early return — 기존과 동일
        τ_bio = self._human_body.compute_torques(dof_pos, dof_vel, cmd, dt)
        τ = τ_pd + self._msk_blend_alpha * τ_bio
        return clamp(τ, -torque_limits, torque_limits)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._human_body.reset(env_ids)
```

**`phc/data/cfg/env/env_im_walk_vic_msk.yaml`** (신규)
```yaml
msk_config:
  muscle_def: "muscle_definitions.yaml"
  patient_profile: "healthy_baseline.yaml"
  blend_alpha: 0.0        # 0.0 = 기존 VIC와 동일
  max_torque_ratio: 0.5
```

**`phc/utils/parse_task.py`** — HumanoidImVICMSK 등록 추가

### 3.3 회귀 테스트 결과

`blend_alpha=0`으로 VIC_PHASE_04 체크포인트 평가:
- reward 545.17, steps 162 → 기존 VIC와 동일 동작 확인

---

## 4. 검증 실험 (`standard_human_model/isaacgym_validation/`)

보행 제어기 통합 전, 각 관절의 근육 특성이 올바르게 반영되는지 정량 검증.

### 4.1 5개 검증 실험 요약

| 실험 | 목적 | 주요 결과 |
|---|---|---|
| **Exp1** 수동 토크 프로파일 | ROM 각도 스윕 시 passive force | 근육 passive force ≈ 0 (l_opt 튜닝 필요) |
| **Exp2** 신장 반사 | 속도별 반사 토크 (healthy vs spastic) | 무릎 baseline 높음 (중립각=0° 문제) |
| **Exp3** R 행렬 검증 | moment arm 매핑 정확도 | **이관절근 토크 비율 = moment arm 비율, 오차 0.0%** ✅ |
| **Exp4** 공동수축 | CC 레벨 → 관절 임피던스 | **고관절/발목 선형 관계 확인, L/R 100% 대칭** ✅ |
| **Exp5** 환자 비교 | 3 프로파일 passive torque/range/reflex | 프로파일 간 차별화 부족 (파라미터 튜닝 필요) |

### 4.2 핵심 발견: 코드 버그 없음

Exp3/Exp4 결과로 **R 행렬 구현과 DOF 매핑은 정확**함이 확인됨.
남은 이슈는 모두 **파라미터 튜닝 레벨** (코드 오류 아님):
- `l_opt=1.0` 정규화 시 ROM 내 근육 길이 변화 부족 → passive force 미발생
- 무릎 실험 시 중립각을 0°이 아닌 70° 사용 필요
- 환자 프로파일 간 인대 파라미터 차별화 미흡

### 4.3 IsaacGym 뷰어 진자 데모 (`demo_knee_pendulum.py`)

**설계:** 3개 SMPL 휴머노이드 나란히 배치, 무릎 관절만 자유(EFFORT 모드), 나머지 고정(POS 모드). 80° 굴곡 + 초기 킥(-5 rad/s)에서 놓아 중력 스윙.

**bio-torque 파라미터 차별화:**

| 프로파일 | 반사 gain | 반사 threshold | 인대 k | 근력 |
|---|---|---|---|---|
| Healthy | 1.0 (기본) | 0.1 | 50 | 100% |
| Spastic | **8.0** | **0.02** | **200** | 100% |
| Flaccid | **0.0** | 999 | **5** | **5%** |

**시뮬레이션 결과 (5초):**

| 프로파일 | 최종 각도 | 최대 bio-torque | 특성 |
|---|---|---|---|
| Healthy | ~74° | ~3 Nm | 적절한 감쇠, 수 회 스윙 후 수렴 |
| **Spastic** | **~82°** | **~25 Nm** | 초기 위치에서 거의 안 움직임 (고경직) |
| **Flaccid** | **~23°** | **~5 Nm** | 자유 진자처럼 크게 스윙 (근력 없음) |

→ 환자 특성이 IsaacGym 시뮬레이션에서 **시각적으로 구별 가능**하게 반영됨 확인.

---

## 5. 발생한 문제 및 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| IsaacGym import 순서 오류 | PyTorch가 먼저 로드됨 | isaacgym을 torch보다 먼저 import |
| UnicodeDecodeError | YAML 파일 내 한글 주석 | open(..., encoding="utf-8") 추가 |
| NaN 발생 (진자 데모) | freejoint root 불안정 | freejoint 제거한 고정 root MJCF 사용 |
| f_max 텐서 타입 오류 | Long 타입 텐서에 Float 연산 | `.float()` 명시적 변환 |
| 진자 거의 안 움직임 | asset angular_damping=0.01 과다 | 0.0으로 낮추고 초기 킥 속도 추가 |

---

## 6. 남은 작업 및 다음 단계

### 단기 (파라미터 튜닝)
- [ ] `l_opt` / `l_slack` 튜닝: ROM 내 passive muscle force 발생하도록 조정
- [ ] 실험 재설계: 무릎 중립각 0° → 70° 변경
- [ ] 환자 프로파일 인대 파라미터 차별화

### 중기 (IsaacGym 통합 심화)
- [ ] `blend_alpha` > 0으로 bio-torque 혼합 실험
- [ ] 전신 실환자 real-to-sim 시연 (stroke, SCI)
- [ ] EMG replay 입력으로 descending_cmd 대체

### 장기
- [ ] IsaacGym → Isaac Lab (PhysX 5) 마이그레이션
- [ ] Closed kinematic chain 지원 (Exoskeleton 통합)
- [ ] Phase 2: Action space → 20 근육 명령으로 완전 전환

---

## 7. 주요 파일 경로

```
PHC/
├── standard_human_model/
│   ├── core/
│   │   ├── human_body.py            # 통합 파이프라인 (메인 클래스)
│   │   ├── muscle_model.py          # Hill 근육 모델
│   │   ├── moment_arm.py            # R 행렬 (20×69)
│   │   ├── activation_dynamics.py   # 활성화 ODE
│   │   ├── reflex_controller.py     # 척수 반사 (신장/GTO/상반억제)
│   │   ├── ligament_model.py        # 인대 soft-limit
│   │   ├── patient_profile.py       # 환자 프로파일 로더
│   │   └── skeleton.py              # SMPL DOF 정의
│   ├── config/
│   │   ├── muscle_definitions.yaml  # 근육군 구조 + R 행렬
│   │   └── healthy_baseline.yaml    # 기준 근육 파라미터
│   ├── profiles/                    # 환자 프로파일 YAML
│   │   ├── stroke/stroke_r_hemiplegia.yaml
│   │   └── sci/sci_t10_complete_flaccid.yaml
│   └── isaacgym_validation/
│       ├── exp1~exp5_*.py           # 5개 수치 검증 실험
│       ├── demo_knee_pendulum.py    # IsaacGym 뷰어 진자 데모
│       ├── smpl_humanoid_fixed.xml  # freejoint 제거 MJCF
│       └── results/                 # PNG 결과 플롯
├── phc/env/tasks/humanoid_im_vic_msk.py   # IsaacGym 통합 태스크
└── phc/data/cfg/env/env_im_walk_vic_msk.yaml
```

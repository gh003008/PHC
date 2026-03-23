# PHC 프레임워크 고도화 로드맵 및 팀 구성 기획

작성일: 2026-03-18
작성자: Claude (연구 기획 제안)

---

## 1. 한 줄 비전

> **"인간-로봇 통합 물리 시뮬레이터를 연구실 공용 인프라로 구축하여, 실험실 데이터 수준의 합성 생체역학 데이터(EMG, GRF, 관절 운동학)를 자동 생성하는 플랫폼"**

---

## 2. 전체 연구 트랙 구성 제안

### 2.1 트랙 구조 (4트랙)

```
┌─────────────────────────────────────────────────────────────┐
│                    공통 인프라 (Core)                         │
│         PHC + standard_human_model + VIC 통합 플랫폼          │
└───────────┬───────────────────┬───────────────────┬─────────┘
            │                   │                   │
    [Track A]           [Track B]           [Track C]
 인간-로봇 통합         환자군 모델링          합성 데이터
  상호작용력            (근골격계 파라미터)      생성기

            └───────────────────┴───────────────────┘
                          [Track D]
                     검증 및 실험 연계
                    (실데이터 vs 시뮬레이션)
```

---

### Track A: 인간-로봇 통합 (Human-Robot Interface)

**담당 인원**: 1~2명 (Isaac Lab / PhysX 5 경험자 우선)

**핵심 목표**: Exoskeleton이 인간에게 가하는 힘과 인간이 Exo에 반응하는 방식을 물리 시뮬레이션으로 정확히 재현

**주요 과제:**
- IsaacGym(PhysX 4) → Isaac Lab(PhysX 5) 마이그레이션
  - 이유: closed kinematic chain(Exo 착용 시 다리+Exo가 폐쇄 체인 형성) 지원
- 인간-Exo 부착점 모델링: rigid attachment vs. soft tissue compliance
- 상호작용력 추정: `net_contact_force_tensor`에서 attachment wrench 추출
- 슬라이딩/마찰력 모델: 실제 하니스/cuff 착용 시 피부 미끄러짐 반영

**마일스톤:**
```
M1: Isaac Lab 기본 환경 이전 (SMPL 휴머노이드 단독)
M2: 단순 Exo 링크 추가 + rigid attachment
M3: VIC 제어기 통합 (인간 측 임피던스 학습)
M4: Exo 제어기 RL 학습 (최종 목표)
```

**왜 독립 트랙?**: Isaac Lab 마이그레이션은 전체 코드베이스에 영향을 줘서 격리가 필요. 이 트랙이 완성되기 전까지 나머지 트랙은 IsaacGym에서 계속 개발 가능.

---

### Track B: 환자군별 근골격계 모델링 (Patient Population)

**담당 인원**: 환자군당 1명 (총 2~3명), 임상/재활 지식 배경 우선

**핵심 목표**: 각 병리 그룹에 대해 문헌 기반의 생리학적으로 타당한 파라미터 세트 확립

**서브트랙:**

| 서브트랙 | 환자군 | 핵심 병리 파라미터 |
|---|---|---|
| B1 | 뇌졸중 편마비 | stretch_gain ↑, reciprocal_inhibition ↓, 편측 f_max ↓ |
| B2 | 척수손상 (SCI) | 병변 레벨별 tau_active=0, passive 구축 진행 |
| B3 | 파킨슨 | co-contraction ↑, tremor 주파수 노이즈 추가 |
| B4 | 근육병증 | f_max 전반적 ↓, fatigue 모델 추가 |

**공통 작업 방법:**
1. 해당 환자군 EMG 문헌에서 근활성화 패턴 수집
2. `profiles/` YAML 파라미터 튜닝 → 시뮬레이션 재현
3. Track D와 협력하여 실측 데이터(EMG, GRF)와 비교 검증
4. 확정된 프로파일은 `healthy_baseline.yaml` 기준 대비 % 변화 표로 문서화

**중요**: 파라미터 변경은 반드시 YAML로만 → 코어 코드 수정 없이 `patient_profile.py`가 로딩

---

### Track C: 합성 데이터 생성기 (Synthetic Data Generator)

**담당 인원**: 1명 (신호 처리 / ML 배경 우선)

**핵심 목표**: 시뮬레이터에서 나오는 물리량을 실험실 계측 장비 데이터 형식으로 변환

#### C1. EMG 합성

**가능 여부: 충분히 가능, 단 모델 단순화 필요**

```
시뮬레이터 출력                →   합성 EMG 신호
근활성화 a(t) [0,1]           →   amplitude envelope
+ 노이즈 (shot + thermal)     →   sEMG-like waveform
+ 볼륨 도체 감쇠 (선택)        →   electrode depth correction
```

현재 `activation_dynamics.py`가 이미 a(t)를 계산 중. 여기에:
- Gaussian 노이즈 + 직류 편향 제거 (BPF 20~500 Hz 시뮬레이션)
- 근육별 electrode 위치 → spatial weighting
- Fatigue 모델 추가 시 주파수 shift까지 재현 가능

참고: Heckman & Enoka (2012) "Motor Unit" 모델 기반 MU discharge 시뮬레이션이 더 정확하지만 연산 비용 큼 → Phase 2에서 검토.

#### C2. GRF (지면반력) 합성

**가능 여부: 매우 높음, IsaacGym에서 직접 추출 가능**

```python
# IsaacGym contact force 추출 (이미 가능)
_contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
contact_forces = gymtorch.wrap_tensor(_contact_force_tensor)
# → (num_envs * num_bodies, 3) in Newtons

# 발바닥 body index 추출 → force plate 데이터 형식으로 저장
foot_contact = contact_forces[foot_body_ids]  # Fx, Fy, Fz
```

실험실 Force Plate와 비교 가능한 지표:
- Vertical GRF (Fz): 보행 중 double-hump pattern
- AP force (Fy): braking/propulsion
- COP (Center of Pressure): 발바닥 접촉 분포 → 추가 구현 필요

#### C3. 관절 운동학 / IMU

```
시뮬레이터 → 실험실 장비 매핑:
DOF angle   → 전통 Goniometry / Vicon 마커 기반 관절각
body accel  → IMU (3축 가속도 + 자이로)
trunk angle → Wearable sensor
```

#### C4. 데이터 출력 표준화

```
output/
└── synthetic_data/
    ├── subject_001_healthy/
    │   ├── emg/     # .csv, 채널별 근활성화
    │   ├── grf/     # .csv, Fx/Fy/Fz/COP
    │   ├── kinematics/  # .csv, 관절각/각속도
    │   └── metadata.yaml  # 프로파일, 보행 속도, step count
    └── subject_002_stroke/
```

표준 포맷을 정해두면 실험실 실측 데이터와 직접 비교 파이프라인 구성 가능.

---

### Track D: 검증 및 실험 연계 (Validation)

**담당 인원**: 전체 공통 (실험 데이터 있는 인원 참여)

**핵심 목표**: 합성 데이터 ↔ 실측 데이터 차이를 정량 지표로 평가하고 모델 파라미터 피드백

**주요 지표:**
- EMG correlation coefficient (합성 vs. 실측, 동일 보행 조건)
- GRF RMSE (N), GRF pattern shape similarity (DTW)
- 관절각 RMSE (deg)

**자동화 파이프라인:**
```
파라미터 변경 (YAML) → 시뮬레이션 실행 → 합성 데이터 생성
                                          → 실측 데이터와 자동 비교
                                          → 지표 리포트 (wandb 연동)
```

---

## 3. 팀 역할 분담 요약

| 인원 | 주 트랙 | 필요 역량 |
|---|---|---|
| 나 (gunhee) | Core + 총괄 + VIC/RL | IsaacGym, RL, Python |
| 인원 A | Track A (인간-로봇) | Isaac Lab, 로봇공학, URDF/MJCF |
| 인원 B | Track B1/B2 (환자군 모델) | 재활의학, 신경과학, 생체역학 |
| 인원 C | Track B3/B4 (환자군 모델) | 신경퇴행, 근육 생리학 |
| 인원 D | Track C (합성 데이터) | 신호처리, ML, Python/numpy |
| 전체 | Track D (검증) | 실험 경험, 데이터 분석 |

---

## 4. 유지보수 개발 전략

### 4.1 아키텍처 원칙

**"Core Stable, Extension Flexible"**

```
stable (공동 관리, PR 리뷰 필수)
├── standard_human_model/core/     # HumanBody 파이프라인
├── phc/env/tasks/humanoid_im_vic*.py  # VIC 통합 인터페이스
└── phc/learning/                  # RL 에이전트

flexible (담당자 자율 개발)
├── standard_human_model/profiles/    # 환자 YAML (Track B 담당)
├── standard_human_model/isaacgym_validation/  # 검증 실험 (Track D)
└── output/synthetic_data/            # 합성 데이터 (Track C)
```

### 4.2 Git 브랜치 전략

```
main (= master)
 ├── dev/human-robot-interface   # Track A 개발
 ├── dev/patient-stroke          # Track B1
 ├── dev/patient-sci             # Track B2
 ├── dev/synthetic-data          # Track C
 └── feature/xxx                 # 단발성 기능 추가
```

- `standard_human_model/core/` 변경 시 → **반드시 팀 전체 PR 리뷰**
- `profiles/` YAML 변경 → 담당자 자율 머지 (코드 영향 없음)
- Core 변경 후 반드시 `isaacgym_validation/` 수치 검증 실험 통과 확인

### 4.3 공동 설계가 필요한 결정들

다음은 담당자 혼자 결정하지 말고 팀이 함께 논의해야 함:

1. **근육 구성 확장**: 20개 → 더 많은 근육군 추가 시 R 행렬 수정 → DOF 매핑 전체에 영향
2. **DOF 정의 변경**: skeleton.py의 NUM_DOFS / joint ordering → 모든 트랙에 영향
3. **Isaac Lab 마이그레이션 타이밍**: 기존 IsaacGym 코드 deprecation 시점 합의
4. **합성 데이터 포맷**: Track C에서 정하면 Track D가 의존 → 사전 합의 필요

### 4.4 문서화 기준 (팀 공통)

- 파라미터 추가 시: `healthy_baseline.yaml`에 기본값 + 주석 필수
- 환자 프로파일 확정 시: `01_research_docs/`에 문헌 근거와 함께 기록
- 검증 실험 결과: `standard_human_model/isaacgym_validation/results/`에 PNG + 수치 테이블 저장

---

## 5. 합성 데이터 생성 타당성 검토

### 5.1 단기 (6개월 내): 충분히 가능

| 데이터 | 기반 | 구현 난이도 | 기대 품질 |
|---|---|---|---|
| GRF | contact_force_tensor (기존 기능) | ★☆☆ | 높음 (물리 직접 출력) |
| 관절각/각속도 | dof_pos/vel (기존 기능) | ★☆☆ | 높음 |
| 근활성화 envelope | activation_dynamics a(t) | ★★☆ | 중간 (노이즈 모델 추가 필요) |

### 5.2 중기 (1년): 추가 모델링 필요

| 데이터 | 추가 필요 | 구현 난이도 |
|---|---|---|
| sEMG 파형 (세부) | Motor Unit discharge 모델 | ★★★ |
| COP (발바닥 분포) | 발바닥 다점 접촉 세분화 | ★★☆ |
| 근육 내 압력 | 근육 체적 모델 | ★★★ |

### 5.3 데이터 품질 로드맵

```
Phase 1 (지금 가능): GRF + 관절각 = 임상 보행분석 수준
Phase 2 (6개월): EMG envelope + 가상 IMU = 웨어러블 연구 수준
Phase 3 (1년+): sEMG 파형 + COP = 실험실 완전 대체 수준
```

실험실 데이터를 "완전히" 대체하는 건 어렵지만,
**"특정 조건의 반복 실험, 극한 상황, 희귀 환자군"** 에서는
시뮬레이션 데이터가 실험 데이터보다 오히려 더 가치 있음.

---

## 6. 단기 실행 계획 (다음 2~3주)

### 즉시 착수 가능 (Core 안정화)

- [ ] `l_opt` / `l_slack` 튜닝 → passive muscle force 발생 확인 (현재 미비)
- [ ] 무릎 중립각 0° → 70° 변경, 검증 실험 재실행
- [ ] GRF 추출 모듈 작성 (`contact_force_tensor` → CSV 저장)
- [ ] `blend_alpha > 0` 실험: bio-torque를 VIC 보행 제어에 혼합

### 팀 온보딩 준비

- [ ] `standard_human_model/README.md` 작성 (영문): 파이프라인 그림 + 퀵스타트
- [ ] 수치 검증 실험 5개 자동 실행 스크립트 (`run_all_validations.sh`)
- [ ] 환자 프로파일 기여 가이드: YAML 형식, 파라미터 범위, 문헌 링크 방식

---

## 7. 결론

이 프레임워크는 현재 **"연구자 1명의 탐색 도구"** 수준이지만,
위 구조를 따르면 **"연구실 공용 인프라"** 로 전환이 가능하다.

핵심은 세 가지:
1. **Core는 팀이 함께 수호**, Extension(프로파일/합성 데이터)은 담당자 자율
2. **YAML 기반 환자 프로파일** → 코드 수정 없이 누구나 환자군 추가 가능
3. **검증 수치 기준 합의** → "이 GRF 오차 이하면 충분하다"는 공통 기준 먼저 정의

생성 데이터가 실험 데이터 수준에 도달하는 순간,
"데이터 부족 문제"라는 재활공학 연구의 가장 큰 병목이 해소된다.

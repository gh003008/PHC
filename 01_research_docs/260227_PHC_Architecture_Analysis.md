# PHC 프로젝트 아키텍처 분석 (2026-02-27)

본 문서는 PHC 프로젝트의 Isaac Gym 환경 설정, 상태/행동 공간 정의, 그리고 가변 임피던스 제어(VIC)를 위한 토크 계산 로직의 위치와 구조를 분석한 결과입니다.

## 1. 개요 (Architecture Overview)

PHC 프로젝트는 Isaac Gym을 기반으로 한 강화학습 환경을 구축하고 있으며, `phc/env/tasks` 폴더 내의 스크립트들이 핵심 로직을 담당합니다.

```mermaid
graph TD
    A[run.py / run_hydra.py] --> B[phc/env/tasks/vec_task.py]
    B --> C[phc/env/tasks/humanoid.py]
    C --> D[phc/env/tasks/humanoid_im_vic.py]
    
    subgraph "Environment Setup (Isaac Gym)"
        C -- "create_sim, _create_envs" --> E((Isaac Gym API))
    end
    
    subgraph "Logic & Control"
        D -- "State Space: _compute_task_obs" --> F[Observations]
        D -- "Action Space: get_action_size" --> G[Actions (Pose + CCF)]
        D -- "Torque Calculation: _compute_torques" --> H[Torque Command]
    end
```

## 2. 주요 구성 요소 및 위치

### 2.1 Isaac Gym 환경 설정 (Environment Setting)
- **핵심 파일**: `phc/env/tasks/humanoid.py` 및 `phc/env/tasks/base_task.py`
- **주요 로직**:
    - `Humanoid.create_sim()`: Isaac Gym 엔진 및 시뮬레이션 환경 초기화.
    - `Humanoid._create_envs()`: 로봇 에셋 로드 및 다중 환경(parallel environments) 배치.
    - `phc/env/tasks/vec_task.py`: 여러 환경을 벡터화하여 관리하는 최상위 래퍼.

### 2.2 상태 및 행동 공간 정의 (State/Action Space)
- **행동 공간(Action Space)**: 
    - **파일**: `phc/env/tasks/humanoid_im_vic.py`
    - **정의**: `get_action_size()`에서 결정되며, VIC 활성 시 Target Pose(DOF 수 만큼)와 CCF(Co-Contraction Factor, 임피던스 조절 파라미터)가 결합된 형태입니다.
- **상태 공간(State Space)**:
    - **파일**: `phc/env/tasks/humanoid_im_vic.py`
    - **정의**: `_compute_task_obs()`에서 계산됩니다. 기본 캐릭터 상태(Joint pos/vel 등)와 함께 참조 동작(Reference motion)과의 차이가 포함됩니다.

### 2.3 가변 임피던스 제어(VIC) 토크 계산 로직
- **파일**: `phc/env/tasks/humanoid_im_vic.py` (라인 1130 부근)
- **로직 상세 (`_compute_torques`)**:
    1. **CCF 추출**: 입력된 Action에서 Target Pose와 CCF를 분리합니다.
    2. **Impedance Scaling**: `impedance_scale = 2^ccf` 공식을 통해 기본 Gain(`p_gains`, `d_gains`)을 스케일링하여 동적 Gain(`kp`, `kd`)을 생성합니다.
    3. **토크 산출**: PD 제어 공식을 사용하여 최종 토크를 계산합니다.
       `torques = kp * (target_pose - current_pose) - kd * current_vel`

## 3. 결론
PHC의 아키텍처는 `humanoid.py`에서 물리 환경의 기초를 다지고, `humanoid_im_vic.py`에서 실제 학습에 필요한 고차원적 제어(VIC)와 상태 관리를 수행하는 계층 구조를 가지고 있습니다.

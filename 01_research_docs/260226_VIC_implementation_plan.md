# VIC (Variable Impedance Control) 구현 계획 (V4 Baseline 기반)

Date: 2026-02-26
Author: Antigravity

## 1. 개요
V4 Baseline(`HumanoidIm`)의 성공적인 보행 엔진을 바탕으로, 관절의 강성(Stiffness)과 감쇠(Damping)를 동적으로 조절하는 **Variable Impedance Control (VIC)**을 구현합니다. 이는 인간의 Co-contraction 기법을 모사하여 외란에 강인하고 에너지 효율적인 보행을 목표로 합니다.

## 2. 주요 아키텍처 변경

### 2.1. 액션 스페이스 확장 (Action Space)
*   **기존**: $[q_{target}]$ (DoF 개수만큼, 약 31~69개)
*   **VIC**: $[q_{target}, \beta]$ ($2 \times$ DoF)
    *   $\beta \in [-1, 1]$ 는 임피던스 스케일링 팩터 (CCF)
    *   $Kp = Kp_{base} \cdot 2^{\beta}$
    *   $Kd = Kd_{base} \cdot 2^{\beta}$

### 2.2. 제어 루프 (Control Loop)
*   Isaac Gym의 내장 PD 제어기 (`set_dof_position_target_tensor`) 대신 **Manual PD 계산** 방식으로 전환합니다.
*   `HumanoidImVIC.pre_physics_step`에서 다음과 같이 토크를 직접 계산하여 인가합니다:
    $$ \tau = K_p \odot (q_{target} - q) + K_d \odot (0 - \dot{q}) $$
*   `self.gym.set_dof_actuation_force_tensor`를 사용하여 직접 토크를 전달합니다.

### 2.3. 커리큘럼 학습 (Curriculum Strategy)
사용자의 요청에 따라 2단계 학습 전략을 적용합니다:
*   **Stage 1 (Warm-up)**: $\beta$ 액션을 무시하고 $\beta=0$ (기본 강성)으로 고정하여 기존 V4와 동일한 보행을 먼저 학습합니다.
*   **Stage 2 (VIC Learning)**: $\beta$ 액션을 활성화하여 강성을 가변적으로 학습합니다. 이때 대사 비용(Metabolic Cost) 리워드를 추가하여 불필요한 고강성을 방지합니다.

## 3. 구현 단계 (Action Items)

### Step 1: 파일 시스템 준비 (완료)
*   [x] `phc/env/tasks/humanoid_im_vic.py` 생성 (복사본)
*   [x] `phc/data/cfg/env/env_im_walk_vic.yaml` 생성
*   [x] `phc/data/cfg/learning/im_walk_vic.yaml` 생성

### Step 2: 환경 코드 수정 (`humanoid_im_vic.py`)
*   [ ] `vic_enabled` 플래그 및 커리큘럼 전환 로직 추가
*   [ ] `get_action_size` 오버라이드 ($2 \times$ DoF)
*   [ ] `pre_physics_step`에서 Manual PD 계산 로직 구현
*   [ ] Metabolic Cost 리워드 함수 추가 (`_compute_reward`)

### Step 3: 설정 및 학습 시작
*   [ ] `env_im_walk_vic.yaml`에 VIC 관련 파라미터 추가
*   [ ] **Stage 1** 학습 시작 및 모니터링

---

## 4. 사용자 검토 필요 사항
1.  **CCF 스케일 범위**: 현재 $2^{\beta}$ (0.5배 ~ 2.0배)로 계획 중입니다. 더 넓은 범위(예: 0.1배 ~ 10배)가 필요하신지 확인 부탁드립니다.
2.  **커리큘럼 전환 시점**: Stage 1에서 어느 정도 성능(예: 5k 에폭)이 나오면 자동으로 Stage 2로 넘어가게 할지, 아니면 수동으로 전환하실지 결정이 필요합니다.

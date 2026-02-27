# VIC01 Project Handoff Summary (260227)

이 문서는 인간 모델 가변 임피던스 제어(Variable Impedance Control, VIC) 프로젝트의 현재 상태와 핵심 설계 결정 사항을 요약하여 다음 세션의 원활한 업무 인계 및 연속성을 보장하기 위해 작성되었습니다.

## 1. 핵심 설계 결정 (Core Design Decisions)

### 수동 PD 제어 시스템 (Manual PD Control Loop)
*   **배경**: 가변 강성 및 댐핑을 실시간으로 조절하기 위해 Isaac Gym의 내장 PD 모드 대신 직접 토크를 계산하는 루프를 구현했습니다.
*   **구현**: `HumanoidImVIC` 클래스에서 `_compute_torques` 메서드를 오버라이드하여, 매 물리 단계마다 액션으로부터 계산된 타겟 포즈와 강성 계수를 기반으로 토크를 생성합니다.
*   **장점**: 제어 루프의 최하단에 접근할수 있어 물리적 정밀도가 높고, 생체역학적 특성(Co-contraction) 반영이 용이합니다.

### 액션 공간 확장 (Action Space Expansion)
*   **구현**: 기존의 [Joint Target Pose] 액션 뒤에 [Co-contraction Factor, CCF] 액션을 추가하여 액션 공간의 크기를 2배($2N$)로 확장했습니다.
*   **범위**: CCF는 -1.0 ~ 1.0 범위로 출력되며, 내부적으로 Baseline Gain을 조절하는 스케일링 팩터로 변환됩니다.

### 커리큘럼 학습 전략 (Curriculum Strategy)
*   **Stage 1 (현 상태)**: 기초 보행 안정성 확보를 위해 CCF를 0으로 고정(Masking)한 상태에서 학습을 진행했습니다.
*   **Stage 2 (예정)**: CCF를 활성화하여 에이전트가 변동하는 외란이나 보행 상태에 따라 스스로 적절한 강성을 선택하도록 유도합니다.

## 2. 현재 구현 상태 (Current Implementation)

*   **학습 데이터**: `VIC01` 실험이 15,000 에폭까지 완료되었으며, 보행 지표(Mean Reward ~200-400)가 안정적으로 수렴했습니다.
*   **평가 결과**: 시각화(GUI) 확인 결과, 확장된 액션 공간에서도 기존 보행 수준의 안정성을 유지하며 전방 보행을 수행합니다.
*   **설정 파일**:
    *   `phc/data/cfg/env/env_im_walk_vic.yaml`: VIC 관련 파라미터 제어.
    *   `phc/data/cfg/learning/im_walk_vic.yaml`: 학습 알고리즘 설정.

## 3. 미해결 이슈 및 주의사항 (Issues & Constraints)

### 시각화 세그멘테이션 폴트 (Visualization Bug)
*   **증상**: `--headless False` 또는 일반 실행 시 가상 디스플레이(Xephyr)와의 GLX 충돌로 인해 `Segmentation fault (core dumped)` 발생.
*   **해결책**: 로컬 실행 시 반드시 `--no_virtual_display` 플래그를 추가하여 실제 디스플레이 장치를 사용하도록 해야 합니다.

### Open3D 렌더러 종속성
*   **이슈**: 메쉬 기반 렌더링(`--render_o3d`) 사용 시 Open3D 라이브러리가 필요하며, 일부 환경에서 드라이버 호환성 문제가 있을 수 있습니다.

### 대사 비용(Metabolic Cost) 정교화
*   **현항**: Stage 2 진입 시 높은 강성 사용을 억제하기 위한 보상 함수(`vic_metabolic_reward_w`)가 코드로만 준비되어 있으며, 실제 학습 효과에 대한 가중치 튜닝이 필요합니다.

## 4. 다음 단계 (Next Steps)

1.  **Stage 2 활성화**: 환경 설정 파일에서 `vic_curriculum_stage: 2`로 변경 후 재학습 진행.
2.  **강인성 테스트**: 보행 도중 일정 주기로 외부 힘(Push)을 가했을 때, 고정 강성 모델(Stage 1) 대비 가변 강성 모델(Stage 2)의 복구 성공률 비교.
3.  **인간-로봇 통합**: 구축된 가변 임피던스 인간 모델에 웨어러블 로봇 모델을 결합하여 협응 에너지 효율 분석.

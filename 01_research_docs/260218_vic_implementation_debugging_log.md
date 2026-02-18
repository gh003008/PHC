# VIC 구현 디버깅 로그 및 현재 상태 보고

## 1. 개요
본 문서는 2026년 2월 18일 진행된 Variable Impedance Control (VIC) 구현 과정에서 발생한 주요 이슈와 해결 방법, 그리고 현재 시스템의 상태를 정리합니다.

## 2. 주요 발생 이슈 및 해결 과정 (Debugging Log)

### 2.1. 설정 파일 (Config) 관련 이슈
| 이슈 현상 | 원인 분석 | 해결 방법 |
|---|---|---|
| `TypeError: 'str' object does not support item assignment` | `env_im_mcp_vic.yaml` 내 `task` 키가 존재하여 `cfg['task']`가 문자열로 파싱됨 (딕셔너리 기대) | Config 파일에서 `task` 키 제거 (Argparse로 전달됨) |
| `KeyError: 'env'`, `KeyError: 'numEnvs'` | 1. Config 파일 구조가 Flat하여 `env` 키 부재<br>2. 코드 내 변수명 불일치 (SnakeCase vs CamelCase) | 1. `env` 블록 생성 및 설정 이동<br>2. `numEnvs`, `episodeLength` 등 CamelCase 키 추가 |
| `AttributeError: 'dict' object has no attribute 'sim'` | `config.py`가 `dict`를 반환하나 코드는 dot notation 사용 | `load_cfg` 함수 반환 시 `EasyDict`로 래핑하도록 수정 |
| `yaml.scanner.ScannerError` | `control` 블록을 `sim` 블록 내부에 잘못된 인덴트로 삽입 | `control` 블록을 최상위 레벨로 올바르게 이동 |
| `AttributeError` 다수 (`disable_multiprocessing`, `domain_rand`, `step_dt`) | 필수 설정 Key 누락 | Config 파일에 해당 키 (`disable_multiprocessing: False`, `domain_rand` 블록, `step_dt`) 추가 |

### 2.2. 코드/로직 관련 이슈
| 이슈 현상 | 원인 분석 | 해결 방법 |
|---|---|---|
| `AttributeError: ... 'p_gains'` | `isaac_pd` 모드 사용 시 Base Class가 `p_gains`를 초기화하지 않음 | `_setup_character_props`에서 `p_gains`, `d_gains`가 없을 경우 기본값(kp=300, kd=30)으로 초기화하는 로직 추가 |
| `AttributeError: ... 'num_dof'` | 초기화 시점 차이로 `self.num_dof` 접근 불가 | `self._dof_size` (Base Class에서 설정됨)를 사용하도록 변경 |
| Stiffness 제어 불가 | `isaac_pd` 모드는 시뮬레이션 내부 PD 제어기를 사용하여 매 스텝 Gain 변경이 어려움 | `control_mode`를 `pd`로 변경하고, Python 레벨에서 Torque를 직접 계산하여 Gain 변조 적용 |

## 3. VIC 구현 로직 (Code Implementation)
- **`HumanoidImMCPVIC` 클래스 구현**:
    - `HumanoidImMCP`를 상속받아 구현 (현재는 임시로 VIC 로직 중복 포함).
    - **Action Space 확장**: 기존 Kinematic Action + Stiffness Action (DoF x 2).
    - **Physics Step 수정**: 
        - `learn_stiffness=False` (Stage 1): Stiffness Action 무시, 기본 Gain 사용.
        - `learn_stiffness=True` (Stage 2): Policy가 출력한 Stiffness Action을 [Lower, Upper] 범위로 매핑하여 PD Gain 조절.
    - **Metabolic Reward**: 과도한 Stiffness 사용 억제를 위한 보상 함수 추가.

## 4. 현재 상태 (Current Status)
- **Stage 1 (Kinematic Adaptation) 학습 시작됨**:
    - 현재 Stiffness Learning은 비활성화 (`learn_stiffness=False`) 상태.
    - 기본 Kinematics 학습이 정상적으로 돌아가는지 확인하는 단계.
    - WandB 프로젝트: `PHC_VIC`
- **알려진 문제점**:
    - `HumanoidImMCPVIC` 클래스에 `HumanoidVIC`의 로직이 중복되어 있음 (상속 구조 문제).

## 5. 향후 계획 (Next Steps)
1. **코드 리팩토링 (상속 구조 개선)**:
    - User 요청 사항 반영: `HumanoidVIC`가 `HumanoidIm`을 상속받도록 구조 변경.
    - `HumanoidImMCP`가 `HumanoidVIC`를 상속받거나, `HumanoidImMCPVIC`가 다중 상속/Mixin을 통해 중복 코드 제거.
2. **Stage 1 학습 완료 후 Stage 2 전환**:
    - Kinematic 성능 검증 후 `learn_stiffness=True` 활성화 및 `metabolic_cost_w` 튜닝.
    - Curriculum Learning 진행.

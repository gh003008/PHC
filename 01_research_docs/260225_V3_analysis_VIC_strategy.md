# 260225 V3 학습 분석 및 VIC 전략 제안

## 1. V3 학습 곡선 및 에폭(Epoch) 단축 제안
- **현상 분석**: V1, V2 및 현재 진행 중인 V3 모두 10,000 에폭 부근에서 보상이 수렴하며 곡선 형태가 유사함.
- **의견**: 
    - **15,000 에폭 제안**: 동의합니다. 단일 모션(Walking) 학습의 경우 10k~15k 사이에서 물리적 안정성이 확보된 후에는 미세한 지터 개선만 이루어지는 경우가 많습니다. 빠른 프로토타이핑을 위해 15k를 'Early Exit' 지점으로 설정하는 것이 효율적입니다.
    - **Step-jump 가능성**: Discriminator가 갑자기 새로운 '부자연스러움'을 찾아내어 Policy가 급격히 적응해야 하는 특수 상황이 아닌 한, 단일 모션에서는 후반부의 드라마틱한 계단식 상승은 드뭅니다.

## 2. VIC (Variable Impedance Control) 구현 원리
- **매커니즘**: 사용자님이 예상하신 대로 **환경(Environment)의 PD 제어기** 단에서 구현됩니다.
- **수식 관계**:
    - $\tau = (K_{p, base} \cdot \text{scale}) (q_{target} - q) - (K_{d, base} \cdot \text{scale}) \dot{q}$
    - 여기서 `scale`은 Policy의 Action 결과값(CCF) 중 하나입니다.
- **코드 구조**: `humanoid_vic.py` 등에 이미 기반 로직이 있으며, `_physics_step` 이전에 `actions`에서 CCF 부분을 분리하여 `p_gains`, `d_gains` 텐서를 업데이트한 뒤 시뮬레이션을 돌리는 방식입니다.

## 3. Action Space 확장에 따른 학습 기법
Action Space가 2배(Target Pos + CCF)로 늘어날 때 고려할 기술적 장치들입니다.

### 3.1. CCF 그룹화 (Group-based CCF)
- 모든 31개 조인트에 각각 CCF를 주는 대신, **주요 부위별로 그룹화**하여 차원을 줄입니다.
    - 예: `[왼다리, 오른다리, 몸통, 팔]` 총 4~6개의 CCF만 출력.
    - 장점: Action Space 증가폭을 최소화하면서도 부위별 co-contraction 효과를 낼 수 있음.

### 3.2. Stiffness Penalty (Metabolic Cost)
- 불필요하게 높은 강성(Stiffness)을 유지는 것에 대해 리워드 감점을 줍니다.
- $R_{stiff} = -w \cdot \sum(CCF^2)$
- 장점: 에너지 효율적인(최소한의 힘으로 걷는) 동작을 유도하며, 학습 초기 탐색 범위를 안정적으로 가이드함.

### 3.3. Architecture Separation (Dual Heads)
- 신경망의 마지막 레이어에서 Kinematics(위치)와 Dynamics(강성)를 출력하는 **Head를 분리**합니다.
- 장점: 위치 추종 특성과 강성 조절 특성이 서로 간섭하지 않고 독립적으로 최적화될 수 있도록 돕습니다.

## 4. 결론 및 향후 계획
1. V3는 30k까지 완주하되(현재 23k 이상), 다음 실험부터는 **15k에서 중단 및 평가**를 기본으로 함.
2. 다음 단계로 **VariableImpedanceController**가 적용된 `HumanoidVIC` 환경을 활성화하여 CCF 학습을 시작.
3. 우선은 모든 조인트가 아닌 **Limb-group CCF**로 시작하여 학습 난이도를 조절할 것을 권장.

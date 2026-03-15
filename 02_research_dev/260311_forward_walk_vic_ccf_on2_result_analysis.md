# VIC_CCF_ON2 결과 분석 (260311)

## 1. 실험 설정

VIC_CCF_ON (CCF sigma=-2.9, std=0.055로 CCF 미학습)의 후속 실험.

변경 사항 (VIC_CCF_ON 대비):
- CCF 8개 그룹에 별도 sigma=-1.0 (std=0.37) 적용 → 탐색 범위 ±0.74로 확대
- PD target(69 dims) sigma=-2.9 유지
- 그 외 (커리큘럼, max_epochs) 동일 유지

유지 사항:
- vic_curriculum_stage: 2 (CCF 학습 활성화)
- reward_curriculum_switch_epoch: 10000
- max_epochs: 20000
- **Phase obs: 미적용** (obs에 phase 정보 없음. CCF는 순전히 reference motion obs로만 학습)

---

## 2. 정량적 평가

평가 환경: num_envs=1, test 모드, 최종 체크포인트 (20000 에폭)

| 지표 | VIC_CCF_ON2 (20k) | VIC_CCF_ON (20k) | VIC11 (25k) |
| :--- | :--- | :--- | :--- |
| 평가 Avg Reward | **939.51** | 945.30 | 947.11 |
| 평가 Avg Steps | **297.39** | 297.98 | 300.1 |

성능은 사실상 동일. CCF sigma 확대가 성능을 손상시키지 않으면서 CCF 학습을 유도했다.

---

## 3. CCF 학습 결과: 체크포인트 mu bias 분석

최종 체크포인트(20k)의 policy mu bias에서 CCF 8개 그룹 직접 추출:

| 그룹 | CCF bias | impedance_scale | VIC_CCF_ON 대비 |
| :--- | :--- | :--- | :--- |
| G0 L_Hip | +0.2385 | **1.180x** | 0.003 → 0.180 (59배) |
| G1 L_Knee | -0.0974 | **0.935x** | -0.028 → -0.065 |
| G2 L_Ankle+Toe | +0.3694 | **1.292x** | +0.040 → +0.292 |
| G3 R_Hip | +0.0638 | **1.045x** | +0.008 → +0.045 |
| G4 R_Knee | -0.1408 | **0.907x** | -0.018 → -0.093 |
| G5 R_Ankle+Toe | +0.3601 | **1.283x** | +0.017 → +0.283 |
| G6 Upper-L | -0.3213 | **0.800x** | -0.029 → -0.200 |
| G7 Upper-R | -0.2194 | **0.859x** | +0.006 → -0.141 |

VIC_CCF_ON에서는 모두 ≈1.0x였지만, CCF_ON2에서는 명확한 분화가 일어났다.

---

## 4. Phase별 임피던스 분석

평가 598 스텝 기록 후 phase 20 구간으로 binning.
그래프: `output/phase_ccf_log_plot.png`

> 주의: 현재 모션은 "멈춤→4걸음→멈춤"의 단발 walk-stop 시퀀스이며,
> phase=0은 정지 자세, phase≈0.15~0.85가 실제 보행 구간이다.
> 이는 주기적 보행(cyclic gait)의 stance/swing cycle과 다르다.

| Group | 초반(0-20%) | 중반(40-60%) | 후반(80-100%) | 전체평균 |
| :--- | :--- | :--- | :--- | :--- |
| L_Hip | 1.027 | **1.367** | 0.972 | 1.187 |
| L_Knee | 0.584 | 0.703 | 0.712 | 0.682 |
| L_Ankle+Toe | 1.018 | 1.241 | **1.312** | 1.278 |
| R_Hip | 0.700 | **0.956** | 0.995 | 0.953 |
| R_Knee | 0.618 | 0.710 | 0.658 | 0.688 |
| R_Ankle+Toe | **1.349** | **1.517** | **1.554** | **1.494** |
| Upper-L | 0.578 | 0.651 | 0.620 | 0.629 |
| Upper-R | 0.621 | 0.702 | 0.657 | 0.668 |

---

## 5. 선행 연구와 비교

### 5-1. 전체 패턴 일치 여부

전반적인 계층 구조 `발목 > 엉덩이 > 무릎 ≈ 상체`가 문헌과 일치한다.

| 관절 | 본 실험 (평균 scale) | 문헌 기대치 | 일치 여부 |
| :--- | :--- | :--- | :--- |
| 발목 | 1.28~1.49x (높음) | 하지 관절 중 가장 높음, stance peak에서 기저치 대비 15~20배 상승 | ✅ |
| 무릎 | 0.68~0.69x (낮음) | swing≈0 Nm/rad, mid-stance에서 수동 잠금 → 평균적으로 낮음 | ✅ |
| 엉덩이 | 0.95~1.19x (중간) | 보행 전반에 걸쳐 균형 제어 → 중간 수준 | ✅ |
| 상체 | 0.63~0.67x (낮음) | 보행 중 near-passive, 체중 부하 없음 | ✅ |

### 5-2. Stance/Swing별 기대 패턴과의 비교

문헌(Rouse 2014, Lee 2016, Sartori 2015)에서 cyclic gait 기준 기대 패턴:

**발목:**
- Stance 후반(push-off, 40~60% gait cycle): stiffness 최대 (~6.5 Nm/rad/kg)
- Swing: stiffness ≈ 0 (stance 대비 15~20배 하락)
- → 본 실험: R_Ankle이 초반(1.35x)부터 후반(1.55x)까지 지속 상승. 보행 중 점진적 stiffness 증가 패턴과 부합.

**무릎:**
- Loading response(0~12%): 짧은 co-contraction 피크 (CCR=0.62, Sartori 2015)
- Mid-stance(12~40%): 수동 잠금 → stiffness 최저
- Swing: stiffness ≈ 0~3.5 Nm/rad
- → 본 실험: L/R_Knee 모두 0.58~0.71x로 일관되게 낮음. 보행 전반에 걸친 무릎 낮은 stiffness와 부합.

**엉덩이:**
- Swing(60~100%): 34~66 Nm/rad (무릎보다 10~20배 높음)
- Stance 전반: 균형 제어를 위한 중간 수준 유지
- → 본 실험: L_Hip 중반부 1.367x 상승이 보행 stride 중의 균형 제어와 해석 가능.

### 5-3. Stance/Swing 비대칭 미포착의 이유

**현재 분석의 한계**: phase는 모션 클립 내 시간적 위치이며, 어느 발이 Stance이고 어느 발이 Swing인지 직접 알 수 없다. Phase 0→1이 한 보행 사이클(우발 stance → 좌발 stance)에 해당하지 않고 walk-stop 시퀀스 전체에 해당하기 때문이다.

Stance/Swing별 임피던스 분화를 확인하려면:
- Contact force 정보를 추가로 로깅하여 각 스텝에서 어느 발이 지면에 닿아있는지 기록
- 또는 주기적 보행 모션(cyclic walk loop)으로 모션 파일 교체

**좌우 비대칭에 대해**: R_Ankle(1.49x) > L_Ankle(1.28x)의 차이(약 16%)는 문헌(Schache 2014)에서 정상인의 10~15% 비대칭이 일반적임을 감안하면 생리학적으로 수용 가능한 범위이다. 우세발 역할 차이의 가능성도 있으나 검증 필요.

---

## 6. 무릎 잠금(Knee Locking) 모델 구현 여부

### SMPL humanoid.xml 관절 범위

```
L/R_Knee_x (굴곡/신전): range [0.0, 145.0]도
L/R_Knee_y, z (외/내반, 회전): range [-5.625, +5.625]도
```

### 결론: Hard Stop만 있고 Progressive Locking 없음

실제 인체의 무릎 잠금(screw-home mechanism):
- 무릎이 완전 신전(0°)에 가까워질수록 경골이 외회전하며 관절면이 close-packed 상태가 됨
- 이는 **점진적으로 stiffness가 증가**하는 passive 메커니즘

SMPL 모델의 구현:
- `stiffness="0"`: 관절 내 스프링 stiffness 없음
- Range limit을 벗어나면 PhysX가 **Hard constraint**(급격한 반발력)으로 처리
- 범위 내에서는 stiffness 없음 → 진정한 screw-home locking 없음

따라서 현재 모델에서는:
- 무릎 과신전 방지는 구현됨 (0° hard stop)
- Mid-stance에서의 점진적 stiffness 증가는 미구현
- CCF가 무릎의 낮은 impedance를 학습한 것은 "locking이 없어서 PD control이 주로 담당"하는 상황을 반영할 수 있음

---

## 7. 종합 해석

CCF_ON2에서 나타난 임피던스 패턴은 선행 연구의 생체역학적 데이터와 **질적으로 일치**한다.

강점:
- 발목 stiffness 상승, 무릎 stiffness 하강, 상체 stiffness 하강이 자발적으로 학습됨
- Phase obs 없이도(모션 reference obs만으로) 이 패턴이 나타남 → VIC가 유의미하게 작동하고 있다는 증거

한계:
- Stance/Swing 개별 phase별 분화는 현재 분석 방법으로 확인 불가
- 모션 파일이 cyclic gait가 아닌 walk-stop이라 보행 사이클 정렬 어려움
- Knee locking이 모델에 미구현되어 생물학적 무릎 거동과 완전 일치하지 않음

---

## 8. 다음 실험 방향

**VIC_PHASE**: Phase obs (sin/cos(2π×phase)) +2 dims 추가
- 에이전트가 cycle 경계를 사전에 인지하여 사이클 전환 부자연스러움 완화
- phase별 CCF 분화가 더 명확해질 것으로 기대 (에이전트가 "지금이 어느 phase"를 알면 임피던스를 phase-dependent하게 조절 가능)

**Contact-aware CCF 분석**: 지면 접촉 여부를 함께 로깅하면 실제 stance leg vs. swing leg 임피던스 비교 가능

**참고 문헌**:
- Rouse et al. (2014) / Lee et al. (2016): ankle impedance measurement. PMC5823694, PMC5067112
- Sartori et al. (2015): ankle/knee stiffness with CCR. PMC4620138
- Vlutters et al. (2022): hip/knee swing-phase impedance. PubMed 35503817
- Shamaei et al. (2013): knee stance quasi-stiffness. PMC3606171
- Schache et al. (2014): L/R asymmetry in healthy gait. PMC4267535
- Winter (2009): Biomechanics and Motor Control of Human Movement
